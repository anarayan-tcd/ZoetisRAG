# extract_brands.py
#
# Turns the messy, vet-entered `Notes` column in Labelled.csv into a clean,
# deduplicated list of real pet-medicine brands to research (brand_seed_list.csv).
#
# Approach:
#   1. Group rows by ProductId and compute a cheap heuristic brand-token
#      guess per group (used only as a hint, never as the final answer).
#   2. Every group is resolved by DeepSeek, which decides the canonical
#      (possibly multi-word) brand name, whether the group is genuinely a
#      purchasable medicine (vs. lab diagnostics / admin / reminder notes),
#      and a short known-product-line hint for later disambiguation.
#   3. Groups the LLM rejects as non-medicine (or fails to parse twice in a
#      row) are excluded.
#   4. Duplicate brands that ended up under different ProductIds (e.g. a
#      brand split across "Frontline" and "Frontline Combo" groups) are
#      collapsed into one row, with ProductId/ProductVariantId/SkuId sets
#      unioned so the final brand list can later be cross-referenced back
#      to Labelled.csv via any of the three id types.

import json
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import os

# LLM-generated reasons/brand names can occasionally contain non-Latin
# script; Windows' console defaults to cp1252, which can't encode it —
# reconfigure stdout so that never crashes a long run partway through.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────
# ENV SETUP
# ─────────────────────────────────────────────────────────────

BACKEND_ROOT = Path(__file__).resolve().parent.parent

env_path = BACKEND_ROOT / "APIKeys.env"
load_dotenv(dotenv_path=env_path)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError(f"Missing DEEPSEEK_API_KEY in {env_path}")

client_llm = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

INPUT_CSV = BACKEND_ROOT / "data" / "raw" / "Labelled.csv"
OUTPUT_CSV = BACKEND_ROOT / "data" / "generated" / "brand_seed_list.csv"

# Resolved groups are checkpointed here as they complete, so a crash (e.g.
# the final CSV write failing because the file is open elsewhere) doesn't
# throw away 176 DeepSeek calls' worth of work — rerunning resumes instead
# of starting over.
CHECKPOINT_FILE = BACKEND_ROOT / "data" / "generated" / "brand_extraction_checkpoint.json"

# Cap on how many distinct notes get sent to the LLM per group, to keep
# prompts a reasonable size for very large groups.
MAX_NOTES_PER_LLM_CALL = 200

# A parse failure gets one retry before the group is given up on, since
# a single bad/garbled response shouldn't permanently drop a real brand.
MAX_LLM_ATTEMPTS = 2

TOKEN_CLEAN_RE = re.compile(r"[^A-Za-z]+")


# ─────────────────────────────────────────────────────────────
# JSON HELPERS
# ─────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """
    Robustly pulls a JSON object out of an LLM response: strips markdown
    fences if present, and falls back to brace-matching if there's stray
    text around the JSON. Returns {} if nothing parseable is found, so
    callers can distinguish "couldn't parse" from a real decision.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {}


# ─────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Returns {product_id_str: resolved_row_dict} for groups already
    resolved in a previous (possibly crashed) run."""
    if not CHECKPOINT_FILE.exists():
        return {}

    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_checkpoint(checkpoint: dict):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


# ─────────────────────────────────────────────────────────────
# STEP 1 — GROUP ROWS AND COMPUTE A HEURISTIC HINT
# ─────────────────────────────────────────────────────────────

def guess_brand_token(note: str) -> str:
    """
    Extracts a candidate brand token from the start of a messy note,
    e.g. 'FRONTLINE SPOT ON SML DOG 3PK' -> 'FRONTLINE'. Used only as a
    hint for the LLM, never emitted directly as a final brand name, since
    real brand names are often more than one word (e.g. 'Wagg & Purr').
    """
    for raw_token in re.split(r"[\s/]+", str(note).strip()):
        cleaned = TOKEN_CLEAN_RE.sub("", raw_token)
        if cleaned:
            return cleaned.upper()
    return ""


def build_group_records(df: pd.DataFrame) -> list:
    """One record per ProductId group: distinct notes, a heuristic hint,
    and the set of ProductVariantIds/SkuIds seen in that group."""

    records = []

    for product_id, group in df.groupby("ProductId"):

        guess_counts = group["brand_guess"].value_counts()
        heuristic_hint = guess_counts.index[0] if len(guess_counts) else ""

        variant_ids = sorted(
            {str(int(v)) for v in group["ProductVariantId"].dropna().unique()},
            key=int
        )
        sku_ids = sorted(
            {str(int(v)) for v in group["SkuId"].dropna().unique()},
            key=int
        )

        records.append({
            "product_id": product_id,
            "total_rows": len(group),
            "heuristic_hint": heuristic_hint,
            "distinct_notes": group["Notes"].drop_duplicates().tolist(),
            "variant_ids": variant_ids,
            "sku_ids": sku_ids,
        })

    return records


# ─────────────────────────────────────────────────────────────
# STEP 2 — LLM RESOLUTION FOR EVERY GROUP
# ─────────────────────────────────────────────────────────────

# Plain (non f-string) template — see knowledge_agent2.py's
# RESEARCH_PROMPT_TEMPLATE for why: the golden examples and output
# schema below are full of literal JSON braces, which would otherwise
# all need escaping as {{ }} in an f-string.
BRAND_RESOLUTION_PROMPT_TEMPLATE = """
### ROLE
You are cleaning messy, vet-entered prescription notes to identify the
real-world pet-medicine brand a group of similar notes refers to.

### CONTEXT
A vet practice management system grouped these notes together under one
internal product category, but that grouping is unreliable — it can mix
multiple unrelated brands, lab diagnostic tests, or non-medicine admin
notes (appointment reminders, subscription/reminder programs, generic
tasks) under a single category. Your output seeds a brand list that gets
researched via live web search, so a wrong or junk brand here wastes a
real search call downstream, and a missed real brand means it never gets
researched at all.

### INFO PROVIDED
- A simple heuristic's guess at the leading brand token: "__HINT__"
  -> use this only as a hint. It is frequently wrong or incomplete (e.g.
  missing a second word of a multi-word brand name like "Wagg & Purr", or
  entirely unrelated to the real brand).
- Distinct note strings in this group:
__NOTES__

### RULES
1. Identify the single dominant real pet-medicine BRAND name referred to
   by most of these notes. Use the full, correct brand name/phrase — not
   just the first word, if the real brand name has more than one word.
2. Decide if this group is genuinely a purchasable pet medicine or
   parasiticide product — NOT lab diagnostics, NOT an admin/reminder/
   subscription program, NOT an unrelated non-medicine item.
3. A group can contain some noise (a handful of unrelated/admin notes)
   without invalidating it — only reject the WHOLE group if there is no
   single dominant real medicine brand at all.
4. Give a short phrase describing the known product line(s)/variant
   types seen in the notes — this will be used later to disambiguate a
   generic-sounding brand name in a live web search, so keep it short
   but distinguishing.

### GOLDEN EXAMPLES

NOTES: "FRONTLINE SPOT ON SML DOG 3PK", "Frontline Combo Dog 10-20Kg 6's", "FRONTLINE COMBO S/ON XL 40-60K 6PIP"
-> {"dominant_brand": "Frontline", "is_medicine": true, "confidence": "high", "known_product_line": "Spot-On, Combo", "reason": "All notes clearly reference Frontline across multiple product lines and pack sizes."}

NOTES: "Practic 2.4-5kg/3pk", "Practic 11-22kg/6pk", "Advise/Discuss Training/Behaviour - Practice Vet Exam"
-> {"dominant_brand": "Prac-tic", "is_medicine": true, "confidence": "high", "known_product_line": "Spot-on for dogs, multiple size variants", "reason": "Majority of notes clearly refer to Prac-tic; one unrelated admin note doesn't invalidate the group."}
(minority noise doesn't sink an otherwise-clear group)

NOTES: "Idexx Canine SpecPLi advantage (internal or external)", "Idexx chem 17+SDMA combo (1)", "Idexx Protein Electrophoresis Serum or Fluid"
-> {"dominant_brand": "", "is_medicine": false, "confidence": "high", "known_product_line": "", "reason": "All notes are Idexx laboratory diagnostic tests, not a purchasable medicine product."}

NOTES: "Irecall Stop Reminder Group All", "iRecall Stop Parasiticide - All Groups", "Stop Irecall Parasiticide"
-> {"dominant_brand": "", "is_medicine": false, "confidence": "high", "known_product_line": "", "reason": "These are reminder/subscription program messages, not an actual medicine product."}

### OUTPUT FORMAT (STRICT JSON ONLY):
{
  "dominant_brand": "brand name, or empty string if none",
  "is_medicine": true,
  "confidence": "high",
  "known_product_line": "short phrase",
  "reason": "one sentence"
}
"""


def resolve_group_with_llm(record: dict) -> dict:
    """
    Asks DeepSeek to identify the canonical brand in a ProductId group,
    whether it's genuinely a purchasable medicine, and a short known
    product-line hint for later web-search disambiguation.
    """
    sample = record["distinct_notes"][:MAX_NOTES_PER_LLM_CALL]
    notes_block = chr(10).join(f"- {n}" for n in sample)

    prompt = (
        BRAND_RESOLUTION_PROMPT_TEMPLATE
        .replace("__HINT__", record["heuristic_hint"])
        .replace("__NOTES__", notes_block)
    )

    response = client_llm.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content.strip()

    return extract_json(raw)


def resolve_all_groups(records: list) -> list:
    checkpoint = load_checkpoint()

    if checkpoint:
        print(f"Resuming from checkpoint — {len(checkpoint)} group(s) already resolved.\n")

    for i, record in enumerate(records):

        product_id = record["product_id"]
        checkpoint_key = str(product_id)

        if checkpoint_key in checkpoint:
            continue

        print(
            f"[{i + 1}/{len(records)}] ProductId {product_id} "
            f"({record['total_rows']} rows, {len(record['distinct_notes'])} distinct notes, "
            f"hint: '{record['heuristic_hint']}')..."
        )

        result = {}

        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            result = resolve_group_with_llm(record)

            if result:
                break

            print(f"  -> could not parse LLM response (attempt {attempt}/{MAX_LLM_ATTEMPTS})")

        if not result:
            print("  -> excluded: no valid response after retry")
            checkpoint[checkpoint_key] = None
            save_checkpoint(checkpoint)
            continue

        brand = str(result.get("dominant_brand", "")).strip()
        is_medicine = bool(result.get("is_medicine"))

        if brand and is_medicine:
            checkpoint[checkpoint_key] = {
                "Brand": brand.title(),
                "KnownProductLines": str(result.get("known_product_line", "")).strip(),
                "SourceProductIds": str(product_id),
                "SourceProductVariantIds": ";".join(record["variant_ids"]),
                "SourceSkuIds": ";".join(record["sku_ids"]),
                "TotalHistoricalRows": record["total_rows"],
                "Method": f"llm ({result.get('confidence', '?')})",
            }
        else:
            print(f"  -> excluded: {result.get('reason', 'no dominant medicine brand')}")
            checkpoint[checkpoint_key] = None

        save_checkpoint(checkpoint)

    return [row for row in checkpoint.values() if row is not None]


# ─────────────────────────────────────────────────────────────
# STEP 3 — COLLAPSE DUPLICATE BRANDS ACROSS PRODUCT IDS
# ─────────────────────────────────────────────────────────────

def merge_duplicate_brands(rows: list) -> list:
    merged = {}

    for row in rows:
        key = row["Brand"].strip().lower()

        variant_ids = set(filter(None, row["SourceProductVariantIds"].split(";")))
        sku_ids = set(filter(None, row["SourceSkuIds"].split(";")))
        known_line = row["KnownProductLines"]

        if key not in merged:
            merged[key] = {
                "Brand": row["Brand"],
                "KnownProductLines": {known_line} if known_line else set(),
                "SourceProductIds": {row["SourceProductIds"]},
                "SourceProductVariantIds": variant_ids,
                "SourceSkuIds": sku_ids,
                "TotalHistoricalRows": row["TotalHistoricalRows"],
                "Methods": {row["Method"]},
            }
        else:
            entry = merged[key]
            if known_line:
                entry["KnownProductLines"].add(known_line)
            entry["SourceProductIds"].add(row["SourceProductIds"])
            entry["SourceProductVariantIds"].update(variant_ids)
            entry["SourceSkuIds"].update(sku_ids)
            entry["TotalHistoricalRows"] += row["TotalHistoricalRows"]
            entry["Methods"].add(row["Method"])

    final_rows = []

    for entry in merged.values():
        final_rows.append({
            "Brand": entry["Brand"],
            "KnownProductLines": "; ".join(sorted(entry["KnownProductLines"])),
            "SourceProductIds": ";".join(sorted(entry["SourceProductIds"], key=int)),
            "SourceProductVariantIds": ";".join(sorted(entry["SourceProductVariantIds"], key=int)),
            "SourceSkuIds": ";".join(sorted(entry["SourceSkuIds"], key=int)),
            "TotalHistoricalRows": entry["TotalHistoricalRows"],
            "Method": ";".join(sorted(entry["Methods"])),
        })

    final_rows.sort(key=lambda r: -r["TotalHistoricalRows"])

    return final_rows


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def build_brand_seed_list():

    print("=" * 60)
    print("BRAND EXTRACTOR — Labelled.csv -> brand_seed_list.csv")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    df["Notes"] = df["Notes"].astype(str)
    df["brand_guess"] = df["Notes"].apply(guess_brand_token)

    records = build_group_records(df)

    print(f"\n{len(records)} ProductId group(s) to resolve via DeepSeek...\n")

    resolved = resolve_all_groups(records)
    final_rows = merge_duplicate_brands(resolved)

    try:
        pd.DataFrame(final_rows).to_csv(OUTPUT_CSV, index=False)
        output_path = OUTPUT_CSV
    except PermissionError:
        fallback_path = OUTPUT_CSV.with_stem(OUTPUT_CSV.stem + "_new")
        pd.DataFrame(final_rows).to_csv(fallback_path, index=False)
        output_path = fallback_path
        print(
            f"\nWARNING: {OUTPUT_CSV} is open in another program and could not be "
            f"overwritten. Wrote to {fallback_path} instead — close the original "
            f"file and rename/replace it manually."
        )

    print("\n" + "=" * 60)
    print(f"Done! {len(final_rows)} unique brand(s) written to {output_path}")
    print(f"({len(records) - len(resolved)} group(s) excluded as non-medicine or unresolved.)")
    print("Run knowledge_agent2.py next to research and store them.")
    print("=" * 60)


if __name__ == "__main__":
    build_brand_seed_list()
