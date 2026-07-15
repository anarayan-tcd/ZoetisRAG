# sku_examples_builder.py
#
# Builds the "sku_examples" ChromaDB collection: the bridge between the two
# ID worlds this system straddles —
#   * Labelled.csv  — messy, real historical vet notes, but the ONLY source
#     of the real ProductId / ProductVariantId / SkuId numbers. Noisy, but
#     it is the source of truth for which id-triplets actually exist.
#   * vet_products  — clean, Gemini-researched product catalogue, but with
#     no id numbers of its own.
#
# TRIPLET-FIRST design (rebuilt from the older brand-first version, which
# grouped brand -> sku -> variant and silently dropped ~1,400 triplets):
#
#   1. Enumerate EVERY unique (ProductId, ProductVariantId, SkuId) triplet in
#      Labelled.csv — nulls kept as the sentinel 0 — so no real id-triplet
#      is ever skipped. Labelled.csv is the source of truth.
#   2. For each triplet, take its cleanest / most representative notes and
#      match it to the Gemini vet_products catalogue:
#        - embedding cosine >= CONFIDENT_SIM  -> confident direct match
#          (a real product name that lines up with the catalogue).
#        - otherwise, if the triplet is legitimate enough to bother with
#          (occurs >= LLM_MIN_COUNT times), it goes to a per-brand DeepSeek
#          batch call that assigns each unresolved note-cluster to a
#          catalogue variant (or "none").
#        - rare + low-similarity triplets (typos, internal codes, admin
#          notes like "Dummy Code", "chiropractic consult") are kept — their
#          notes are still stored so the source of truth is never thinned —
#          but left UNLINKED, with no catalogue variant and no probability.
#   3. A triplet may legitimately map to MORE THAN ONE catalogue variant
#      (n:m — e.g. the same historical id across regional product names), so
#      co-matches within CO_MATCH_MARGIN of the best are also linked.
#   4. Occurrence probabilities: within each catalogue variant's group of
#      mapped triplets, each triplet's probability = its historical row count
#      / the group's total. So a variant backed by triplets at 55% / 40% / 5%
#      carries all three, weighted — Phase 2/3b (and the DeepSeek judge) use
#      these to pick the most likely id-triplet for a confirmed match.
#
# Rebuild-safe: the whole thing is deterministic from Labelled.csv except the
# per-brand LLM batch pass, which is checkpointed to disk so a re-run resumes
# instead of re-spending credits. Builds into TARGET_COLLECTION (a v2 name by
# default) so the live collection keeps serving until a validated swap.

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# sku_examples_builder.py lives in backend/knowledge/ but chromadb_client.py
# lives in backend/ — add the parent directory to sys.path so it can be
# imported without turning this into a full installable package.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from chromadb_client import (
    get_embedding_model, get_collection,
    VET_PRODUCTS_COLLECTION, SKU_EXAMPLES_COLLECTION,
)

# Some brand names in brand_seed_list.csv are non-Latin script (e.g.
# Japanese-market products). Windows' console defaults to cp1252, which
# can't encode them — reconfigure stdout so printing a brand name never
# crashes the run partway through.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

LABELLED_CSV = BACKEND_ROOT / "data" / "raw" / "Labelled.csv"
BRAND_SEED_CSV = BACKEND_ROOT / "data" / "generated" / "brand_seed_list.csv"
LLM_CHECKPOINT = BACKEND_ROOT / "data" / "generated" / "sku_examples_llm_checkpoint.json"

# Build into a v2 collection so the live "sku_examples" keeps serving until a
# validated swap. Override with the SKU_EXAMPLES_TARGET env var (set it to
# SKU_EXAMPLES_COLLECTION's value to build in place).
TARGET_COLLECTION = os.getenv("SKU_EXAMPLES_TARGET", SKU_EXAMPLES_COLLECTION + "_v2")

# How many representative Notes to store per triplet.
EXAMPLES_PER_TRIPLET = 5

# Matching thresholds (tuned from a dry-run over all triplets). Embedding
# similarity reliably identifies the brand/product LINE but NOT the exact
# size/pack — within one brand every variant shares most of its words, so the
# discriminating detail (weight band, pipette count) is low-weight in the
# embedding. So embeddings are only trusted when a note is essentially a
# VERBATIM copy of the catalogue name (very high sim); everything legitimate
# below that goes to the LLM, which can actually read "4-10kg 4Pip" and pick
# the 4-10kg variant. n:m regional aliasing is left entirely to the LLM too.
CONFIDENT_SIM = 0.80        # embedding cosine at/above this = verbatim-confident match
LLM_MIN_COUNT = 3           # only triplets seen >= this many times are worth an LLM call
LLM_BATCH_SIZE = 20         # unresolved note-clusters per DeepSeek batch call
MAX_WORKERS = 6             # parallel DeepSeek batch calls

# ─────────────────────────────────────────────────────────────
# DEEPSEEK CLIENT (self-contained, mirrors extract_brands.py — keeps the
# knowledge/ scripts independent of the query-time pipeline package)
# ─────────────────────────────────────────────────────────────

load_dotenv(dotenv_path=BACKEND_ROOT / "APIKeys.env")
_DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
if not _DEEPSEEK_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY in APIKeys.env")
_client_llm = OpenAI(api_key=_DEEPSEEK_KEY, base_url="https://api.deepseek.com")


def deepseek_json(prompt: str, attempts: int = 3) -> dict:
    """Calls DeepSeek and parses its JSON reply, retrying on empty/malformed
    responses. Returns {} if it never parses (the caller treats that batch as
    'no assignments' rather than crashing the whole build)."""
    last = None
    for _ in range(attempts):
        resp = _client_llm.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            last = "empty response"
            continue
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            last = e
    print(f"  (DeepSeek batch returned no valid JSON after {attempts} attempts: {last})")
    return {}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def parse_id_list(raw) -> list:
    if raw is None or pd.isna(raw):
        return []
    return [int(x) for x in str(raw).split(";") if x.strip().isdigit()]


def representative_notes(notes: pd.Series, n: int = EXAMPLES_PER_TRIPLET) -> list:
    """The most representative Notes for a triplet: the most frequent entries
    (the most common way vets actually wrote it), length-filtered and
    deduplicated case-insensitively. Falls back to the raw notes if the
    length filter removes everything (so a triplet is never left with zero
    examples just because its phrasings are all short/long)."""
    clean = notes.dropna().astype(str)
    sane = clean[clean.str.len().between(5, 120)]
    pool = sane if not sane.empty else clean
    if pool.empty:
        return []

    seen, results = set(), []
    for note in pool.value_counts().head(n * 3).index.tolist():
        key = note.lower().strip()
        if key not in seen:
            seen.add(key)
            results.append(note.strip())
        if len(results) >= n:
            break
    return results


# ─────────────────────────────────────────────────────────────
# STEP 1 — enumerate every id-triplet in Labelled.csv
# ─────────────────────────────────────────────────────────────

def enumerate_triplets(df: pd.DataFrame, pid_to_brand: dict) -> list:
    """One record per unique (ProductId, ProductVariantId, SkuId), nulls kept
    as the sentinel 0. Skips triplets whose ProductId isn't a real brand in
    the seed list (a handful of admin/diagnostic ProductIds)."""
    df = df.copy()
    df["ProductId"] = df["ProductId"].astype(int)
    df["ProductVariantId"] = df["ProductVariantId"].fillna(0).astype(int)
    df["SkuId"] = df["SkuId"].fillna(0).astype(int)

    records = []
    for (pid, vid, sid), sub in df.groupby(["ProductId", "ProductVariantId", "SkuId"]):
        brand = pid_to_brand.get(pid)
        if not brand:
            continue

        pack_series = sub["PackSize"].dropna().astype(str) if "PackSize" in sub else pd.Series([], dtype=str)
        pack_size = pack_series.value_counts().index[0] if not pack_series.empty else ""

        records.append({
            "brand": brand,
            "product_id": pid,
            "product_variant_id": vid,
            "sku_id": sid,
            "pack_size": pack_size,
            "occurrence_count": len(sub),
            "notes": representative_notes(sub["Notes"]),
            # filled in by the matching stage:
            "gemini_variants": [],          # list of matched vet_products variant_names (n:m)
            "match_method": "unmatched",
        })
    return records


# ─────────────────────────────────────────────────────────────
# STEP 2a — embedding match against the Gemini catalogue
# ─────────────────────────────────────────────────────────────

def build_brand_variant_index(embedding_model) -> dict:
    """brand -> (variant_names, normalized embedding matrix) for every
    vet_products variant, embedded once."""
    import numpy as np

    vp = get_collection(VET_PRODUCTS_COLLECTION)
    data = vp.get(include=["documents", "metadatas"])

    per_brand = {}
    for doc, meta in zip(data["documents"], data["metadatas"]):
        per_brand.setdefault(meta.get("brand"), {"names": [], "docs": []})
        per_brand[meta.get("brand")]["names"].append(meta.get("variant_name", ""))
        per_brand[meta.get("brand")]["docs"].append(doc)

    index = {}
    for brand, d in per_brand.items():
        embs = np.asarray(embedding_model.encode(d["docs"], normalize_embeddings=True))
        index[brand] = (d["names"], embs)
    return index


def embedding_match(records: list, brand_index: dict, embedding_model) -> list:
    """Assigns confident embedding matches in place; returns the residual
    (legitimate-but-unmatched triplets that need the LLM batch pass)."""
    import numpy as np

    matchable = [r for r in records if r["brand"] in brand_index and r["notes"]]
    note_embs = np.asarray(
        embedding_model.encode([r["notes"][0] for r in matchable], normalize_embeddings=True)
    )

    residual = []
    for r, ne in zip(matchable, note_embs):
        names, vembs = brand_index[r["brand"]]
        sims = vembs @ ne
        best = int(np.argmax(sims))
        best_sim = float(sims[best])

        if best_sim >= CONFIDENT_SIM:
            # verbatim-confident: link the single best variant only. Sibling
            # sizes sit just below in similarity, so co-linking them here would
            # wrongly attach a note to the wrong pack — n:m is the LLM's job.
            r["gemini_variants"] = [names[best]]
            r["match_method"] = "embedding"
        elif r["occurrence_count"] >= LLM_MIN_COUNT:
            residual.append(r)
        # else: rare + low-sim -> left unmatched (notes still stored later)

    return residual


# ─────────────────────────────────────────────────────────────
# STEP 2b — DeepSeek batch pass for the residual, per brand
# ─────────────────────────────────────────────────────────────

LLM_BATCH_PROMPT = """
### ROLE
You are matching messy historical veterinary prescription note-clusters to a
clean product catalogue for ONE brand. Each note-cluster is how vets actually
typed one specific historical product/pack; your job is to say which catalogue
variant (if any) each cluster refers to.

### BRAND
__BRAND__

### CATALOGUE VARIANTS (the only valid targets)
__VARIANTS__

### NOTE CLUSTERS TO ASSIGN
__CLUSTERS__

### RULES
1. For each cluster number, return the catalogue variant number it best
   matches, based on product line, species, weight/size band, and pack count.
2. If the note gives some details but omits others — e.g. it states the
   weight band but not the pack count, or names the product line but not the
   size — STILL match it to the closest-fitting variant of that line/size.
   Pick the single best variant; do NOT return 0 just because a pack count or
   minor detail is missing. Under-specified but clearly-real products must be
   matched, not dropped.
3. Return 0 ONLY when the cluster is not a real product from this catalogue
   at all — an internal code ("TROCO6"), an admin/reminder/booking note
   ("chiropractic consult", "Dummy Code", "stop reminders", "Procedure"), or
   text with no plausible variant. 0 is for genuine junk, never for a real
   product that merely lacks a detail.
4. A cluster may match more than one variant ONLY if they are genuinely the
   same product under different regional names — then list those numbers.
   Otherwise give exactly one number.
5. Use ONLY the variant numbers shown. Never invent a number.

### OUTPUT FORMAT (STRICT JSON ONLY)
{"assignments": {"1": [variant_numbers], "2": [variant_numbers], ...}}
Every cluster number must appear. Use [0] for genuine "no match / junk".
"""


def _run_llm_batch(brand: str, variant_names: list, batch: list) -> dict:
    """One DeepSeek call for a chunk of a brand's residual clusters. Returns
    {cluster_local_index: [matched variant_name, ...]}."""
    variants_block = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(variant_names))
    clusters_block = "\n".join(
        f"{i + 1}. " + " | ".join(f'"{n}"' for n in r["notes"][:3])
        for i, r in enumerate(batch)
    )
    prompt = (
        LLM_BATCH_PROMPT
        .replace("__BRAND__", brand)
        .replace("__VARIANTS__", variants_block)
        .replace("__CLUSTERS__", clusters_block)
    )

    result = deepseek_json(prompt)
    assignments = result.get("assignments", {}) if isinstance(result, dict) else {}

    resolved = {}
    for i in range(len(batch)):
        raw = assignments.get(str(i + 1), [])
        nums = raw if isinstance(raw, list) else [raw]
        picked = []
        for num in nums:
            try:
                idx = int(num) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(variant_names):
                picked.append(variant_names[idx])
        resolved[i] = sorted(set(picked))
    return resolved


def _batch_key(brand: str, chunk_index: int) -> str:
    return f"{brand}##{chunk_index}"


def llm_residual_pass(residual: list, brand_index: dict):
    """Assigns residual triplets to catalogue variants via per-brand DeepSeek
    batch calls, parallelised and checkpointed so a re-run resumes."""
    # group residual by brand, split into fixed-size chunks
    by_brand = {}
    for r in residual:
        by_brand.setdefault(r["brand"], []).append(r)

    jobs = []   # (brand, chunk_index, [records])
    for brand, recs in by_brand.items():
        for c in range(0, len(recs), LLM_BATCH_SIZE):
            jobs.append((brand, c // LLM_BATCH_SIZE, recs[c:c + LLM_BATCH_SIZE]))

    checkpoint = {}
    if LLM_CHECKPOINT.exists():
        try:
            checkpoint = json.loads(LLM_CHECKPOINT.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            checkpoint = {}

    pending = [j for j in jobs if _batch_key(j[0], j[1]) not in checkpoint]
    print(f"\nLLM residual: {len(residual)} triplet(s) in {len(jobs)} batch(es) "
          f"across {len(by_brand)} brand(s) — {len(jobs) - len(pending)} already checkpointed, "
          f"{len(pending)} to run.")

    def work(job):
        brand, chunk_index, batch = job
        names = brand_index[brand][0]
        resolved = _run_llm_batch(brand, names, batch)
        # serialise by the batch's stable position so the checkpoint can be replayed
        return _batch_key(brand, chunk_index), {str(i): v for i, v in resolved.items()}

    done = 0
    if pending:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(work, j): j for j in pending}
            for fut in as_completed(futures):
                key, resolved = fut.result()
                checkpoint[key] = resolved
                done += 1
                if done % 10 == 0 or done == len(pending):
                    LLM_CHECKPOINT.write_text(json.dumps(checkpoint), encoding="utf-8")
                    print(f"  ...{done}/{len(pending)} batches done")
    LLM_CHECKPOINT.write_text(json.dumps(checkpoint), encoding="utf-8")

    # apply checkpointed assignments back onto the records
    for brand, chunk_index, batch in jobs:
        resolved = checkpoint.get(_batch_key(brand, chunk_index), {})
        for i, r in enumerate(batch):
            variants = resolved.get(str(i), [])
            if variants:
                r["gemini_variants"] = sorted(set(variants))
                r["match_method"] = "llm"


# ─────────────────────────────────────────────────────────────
# STEP 3 — occurrence probabilities within each catalogue variant group
# ─────────────────────────────────────────────────────────────

def compute_probabilities(records: list) -> dict:
    """Returns {(brand, variant_name): {triplet_key: probability}} where a
    triplet's probability is its historical row count as a share of every
    triplet mapped to that same catalogue variant."""
    from collections import defaultdict

    group_totals = defaultdict(int)
    for r in records:
        for v in r["gemini_variants"]:
            group_totals[(r["brand"], v)] += r["occurrence_count"]

    probs = defaultdict(dict)
    for r in records:
        key = (r["product_id"], r["product_variant_id"], r["sku_id"])
        for v in r["gemini_variants"]:
            total = group_totals[(r["brand"], v)]
            probs[(r["brand"], v)][key] = r["occurrence_count"] / total if total else 0.0
    return probs


# ─────────────────────────────────────────────────────────────
# STEP 4 — store into ChromaDB
# ─────────────────────────────────────────────────────────────

def store(records: list, probs: dict, embedding_model):
    collection = get_collection(TARGET_COLLECTION)

    # Full rebuild: this collection is entirely regenerable from Labelled.csv
    # (+ the checkpointed LLM pass), so wipe any prior v2 build first.
    existing = collection.get(include=[])["ids"]
    if existing:
        collection.delete(ids=existing)
        print(f"Cleared {len(existing)} pre-existing entr(y/ies) from '{TARGET_COLLECTION}'.")

    stored = 0
    for r in records:
        key = (r["product_id"], r["product_variant_id"], r["sku_id"])

        # top matched variant (highest occurrence-probability) is stored as the
        # scalar link; the full n:m list is preserved as JSON for completeness.
        top_variant = ""
        top_prob = 0.0
        for v in r["gemini_variants"]:
            p = probs.get((r["brand"], v), {}).get(key, 0.0)
            if p >= top_prob:
                top_prob, top_variant = p, v

        for i, note in enumerate(r["notes"]):
            # product_id MUST be in the id: a brand with multiple historical
            # ProductIds (16 of them do, e.g. Frontline -> [32, 182]) can
            # have the SAME (variant_id, sku_id) pair — often both 0/0, the
            # "nothing recorded" sentinel — under two different ProductIds.
            # Without product_id here those triplets collide and upsert()
            # silently drops one, which is exactly the data loss this
            # rewrite exists to eliminate.
            entry_id = (
                f"example__{slugify(r['brand'])}__pid{r['product_id']}"
                f"__pv{r['product_variant_id']}__sk{r['sku_id']}__{i}"
            )
            embedding = embedding_model.encode(note, normalize_embeddings=True).tolist()
            collection.upsert(
                documents=[note],
                embeddings=[embedding],
                ids=[entry_id],
                # Plain-Python casts matter: these values arrive as numpy
                # int32/int64 from pandas groupbys, which ChromaDB's
                # metadata validation rejects outright.
                metadatas=[{
                    "brand": str(r["brand"]),
                    "product_id": int(r["product_id"]),
                    "product_variant_id": int(r["product_variant_id"]),
                    "sku_id": int(r["sku_id"]),
                    "pack_size": str(r["pack_size"]),
                    "occurrence_count": int(r["occurrence_count"]),
                    "occurrence_probability": float(round(top_prob, 4)),
                    "gemini_variant": str(top_variant),
                    "gemini_variants": json.dumps(r["gemini_variants"]),
                    "match_method": str(r["match_method"]),
                }],
            )
            stored += 1
    return stored, collection


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def build_sku_examples():
    print("=" * 64)
    print("SKU EXAMPLES BUILDER (triplet-first + hybrid catalogue match)")
    print(f"Target collection: '{TARGET_COLLECTION}'")
    print("=" * 64)

    if not BRAND_SEED_CSV.exists():
        raise FileNotFoundError(f"{BRAND_SEED_CSV} not found — run extract_brands.py first.")

    embedding_model = get_embedding_model()

    seed_df = pd.read_csv(BRAND_SEED_CSV)
    pid_to_brand = {}
    for _, row in seed_df.iterrows():
        brand = str(row["Brand"]).strip()
        for pid in parse_id_list(row.get("SourceProductIds")):
            pid_to_brand[pid] = brand

    df = pd.read_csv(LABELLED_CSV)
    df["Notes"] = df["Notes"].astype(str)

    print("\n[1/4] Enumerating id-triplets...")
    records = enumerate_triplets(df, pid_to_brand)
    print(f"      {len(records)} legitimate triplet(s) enumerated "
          f"(covering brands present in the seed list).")

    print("[2/4] Building catalogue index + embedding match...")
    brand_index = build_brand_variant_index(embedding_model)
    residual = embedding_match(records, brand_index, embedding_model)
    n_embed = sum(1 for r in records if r["match_method"] == "embedding")
    print(f"      {n_embed} confident embedding match(es); {len(residual)} to the LLM pass.")

    print("[3/4] DeepSeek residual batch pass...")
    if residual:
        llm_residual_pass(residual, brand_index)
    n_llm = sum(1 for r in records if r["match_method"] == "llm")
    print(f"      {n_llm} additional match(es) from DeepSeek.")

    probs = compute_probabilities(records)

    print("[4/4] Storing to ChromaDB...")
    stored, collection = store(records, probs, embedding_model)

    linked = sum(1 for r in records if r["gemini_variants"])
    unlinked = len(records) - linked
    print("\n" + "=" * 64)
    print(f"Done. {len(records)} triplet(s): {linked} linked to the catalogue "
          f"({n_embed} embedding + {n_llm} LLM), {unlinked} kept unlinked.")
    print(f"{stored} representative-note document(s) written.")
    print(f"'{TARGET_COLLECTION}' now holds {collection.count()} entr(y/ies).")
    print("=" * 64)


if __name__ == "__main__":
    build_sku_examples()
