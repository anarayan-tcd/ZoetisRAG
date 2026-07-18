# structured/phase2_structured.py — product selection for the structured
# pipeline. One DeepSeek call sees the confirmed brand's COMPLETE catalog
# variant list (catalogs top out at ~31 variants/brand, so nothing is ever
# shortlisted away — no vector search, no candidate starvation, none of the
# widening machinery the raw pipeline needs).
#
# This stage answers "WHICH REAL PRODUCT is this?", NOT "which catalog row
# is it?" — deliberately different questions. The same real product is
# routinely listed under several VariantIds (Capstar's 11.4mg small
# dog/cat item exists as VariantId 102 "CAPSTAR TABS 11MG", 465 "Capstar
# tablets Small (Dogs and Cats)", 467 "Exelpet Capstar Cats and Small
# Dogs", 586 and 588 — rebrands, private-label duplicates and catalog
# re-codes accumulated over time). Picking one of those as "the" answer
# means inventing a distinction the catalog doesn't make, and the earlier
# single-pick design demonstrably got it wrong: it chose 465 over the
# 73%-dominant 102 purely because 465 had richer researched text attached.
# So this stage returns EVERY VariantId that is the same real product, and
# a later deterministic step enumerates their ids. What it must still do
# strictly is exclude variants that are genuinely DIFFERENT products —
# Capstar 103/466/468/587 sit right beside those in the same list and are
# the 57mg large-dog item.

from functools import lru_cache

from ..shared.resources import (
    sku_examples_collection, deepseek_json, id_str,
    detect_language_hints, classify_probability_tier,
)
from ..shared.query_utils import append_color_if_missing

MAX_PHRASINGS_PER_VARIANT = 2

NEW_PRODUCT_SENTINEL = "New Product"


# ─────────────────────────────────────────────────────────────
# REAL VET PHRASINGS (labelled.csv, via the sku_examples collection)
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _phrasings_by_variant(product_id: int) -> dict:
    """variant_id -> up to MAX_PHRASINGS_PER_VARIANT real vet-entered
    phrasings, one bulk fetch per ProductId (brand) then grouped in memory
    — not one ChromaDB call per candidate variant."""
    try:
        data = sku_examples_collection.get(
            where={"product_id": int(product_id)},
            include=["documents", "metadatas"],
        )
    except Exception:
        return {}

    grouped = {}
    for doc, meta in zip(data["documents"], data["metadatas"]):
        vid = meta.get("product_variant_id")
        if not vid:
            continue
        notes = grouped.setdefault(int(vid), [])
        if doc not in notes and len(notes) < MAX_PHRASINGS_PER_VARIANT:
            notes.append(doc)
    return grouped


# ─────────────────────────────────────────────────────────────
# CANDIDATE BLOCK
# ─────────────────────────────────────────────────────────────

def _pack_options(variant: dict) -> str:
    if variant.get("is_new_product"):
        return "n/a (not in catalog)"
    opts = []
    for s in variant["skus"]:
        unit = f"{s['pack_units']} units" if s["pack_units"] else ""
        ml = f"{s['ml_volume']:g}ml" if s["ml_volume"] else ""
        label = " / ".join(x for x in (unit, ml) if x) or "unspecified"
        opts.append(label)
    return ", ".join(dict.fromkeys(opts)) or "unspecified"


def build_candidates_block(variants: list) -> str:
    total = sum(v["occurrences"] for v in variants) or 0

    lines = []
    for v in variants:
        vid = v["variant_id"] if v["variant_id"] != "" else NEW_PRODUCT_SENTINEL
        parts = [f'VariantId {vid} | ProductId {v["product_id"] or NEW_PRODUCT_SENTINEL} | "{v["description"]}"']

        if v.get("gemini_name") and v["gemini_name"] != v["description"]:
            parts.append(f'market name: "{v["gemini_name"]}"')
        if v.get("gemini_species"):
            parts.append(f'species: {v["gemini_species"]}')
        if v.get("gemini_region"):
            parts.append(f'region: {v["gemini_region"]}')
        if v.get("gemini_color"):
            parts.append(f'pack color: {v["gemini_color"]}')

        parts.append(f'pack options: {_pack_options(v)}')

        if total and v["occurrences"]:
            parts.append(f'prescribed {v["occurrences"]}x ({round(100 * v["occurrences"] / total)}% of this brand\'s scripts)')
        elif v.get("is_new_product"):
            parts.append("NEW PRODUCT — researched on the market but absent from the catalog (no ids)")

        phrasings = []
        if v["product_id"]:
            phrasings = _phrasings_by_variant(int(v["product_id"])).get(int(v["variant_id"]), [])
        if phrasings:
            parts.append("real vet phrasings: " + "; ".join(f'"{p}"' for p in phrasings))

        lines.append("  - " + "\n    ".join(parts))

    guidance = ""
    shares = sorted(
        ((v["variant_id"], v["occurrences"] / total) for v in variants if total and v["occurrences"]),
        key=lambda x: x[1], reverse=True,
    )
    if shares:
        tier = classify_probability_tier(shares)
        if tier == "dominant":
            vid, p = shares[0]
            guidance = f"\n(Historical signal: VariantId {vid} is this brand's dominant prescription at {round(p * 100)}%.)"
        elif tier == "tied":
            (va, pa), (vb, pb) = shares[0], shares[1]
            guidance = (f"\n(Historical signal: roughly tied between VariantId {va} ({round(pa * 100)}%) "
                        f"and VariantId {vb} ({round(pb * 100)}%).)")

    return "\n".join(lines) + guidance


# ─────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────

PHASE2_STRUCTURED_PROMPT = """
### ROLE
You are a veterinary product matching system working from an AUTHORITATIVE
product catalog. The brand is already confirmed. Below is that brand's
COMPLETE variant list. Identify WHICH REAL PRODUCT the messy vet
prescription refers to, and list EVERY VariantId in the catalog that is
that same real product.

### CONTEXT
This catalog accumulated over years, so ONE real product is often listed
under SEVERAL VariantIds — rebrands, private-label duplicates, re-codes.
Their descriptions differ in wording and in how much detail they carry,
but they are the same physical item. You are NOT being asked to choose
between those; a later step reports all their ids. You ARE being asked to
draw the line between them and the genuinely DIFFERENT products sitting
next to them in the list (a different strength, species, or weight band).

Nothing is missing from the list: if the query names a strength, weight
band, or species and no candidate carries it, that product genuinely does
not exist here — say so honestly rather than force-fitting. Some candidates
are marked "NEW PRODUCT": real market products absent from the catalog,
with no ids; match one ONLY when it clearly fits better than every catalog
variant. Vets use heavy shorthand ("3pk" = 3-pack, "S/on" = Spot On,
"sml/med/lge" sizes) and sometimes other languages.

### INFO PROVIDED
BRAND (confirmed): __BRAND__

USER QUERY:
__QUERY__

MULTILINGUAL TERM HINT (deterministic scan — non-English words found in the
query, with English meanings): __SPECIES_HINT__

CANDIDATE VARIANTS (complete list for this brand):
__CANDIDATES__

### RULES
1. Identify the ONE real product the query refers to. Describe it in the
   product fields (name, species, size, strength). Do NOT put any id in
   these fields.
2. In "variant_ids", list EVERY VariantId from the list above that is that
   SAME real product — same strength, same species, same weight band —
   even where the wording differs a lot ("CAPSTAR TABS 11MG", "Capstar
   tablets Small (Dogs and Cats)" and "Exelpet Capstar Cats and Small
   Dogs" are all the same 11.4mg small dog/cat tablet). Copy the digits
   EXACTLY as printed. Usually this is one id; include several only when
   they genuinely are the same item.
3. NEVER include a variant that differs in any real characteristic —
   different strength, species, or weight band. Capstar's 57mg large-dog
   variants are NOT the 11.4mg small dog/cat variants, no matter how
   similar their names look. When in doubt about two candidates being the
   same item, leave the doubtful one OUT: a wrong id in this list is worse
   than a missing one.
4. Do NOT use description richness to rank or exclude. A candidate whose
   text spells out "11.4mg" is not a better match than one that only says
   "11MG" — a terser description means less research text was attached to
   that row, not a different product. Both belong in "variant_ids".
5. Match on species, weight band, and strength FIRST. A hinted species (see
   MULTILINGUAL TERM HINT) is binding: never pick a dog product for a
   cat-hinted query or vice versa. Pack color is supporting evidence only —
   never decisive against dosage/weight/species.
6. Do NOT decide pack size here — a later stage does, and it applies to
   every variant you list. Ignore pack counts entirely unless they're the
   only thing separating two candidates as real products.
7. "products" = SHORT product line name (1-3 words, no size/species/count);
   "full_product_name" = the clearest complete description of the product
   (prefer a market name when one is shown). They must never be identical.
8. For a NEW PRODUCT match, set "variant_ids" to ["New Product"].
9. "animal_age" is always "". Confidence: High, Medium, Low, or Very Low
   only. Don't fill in "region" — it's filled afterward from catalog data.
10. If nothing plausibly matches, return Very Low confidence, an empty
    "variant_ids", and explain — never force a wrong match.

### GOLDEN EXAMPLES

QUERY: "CAPSTAR TABS 6X11.4MG"
CANDIDATES include:
  - VariantId 102 | "CAPSTAR TABS 11MG" | prescribed 1043x (73%)
  - VariantId 103 | "CAPSTAR TABS 57MG" | prescribed 402x (28%)
  - VariantId 465 | "Capstar tablets Small (Dogs and Cats)" | market name: "...2-25 lbs (1-11 kg) 11.4mg" | prescribed 121x (9%)
  - VariantId 466 | "Capstar tablets Large Dogs" | market name: "...25.1-125 lbs 57mg" | prescribed 81x (6%)
  - VariantId 467 | "Exelpet Capstar Cats and Small Dogs" | prescribed 39x (3%)
  - VariantId 586 | "Capstar Flea Treatment Tablets (2 - 25 lbs) Dog" | prescribed 53x (4%)
  - VariantId 588 | "Capstar Flea Treatment Tablets (All weights) Cat" | prescribed 9x (1%)
-> variant_ids: ["102", "465", "467", "586", "588"], confidence High,
   reason: "Query's 11.4mg is the small dog/cat strength (11mg and 11.4mg are the same real tablet). VariantIds 102, 465, 467, 586 and 588 are all that same item under different catalog listings. 103, 466 and 468 are the 57mg large-dog product and are excluded."
(all five are returned rather than picking the dominant 102 — they are one product, and nothing in the query distinguishes which listing the vet meant. The 57mg ones are excluded because 57mg is a genuinely different strength.)

QUERY: "milbemax chewy dog 12.5/125mg"
CANDIDATES include: VariantId 86 | "MILBEMAX CHEWY DOG 12.5/125MG" | prescribed 1104x (61%), VariantId 87 | "MILBEMAX CHEWY SML DOG 2.5/25MG" | prescribed 702x (39%)
-> variant_ids: ["86"], confidence High,
   reason: "Query names the 12.5/125mg strength exactly, matching VariantId 86. VariantId 87 is the 2.5/25mg small-dog strength — a different product."
(a single id here: the catalog lists this product only once. Its several SkuIds are the next stage's business, not this one's.)

### OUTPUT FORMAT (STRICT JSON ONLY — no id fields inside product_match):
{
  "product_match": {
    "brand": "__BRAND__",
    "products": "",
    "full_product_name": "",
    "species": "",
    "animal_size": "",
    "pack_size": "",
    "pack_quantity": "",
    "animal_age": "",
    "confidence": "",
    "reason": ""
  },
  "variant_ids": [],
  "variant_ids_reason": ""
}
"""


# ─────────────────────────────────────────────────────────────
# CALL + VALIDATION
# ─────────────────────────────────────────────────────────────

_EMPTY_MATCH = {
    "brand": "", "products": "", "full_product_name": "", "region": "",
    "species": "", "animal_size": "", "pack_size": "", "pack_quantity": "",
    "animal_age": "", "confidence": "Low", "reason": "",
}


def _find_new_product(full_product_name: str, new_products_by_name: dict) -> dict:
    """New Product candidates have no id to look up by, so match on the
    name text the LLM was told to copy verbatim — exact first, then a
    loose substring fallback in case it paraphrased slightly."""
    if not full_product_name:
        return {}
    if full_product_name in new_products_by_name:
        return new_products_by_name[full_product_name]

    lowered = full_product_name.lower()
    for name, variant in new_products_by_name.items():
        if name.lower() in lowered or lowered in name.lower():
            return variant
    return {}


def ask_deepseek_phase2_structured(query: str, brand: str, variants: list) -> dict:
    """Returns {"product_match": {...}, "variant_ids": [...],
    "variant_ids_reason": str, "is_new_product": bool}. variant_ids is
    validated against what was actually shown — a hallucinated or
    borrowed id is dropped rather than reported."""
    prompt = (
        PHASE2_STRUCTURED_PROMPT
        .replace("__CANDIDATES__", build_candidates_block(variants))
        .replace("__QUERY__", query)
        .replace("__SPECIES_HINT__", detect_language_hints(query) or "(none detected)")
        .replace("__BRAND__", brand)
    )

    result = deepseek_json(prompt)

    match = dict(_EMPTY_MATCH)
    raw_match = result.get("product_match", {}) or {}
    match.update({k: raw_match.get(k, "") or "" for k in _EMPTY_MATCH})
    match["brand"] = brand

    raw_ids = [str(v).strip() for v in (result.get("variant_ids") or [])]
    reason = str(result.get("variant_ids_reason") or "")

    new_products_by_name = {v["description"]: v for v in variants if v.get("is_new_product")}

    if any(v == NEW_PRODUCT_SENTINEL for v in raw_ids):
        new_product = _find_new_product(match["full_product_name"], new_products_by_name)
        match["region"] = new_product.get("gemini_region", "")
        match["full_product_name"] = append_color_if_missing(
            match["full_product_name"], new_product.get("gemini_color", "")
        )
        return {
            "product_match": match,
            "variant_ids": [],
            "variant_ids_reason": reason,
            "is_new_product": True,
        }

    valid_ids = {str(v["variant_id"]) for v in variants if v["variant_id"] != ""}
    variant_ids = [v for v in raw_ids if v in valid_ids]

    dropped = [v for v in raw_ids if v not in valid_ids]
    if dropped:
        print(f"  Dropped {len(dropped)} hallucinated VariantId(s) for '{query}': {dropped}")
        if match["confidence"] in ("High", "Medium") and not variant_ids:
            match["confidence"] = "Low"

    # Region/color come from the catalog's own data, never from the LLM —
    # same principle as the ids themselves. Taken from the first matched
    # variant that actually HAS each field rather than simply the first:
    # these are duplicate listings of one product, and researched text is
    # attached unevenly across them (Capstar's dominant VariantId 102 has
    # none at all, while its duplicate 465 carries the full region list) —
    # so the product's real region shouldn't vanish on the luck of which
    # listing happened to sort first.
    matched = [v for v in variants if str(v["variant_id"]) in set(variant_ids)]

    if matched:
        match["region"] = next((v["gemini_region"] for v in matched if v.get("gemini_region")), "")
        color = next((v["gemini_color"] for v in matched if v.get("gemini_color")), "")
        match["full_product_name"] = append_color_if_missing(match["full_product_name"], color)

    return {
        "product_match": match,
        "variant_ids": variant_ids,
        "variant_ids_reason": reason,
        "is_new_product": False,
    }
