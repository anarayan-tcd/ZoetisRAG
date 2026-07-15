# phase3b_sku.py — Phase 3a (deterministic) and Phase 3b (focused LLM
# call) of the query-time matching pipeline: resolve the exact pack size
# (SkuId) for a match whose product_id/product_variant_id have already
# been assigned by Phase 2. Phase 3a is just a lookup — if the confirmed
# variant has only one real SkuId on file, use it directly, no LLM call
# needed. Phase 3b only fires when there's genuine ambiguity (more than
# one real SkuId), and only for the primary match.

from .resources import sku_examples_collection, deepseek_json, id_str, classify_probability_tier


def _candidate_skus_for_variant(brand: str, variant_id: str) -> dict:
    """Every distinct SkuId under this (brand, product_variant_id), each with
    its product_id, up to 3 representative real phrasings, and how many times
    that pack was actually prescribed historically. Feeds Phase 3b.
    Returns {sku_id_str: {"product_id": str, "notes": [...], "occurrences": int}}.

    occurrences comes from the triplet-first builder's occurrence_count
    metadata (real historical row count for that exact pack) — a far better
    "most commonly prescribed" signal than counting stored phrasings, which
    is capped and so barely varies between packs. Defaults to 0 on
    pre-rebuild rows that lack the field."""
    try:
        data = sku_examples_collection.get(
            where={"$and": [{"brand": brand}, {"product_variant_id": int(variant_id)}]},
            include=["documents", "metadatas"],
        )
    except (ValueError, Exception):
        return {}

    skus = {}

    for doc, meta in zip(data["documents"], data["metadatas"]):
        sid = id_str(meta.get("sku_id"))
        if not sid:
            continue
        entry = skus.setdefault(
            sid,
            {"product_id": id_str(meta.get("product_id")), "notes": [], "occurrences": 0},
        )
        # occurrence_count is per-triplet, repeated on each of that triplet's
        # note rows — take the max rather than summing over duplicated rows.
        entry["occurrences"] = max(entry["occurrences"], int(meta.get("occurrence_count") or 0))
        if doc not in entry["notes"] and len(entry["notes"]) < 3:
            entry["notes"].append(doc)

    return skus


# Phase 3b — a SMALL, focused LLM call that runs ONLY for the primary match
# and ONLY when the confirmed variant has more than one real SkuId (pack
# size). It picks the pack size; product_id/product_variant_id come from
# Phase 2, not from here.
PHASE3B_PROMPT_TEMPLATE = """
### ROLE
You are a veterinary product matching system. The brand and product VARIANT
are already confirmed. Your ONLY job: pick the correct pack size (SkuId)
from candidates that all belong to that variant and differ ONLY in pack
size/quantity.

### CONTEXT
Every candidate is the same product line, species, and region — they only
differ in how much product is in the pack (100ml vs 500ml, 3-pipette vs
6-pipette). Each candidate shows real historical phrasings AND how often
it was actually prescribed (occurrence count / % share). Vets use
shorthand: "3pk"/"3 pack" = 3-pack, "6s" = 6-pack, a bare "100" next to
"ml" = 100ml.

### INFO PROVIDED
BRAND (confirmed): __BRAND__
CONFIRMED VARIANT (confirmed): __FULL_PRODUCT_NAME__
ORIGINAL QUERY: __QUERY__
CANDIDATE SKUs: __SKU_CANDIDATES__

### RULES

1. DISCRIMINATE FIRST. A number in the query only "specifies a pack size"
   if it's true of SOME candidates and false of others. If every
   candidate's phrasing implies the same count as the query (e.g. query
   says "1 pack" and every candidate is singular), that number carries no
   information — treat it as if the query specified nothing, and skip to
   Rule 3. This usually means the candidates are duplicate/legacy SkuId
   codes for the same physical pack.

2. WEIGH THE MATCH AGAINST THE EVIDENCE. Even when a match does
   discriminate, check the occurrence share of the matching candidate
   against the top non-matching candidate. If the ONLY candidate matching
   the query's stated count has a small share (roughly <10%, or single-
   digit occurrences) while a candidate with a real majority share
   (roughly >50%) does NOT match, be suspicious of the literal match
   rather than trusting it automatically — for repeat/monthly products
   sold in a fixed multi-count pack, a bare number in the query more
   often describes a DOSE ("give 1 tablet") than the pack the vet
   actually ordered. In that situation: prefer the dominant candidate,
   confidence = Medium, and say in "reason" that the stated count likely
   describes dosing, not the pack purchased.

3. NO PACK SIZE STATED (or Rule 1 ruled it out): use the "(Historical
   signal: ...)" guidance line:
   - "dominant" -> pick it, confidence = Medium.
   - "roughly tied" -> use any other query detail to break the tie; if
     truly nothing else to go on, pick the higher one, confidence =
     Medium, and say so honestly.
   - "weak/fragmented" -> still pick the highest-share candidate,
     confidence = Low.
   - no guidance line at all -> pick the candidate with the most
     phrasings recorded, confidence = Medium.

4. NO MATCH: if the query states a pack size that discriminates (Rule 1)
   and genuinely no candidate has it, return an empty sku_id, confidence
   = Low, and explain — never force-fit the closest wrong pack.

5. Never invent a SkuId — only return one from the candidate list.
6. Use ONLY: High, Medium, Low for confidence.

### GOLDEN EXAMPLES

QUERY: "frontline spray 500ml"
CANDIDATES: SkuId 229 ["...100ml"], SkuId 219 ["...250ml"], SkuId 220 ["...500ml"]
-> {"sku_id": "220", "confidence": "High", "reason": "Query specifies 500ml; SkuId 220 is the 500ml pack."}

QUERY: "frontline s/on sml dog 3pk"
CANDIDATES: SkuId 1227 [412x, 78%, "...SML DOG 3PK"], SkuId 1228 [116x, 22%, "...SML DOG 6PK"]
-> {"sku_id": "1227", "confidence": "High", "reason": "Shorthand '3pk' = 3 Pipette Pack, matching SkuId 1227, which is also the historically dominant pack."}

QUERY: "frontline spray" (no size given)
CANDIDATES: SkuId 229 [612x, 68%, "...100ml"], SkuId 219 [90x, 10%, "...250ml"], SkuId 220 [198x, 22%, "...500ml"]
(Historical signal: SkuId 229 is dominant at 68%.)
-> {"sku_id": "229", "confidence": "Medium", "reason": "No pack size specified; SkuId 229 is the dominant historical pack at 68%."}

QUERY: "advocate cat" (no size given, two packs close in share)
CANDIDATES: SkuId 88 [210x, 55%, "...4-8kg"], SkuId 89 [172x, 45%, "...over 8kg"]
(Historical signal: roughly tied between SkuId 88 (55%) and SkuId 89 (45%).)
-> {"sku_id": "88", "confidence": "Medium", "reason": "No weight given and the two packs are near-tied (55/45); defaulted to the marginally more common SkuId 88."}

QUERY: "frontline spray 1litre"
CANDIDATES: SkuId 229 ["...100ml"], SkuId 219 ["...250ml"], SkuId 220 ["...500ml"]
-> {"sku_id": "", "confidence": "Low", "reason": "Query requests 1 litre; no candidate pack matches that size."}

QUERY: "bravecto gato spot-on 112.5 mg (2.6-6.2 lb) uno pack" (Rule 1: non-discriminating)
CANDIDATES: SkuId 598 [2x, 0%, "...1pk"], SkuId 1258 [300x, 23%, "...1 PIP"], SkuId 1264 [101x, 8%, "...112.5EA"], SkuId 3378 [884x, 69%, "...1 Pip."]
(Historical signal: SkuId 3378 is dominant at 69%.)
-> {"sku_id": "3378", "confidence": "Medium", "reason": "'uno pack' (1) matches every candidate equally — they're all singular packs — so it doesn't discriminate. Falling back to the historical signal: SkuId 3378 is dominant at 69%."}

QUERY: "comfortablis tablet - one hundred and eigthy mg - 1 pc" (Rule 2: weak match vs strong dominance)
CANDIDATES: SkuId 61 [591x, 71%, "...180MG TABS...6PK"], SkuId 2392 [235x, 28%, "...180MG...6pk"], SkuId 2674 [3x, 0%, "...Single Tablet"]
(Historical signal: SkuId 61 is dominant at 71%.)
-> {"sku_id": "61", "confidence": "Medium", "reason": "Query's '1 pc' literally matches only SkuId 2674's 'Single Tablet' phrasing, but that candidate has just 3 historical occurrences against SkuId 61's 591 (71% share) — for a monthly chewable normally sold in a 6-count pack, '1 pc' more likely means 'give one tablet' (a dosing note) than a request for the rare singles SKU. Deferring to the dominant historical pack, SkuId 61."}
(do NOT let a 3-occurrence literal match override a 591-occurrence majority — a match that thin is more likely coincidental phrasing than a genuine pack-size request)

### OUTPUT FORMAT (STRICT JSON ONLY):
{
  "sku_id": "",
  "confidence": "",
  "reason": ""
}
"""


def build_sku_candidates_block(candidates: dict) -> str:
    total = sum(info.get("occurrences", 0) for info in candidates.values())

    lines = []
    for sid, info in candidates.items():
        phrasings = "; ".join(f'"{n}"' for n in info["notes"]) or "(no phrasing recorded)"
        occ = info.get("occurrences", 0)
        share = f", {round(100 * occ / total)}% of historical scripts" if total else ""
        lines.append(f"SkuId {sid} [prescribed {occ}x{share}] — {phrasings}")

    guidance = ""
    if total:
        shares = sorted(
            ((sid, info.get("occurrences", 0) / total) for sid, info in candidates.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        tier = classify_probability_tier(shares)

        if tier == "dominant":
            sid, p = shares[0]
            guidance = f"\n(Historical signal: SkuId {sid} is dominant at {round(p * 100)}% -- prefer it when the query doesn't specify a pack size.)"
        elif tier == "tied":
            (sid_a, p_a), (sid_b, p_b) = shares[0], shares[1]
            guidance = (
                f"\n(Historical signal: roughly tied between SkuId {sid_a} ({round(p_a * 100)}%) "
                f"and SkuId {sid_b} ({round(p_b * 100)}%) -- use the query's other details to decide; "
                f"do not default to the higher one blindly.)"
            )
        elif tier == "low":
            guidance = "\n(Historical signal is weak/fragmented across many packs -- use whichever candidate best matches the query's own details.)"

    return "\n".join(lines) + guidance


def ask_deepseek_phase3b(query: str, brand: str, match: dict, candidates: dict) -> str:
    """Small, focused LLM call: pick the SkuId (pack size) for the primary
    match among candidates that all share the confirmed variant. Returns a
    sku_id from the candidate set, or "" if the model can't match / errors."""
    prompt = (
        PHASE3B_PROMPT_TEMPLATE
        .replace("__BRAND__", brand)
        .replace("__FULL_PRODUCT_NAME__", match.get("full_product_name") or "")
        .replace("__QUERY__", query)
        .replace("__SKU_CANDIDATES__", build_sku_candidates_block(candidates))
    )

    try:
        result = deepseek_json(prompt)
    except Exception:
        return ""

    sku = id_str(result.get("sku_id"))
    return sku if sku in candidates else ""


def resolve_sku(query: str, brand: str, match: dict, run_pack_llm: bool) -> dict:
    """
    Resolves sku_id (exact pack size) for a match whose product_id /
    product_variant_id have ALREADY been assigned by Phase 2's LLM
    (Rule 8, matched at the product-line level) and pair-validated:
      - No product_variant_id known -> nothing to resolve, sku_id blank.
      - Exactly one real SkuId under this (brand, variant) -> take it
        directly, no LLM call needed. (Phase 3a.)
      - More than one real SkuId (the variant has multiple pack sizes)
        -> Phase 3b, a small focused LLM call, picks the pack size — but
        ONLY when run_pack_llm (primary match only). Alternatives leave
        sku_id blank rather than guess a pack size.
    """
    match.setdefault("sku_id", "")

    vid = match.get("product_variant_id") or ""

    if not vid:
        return match

    candidates = _candidate_skus_for_variant(brand, vid)

    if len(candidates) == 1:
        only_sku = next(iter(candidates))
        match["sku_id"] = only_sku
        match["product_id"] = candidates[only_sku]["product_id"] or match.get("product_id", "")
    elif len(candidates) > 1 and run_pack_llm:
        chosen = ask_deepseek_phase3b(query, brand, match, candidates)
        if chosen:
            match["sku_id"] = chosen
            match["product_id"] = candidates[chosen]["product_id"] or match.get("product_id", "")
    # else: multiple candidates on a non-primary match, or zero candidates
    # for this variant -> leave sku_id blank rather than guess.

    return match
