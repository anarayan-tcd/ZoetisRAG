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
You are a veterinary product matching system. The brand and the exact
product VARIANT (product line) have already been confirmed. Your ONLY job
is to pick the correct pack size — the specific SkuId — from a small list
of candidate SkuIds that ALL belong to that one confirmed variant and
differ ONLY in pack size / pack quantity.

### CONTEXT
Every candidate below is the same product line, same species, same region
— the only thing separating them is how much product is in the pack (e.g.
100ml vs 250ml vs 500ml, or a 3-pipette vs 6-pipette pack). Each candidate
SkuId is shown with real historical vet-entered phrasings so you can see
how that pack size was actually written. Vets frequently use shorthand:
"3pk"/"3 pack" = 3-pack, "6s" = 6-pack, a bare "100" next to an "ml"
product = 100ml. Match on the NUMBER and UNIT of the pack, not on the
surrounding product words (those are identical across every candidate).

### INFO PROVIDED
BRAND (confirmed): __BRAND__
CONFIRMED VARIANT (confirmed): __FULL_PRODUCT_NAME__

ORIGINAL QUERY (the source of truth for which pack the vet wants):
__QUERY__

CANDIDATE SKUs (all belong to the confirmed variant — pick ONE by pack size):
__SKU_CANDIDATES__

### RULES
1. Choose the SkuId whose real phrasings match the pack size/quantity in the ORIGINAL QUERY. Return its sku_id exactly as given.
2. Never invent a SkuId — only return one from the candidate list.
3. If the query clearly specifies a pack size and one candidate matches it, confidence = High.
4. If the query does NOT specify any pack size/quantity at all, use the "(Historical signal: ...)" guidance line shown after the candidates, if present:
   - "dominant" -> pick that SkuId directly, confidence = Medium.
   - "roughly tied" -> look for ANY other query detail that favors one of the two tied SkuIds; if truly nothing else to go on, pick the higher one, confidence = Medium, and mention the near-tie in your reason.
   - "weak/fragmented" -> still pick the highest-share candidate (better than guessing blind), confidence = Low, and say the historical signal was weak.
   If no guidance line is shown at all, pick the candidate with the most phrasings recorded, confidence = Medium.
5. If the query specifies a pack size that NO candidate matches, return an empty sku_id with confidence = Low and explain — do NOT force-fit onto the closest wrong pack.
6. Use ONLY: High, Medium, Low for confidence.

### GOLDEN EXAMPLES

QUERY: "frontline spray 500ml"
CANDIDATES: SkuId 229 ["...100ml"], SkuId 219 ["...250ml"], SkuId 220 ["...500ml"]
-> {"sku_id": "220", "confidence": "High", "reason": "Query specifies 500ml; SkuId 220 is the 500ml pack."}

QUERY: "frontline s/on sml dog 3pk"
CANDIDATES: SkuId 1227 ["...SML DOG 3PK"], SkuId 1228 ["...SML DOG 6PK"]
-> {"sku_id": "1227", "confidence": "High", "reason": "Shorthand '3pk' = 3 Pipette Pack, matching SkuId 1227."}

QUERY: "frontline spray" (no size given)
CANDIDATES: SkuId 229 [prescribed 612x, 68% of historical scripts, "...100ml"], SkuId 219 [prescribed 90x, 10% of historical scripts, "...250ml"], SkuId 220 [prescribed 198x, 22% of historical scripts, "...500ml"]
(Historical signal: SkuId 229 is dominant at 68% -- prefer it when the query doesn't specify a pack size.)
-> {"sku_id": "229", "confidence": "Medium", "reason": "No pack size specified; SkuId 229 (100ml) is the dominant historical pack at 68%."}

QUERY: "advocate cat" (no size given, two packs close in share)
CANDIDATES: SkuId 88 [prescribed 210x, 55% of historical scripts, "...4-8kg"], SkuId 89 [prescribed 172x, 45% of historical scripts, "...over 8kg"]
(Historical signal: roughly tied between SkuId 88 (55%) and SkuId 89 (45%) -- use the query's other details to decide; do not default to the higher one blindly.)
-> {"sku_id": "88", "confidence": "Medium", "reason": "No weight given in the query and the two packs are historically near-tied (55/45); defaulted to the marginally more common SkuId 88 since nothing else in the query distinguishes them."}
(a near-tie is still resolved — never leave sku_id blank just because it's close — but the reason must say so honestly rather than implying certainty)

QUERY: "frontline spray 1litre"
CANDIDATES: SkuId 229 ["...100ml"], SkuId 219 ["...250ml"], SkuId 220 ["...500ml"]
-> {"sku_id": "", "confidence": "Low", "reason": "Query requests 1 litre; no candidate pack matches that size."}

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
