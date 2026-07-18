# llm_judge.py — final quality check on High/Medium confidence results,
# run as a SEPARATE call rather than folding a self-check into Phase 2's
# own prompt: a model self-critiquing inside the same generation that just
# produced the answer tends to rationalize rather than genuinely re-examine
# it. A fresh call, given only the proposed match plus historical grounding,
# is a real second opinion. Scoped tightly to control cost: skipped for
# Low/Very Low confidence results, which already admit uncertainty and gain
# nothing from judging.
#
# The judge approves or flags (downgrades confidence); it cannot correct.
# It used to be able to promote a better-grounded alternative into the
# primary slot, but the pipeline no longer produces alternative matches —
# it returns one product plus every id combination that product maps to —
# so there is nothing to promote. Re-running variant selection from here
# would be a second Phase 2, not a cheap quality-control pass, so a match
# the judge can't support is flagged for a human rather than silently
# swapped.

from ..shared.resources import sku_examples_collection, deepseek_json, detect_language_hints

JUDGE_PROMPT_TEMPLATE = """
### ROLE
You are a quality-control judge for a veterinary product matching system.
A separate stage has already proposed a specific product match for a
prescription query. Your job is to check whether that match plausibly fits
real historical prescribing records. You do NOT propose a different match.

### CONTEXT
The proposed match below was assigned High or Medium confidence by the
matching stage. Before this result is shown to the end user, you act as an
independent second check: does this brand + product line + pack
size/species/region combination actually look like something vets have
plausibly prescribed, based on the real historical examples for this exact
variant? A technically-valid id can still be a poor semantic fit — e.g. if
the species or pack size context doesn't line up with how this variant has
actually been used historically.

### INFO PROVIDED
ORIGINAL QUERY: __QUERY__
MULTILINGUAL TERM HINT (deterministic token scan): __SPECIES_HINT__
  -> Non-English words found in the query with their English meaning
  ("perro" = dog, "chat" = cat, "caja" = pack/box, "comprimes" = tablets).
  Treat a hinted species as an explicit species statement when checking for
  species mismatches, and hinted form words (tablet, pipette, pack...) as
  the query specifying that formulation. Ignore if "(none detected)".

PROPOSED MATCH:
  Brand: __BRAND__
  Product line: __PRODUCTS__
  Full product name: __FULL_PRODUCT_NAME__
  Species: __SPECIES__
  Region: __REGION__
  Assigned confidence: __CONFIDENCE__
  Original reasoning: __REASON__

REAL HISTORICAL PHRASINGS FOR THIS EXACT PRODUCT VARIANT:
__GROUNDING__

### RULES
1. If the match is well-supported by its historical phrasings
   (species/pack size/naming line up, or the query clearly matches at
   least one historical phrasing), verdict = "approved" and keep the
   original confidence.
2. If the match is NOT well supported (wrong species, wrong pack size, or
   the phrasings clearly point elsewhere), verdict = "flagged": lower the
   confidence by exactly one level (High -> Medium, Medium -> Low) and say
   specifically what doesn't line up. Do not propose a replacement — a
   human reviews flagged rows.
3. If there are NO real historical phrasings available, you cannot confirm
   OR deny anything — verdict = "approved" (there is nothing to contradict
   it), but note the lack of grounding in your reason.
4. Be conservative: only flag when you have a genuine, specific concern
   grounded in the evidence shown, not a vague feeling. Most well-formed
   matches should be approved.

### GOLDEN EXAMPLES

QUERY: "frontline spray 500ml"
MATCH: Frontline / Spray / Spray 500ml, species Dog, High confidence
GROUNDING: "FRONTLINE SPRAY 500ML", "Frontline Spray 500ml for dogs"
-> {"verdict": "approved", "adjusted_confidence": "High", "reason": "Multiple real historical records exactly match this product and pack size."}

QUERY: "advocate cat spot on"
MATCH: Advocate / Spot-On / Spot-On for Extra Large Dogs (over 25kg), species Dog, Medium confidence
GROUNDING: "ADVOCATE DOG EX LGE", "Advocate dog 25kg+"
-> {"verdict": "flagged", "adjusted_confidence": "Low", "reason": "Query explicitly says 'cat', but the match and its grounding are dog-specific — a species mismatch."}

QUERY: "bravecto chew 2-4.5kg"
MATCH: Bravecto / Chews / Chews for Dogs 2-4.5kg (Very Small Dog), species Dog, High confidence
GROUNDING: (none available)
-> {"verdict": "approved", "adjusted_confidence": "High", "reason": "No historical grounding available for this specific variant, so nothing contradicts the match; the weight band in the query matches the proposed variant exactly."}

### OUTPUT FORMAT (STRICT JSON ONLY):
{
  "verdict": "",
  "adjusted_confidence": "",
  "reason": ""
}
"""


def _grounding_examples_for_variant(brand: str, variant_id, limit: int = 8) -> list:
    """Real historical phrasings tied to this SPECIFIC (brand, variant) —
    a fuller, dedicated view for the judge, distinct from Phase 2's
    brand-wide MAX_EXAMPLE_HINTS sample."""
    variant_id = str(variant_id or "")

    if not variant_id:
        return []

    try:
        data = sku_examples_collection.get(
            where={"$and": [{"brand": brand}, {"product_variant_id": int(variant_id)}]},
            include=["documents"],
        )
    except (ValueError, Exception):
        return []

    seen = []
    for doc in data["documents"]:
        if doc not in seen:
            seen.append(doc)
        if len(seen) >= limit:
            break

    return seen


def build_judge_grounding_block(examples: list) -> str:
    if not examples:
        return "(none available)"

    return "\n".join(f'- "{e}"' for e in examples)


def ask_deepseek_judge(query: str, brand: str, match: dict, variant_id: str) -> dict:
    """The judge call itself. Returns {} (a no-op) on any failure — a
    judging error should never take down the underlying match."""

    grounding = _grounding_examples_for_variant(brand, variant_id)

    prompt = (
        JUDGE_PROMPT_TEMPLATE
        .replace("__QUERY__", query)
        .replace("__SPECIES_HINT__", detect_language_hints(query) or "(none detected)")
        .replace("__BRAND__", brand)
        .replace("__PRODUCTS__", match.get("products") or "")
        .replace("__FULL_PRODUCT_NAME__", match.get("full_product_name") or "")
        .replace("__SPECIES__", match.get("species") or "")
        .replace("__REGION__", match.get("region") or "")
        .replace("__CONFIDENCE__", match.get("confidence") or "")
        .replace("__REASON__", match.get("reason") or "")
        .replace("__GROUNDING__", build_judge_grounding_block(grounding))
    )

    try:
        return deepseek_json(prompt)
    except Exception:
        return {}


def apply_llm_judge(query: str, brand: str, result: dict) -> dict:
    """
    Final quality-control pass over the assembled result, right before it
    is returned: reviews the product match against real historical phrasing
    for its variant, only when confidence is High or Medium. Approves as-is
    or downgrades confidence ("flagged").
    """
    match = result.get("product_match")

    if not match or brand == "Unknown" or match.get("confidence") not in ("High", "Medium"):
        return result

    # The variant to ground against lives on the id combination now, not
    # the match itself — raw mode always has exactly one.
    combinations = result.get("id_combinations") or []
    variant_id = combinations[0].get("product_variant_id") if combinations else ""

    verdict = ask_deepseek_judge(query, brand, match, variant_id)

    if verdict.get("verdict") == "flagged":
        downgrade = {"High": "Medium", "Medium": "Low"}
        new_confidence = verdict.get("adjusted_confidence") or downgrade.get(match["confidence"], match["confidence"])
        judge_reason = verdict.get("reason", "")

        print(f"  LLM judge flagged '{query}': {judge_reason}")

        match["confidence"] = new_confidence
        if judge_reason:
            match["reason"] = f"{match.get('reason', '')} [Judge: {judge_reason}]".strip()

    return result
