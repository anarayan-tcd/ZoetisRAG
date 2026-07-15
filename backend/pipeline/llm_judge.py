# llm_judge.py — final quality check on High/Medium confidence results,
# run as a SEPARATE call rather than folding a self-check into Phase 2's
# own prompt: a model self-critiquing inside the same generation that just
# produced the answer tends to rationalize rather than genuinely re-examine
# it. A fresh call, given only the proposed match plus historical grounding,
# is a real second opinion. Scoped tightly to control cost: primary match
# only (never alternatives), and skipped for Low/Very Low confidence
# results, which already admit uncertainty and gain nothing from judging.
#
# Beyond approve/flag, the judge can also CORRECT: if the primary match
# looks wrong but one of Phase 2's own alternative_matches is well
# supported by ITS historical grounding, the judge promotes that
# alternative to primary instead of just downgrading confidence. This is
# deliberately bounded to candidates Phase 2 already vetted for the SAME
# brand — the judge never invents a new match, and can't fix a wrong
# BRAND pick from Phase 1 (that would need a fresh brand search, which is
# out of scope for a cheap quality-control pass).

from .resources import sku_examples_collection, deepseek_json, detect_language_hints
from .phase3b_sku import resolve_sku

JUDGE_PROMPT_TEMPLATE = """
### ROLE
You are a quality-control judge for a veterinary product matching system.
A separate stage has already proposed a specific product match for a
prescription query, plus up to two alternative candidates it also
considered. Your job is to check whether the proposed PRIMARY match
plausibly fits real historical prescribing records — and if it doesn't,
to check whether one of the ALTERNATIVES fits better before giving up and
just flagging it. You do NOT invent a new match beyond what's listed
below.

### CONTEXT
The proposed primary match below was assigned High or Medium confidence
by the matching stage. Before this result is shown to the end user, you
act as an independent second check: does this brand + product line + pack
size/species/region combination actually look like something vets have
plausibly prescribed, based on the real historical examples for this
exact variant? A technically-valid id can still be a poor semantic fit —
e.g. if the species or pack size context doesn't line up with how this
variant has actually been used historically.

If the primary match does NOT hold up, look at the alternatives (each
shown with its OWN historical grounding). If one of them clearly fits the
query better than the primary, you may promote it instead of merely
flagging the primary — but only when that alternative's own grounding
genuinely supports it, not as a default fallback.

### INFO PROVIDED
ORIGINAL QUERY: __QUERY__
MULTILINGUAL TERM HINT (deterministic token scan): __SPECIES_HINT__
  -> Non-English words found in the query with their English meaning
  ("perro" = dog, "chat" = cat, "caja" = pack/box, "comprimes" = tablets).
  Treat a hinted species as an explicit species statement when checking for
  species mismatches, and hinted form words (tablet, pipette, pack...) as
  the query specifying that formulation. Ignore if "(none detected)".

PROPOSED PRIMARY MATCH:
  Brand: __BRAND__
  Product line: __PRODUCTS__
  Full product name: __FULL_PRODUCT_NAME__
  Species: __SPECIES__
  Region: __REGION__
  Assigned confidence: __CONFIDENCE__
  Original reasoning: __REASON__

REAL HISTORICAL PHRASINGS FOR THIS EXACT PRODUCT VARIANT:
__GROUNDING__

ALTERNATIVE 1: __ALT1_SUMMARY__
  Historical phrasings: __ALT1_GROUNDING__

ALTERNATIVE 2: __ALT2_SUMMARY__
  Historical phrasings: __ALT2_GROUNDING__

### RULES
1. If the primary match is well-supported by its historical phrasings
   (species/pack size/naming line up, or the query clearly matches at
   least one historical phrasing), verdict = "approved" and keep the
   original confidence. Ignore the alternatives entirely in this case.
2. If the primary match is NOT well supported (wrong species, wrong pack
   size, or the phrasings clearly point elsewhere) AND one specific
   alternative IS well supported by its OWN historical phrasings for this
   exact query, verdict = "corrected": set "corrected_to" to that
   alternative's number ("1" or "2"), and "adjusted_confidence" to the
   confidence the promoted match deserves (usually High if its grounding
   is strong, Medium otherwise).
3. If the primary match is NOT well supported and NEITHER alternative is
   clearly better either, verdict = "flagged": lower the confidence by
   exactly one level (High -> Medium, Medium -> Low), and leave
   "corrected_to" empty. Never invent a different match beyond what's
   listed above.
4. If there are NO real historical phrasings available for the primary OR
   either alternative, you cannot confirm OR deny anything — verdict =
   "approved" (there is nothing to contradict it), but note the lack of
   grounding in your reason.
5. Be conservative: only flag or correct when you have a genuine, specific
   concern grounded in the evidence shown, not a vague feeling. Most
   well-formed matches should be approved. Only correct when the chosen
   alternative's OWN grounding is clearly stronger than the primary's —
   never promote an alternative just because the primary looks weak.

### GOLDEN EXAMPLES

QUERY: "frontline spray 500ml"
PRIMARY: Frontline / Spray / Spray 500ml, species Dog, High confidence
GROUNDING: "FRONTLINE SPRAY 500ML", "Frontline Spray 500ml for dogs"
-> {"verdict": "approved", "adjusted_confidence": "High", "corrected_to": "", "reason": "Multiple real historical records exactly match this product and pack size."}

QUERY: "advocate cat spot on"
PRIMARY: Advocate / Spot-On / Spot-On for Extra Large Dogs (over 25kg), species Dog, Medium confidence
GROUNDING: "ADVOCATE DOG EX LGE", "Advocate dog 25kg+"
ALTERNATIVE 1: Advocate / Spot-On / Spot-On for Cats (over 4kg), species Cat, Medium confidence
ALTERNATIVE 1 GROUNDING: "ADVOCATE CAT >4KG", "advocate cat large"
-> {"verdict": "corrected", "adjusted_confidence": "High", "corrected_to": "1", "reason": "Query explicitly says 'cat', but the primary and its grounding are dog-specific — a species mismatch. Alternative 1 is the cat variant and its own historical phrasings directly support this query."}

QUERY: "bravecto chew 2-4.5kg"
PRIMARY: Bravecto / Chews / Chews for Dogs 2-4.5kg (Very Small Dog), species Dog, High confidence
GROUNDING: (none available)
-> {"verdict": "approved", "adjusted_confidence": "High", "corrected_to": "", "reason": "No historical grounding available for this specific variant, so nothing contradicts the match; the weight band in the query matches the proposed variant exactly."}

QUERY: "stronhold dog dose 120mg"
PRIMARY: Program Plus / Tablet / Program Plus Tablet, species Dog, Medium confidence
GROUNDING: (none available)
ALTERNATIVE 1: Program Plus / Tablet / Program Plus Tablet Large Dog, species Dog, Low confidence
ALTERNATIVE 1 GROUNDING: (none available)
ALTERNATIVE 2: (no alternative available)
-> {"verdict": "flagged", "adjusted_confidence": "Low", "corrected_to": "", "reason": "Query says 'stronhold', likely a misspelling of the different brand Stronghold, but every candidate shown is Program Plus with no grounding to confirm or deny either — no alternative here actually fixes the brand mismatch, so this can only be flagged, not corrected."}

### OUTPUT FORMAT (STRICT JSON ONLY):
{
  "verdict": "",
  "adjusted_confidence": "",
  "corrected_to": "",
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


def _alt_summary(alt: dict) -> str:
    return (
        f"{alt.get('brand') or ''} / {alt.get('products') or ''} / "
        f"{alt.get('full_product_name') or ''}, species {alt.get('species') or ''}, "
        f"{alt.get('confidence') or ''} confidence"
    )


def ask_deepseek_judge(query: str, brand: str, match: dict, alternatives: list) -> dict:
    """The judge call itself. Returns {} (a no-op) on any failure — a
    judging error should never take down the underlying match."""

    grounding = _grounding_examples_for_variant(brand, match.get("product_variant_id"))

    alt_slots = (alternatives + [None, None])[:2]
    alt_summaries = []
    alt_groundings = []

    for alt in alt_slots:
        if alt is None:
            alt_summaries.append("(no alternative available)")
            alt_groundings.append("(none available)")
        else:
            alt_summaries.append(_alt_summary(alt))
            alt_groundings.append(
                build_judge_grounding_block(
                    _grounding_examples_for_variant(brand, alt.get("product_variant_id"))
                )
            )

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
        .replace("__ALT1_SUMMARY__", alt_summaries[0])
        .replace("__ALT1_GROUNDING__", alt_groundings[0])
        .replace("__ALT2_SUMMARY__", alt_summaries[1])
        .replace("__ALT2_GROUNDING__", alt_groundings[1])
    )

    try:
        return deepseek_json(prompt)
    except Exception:
        return {}


def _parse_alt_index(value) -> int:
    """Converts the judge's 1-based "corrected_to" string to a 0-based
    list index. Returns -1 (an always-invalid index) on anything
    unparseable, so callers can bounds-check with a single comparison."""
    try:
        idx = int(str(value).strip()) - 1
    except (TypeError, ValueError):
        return -1
    return idx if idx >= 0 else -1


def _promote_alternative(query: str, brand: str, result: dict, idx: int, verdict: dict) -> dict:
    """Swaps a well-grounded alternative into the primary slot, demoting
    the old primary into alternative_matches instead of discarding it."""
    primary = result["primary_match"]
    alternatives = result["alternative_matches"]
    chosen = alternatives.pop(idx)

    judge_reason = verdict.get("reason", "")
    print(f"  LLM judge corrected '{query}': promoted alternative {idx + 1} — {judge_reason}")

    chosen["confidence"] = verdict.get("adjusted_confidence") or chosen.get("confidence") or "Medium"
    if judge_reason:
        chosen["reason"] = f"{chosen.get('reason', '')} [Judge corrected: {judge_reason}]".strip()

    # Alternatives never get Phase 3b's pack-size LLM call — now that this
    # one is becoming the primary, resolve its exact sku_id properly.
    chosen = resolve_sku(query, brand, chosen, run_pack_llm=True)

    primary["reason"] = f"{primary.get('reason', '')} [Judge: replaced by a better-grounded alternative]".strip()
    alternatives.append(primary)

    result["primary_match"] = chosen
    result["alternative_matches"] = alternatives
    return result


def apply_llm_judge(query: str, brand: str, result: dict) -> dict:
    """
    Final quality-control pass over the assembled result, right before it
    is returned: reviews the PRIMARY match against real historical
    phrasing for its specific variant, only when confidence is High or
    Medium. Can approve as-is, downgrade confidence ("flagged"), or
    promote a well-grounded alternative to primary ("corrected") — it
    never invents a match beyond primary_match/alternative_matches as
    already produced by Phase 2.
    """
    primary = result.get("primary_match")

    if not primary or brand == "Unknown" or primary.get("confidence") not in ("High", "Medium"):
        return result

    alternatives = result.get("alternative_matches", [])
    verdict = ask_deepseek_judge(query, brand, primary, alternatives)
    action = verdict.get("verdict")

    if action == "corrected":
        idx = _parse_alt_index(verdict.get("corrected_to"))
        if 0 <= idx < len(alternatives):
            result = _promote_alternative(query, brand, result, idx, verdict)
        # else: malformed/out-of-range response — treat as a no-op rather
        # than risk swapping in the wrong candidate.

    elif action == "flagged":
        downgrade = {"High": "Medium", "Medium": "Low"}
        new_confidence = verdict.get("adjusted_confidence") or downgrade.get(primary["confidence"], primary["confidence"])
        judge_reason = verdict.get("reason", "")

        print(f"  LLM judge flagged '{query}': {judge_reason}")

        primary["confidence"] = new_confidence
        if judge_reason:
            primary["reason"] = f"{primary.get('reason', '')} [Judge: {judge_reason}]".strip()

    return result
