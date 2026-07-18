# structured/phase3_structured.py — PACK SIZE resolution for the
# structured pipeline. Deliberately answers "how big a pack?", not "which
# SkuId?": several SkuIds routinely describe the same physical pack
# (Milbemax VariantId 86 has six SkuIds at PackUnits 4, all with identical
# description text), so asking which SkuId is asking the model to invent a
# distinction the catalog doesn't make. Asking which PACK is a real
# question with a real answer, and the catalog's own PackUnits/MlVolume
# fields answer it — nothing is inferred from historical phrasing.
#
# The pack must be fixed BEFORE ids are enumerated, so that a stated "6
# pack" can never drag a 1-pack code into the results and no legitimate
# 6-pack code is skipped.
#
# One pack option -> take it, no LLM call. Several -> a small DeepSeek call
# carrying the rules that took several real bugs to get right: a stated
# count only counts when it discriminates, unambiguous pack wording wins
# even at zero history, and a bare number is treated as a possible dose.

from ..shared.resources import deepseek_json, classify_probability_tier
from ..shared.query_utils import apply_resolved_pack


PHASE3_STRUCTURED_PROMPT = """
### ROLE
You are a veterinary product matching system. Brand and product are already
confirmed. Pick which PACK SIZE of that product the query refers to, from
the real pack options below. Every option is the same product — they differ
only in how much is in the pack: the number of units (PackUnits) and/or
liquid volume (MlVolume), both taken from the authoritative catalog.

### INFO PROVIDED
BRAND: __BRAND__
CONFIRMED PRODUCT: __PRODUCT__
ORIGINAL QUERY: __QUERY__
PACK OPTIONS: __PACKS__

### RULES
1. DISCRIMINATE FIRST. A number in the query only "specifies a pack" if it
   matches SOME options and not others. A count every option shares carries
   no information — treat it as unstated (Rule 4).
2. UNAMBIGUOUS PACK WORDING ALWAYS WINS, EVEN AT 0 OCCURRENCES. If the
   query attaches an explicit PACK-SIZE WORD to the number — "pack",
   "count"/"ct", "box", "carton", "PK", or a bare "x" between two numbers
   like "6x11.4mg" — that is a direct statement of the pack the vet is
   ordering, never a dose. Match it to the option with that exact
   PackUnits and use it with High confidence, REGARDLESS of that option's
   historical share: a real catalog pack with 0 recorded prescriptions is
   still correct when the query names it explicitly. Low history means
   rarely prescribed, not wrong.
3. AMBIGUOUS BARE NUMBERS DEFER TO A DOMINANT SIGNAL. Only when the
   stated count has NO pack-word attached — a bare "1 pc", "1 tablet",
   "2 each" ("tablets"/"chews" alone still counts as ambiguous: it as
   easily means a dose as a pack) — treat it as potentially a DOSING
   instruction rather than a pack request. In that case: if the
   "(Historical signal: ...)" line marks a DIFFERENT option as dominant,
   prefer that dominant option instead of the literal match, confidence
   Medium, and say the stated count more likely describes dosing than the
   pack ordered. This applies REGARDLESS of how large the literal match's
   own share is (28% is not "small" in isolation, but it still isn't
   dominant) — "dominant" already means most vets order that standard
   pack, and one ambiguous number in one query shouldn't override that.
   Only let the bare number decide the pack when there's no dominant
   signal (tied or weak) or when it already matches the dominant option.
   This rule does NOT apply when Rule 2's pack-word test is met.
4. NO PACK STATED (or ruled out by Rule 1): follow the "(Historical
   signal: ...)" line — dominant -> pick it (Medium); tied -> break with any
   other query detail, else the higher share (Medium, say so); weak -> the
   highest share (Low). No signal line -> the highest-occurrence option
   (Medium).
5. If a discriminating stated pack matches NO option, return empty
   pack_units AND ml_volume (Low) and explain — never force-fit the
   closest wrong pack.
6. Only return a pack size from the options listed. Confidence: High,
   Medium, or Low only. A clean discriminating match (Rule 2) = High.

### GOLDEN EXAMPLES

QUERY: "seresto cat collar 6 pack"
PACK OPTIONS: PackUnits 1 [prescribed 1590x, 100%], PackUnits 6 [prescribed 0x, 0%]
-> {"pack_units": "6", "ml_volume": "", "confidence": "High", "reason": "Query explicitly says '6 pack' (Rule 2) — the 6-count pack exists in the catalog. Its 0 recorded prescriptions don't matter: the query names it directly."}
(do NOT default to the 1-count pack just because it's historically dominant — "6 pack" is not a dosing phrase, so Rule 3 never applies)

QUERY: "comfortis 180mg 1 pc"
PACK OPTIONS: PackUnits 6 [prescribed 591x, 72%], PackUnits 1 [prescribed 235x, 28%]
(Historical signal: PackUnits 6 is dominant at 72%.)
-> {"pack_units": "6", "ml_volume": "", "confidence": "Medium", "reason": "Query's '1 pc' has no pack-word attached (Rule 3, not Rule 2) — for a monthly chewable normally sold in a 6-count pack, this bare number more likely means 'give one tablet' than a request for the singles pack. PackUnits 1 has a real 28% share, not a negligible one, but PackUnits 6 is still the dominant historical pack at 72%, so it's preferred over the literal match."}
(the difference from the example above: "1 pc" is a bare number, "6 pack" names the pack explicitly — that distinction decides which rule applies. Note this is NOT decided by 28% being "small enough" — Rule 3 defers to the dominant tier whenever one exists, regardless of the literal match's own share)

QUERY: "frontline spray"
PACK OPTIONS: 100ml [prescribed 612x, 68%], 250ml [prescribed 90x, 10%], 500ml [prescribed 198x, 22%]
-> {"pack_units": "", "ml_volume": "100", "confidence": "Medium", "reason": "No pack size specified; 100ml is the dominant historical pack at 68%."}

### OUTPUT (STRICT JSON ONLY)
{"pack_units": "", "ml_volume": "", "confidence": "", "reason": ""}
"""


def build_packs_block(packs: list) -> str:
    total = sum(p["occurrences"] for p in packs)

    lines = []
    for p in packs:
        size = []
        if p["pack_units"]:
            size.append(f'PackUnits {p["pack_units"]}')
        if p["ml_volume"]:
            size.append(f'{p["ml_volume"]:g}ml')
        label = ", ".join(size) or "pack size unspecified"

        share = f', {round(100 * p["occurrences"] / total)}%' if total else ""
        # sku_count is surfaced so the model can see that a pack maps to
        # several interchangeable catalog codes — and that this is normal,
        # not a reason to prefer a "cleaner" single-code option.
        codes = f' [{p["sku_count"]} catalog code(s) for this pack]' if p["sku_count"] > 1 else ""
        lines.append(f'{label} [prescribed {p["occurrences"]}x{share}]{codes}')

    guidance = ""
    if total:
        shares = sorted(
            (((p["pack_units"], p["ml_volume"]), p["occurrences"] / total) for p in packs),
            key=lambda x: x[1], reverse=True,
        )
        tier = classify_probability_tier(shares)

        def fmt(key):
            units, ml = key
            return f"PackUnits {units}" if units else f"{ml:g}ml"

        if tier == "dominant":
            key, p = shares[0]
            guidance = f"\n(Historical signal: {fmt(key)} is dominant at {round(p * 100)}%.)"
        elif tier == "tied":
            (ka, pa), (kb, pb) = shares[0], shares[1]
            guidance = (f"\n(Historical signal: roughly tied between {fmt(ka)} ({round(pa * 100)}%) "
                        f"and {fmt(kb)} ({round(pb * 100)}%).)")
        elif tier == "low":
            guidance = "\n(Historical signal is weak/fragmented — rely on the query's own details.)"

    return "\n".join(lines) + guidance


def _as_number(value, cast):
    s = str(value or "").strip()
    if not s:
        return None
    try:
        return cast(float(s))
    except ValueError:
        return None


def resolve_pack_structured(query: str, brand: str, product_name: str, packs: list) -> dict:
    """Picks one pack size from `packs` (see catalog.pack_options_for_variants).

    Returns {"pack_units": int|None, "ml_volume": float|None,
    "confidence": str, "reason": str}. Both None means unresolved — the
    caller then enumerates every pack's ids rather than guessing."""
    if not packs:
        return {"pack_units": None, "ml_volume": None, "confidence": "Low",
                "reason": "No pack options recorded for this product."}

    if len(packs) == 1:
        only = packs[0]
        return {
            "pack_units": only["pack_units"] or None,
            "ml_volume": only["ml_volume"] or None,
            "confidence": "High",
            "reason": "Only one pack size exists for this product in the catalog.",
        }

    prompt = (
        PHASE3_STRUCTURED_PROMPT
        .replace("__BRAND__", brand)
        .replace("__PRODUCT__", product_name or "")
        .replace("__QUERY__", query)
        .replace("__PACKS__", build_packs_block(packs))
    )

    try:
        result = deepseek_json(prompt)
    except Exception:
        return {"pack_units": None, "ml_volume": None, "confidence": "Low",
                "reason": "Pack-size resolution failed; reporting every pack's ids."}

    units = _as_number(result.get("pack_units"), int)
    ml = _as_number(result.get("ml_volume"), float)

    # Only honour a pack that actually exists in the options shown.
    valid = any(
        (units is not None and p["pack_units"] == units) or (ml is not None and p["ml_volume"] == ml)
        for p in packs
    )
    if not valid:
        return {"pack_units": None, "ml_volume": None,
                "confidence": result.get("confidence") or "Low",
                "reason": result.get("reason") or "No pack option matched the query."}

    return {
        "pack_units": units,
        "ml_volume": ml,
        "confidence": result.get("confidence") or "Medium",
        "reason": result.get("reason") or "",
    }


def apply_pack_to_match(match: dict, pack_units, ml_volume) -> None:
    """Writes the resolved pack back onto the product match's own fields —
    ground truth from the catalog, overriding whatever Phase 2 guessed
    before the pack was known."""
    if pack_units is None and ml_volume is None:
        return
    apply_resolved_pack(match, {"pack_units": pack_units or 0, "ml_volume": ml_volume or 0})
