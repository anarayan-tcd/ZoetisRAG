# shared/query_utils.py — small pieces both matching engines (raw and
# structured) need before/after their own reasoning: input normalization
# and the empty-result shape.

import difflib
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_DISALLOWED_CHARS_RE = re.compile(r"[^a-zA-Z0-9\s.\-/%(),'&+]")


# ─────────────────────────────────────────────────────────────
# SPELLED-OUT NUMBERS -> DIGITS (deterministic — no LLM, no API cost)
# ─────────────────────────────────────────────────────────────
# Real vet notes sometimes spell dosages/pack counts out in words instead
# of digits ("One Hundred and Eigthy MG", "forty eight") — confirmed on
# real failures: "forty eight" was read as pack quantity 1, not 48; a
# spelled-out strength was similarly missed. An LLM asked to read the raw
# text can't be relied on to always convert these correctly, especially
# compound ones ("one hundred and eighty" = 180) or ones with a typo baked
# in ("Eigthy") — the same class of gap the fuzzy garbled-brand recovery
# in phase1_brand.py exists to patch deterministically, applied here to
# numbers instead of brand names. Runs once, in clean_query, so every
# downstream stage (retrieval and every LLM prompt) already sees "180"
# rather than having to solve this itself.

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000}

_NUMBER_WORD_MIN_LEN = 5   # below this, only an exact spelling is trusted
                           # (typo-correcting "ten"/"one"/"six" is too
                           # collision-prone against ordinary short words)
_NUMBER_WORD_MATCH_THRESHOLD = 0.80

# Only words long enough to typo-correct safely (see _NUMBER_WORD_MIN_LEN).
_FUZZY_NUMBER_WORDS = {
    w: (d, v) for d, wv in (("ones", _ONES), ("tens", _TENS), ("scale", _SCALES))
    for w, v in wv.items() if len(w) >= _NUMBER_WORD_MIN_LEN
}


def _match_number_word(word: str):
    """Returns (kind, value) for a recognized number word — exact match
    always, a fuzzy typo-corrected match only for words long enough that
    collision risk against unrelated real words is low. (None, 0) if the
    word isn't a number word at all."""
    core = word.strip(".,()'")

    if core in _ONES:
        return ("ones", _ONES[core])
    if core in _TENS:
        return ("tens", _TENS[core])
    if core in _SCALES:
        return ("scale", _SCALES[core])
    if core == "and":
        return ("and", 0)

    if len(core) >= _NUMBER_WORD_MIN_LEN:
        best_word, best_ratio = None, 0.0
        for candidate in _FUZZY_NUMBER_WORDS:
            ratio = difflib.SequenceMatcher(None, core, candidate).ratio()
            if ratio > best_ratio:
                best_ratio, best_word = ratio, candidate
        if best_ratio >= _NUMBER_WORD_MATCH_THRESHOLD:
            return _FUZZY_NUMBER_WORDS[best_word]

    return (None, 0)


def _parse_number_phrase(words: list, start: int):
    """Greedily consumes a run of number words starting at `start`
    (standard word2number accumulation: ones/tens add into the current
    segment, "hundred" multiplies it, "thousand" banks it and resets,
    "and" is a no-op connector allowed only between two number words).
    Returns (value, words_consumed) — consumed is 0 if `start` isn't the
    beginning of a number phrase at all."""
    total = 0
    current = 0
    matched_any = False
    i = start

    while i < len(words):
        kind, value = _match_number_word(words[i])

        if kind is None:
            break

        if kind == "and":
            if not matched_any:
                break
            nxt_kind = _match_number_word(words[i + 1])[0] if i + 1 < len(words) else None
            if nxt_kind is None:
                break
            i += 1
            continue

        if kind == "scale":
            if value == 100:
                current = (current or 1) * 100
            else:  # thousand
                total += (current or 1) * value
                current = 0
        else:  # ones or tens
            current += value

        matched_any = True
        i += 1

    if not matched_any:
        return (0, 0)

    return (total + current, i - start)


def spell_out_numbers_to_digits(text: str) -> str:
    """Replaces every spelled-out number in `text` with its digit form —
    "sixty" -> "60", "one hundred and eighty" -> "180", a typo'd "eigthy"
    corrected en route. Operates on already-lowercased, single-space
    text (the state clean_query is in by the time it calls this)."""
    if not text:
        return text

    words = text.split(" ")
    out = []
    i = 0

    while i < len(words):
        value, consumed = _parse_number_phrase(words, i)
        if consumed:
            out.append(str(value))
            i += consumed
        else:
            out.append(words[i])
            i += 1

    return " ".join(out)


def clean_query(text: str) -> str:
    """
    Normalizes a raw prescription string before it is embedded or
    reasoned over: strips stray unicode (curly quotes, accents),
    drops characters that aren't meaningful for product matching,
    collapses repeated whitespace, and converts spelled-out numbers
    to digits.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    text = _DISALLOWED_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)

    # Lowercased so differently-cased duplicates (common across vet
    # practice exports) share one cache entry and one DeepSeek call.
    text = text.strip().lower()

    return spell_out_numbers_to_digits(text)


# A catalog VariantId spans every pack size of one product line, but the
# researched market name attached to it (structured/catalog.py) may
# describe only ONE specific pack ("...45mg) 6 Pipette Pack") — the two
# functions below keep full_product_name honest: strip that baked-in
# count wherever it can't be trusted yet (variant chosen, SKU not), then
# re-append the CORRECT one once the actual SKU is known.
_PACK_PHRASE_RE = re.compile(
    r"[\s,]*\(?\d+[\s-]*(pipette\s*pack|pipettes?|packs?|doses?|tablets?|chews?|capsules?|counts?)\)?\s*$",
    re.IGNORECASE,
)


def strip_pack_phrase(name: str) -> str:
    return _PACK_PHRASE_RE.sub("", name or "").strip()


def apply_resolved_pack(match: dict, sku: dict) -> None:
    """Ground-truths a match's pack_quantity and full_product_name from
    the SKU that was ACTUALLY assigned, overriding whatever Phase 2's LLM
    guessed before the SKU was known — that guess is frequently wrong
    when Phase 3 doesn't end up picking the same pack the researched
    market name happened to describe (e.g. name says "6 Pipette Pack",
    the query/history actually supports the 3-pack SKU)."""
    pack_units = sku.get("pack_units") or 0
    ml_volume = sku.get("ml_volume") or 0

    if not pack_units and not ml_volume:
        return

    stripped = strip_pack_phrase(match.get("full_product_name") or "")

    if pack_units:
        match["pack_quantity"] = str(pack_units)
        match["full_product_name"] = f"{stripped} (Pack of {pack_units})".strip()
    elif ml_volume:
        match["pack_size"] = f"{ml_volume:g}ml"
        match["full_product_name"] = f"{stripped} ({ml_volume:g}ml)".strip()


def append_color_if_missing(full_product_name: str, color: str) -> str:
    """
    Guarantees pack color shows up in the OUTPUT name whenever the source
    data has it, regardless of whether the LLM happened to include it in
    the text it returned — deterministic, not prompt-dependent. A no-op
    when there's no color, or the name already mentions it (many
    researched variant names already bake color into the text itself,
    e.g. "...for Dogs 1-22 lbs (Orange)" — this must not double it up).
    """
    if not color or not full_product_name:
        return full_product_name

    if color.lower() in full_product_name.lower():
        return full_product_name

    return f"{full_product_name} ({color})"


def unknown_result(reason: str, brand: str = "Unknown", confidence: str = "Low") -> dict:
    """The empty-result shape, matching the pipelines' real output: one
    product match carrying no ids of its own, plus a separate (here empty)
    list of the id combinations it could correspond to."""
    empty = {
        "brand": brand, "products": "", "full_product_name": "", "region": "",
        "species": "", "animal_size": "", "pack_size": "", "pack_quantity": "",
        "animal_age": "", "confidence": confidence, "reason": reason,
    }
    return {"product_match": empty, "id_combinations": [], "id_combinations_reason": ""}
