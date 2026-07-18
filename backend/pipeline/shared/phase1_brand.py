# phase1_brand.py — Phase 1 of the query-time matching pipeline: identify
# WHICH BRAND a messy prescription string most likely refers to. Vector
# search narrows candidates, a deterministic fuzzy-text pass recovers any
# known brand the vector search missed due to heavy input noise, and
# DeepSeek picks (or declines to pick) one from the combined list.

import difflib
import re

from .resources import (
    collection, embedding_model, VECTOR_THRESHOLD, MAX_BRAND_CANDIDATES,
    KNOWN_BRAND_WORDS, deepseek_json,
)


def search_brands_batch(queries: list, top_n: int = MAX_BRAND_CANDIDATES) -> list:
    """
    Embeds and searches many queries in a single ChromaDB round trip.
    Casts a wide net across every variant, then groups hits by brand and
    keeps only each brand's best-matching variant — so Phase 1 reasons
    over candidate BRANDS, not individual variants. Returns a list of
    candidate lists, aligned index-for-index with the input queries.
    """

    if not queries or collection.count() == 0:
        return [[] for _ in queries]

    embeddings = embedding_model.encode(
        queries,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=embeddings,
        n_results=min(collection.count(), 100)
    )

    all_candidates = []

    for doc_list, meta_list, dist_list in zip(
        results["documents"], results["metadatas"], results["distances"]
    ):

        best_per_brand = {}

        for doc, meta, distance in zip(doc_list, meta_list, dist_list):

            brand = meta.get("brand")

            if not brand:
                # Defensive: ignore any stray document that doesn't match
                # the per-variant brand schema (e.g. leftover test data).
                continue

            similarity = 1 - distance

            if similarity < VECTOR_THRESHOLD:
                continue

            if brand not in best_per_brand or similarity > best_per_brand[brand]["similarity"]:
                best_per_brand[brand] = {
                    "brand": brand,
                    "similarity": round(similarity, 3),
                    "example_variant": meta.get("variant_name", ""),
                    "description": doc,
                }

        ranked = sorted(best_per_brand.values(), key=lambda x: x["similarity"], reverse=True)

        all_candidates.append(ranked[:top_n])

    return all_candidates


def search_brands(query: str, top_n: int = MAX_BRAND_CANDIDATES) -> list:
    return search_brands_batch([query], top_n)[0]


# ─────────────────────────────────────────────────────────────
# FUZZY GARBLED-BRAND RECOVERY
# ─────────────────────────────────────────────────────────────
# Heavy input noise — leetspeak digit substitution ("Str0ngh0ld"), missing
# letters ("Profndr"), decorative symbols spliced mid-word ("Front-lïne",
# "Bravct0") — can drag a brand's embedding similarity so low the vector
# search never surfaces it as a Phase 1 candidate at all. When that
# happens, DeepSeek can't pick the right brand no matter how good its
# reasoning is — it can only choose from what it's shown (Rule 2). This
# recovers that candidate deterministically via plain string comparison
# against every known brand name, at no extra DB/LLM cost, and injects it
# so Phase 1's existing "explicit textual match wins" rule can act on it.

_FUZZY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LEETSPEAK_TRANSLATION = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s", "@": "a",
})
_FUZZY_MIN_TOKEN_LEN = 5       # 4-letter generic words (e.g. "pack", "dose") collide too easily
_FUZZY_MIN_ALPHA_RATIO = 0.5   # excludes dose/weight tokens like "125mg" (mostly digits)
_FUZZY_MATCH_THRESHOLD = 0.80
FUZZY_MATCH_LABEL = "text-match (garbled/misspelled brand token recognized in query)"


def _is_wordlike(tok: str) -> bool:
    """Mostly-letters, not a dose/weight number like '125mg'."""
    return len(tok) >= 3 and sum(c.isalpha() for c in tok) / len(tok) >= _FUZZY_MIN_ALPHA_RATIO


def _fuzzy_match_tokens(query: str) -> list:
    """Extracts letters-only fuzzy-match candidates from a query: raw
    alphanumeric tokens, leetspeak-decoded and stripped of any remaining
    digits. A token only stands alone once it's >=5 letters (a shorter
    one is too likely to collide with an unrelated brand fragment purely
    by coincidence, e.g. "pack" vs "prac"), but two SHORT adjacent
    word-like tokens are still tried concatenated, since clean_query's
    punctuation-stripping can split one garbled brand token across a
    space (e.g. "Stong_hold" -> "stong hold", "Front-line" -> "front
    line") — the combined string is what actually gets matched, so each
    half only needs to look word-ish on its own."""
    raw_tokens = _FUZZY_TOKEN_RE.findall(query.lower())

    def normalize(s: str) -> str:
        return re.sub(r"[0-9]", "", s.translate(_LEETSPEAK_TRANSLATION))

    candidates = []

    for i, tok in enumerate(raw_tokens):
        if len(tok) >= _FUZZY_MIN_TOKEN_LEN and _is_wordlike(tok):
            candidates.append(normalize(tok))

        if i + 1 < len(raw_tokens):
            nxt = raw_tokens[i + 1]
            if _is_wordlike(tok) and _is_wordlike(nxt):
                candidates.append(normalize(tok + nxt))

    return [c for c in candidates if len(c) >= _FUZZY_MIN_TOKEN_LEN]


def fuzzy_brand_match(query: str, known_brand_words: dict = KNOWN_BRAND_WORDS) -> str:
    """Best-matching canonical brand name from known_brand_words (see
    resources.KNOWN_BRAND_WORDS), or "" if nothing clears the similarity
    threshold."""
    best_brand, best_ratio = "", 0.0

    for tok in _fuzzy_match_tokens(query):
        for word, brand in known_brand_words.items():
            ratio = difflib.SequenceMatcher(None, tok, word).ratio()
            if ratio > best_ratio:
                best_ratio, best_brand = ratio, brand

    return best_brand if best_ratio >= _FUZZY_MATCH_THRESHOLD else ""


def _example_for_brand(brand: str) -> tuple:
    """One representative (example_variant, description) pair for a
    fuzzy-recovered brand that has no vector-search hit of its own to
    borrow one from."""
    try:
        data = collection.get(where={"brand": brand}, limit=1, include=["documents", "metadatas"])
    except Exception:
        return "", ""

    if not data["documents"]:
        return "", ""

    return data["metadatas"][0].get("variant_name", ""), data["documents"][0]


def augment_with_fuzzy_brand(query: str, candidates: list) -> list:
    """
    Adds a fuzzy-recovered brand candidate to the front of the vector-
    search results when one is found and isn't already present. Never
    replaces or reorders the existing candidates — it's purely additive,
    so a rare spurious fuzzy match just gives DeepSeek one extra
    (ignorable) option rather than displacing a real one.
    """
    already_present = {c["brand"] for c in candidates}

    matched_brand = fuzzy_brand_match(query)

    if not matched_brand or matched_brand in already_present:
        return candidates

    example_variant, description = _example_for_brand(matched_brand)

    fuzzy_candidate = {
        "brand": matched_brand,
        "similarity": FUZZY_MATCH_LABEL,
        "example_variant": example_variant,
        "description": description,
    }

    return [fuzzy_candidate] + candidates


def normalise_confidence(conf) -> str:
    """Converts any confidence value to High/Medium/Low/Very Low."""

    if not conf:
        return "Low"

    s = str(conf).lower().replace("%", "").strip()

    try:
        v = float(s)
        if v > 1:
            v = v / 100.0
        if v >= 0.7:
            return "High"
        elif v >= 0.5:
            return "Medium"
        else:
            return "Low"
    except ValueError:
        if "very low" in s:
            return "Very Low"
        elif "high" in s:
            return "High"
        elif "medium" in s or "mid" in s:
            return "Medium"
        else:
            return "Low"


# Plain (non f-string) template — see phase2_variant.py's PHASE2_PROMPT_TEMPLATE for why.
PHASE1_PROMPT_TEMPLATE = """
### ROLE
You are a veterinary product matching system. Your job is to identify
which BRAND a messy, real-world vet prescription entry most likely
refers to, working only from a shortlist of candidate brands retrieved
from a vector database.

### CONTEXT
Vet practices enter product names inconsistently — abbreviated, misspelled,
reordered, or missing pack/size details entirely (e.g. "FRONTLINE SPOT ON
XL DOG 3PK", "bravecto chew lge", "advocate cat <4kg"). This brand
identification feeds a second stage that picks the exact product variant,
so getting the BRAND right here is critical — a wrong brand here means
the second stage searches an entirely wrong product family.

### INFO PROVIDED
USER QUERY: __QUERY__

CANDIDATES:
__CONTEXT__

### RULES
1. If the query contains an explicit, recognizable brand name (even
   abbreviated, misspelled, or obscured by inserted digits/symbols — e.g.
   "Str0ngh0ld", "Bravct0", "Front-lïne" — all unambiguous once the noise
   is looked past), that brand MUST be the match if it appears among the
   candidates — don't let a higher similarity score on a different
   candidate override an explicit textual match. A candidate marked
   "text-match" instead of a numeric similarity was recovered specifically
   because it's an explicit (if noisy) textual match Rule 1 applies to —
   treat it exactly as strongly as a high numeric similarity would be.
2. Only pick from the candidates given — never invent a brand that isn't
   listed, even if you recognize a "better" real-world brand name.
3. Confidence guide:
   - High: the query explicitly names the brand (or an unambiguous
     abbreviation/misspelling of it)
   - Medium: the query strongly implies this brand via active ingredient,
     packaging language, or a partial/ambiguous name match
   - Low: best-guess from vague symptoms/species/format only, no direct
     brand signal
   - Very Low: none of the candidates plausibly match the query
4. Use ONLY: High, Medium, Low, Very Low for confidence — no numbers or
   percentages.

### GOLDEN EXAMPLES

QUERY: "FRONTLINE SPOT ON XL DOG 3PK"
CANDIDATES: Frontline (0.81), Bravecto (0.42), Advocate (0.38)
-> {"brand": "Frontline", "confidence": "High", "reason": "Query explicitly names 'Frontline'."}

QUERY: "bravecto chew lge dog"
CANDIDATES: Bravecto (0.77), Nexgard (0.51)
-> {"brand": "Bravecto", "confidence": "High", "reason": "Explicit brand name 'bravecto', despite lowercase/informal phrasing."}

QUERY: "advocate cat <4kg"
CANDIDATES: Advocate (0.79), Stronghold (0.44), Frontline (0.36)
-> {"brand": "Advocate", "confidence": "High", "reason": "Query explicitly names 'advocate'."}

QUERY: "nexgard spectra med dog 3pk"
CANDIDATES: Nexgard Spectra (0.83), Nexgard (0.71), Simparica (0.40)
-> {"brand": "Nexgard Spectra", "confidence": "High", "reason": "Query names 'nexgard spectra' explicitly; 'Spectra' is a distinct line from plain Nexgard, so the more specific sibling brand is chosen over the shorter name."}

QUERY: "seresto collar large dog"
CANDIDATES: Seresto (0.80), Scalibor (0.41)
-> {"brand": "Seresto", "confidence": "High", "reason": "Query explicitly names 'seresto'; 'collar' matches its product format."}

QUERY: "stronhold puppy 3 pack"
CANDIDATES: Stronghold (0.74), Advocate (0.40)
-> {"brand": "Stronghold", "confidence": "High", "reason": "'stronhold' is an unambiguous misspelling of Stronghold."}

QUERY: "Str0ngh0ld chat dose 45 mg pack 6"
CANDIDATES: Heartgard Plus (0.52), Stronghold (text-match (garbled/misspelled brand token recognized in query))
-> {"brand": "Stronghold", "confidence": "High", "reason": "'Str0ngh0ld' is a leetspeak-obscured but unambiguous reference to Stronghold, recognized via explicit text match despite low vector similarity — Rule 1 overrides the higher-similarity Heartgard Plus candidate."}

QUERY: "panacur 10% susp 100ml"
CANDIDATES: Panacur (0.80), Drontal (0.38)
-> {"brand": "Panacur", "confidence": "High", "reason": "Query explicitly names 'panacur'; '10% susp' matches its liquid dewormer format."}

QUERY: "selamectin spot on cat 3 pipettes"
CANDIDATES: Stronghold (0.58), Revolution (0.56), Advocate (0.50)
-> {"brand": "Stronghold", "confidence": "Medium", "reason": "No brand named, but 'selamectin' is the active ingredient of Stronghold/Revolution; picked the highest-similarity selamectin candidate — implied, not explicit."}

QUERY: "flea tablet large dog monthly"
CANDIDATES: Nexgard (0.55), Bravecto (0.53), Simparica (0.50)
-> {"brand": "Nexgard", "confidence": "Low", "reason": "No brand named explicitly; picked highest-similarity candidate, but multiple oral flea brands plausibly fit a generic 'monthly flea tablet'."}

QUERY: "worming tablet for cat"
CANDIDATES: Panacur (0.31), Drontal (0.29)
-> {"brand": "Drontal", "confidence": "Very Low", "reason": "No brand indicated at all; extremely generic description, low confidence in any specific candidate."}

### OUTPUT FORMAT (STRICT JSON ONLY):
{
  "brand": "",
  "confidence": "",
  "reason": ""
}
"""


def ask_deepseek_phase1(query: str, brand_candidates: list) -> dict:
    """DeepSeek reasons over Phase 1 candidates to identify the brand."""

    context = ""

    for i, c in enumerate(brand_candidates):
        context += f"""
CANDIDATE {i + 1}
BRAND: {c['brand']}
SIMILARITY: {c['similarity']}
EXAMPLE VARIANT: {c['example_variant']}
"""

    prompt = (
        PHASE1_PROMPT_TEMPLATE
        .replace("__QUERY__", query)
        .replace("__CONTEXT__", context)
    )

    result = deepseek_json(prompt)
    result["confidence"] = normalise_confidence(result.get("confidence"))

    return result
