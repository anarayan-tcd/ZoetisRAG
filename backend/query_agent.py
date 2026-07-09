import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

from chromadb_client import get_embedding_model, get_collection, SKU_EXAMPLES_COLLECTION
from knowledge.extract_brands import guess_brand_token


# ─────────────────────────────────────────────────────────────
# ENV SETUP
# ─────────────────────────────────────────────────────────────

env_path = Path(__file__).resolve().parent / "APIKeys.env"
load_dotenv(dotenv_path=env_path)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError(f"Missing DEEPSEEK_API_KEY in {env_path}")


# ─────────────────────────────────────────────────────────────
# LLM CLIENT (DEEPSEEK)
# ─────────────────────────────────────────────────────────────

client_llm = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


# ─────────────────────────────────────────────────────────────
# EMBEDDING MODEL + CHROMADB
# ─────────────────────────────────────────────────────────────

embedding_model = get_embedding_model()
collection = get_collection()                          # vet_products
sku_examples_collection = get_collection(SKU_EXAMPLES_COLLECTION)


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

VECTOR_THRESHOLD = 0.30          # minimum similarity to consider a brand/variant at all
MAX_BRAND_CANDIDATES = 5         # distinct brands considered in Phase 1
MAX_VARIANT_CANDIDATES = 8       # variants considered in Phase 2, within the chosen brand
MAX_EXAMPLE_HINTS = 3            # real historical phrasing examples pulled from sku_examples
MAX_WORKERS = 6                  # parallel DeepSeek calls during batch processing

# Query-time live-research fallback (see project_pipeline_recommendations
# memory, recommendation 3): when Phase 1 can't confidently identify a
# brand, optionally research it live via Gemini search grounding and
# merge into ChromaDB on the spot, rather than waiting for the next
# knowledge_agent2.py batch run. Off by default — auto-merging unreviewed
# live output into the shared knowledge base trades away "minimum
# hardcoding, maximum user input" for coverage, so this is opt-in.
# It's a per-call parameter (see identify_product/identify_products_batch)
# rather than a global switch — this is just the default when unspecified.
DEFAULT_ENABLE_LIVE_FALLBACK = False


# ─────────────────────────────────────────────────────────────
# STEP 0 — INPUT CLEANING
# ─────────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")
_DISALLOWED_CHARS_RE = re.compile(r"[^a-zA-Z0-9\s.\-/%(),'&+]")


def clean_query(text: str) -> str:
    """
    Normalizes a raw prescription string before it is embedded or
    reasoned over: strips stray unicode (curly quotes, accents),
    drops characters that aren't meaningful for product matching,
    and collapses repeated whitespace.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    text = _DISALLOWED_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)

    # Lowercased so differently-cased duplicates (common across vet
    # practice exports) share one cache entry and one DeepSeek call.
    return text.strip().lower()


# ─────────────────────────────────────────────────────────────
# PHASE 1 — BRAND IDENTIFICATION
# ─────────────────────────────────────────────────────────────

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


def normalise_confidence(conf) -> str:
    """Converts any confidence value to High/Medium/Low/Very Low."""

    if not conf:
        return "Low"

    s = str(conf).lower().replace("%", "").strip()

    try:
        v = float(s)
        if v >= 0.7 or v >= 70:
            return "High"
        elif v >= 0.5 or v >= 50:
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


# Plain (non f-string) template — see PHASE2_PROMPT_TEMPLATE below for why.
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
   abbreviated or misspelled), that brand MUST be the match if it
   appears among the candidates — don't let a higher similarity score on
   a different candidate override an explicit textual match.
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

QUERY: "flea tablet large dog monthly"
CANDIDATES: Nexgard (0.55), Bravecto (0.53), Simparica (0.50)
-> {"brand": "Nexgard", "confidence": "Low", "reason": "No brand named explicitly; picked highest-similarity candidate, but multiple brands plausibly fit a generic 'monthly flea tablet' description."}

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

    response = client_llm.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content.strip())
    result["confidence"] = normalise_confidence(result.get("confidence"))

    return result


# ─────────────────────────────────────────────────────────────
# PHASE 2 — VARIANT / SKU IDENTIFICATION WITHIN THE CHOSEN BRAND
# ─────────────────────────────────────────────────────────────

def search_variants_for_brand(query: str, brand: str, n_results: int = MAX_VARIANT_CANDIDATES) -> list:
    """
    Searches only the chosen brand's variants, using ChromaDB's native
    `where` + `query_embeddings` filtering in a single call — not a
    manual pull-everything-then-rank-in-Python loop.
    """

    if collection.count() == 0:
        return []

    embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        where={"brand": brand},
        n_results=min(n_results, collection.count())
    )

    candidates = []

    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        candidates.append({
            "variant_name": meta.get("variant_name", ""),
            "region": meta.get("region", ""),
            "active_ingredients": meta.get("active_ingredients", ""),
            "species": meta.get("species", ""),
            "pack_size": meta.get("pack_size", ""),
            "pack_quantity": meta.get("pack_quantity", ""),
            "official_brand_url": meta.get("official_brand_url", ""),
            "description": doc,
            "similarity": round(1 - distance, 3),
        })

    candidates.sort(key=lambda x: x["similarity"], reverse=True)

    return candidates


def search_real_examples(query: str, brand: str, n: int = MAX_EXAMPLE_HINTS) -> list:
    """
    Pulls real historical vet-entered phrasing for this brand from the
    sku_examples collection (see sku_examples_builder.py) — a Gemini-
    researched description doesn't capture how vets actually abbreviate
    things, so a few authentic examples help Phase 2 disambiguate. Each
    example carries the real ProductId/SkuId/ProductVariantId it was
    entered under in Labelled.csv, which is the only place those ids
    exist — the live-researched vet_products variants have no id system
    of their own.
    """

    if sku_examples_collection.count() == 0:
        return []

    embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

    try:
        results = sku_examples_collection.query(
            query_embeddings=[embedding],
            where={"brand": brand},
            n_results=min(n, sku_examples_collection.count())
        )
    except Exception:
        return []

    if not results["documents"] or not results["documents"][0]:
        return []

    examples = []

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        examples.append({
            "text": doc,
            "product_id": meta.get("product_id") or "",
            "sku_id": meta.get("sku_id") or "",
            # sku_examples_builder.py stores 0 as a "not available" sentinel
            "product_variant_id": meta.get("product_variant_id") or "",
        })

    return examples


def build_context_phase2(candidates: list) -> str:

    context = ""

    for i, c in enumerate(candidates):
        context += f"""
────────────────────────────────────────────
CANDIDATE {i + 1}
VARIANT: {c['variant_name']}
REGION: {c['region']}
SIMILARITY: {c['similarity']}

FULL DETAILS:
{c['description']}
"""

    return context


def build_examples_block(examples: list) -> str:

    if not examples:
        return "(none available)"

    lines = []

    for e in examples:
        lines.append(
            f'- "{e["text"]}" '
            f'[ProductId: {e["product_id"]}, SkuId: {e["sku_id"]}, ProductVariantId: {e["product_variant_id"]}]'
        )

    return "\n".join(lines)


# Plain (non f-string) template — the golden examples and output schema
# below are full of literal JSON braces, which would otherwise all need
# escaping as {{ }} in an f-string. Substitution is done with .replace()
# on the __TOKEN__ placeholders instead (same pattern as
# knowledge/knowledge_agent2.py's RESEARCH_PROMPT_TEMPLATE).
PHASE2_PROMPT_TEMPLATE = """
### ROLE
You are a veterinary product matching system. Your job is to pick the
exact product variant (pack size, species, region-specific naming) that
a messy vet prescription entry refers to, from a shortlist of candidate
variants already narrowed down to one confirmed brand.

### CONTEXT
The brand has already been identified by an earlier stage — your job is
purely to disambiguate WHICH variant of that brand's product line the
query refers to (size, pack count, species, region-specific name). You
are also given real historical vet-entered examples of how this brand's
products have actually been prescribed in the past, tagged with their
real ProductId/SkuId/ProductVariantId — these are a phrasing/style
reference AND a source of authoritative ids when they clearly match.
Often shorthand WILL be used in the query (e.g. "3pk" for "3 Pipette Pack") or "S/on" for "Spot On" — you must interpret these correctly. 
If the query is ambiguous about pack size, species, or other details, pick the most commonly prescribed variant as primary with Medium confidence, 
and list the other plausible variants as alternatives. 
Never invent a variant that isn't in the candidate list.

### INFO PROVIDED
BRAND (already confirmed): __BRAND__

USER QUERY:
__QUERY__

CANDIDATE VARIANTS:
__CONTEXT__

REAL HISTORICAL PHRASING EXAMPLES FOR THIS BRAND:
__EXAMPLES__

### RULES
1. Identify the BEST matching product variant from the candidates
2. Also identify TWO additional plausible alternatives from the candidates
3. Rank them by likelihood: primary = highest confidence, alternatives = lower or equal
4. All 3 must be different variants if possible; if only 1 real match exists, repeat it as alternatives with LOWER confidence
5. Use ONLY the provided candidate data — never invent variants, sizes, or ids
6. "animal_age" is not available in this data — always return it as an empty string, never guess a value for it
7. "products" and "full_product_name" are DIFFERENT fields and must NEVER contain the same text:
   - "products" is the SHORT product line name only — 1-3 words, no size, no
     species, no pack count. Examples: "Spot On", "Combo", "Spray", "Plus", "Gold".
   - "full_product_name" is the COMPLETE variant name exactly as it appears in
     the matched candidate, including species/size/pack details. Example:
     "Spot On Medium Dog (10-20kg) 6 Pipette Pack".
   - If you find yourself about to write the same string into both fields,
     you have the wrong value for "products" — go back and extract just the
     product line portion of the name.
8. "product_id", "sku_id", and "product_variant_id" should be filled in ONLY
   if one of the real historical examples clearly corresponds to the same
   variant you are matching — copy its ids exactly in that case. A brand can
   legitimately have MULTIPLE different real ids across its history (e.g.
   different ids for different product lines or eras) — there is no single
   "canonical" id for a brand, so match the id to the SPECIFIC variant, not
   to the brand as a whole. If no example confidently matches this specific
   variant, return all three as empty strings. Never invent or guess an id
   number, and never reuse an id from an example that matches a different
   variant just because it's the same brand.
9. "animal_size" should include tags like S, M, L, XL if known, plus the weight range if present, eg: 2-10kg, 10-20kg
10. "region" must be copied exactly from the candidate's REGION field — never invent or guess a region
11. Use ONLY: High, Medium, Low, Very Low for confidence — no numbers or percentages

### GOLDEN EXAMPLES

QUERY: "frontline spot on small dog 3pk"
CANDIDATES include: "Frontline Spot On Small Dog (2-10kg) 3 Pipette Pack" [region: UK/IE], "...6 Pipette Pack" [region: UK/IE]
HISTORICAL EXAMPLES include: "FRONTLINE SPOT ON SML DOG 3PK" [ProductId: 32, SkuId: 1227, ProductVariantId: 197]
-> primary_match: products "Spot On", full_product_name "Spot On Small Dog (2-10kg) 3 Pipette Pack", ids copied exactly (32/1227/197), region "UK/IE", confidence High
(pack count "3pk" directly matched a candidate AND a historical example -> ids copied exactly; note "products" holds only "Spot On", NOT the full name)

QUERY: "frontline spot on small dog" (no pack count given)
CANDIDATES include both the 3-pack and 6-pack Small Dog variants
-> pick the more commonly-prescribed pack size as primary with MEDIUM confidence (not High — the query didn't specify), list the other pack size as an alternative. Never silently guess a specific pack size at High confidence when the query is ambiguous about it.

QUERY: "frontline gold for dogs 45-88lbs"
CANDIDATES only include UK/IE Spot On and Combo variants (no US "Gold" line found for this brand this run)
-> primary_match: confidence "Very Low", reason: "Query references a 'Gold' product line not present in the candidate list for this brand."
(never force-fit a query onto a wrong-region candidate just because it shares a brand — if nothing plausibly matches, say so honestly)

QUERY: "bravecto chew large dog"
CANDIDATES include: "Bravecto Chews for Dogs 20-40kg (44-88 lbs) (Blue)" [no historical example for this exact variant]
HISTORICAL EXAMPLES for this brand exist, but none match this specific size/variant
-> primary_match: products "Chews", full_product_name "Chews for Dogs 20-40kg (44-88 lbs) (Blue)", product_id/sku_id/product_variant_id all "" (left blank — a historical example existed for the brand generally, but not for THIS specific variant, so the ids are not copied)

### OUTPUT FORMAT (STRICT JSON ONLY):

{
  "primary_match": {
    "brand": "__BRAND__",
    "products": "",
    "full_product_name": "",
    "product_id": "",
    "sku_id": "",
    "product_variant_id": "",
    "region": "",
    "species": "",
    "animal_size": "",
    "pack_size": "",
    "pack_quantity": "",
    "animal_age": "",
    "confidence": "",
    "reason": ""
  },
  "alternative_matches": [
    {
        "brand": "__BRAND__",
        "products": "",
        "full_product_name": "",
        "product_id": "",
        "sku_id": "",
        "product_variant_id": "",
        "region": "",
        "species": "",
        "animal_size": "",
        "pack_size": "",
        "pack_quantity": "",
        "animal_age": "",
        "confidence": "",
        "reason": ""
    },
    {
        "brand": "__BRAND__",
        "products": "",
        "full_product_name": "",
        "product_id": "",
        "sku_id": "",
        "product_variant_id": "",
        "region": "",
        "species": "",
        "animal_size": "",
        "pack_size": "",
        "pack_quantity": "",
        "animal_age": "",
        "confidence": "",
        "reason": ""
    }
  ]
}
"""


def _get_valid_id_triplets_for_brand(brand: str) -> set:
    """
    The set of (product_id, sku_id, product_variant_id) triplets that
    genuinely exist in the sku_examples collection for this brand — used
    to catch DeepSeek citing an id combination that doesn't actually
    correspond to any real historical record. Pure metadata lookup, no
    embedding/similarity involved.

    Uses the same `meta.get(...) or ""` normalization as
    search_real_examples() so a real record with no ProductVariantId
    (stored as the sentinel 0) is represented the same way here as it
    was when shown to the LLM as a candidate example.
    """
    if sku_examples_collection.count() == 0:
        return set()

    try:
        data = sku_examples_collection.get(where={"brand": brand}, include=["metadatas"])
    except Exception:
        return set()

    return {
        (
            str(meta.get("product_id") or ""),
            str(meta.get("sku_id") or ""),
            str(meta.get("product_variant_id") or ""),
        )
        for meta in data["metadatas"]
    }


def _validate_and_correct_ids(match: dict, valid_triplets: set, query: str) -> dict:
    """
    Deterministic (non-LLM) safety net: if DeepSeek claimed a
    product_id/sku_id/product_variant_id combination that doesn't match
    any real record for this brand, clear all three rather than let a
    hallucinated or mismatched id through.
    """
    triplet = (
        str(match.get("product_id") or ""),
        str(match.get("sku_id") or ""),
        str(match.get("product_variant_id") or ""),
    )

    if triplet == ("", "", ""):
        return match

    if triplet not in valid_triplets:
        print(
            f"  Correcting mismatched ids for '{query}': claimed {triplet} "
            f"doesn't match any real record — clearing."
        )
        match["product_id"] = ""
        match["sku_id"] = ""
        match["product_variant_id"] = ""

    return match


def ask_deepseek_phase2(query: str, brand: str, candidates: list, real_examples: list) -> dict:

    context = build_context_phase2(candidates)
    examples_block = build_examples_block(real_examples)

    prompt = (
        PHASE2_PROMPT_TEMPLATE
        .replace("__BRAND__", brand)
        .replace("__QUERY__", query)
        .replace("__CONTEXT__", context)
        .replace("__EXAMPLES__", examples_block)
    )

    response = client_llm.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content.strip()
    result = json.loads(raw)

    valid_triplets = _get_valid_id_triplets_for_brand(brand)

    if "primary_match" in result:
        result["primary_match"] = _validate_and_correct_ids(result["primary_match"], valid_triplets, query)

    result["alternative_matches"] = [
        _validate_and_correct_ids(alt, valid_triplets, query)
        for alt in result.get("alternative_matches", [])
    ]

    return result


# ─────────────────────────────────────────────────────────────
# QUERY-TIME LIVE-RESEARCH FALLBACK (opt-in per call, see enable_live_fallback)
# ─────────────────────────────────────────────────────────────

def _live_research_fallback(guessed_brand: str) -> bool:
    """
    Researches a hypothesis brand name live via Gemini search grounding
    and merges any results into ChromaDB, reusing knowledge_agent2.py's
    own research/store functions — imported lazily so a normal query run
    (fallback disabled) never requires a Gemini API key at all.
    """

    from knowledge.knowledge_agent2 import research_brand_with_gemini, store_variant_in_chromadb
    from chromadb_client import get_all_variant_documents

    print(f"Phase 1 confidence too low — live-researching '{guessed_brand}'...")

    data = research_brand_with_gemini(guessed_brand)
    variants = data.get("variants", [])

    if not variants:
        return False

    existing_docs = get_all_variant_documents()
    updated = sum(store_variant_in_chromadb(guessed_brand, v, existing_docs) for v in variants)

    print(f"Live research added/updated {updated} variant(s) for '{guessed_brand}'.")

    return updated > 0


# ─────────────────────────────────────────────────────────────
# RESULT CACHE
# ─────────────────────────────────────────────────────────────
# Vet prescription lists repeat the same product strings constantly
# (many patients on the same product). Caching by cleaned query text
# means repeats are served instantly instead of re-hitting DeepSeek.

_result_cache = {}


def cache_size() -> int:
    return len(_result_cache)


def clear_cache() -> None:
    _result_cache.clear()


def _is_low_confidence_result(result: dict) -> bool:
    primary = result.get("primary_match", {})
    return (
        not primary.get("brand")
        or primary.get("brand") == "Unknown"
        or primary.get("confidence") in ("Low", "Very Low")
    )


def _cache_usable(cleaned_query: str, enable_live_fallback: bool) -> bool:
    """
    A cache hit is only skipped when the caller now wants live fallback
    AND the cached result was low-confidence — otherwise a query that was
    weakly resolved once (fallback off) would silently keep returning
    that same stale answer forever, even after the caller asks this call
    to try harder. Good cached answers are always reused regardless.
    """
    if cleaned_query not in _result_cache:
        return False

    if enable_live_fallback and _is_low_confidence_result(_result_cache[cleaned_query]):
        return False

    return True


def _unknown_result(reason: str, brand: str = "Unknown", confidence: str = "Low") -> dict:
    empty = {
        "brand": brand, "products": "", "full_product_name": "",
        "product_id": "", "sku_id": "", "product_variant_id": "", "region": "",
        "species": "", "animal_size": "", "pack_size": "", "pack_quantity": "",
        "animal_age": "", "confidence": confidence, "reason": reason,
    }
    return {"primary_match": empty, "alternative_matches": []}


def _resolve_with_brand_candidates(
    cleaned_query: str,
    brand_candidates: list,
    enable_live_fallback: bool = DEFAULT_ENABLE_LIVE_FALLBACK
) -> dict:

    phase1_result = ask_deepseek_phase1(cleaned_query, brand_candidates) if brand_candidates else None

    low_confidence = (
        not brand_candidates
        or phase1_result is None
        or phase1_result.get("confidence") in ("Low", "Very Low")
    )

    if low_confidence and enable_live_fallback:

        guessed_brand = (
            phase1_result.get("brand") if phase1_result and phase1_result.get("brand")
            else guess_brand_token(cleaned_query).title()
        )

        if guessed_brand and _live_research_fallback(guessed_brand):
            brand_candidates = search_brands(cleaned_query)
            phase1_result = ask_deepseek_phase1(cleaned_query, brand_candidates) if brand_candidates else None

    if not brand_candidates or not phase1_result or not phase1_result.get("brand"):
        return _unknown_result("No matching brand found in ChromaDB.")

    brand = phase1_result["brand"]

    variant_candidates = search_variants_for_brand(cleaned_query, brand)

    if not variant_candidates:
        return _unknown_result(
            f"Brand identified as {brand}, but no stored variants matched.",
            brand=brand,
            confidence=phase1_result.get("confidence", "Low")
        )

    real_examples = search_real_examples(cleaned_query, brand)

    return ask_deepseek_phase2(cleaned_query, brand, variant_candidates, real_examples)


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE — SINGLE QUERY
# ─────────────────────────────────────────────────────────────

def identify_product(
    query: str,
    use_cache: bool = True,
    enable_live_fallback: bool = DEFAULT_ENABLE_LIVE_FALLBACK
) -> dict:

    if not query or not query.strip():
        return {"error": "Empty query"}

    cleaned = clean_query(query)

    if not cleaned:
        return {"error": "Empty query after cleaning"}

    if use_cache and _cache_usable(cleaned, enable_live_fallback):
        return _result_cache[cleaned]

    brand_candidates = search_brands(cleaned)
    result = _resolve_with_brand_candidates(cleaned, brand_candidates, enable_live_fallback)

    if use_cache:
        _result_cache[cleaned] = result

    return result


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE — LARGE FILE BATCHES
# ─────────────────────────────────────────────────────────────

def identify_products_batch(
    queries: list,
    max_workers: int = MAX_WORKERS,
    progress_callback=None,
    enable_live_fallback: bool = DEFAULT_ENABLE_LIVE_FALLBACK
):
    """
    Efficiently classifies a large list of raw prescription strings:

    1. Cleans and de-duplicates the queries — identical entries are
       only reasoned over once, and reuse the shared result cache.
    2. Embeds + searches ChromaDB for Phase 1 brand candidates for every
       unique query in ONE round trip instead of one call per row.
    3. Fans the remaining two-phase DeepSeek reasoning out across a
       thread pool (DeepSeek is I/O-bound, so this overlaps network
       latency instead of processing the file one row at a time).

    progress_callback(completed, total_unique_calls) is invoked as each
    query finishes its reasoning, so a UI can render live progress.

    Returns (results, stats) where results is aligned index-for-index
    with `queries`, and stats reports how much work was skipped.
    """

    total = len(queries)
    cleaned = [clean_query(q) for q in queries]

    unique_all = list(dict.fromkeys(c for c in cleaned if c))
    unique_pending = [c for c in unique_all if not _cache_usable(c, enable_live_fallback)]

    if unique_pending:

        brand_candidates_batch = search_brands_batch(unique_pending)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:

            futures = {
                pool.submit(_resolve_with_brand_candidates, cq, cands, enable_live_fallback): cq
                for cq, cands in zip(unique_pending, brand_candidates_batch)
            }

            for future in as_completed(futures):
                cq = futures[future]
                _result_cache[cq] = future.result()
                completed += 1

                if progress_callback:
                    progress_callback(completed, len(unique_pending))

    results = []

    for c in cleaned:
        if not c:
            results.append({"error": "Empty query"})
        else:
            results.append(_result_cache.get(c, {"error": "Processing failed"}))

    stats = {
        "total_rows": total,
        "unique_queries": len(unique_all),
        "llm_calls": len(unique_pending),
        "cache_hits": len(unique_all) - len(unique_pending),
    }

    return results, stats


# ─────────────────────────────────────────────────────────────
# TESTING
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_queries = [
        "Frontline Spray 100ml for dogs",
        "Frontline spot on large dog",
        "fleas treatment cat 3 pipettes",
        "frontline 250ml spray",
        "tick treatment small dog frontline",
        "frontline spot on dog xl 6",
        "frontline plus dog 3 pack"
    ]

    print("=" * 60)
    print("DEEPSEEK VETERINARY PRODUCT CLASSIFIER (two-phase)")
    print("=" * 60)

    for q in test_queries:

        print(f"\nQUERY: {q}")

        result = identify_product(q)

        print(json.dumps(result, indent=2))
        print("-" * 60)
