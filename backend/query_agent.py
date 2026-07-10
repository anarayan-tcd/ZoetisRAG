import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
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
# BRAND -> PRODUCT ID LOOKUP (from brand_seed_list.csv)
# ─────────────────────────────────────────────────────────────
# Every brand has at least one real ProductId in the historical data —
# it's the brand-level id, one tier above sku_id/product_variant_id (see
# ProductId -> ProductVariantId -> SkuId hierarchy: one ProductVariantId
# can have several SkuIds, e.g. Frontline Spray's 100ml/250ml/500ml packs).
# This is used so Phase 2 can guarantee product_id even when no historical
# example confidently matches the specific variant/pack size — that
# uncertainty only ever concerns sku_id/product_variant_id, never
# product_id itself.
#
# A minority of brands (16/146) have MORE THAN ONE historical ProductId,
# merged under one canonical brand name by extract_brands.py (different
# catalog eras/groupings, e.g. Frontline -> [32, 182]). For those, the id
# used by the most historical rows is picked as the default.

BRAND_SEED_CSV = Path(__file__).resolve().parent / "data" / "generated" / "brand_seed_list.csv"
LABELLED_CSV = Path(__file__).resolve().parent / "data" / "raw" / "Labelled.csv"


def _load_brand_product_ids() -> dict:
    """Returns {brand: {"primary": id, "all": [ids...]}}."""
    if not BRAND_SEED_CSV.exists() or not LABELLED_CSV.exists():
        return {}

    seed_df = pd.read_csv(BRAND_SEED_CSV)
    id_counts = pd.read_csv(LABELLED_CSV, usecols=["ProductId"])["ProductId"].value_counts()

    mapping = {}

    for _, row in seed_df.iterrows():
        brand = str(row["Brand"]).strip()
        raw = row.get("SourceProductIds")

        if not raw or pd.isna(raw):
            continue

        ids = [int(x) for x in str(raw).split(";") if x.strip().isdigit()]

        if not ids:
            continue

        primary = max(ids, key=lambda pid: id_counts.get(pid, 0))
        mapping[brand] = {"primary": primary, "all": ids}

    return mapping


BRAND_PRODUCT_IDS = _load_brand_product_ids()


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


def _deepseek_json(prompt: str, attempts: int = 3) -> dict:
    """
    Calls DeepSeek and parses its JSON reply, retrying on an empty or
    malformed response before giving up. DeepSeek occasionally returns
    empty content (the cause of "Expecting value: line 1 column 1" errors),
    which is usually transient — a retry clears it. Raises after the final
    attempt so the caller (or the batch loop's per-row guard) can decide
    how to handle a genuine, repeated failure.
    """
    last_err = None

    for _ in range(attempts):
        response = client_llm.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw = (response.choices[0].message.content or "").strip()

        if not raw:
            last_err = ValueError("empty response content")
            continue

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            last_err = e

    raise ValueError(f"DeepSeek returned no valid JSON after {attempts} attempt(s): {last_err}")


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

    result = _deepseek_json(prompt)
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
    """Real historical phrasings for Phase 2, WITH their real
    ProductId/ProductVariantId — Phase 2 matches these at the product-LINE
    level to assign product_id/product_variant_id (Rule 8). SkuId is
    deliberately not shown here: exact pack size is Phase 3b's job, once
    the variant is confirmed."""

    if not examples:
        return "(none available)"

    lines = []
    for e in examples:
        lines.append(
            f'- "{e["text"]}" '
            f'[ProductId: {e["product_id"]}, ProductVariantId: {e["product_variant_id"]}]'
        )
    return "\n".join(lines)


def build_known_product_id_block(brand: str) -> str:
    """Renders this brand's known ProductId(s) from BRAND_PRODUCT_IDS for
    the Phase 2 prompt — the guaranteed brand-level fallback, restored
    alongside Rule 8."""

    info = BRAND_PRODUCT_IDS.get(brand)

    if not info:
        return "(no known ProductId on file for this brand)"

    if len(info["all"]) == 1:
        return str(info["primary"])

    others = [str(i) for i in info["all"] if i != info["primary"]]
    return f'{info["primary"]} (most common) — other known ids for this brand: {", ".join(others)}'


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
variants already narrowed down to one confirmed brand, and to identify
the ProductId and ProductVariantId for that variant (product-LINE level).
You do NOT decide the exact pack size's SkuId here — that is resolved by
a separate, later stage once the variant/product line is confirmed.

### CONTEXT
The brand has already been identified by an earlier stage — your job is
to disambiguate WHICH variant of that brand's product line the query
refers to (size, pack count, species, region-specific name), and to
attach its ProductId/ProductVariantId using real historical vet-entered
examples of how this brand's products have actually been prescribed,
each tagged with its real ProductId/ProductVariantId. These ids form a
hierarchy: ProductId is the BRAND-level id (one per brand, occasionally
a couple across catalog eras — always known); ProductVariantId
identifies a specific product LINE within the brand (e.g. "Spray" vs
"Combo Cat") — match at THIS level, ignoring exact pack size/count
differences (a historical example for the 250ml pack still confirms the
same ProductVariantId as the 500ml pack you're matching, since they're
the same product line). Exact pack size (SkuId) is deliberately not your
concern — a later stage handles it once your variant match is locked in.
Often shorthand WILL be used in the query (e.g. "3pk" for "3 Pipette Pack",
"S/on" for "Spot On", "sml/med/lge" for size) — you must interpret these
correctly. If the query is ambiguous about pack size, species, or other
details, pick the most commonly prescribed variant as primary with Medium
confidence, and list the other plausible variants as alternatives. Never
invent a variant that isn't in the candidate list.

### INFO PROVIDED
BRAND (already confirmed): __BRAND__

KNOWN PRODUCT ID(S) FOR THIS BRAND (from historical records): __KNOWN_PRODUCT_ID__

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
5. Use ONLY the provided candidate data — never invent variants or sizes
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
8. Every brand has at least one known real ProductId — given above as
   "KNOWN PRODUCT ID(S) FOR THIS BRAND" — so "product_id" must NEVER be
   left blank once a brand has been confirmed:
   - If one of the real historical examples clearly corresponds to the
     SAME product LINE you are matching (species + product line — pack
     size/count does NOT need to match), copy that example's product_id
     AND product_variant_id exactly, together from that one record.
   - Otherwise, still set "product_id" to the known id given above (use
     the one marked "most common" if more than one is listed) — but
     leave "product_variant_id" as an empty string, since that depends
     on a real historical example matching the same product line, which
     didn't happen here.
   Never invent or guess an id number, and never reuse product_variant_id
   from an example that matches a DIFFERENT product line just because
   it's the same brand — but product_id itself always comes from either
   a matching example or the known brand-level id above.
9. "animal_size" should include tags like S, M, L, XL if known, plus the weight range if present, eg: 2-10kg, 10-20kg
10. "region" must be copied exactly from the candidate's REGION field — never invent or guess a region
11. Use ONLY: High, Medium, Low, Very Low for confidence — no numbers or percentages
12. Do NOT output a sku_id field — pack size is resolved by a later stage.

### GOLDEN EXAMPLES

QUERY: "frontline spot on small dog 3pk"
CANDIDATES include: "Frontline Spot On Small Dog (2-10kg) 3 Pipette Pack" [region: UK/IE], "...6 Pipette Pack" [region: UK/IE]
HISTORICAL EXAMPLES include: "FRONTLINE SPOT ON SML DOG 3PK" [ProductId: 32, ProductVariantId: 197]
-> primary_match: products "Spot On", full_product_name "Spot On Small Dog (2-10kg) 3 Pipette Pack", ids copied exactly (32/197), region "UK/IE", confidence High
("products" holds only "Spot On", NOT the full name; pack count itself is NOT part of the id-matching decision — species + product line matched)

QUERY: "frontline spot on small dog" (no pack count given)
CANDIDATES include both the 3-pack and 6-pack Small Dog variants
-> pick the more commonly-prescribed pack size as primary with MEDIUM confidence (query didn't specify), list the other pack size as an alternative. product_id/product_variant_id are unaffected by this ambiguity — they're decided by product line, not pack size.

QUERY: "advocate dog ex lge >25kg pk6"
CANDIDATES include: "Advocate Spot-On for Extra Large Dogs (over 25kg) 6 Pack" [region: UK/IE]
HISTORICAL EXAMPLES include: "ADVOCATE DOG 400 EX LGE (>25KG) [PK3]" [ProductId: 1, ProductVariantId: 7] (a DIFFERENT pack count, same product line)
-> primary_match: products "Spot-On", full_product_name "Spot-On for Extra Large Dogs (over 25kg) 6 Pack", ids copied exactly (1/7), region "UK/IE", confidence High
(the historical example is a 3-pack, not the 6-pack being matched — but it's the SAME product line/species/weight-band, so the ids still apply)

QUERY: "drontal cat xl"
CANDIDATES include: "Drontal Cat XL Wormer Tablet for Large Cats (over 6kg) 1 Tablet" [region: UK/IE]
HISTORICAL EXAMPLES include: "DRONTAL CAT XL TABS" [ProductId: 3, ProductVariantId: 22]
-> primary_match: products "Cat XL", full_product_name "Cat XL Wormer Tablet for Large Cats (over 6kg) 1 Tablet", ids copied exactly (3/22), confidence High
(a tablet wormer, not a spot-on; "products" is the short line "Cat XL")

QUERY: "seresto collar lge dog >8kg"
CANDIDATES include: "Seresto Flea & Tick Collar for Large Dogs (over 8kg)" [region: UK/IE], "...for Small Dogs (under 8kg)" [region: UK/IE]
HISTORICAL EXAMPLES include: "seresto collar lg dog > 8kg" [ProductId: 6, ProductVariantId: 47], "seresto collar sml" [ProductId: 6, ProductVariantId: 49]
-> primary_match: products "Collar", full_product_name "Flea & Tick Collar for Large Dogs (over 8kg)", ids copied exactly (6/47), confidence High
(the Large-dog and Small-dog collars are DIFFERENT product lines/ids — pick the one matching species+size, not just brand)

QUERY: "milbemax chewy dog" (no size/strength given)
CANDIDATES include the Small Dog and the Large Dog chewable variants
-> pick the more commonly-prescribed size as primary with MEDIUM confidence — the same ambiguity rule as pack size, applied to species/weight. If the Small/Large variants have DIFFERENT ProductVariantIds, only assign the id matching the chosen size; if uncertain which, leave product_variant_id blank rather than guess.

QUERY: "bravecto chew large dog"
CANDIDATES include: "Bravecto Chews for Dogs 20-40kg (44-88 lbs) (Blue) 1 Tablet" [no historical example for this product line/size band]
KNOWN PRODUCT ID(S) FOR THIS BRAND: 22 (most common) — other known ids for this brand: 89
-> primary_match: products "Chews", full_product_name "Chews for Dogs 20-40kg (44-88 lbs) (Blue) 1 Tablet", product_id "22" (known brand-level id — always filled), product_variant_id "" (no historical example matched this product line/size band)

QUERY: "frontline gold for dogs 45-88lbs"
CANDIDATES only include UK/IE Spot On and Combo variants (no US "Gold" line found for this brand this run)
-> primary_match: confidence "Very Low", reason: "Query references a 'Gold' product line not present in the candidate list for this brand."
(never force-fit a query onto a wrong-region candidate just because it shares a brand — if nothing plausibly matches, say so honestly)

### OUTPUT FORMAT (STRICT JSON ONLY, no sku_id field):

{
  "primary_match": {
    "brand": "__BRAND__",
    "products": "",
    "full_product_name": "",
    "product_id": "",
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


# Phase 3b — a SMALL, focused LLM call that runs ONLY for the primary match
# and ONLY when the confirmed variant has more than one real SkuId (pack
# size). It picks the pack size; product_id/product_variant_id come from the
# deterministic Phase 3a lookup, not from here.
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
4. If the query does NOT specify any pack size/quantity at all, pick the candidate with the most historical phrasings (the most commonly prescribed pack) and set confidence = Medium.
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
CANDIDATES: SkuId 229 [3 phrasings, "...100ml"], SkuId 219 [1 phrasing, "...250ml"], SkuId 220 [2 phrasings, "...500ml"]
-> {"sku_id": "229", "confidence": "Medium", "reason": "No pack size specified; SkuId 229 (100ml) has the most historical phrasings, so it's the most commonly prescribed pack."}

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


def _get_valid_variant_pairs_for_brand(brand: str) -> set:
    """
    The set of (product_id, product_variant_id) pairs that genuinely exist
    in sku_examples for this brand — used to catch DeepSeek citing a
    product-line id combination (Phase 2's job) that doesn't correspond to
    any real historical record. Product-line level, NOT pack-size level —
    SkuId is validated separately once Phase 3b resolves it.
    """
    if sku_examples_collection.count() == 0:
        return set()

    try:
        data = sku_examples_collection.get(where={"brand": brand}, include=["metadatas"])
    except Exception:
        return set()

    return {
        (str(meta.get("product_id") or ""), str(meta.get("product_variant_id") or ""))
        for meta in data["metadatas"]
    }


def _validate_and_correct_variant_pair(match: dict, valid_pairs: set, query: str) -> dict:
    """
    Deterministic (non-LLM) safety net for Phase 2's Rule 8: if DeepSeek
    claimed a product_id/product_variant_id pair that doesn't match any
    real record for this brand, clear product_variant_id (product_id is
    backstopped separately via the brand-level fallback, so it's left
    alone here). No claim (product_variant_id blank) needs no check.
    """
    pid = str(match.get("product_id") or "")
    vid = str(match.get("product_variant_id") or "")

    if vid == "":
        return match

    if (pid, vid) not in valid_pairs:
        print(
            f"  Correcting mismatched variant id for '{query}': claimed "
            f"({pid}, {vid}) doesn't match any real record — clearing."
        )
        match["product_variant_id"] = ""

    return match


def _get_valid_id_triplets_for_brand(brand: str) -> set:
    """
    The set of (product_id, sku_id, product_variant_id) triplets that
    genuinely exist in the sku_examples collection for this brand — the
    final backstop once Phase 3b has (or hasn't) resolved a specific
    sku_id. Pure metadata lookup, no embedding/similarity involved.
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
    Deterministic (non-LLM) final backstop: only runs when a SPECIFIC
    sku_id was actually assigned (by Phase 3b, or the single-SKU
    shortcut) — that's the only case making a claim valid_triplets can
    verify. A blank sku_id (variant known but pack size not resolved,
    e.g. an alternative match, or a variant with no real SKUs) makes no
    specific-record claim and needs no check.
    """
    triplet = (
        str(match.get("product_id") or ""),
        str(match.get("sku_id") or ""),
        str(match.get("product_variant_id") or ""),
    )

    if triplet[1] == "":
        return match

    if triplet not in valid_triplets:
        print(
            f"  Correcting mismatched ids for '{query}': claimed {triplet} "
            f"doesn't match any real record — clearing sku_id."
        )
        match["sku_id"] = ""

    return match


def _brand_fallback_product_id(brand: str) -> str:
    """The known brand-level ProductId (most common if a brand has several)
    — the guarantee behind Rule 8: product_id is never blank once the
    brand is known, even if no historical example matched the variant."""
    info = BRAND_PRODUCT_IDS.get(brand)
    return str(info["primary"]) if info else ""


def _id_str(v) -> str:
    """Normalizes a ChromaDB metadata id to a plain string; the sentinel 0
    (no ProductVariantId/SkuId recorded) and None both become ""."""
    if v in (None, 0, "0"):
        return ""
    return str(v)


def _candidate_skus_for_variant(brand: str, variant_id: str) -> dict:
    """Every distinct SkuId under this (brand, product_variant_id), each with
    its product_id and up to 3 representative real phrasings. Feeds Phase 3b.
    Returns {sku_id_str: {"product_id": str, "notes": [str, ...]}}."""
    try:
        data = sku_examples_collection.get(
            where={"$and": [{"brand": brand}, {"product_variant_id": int(variant_id)}]},
            include=["documents", "metadatas"],
        )
    except (ValueError, Exception):
        return {}

    skus = {}

    for doc, meta in zip(data["documents"], data["metadatas"]):
        sid = _id_str(meta.get("sku_id"))
        if not sid:
            continue
        entry = skus.setdefault(sid, {"product_id": _id_str(meta.get("product_id")), "notes": []})
        if doc not in entry["notes"] and len(entry["notes"]) < 3:
            entry["notes"].append(doc)

    return skus


# ── PHASE 3b — focused LLM pack-size pick (primary match only) ────

def build_sku_candidates_block(candidates: dict) -> str:
    lines = []
    for sid, info in candidates.items():
        phrasings = "; ".join(f'"{n}"' for n in info["notes"]) or "(no phrasing recorded)"
        lines.append(f"SkuId {sid} [{len(info['notes'])} phrasing(s)] — {phrasings}")
    return "\n".join(lines)


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
        result = _deepseek_json(prompt)
    except Exception:
        return ""

    sku = _id_str(result.get("sku_id"))
    return sku if sku in candidates else ""


def _resolve_sku(query: str, brand: str, match: dict, run_pack_llm: bool) -> dict:
    """
    Resolves sku_id (exact pack size) for a match whose product_id /
    product_variant_id have ALREADY been assigned by Phase 2's LLM
    (Rule 8, matched at the product-line level) and pair-validated:
      - No product_variant_id known -> nothing to resolve, sku_id blank.
      - Exactly one real SkuId under this (brand, variant) -> take it
        directly, no LLM call needed.
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


def _finalize_match_ids(query: str, brand: str, match: dict, valid_pairs: set, run_pack_llm: bool) -> dict:
    """
    Full id-resolution pipeline for one match, run after Phase 2's LLM has
    already picked product_id/product_variant_id (Rule 8):
      1. Validate the (product_id, product_variant_id) pair against real
         sku_examples records for this brand — clears product_variant_id
         if DeepSeek claimed a pair that doesn't actually exist.
      2. Guarantee product_id via the brand-level fallback if still blank.
      3. Resolve sku_id (Phase 3b or the single-SKU shortcut).
      4. Final triplet backstop — only meaningful once sku_id is set.
    """
    match = _validate_and_correct_variant_pair(match, valid_pairs, query)
    match["product_id"] = match.get("product_id") or _brand_fallback_product_id(brand)
    match = _resolve_sku(query, brand, match, run_pack_llm)
    return match


def ask_deepseek_phase2(query: str, brand: str, candidates: list, real_examples: list) -> dict:
    """
    Phase 2: the LLM picks the variant/characteristics AND product_id /
    product_variant_id (Rule 8, matched at the product-line level, real
    historical examples as the source of truth) in one call. Exact pack
    size (sku_id) is then resolved separately in _finalize_match_ids —
    directly if the variant has only one real SkuId, otherwise via
    Phase 3b (a small, focused LLM call, primary match only).
    """
    context = build_context_phase2(candidates)
    examples_block = build_examples_block(real_examples)
    known_product_id_block = build_known_product_id_block(brand)

    prompt = (
        PHASE2_PROMPT_TEMPLATE
        .replace("__BRAND__", brand)
        .replace("__QUERY__", query)
        .replace("__CONTEXT__", context)
        .replace("__EXAMPLES__", examples_block)
        .replace("__KNOWN_PRODUCT_ID__", known_product_id_block)
    )

    result = _deepseek_json(prompt)

    valid_pairs = _get_valid_variant_pairs_for_brand(brand)

    if "primary_match" in result:
        result["primary_match"] = _finalize_match_ids(query, brand, result["primary_match"], valid_pairs, run_pack_llm=True)

    result["alternative_matches"] = [
        _finalize_match_ids(query, brand, alt, valid_pairs, run_pack_llm=False)
        for alt in result.get("alternative_matches", [])
    ]

    # Final deterministic backstop against a hallucinated sku_id specifically.
    valid_triplets = _get_valid_id_triplets_for_brand(brand)

    if "primary_match" in result:
        result["primary_match"] = _validate_and_correct_ids(result["primary_match"], valid_triplets, query)

    result["alternative_matches"] = [
        _validate_and_correct_ids(alt, valid_triplets, query)
        for alt in result["alternative_matches"]
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

    # Rows that failed this run (e.g. a repeated malformed DeepSeek reply)
    # go here, NOT into _result_cache — so one bad row can't abort the
    # whole batch, and a later re-run retries it instead of serving a
    # cached failure. Only genuine successes are cached.
    failures = {}
    failed_count = 0

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

                try:
                    _result_cache[cq] = future.result()
                except Exception as e:
                    failures[cq] = _unknown_result(f"Processing failed: {e}")
                    failed_count += 1

                completed += 1

                if progress_callback:
                    progress_callback(completed, len(unique_pending))

    results = []

    for c in cleaned:
        if not c:
            results.append({"error": "Empty query"})
        elif c in _result_cache:
            results.append(_result_cache[c])
        else:
            results.append(failures.get(c, _unknown_result("Processing failed")))

    stats = {
        "total_rows": total,
        "unique_queries": len(unique_all),
        "llm_calls": len(unique_pending),
        "cache_hits": len(unique_all) - len(unique_pending),
        "failed": failed_count,
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
