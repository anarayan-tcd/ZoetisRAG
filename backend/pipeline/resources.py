# resources.py — shared setup and small cross-phase utilities for the
# query-time matching pipeline (query_agent.py + phase1_brand.py +
# phase2_variant.py + phase3b_sku.py + llm_judge.py). Centralized here so
# every phase module imports FROM this one file rather than reaching into
# each other or back into query_agent.py, which would create import cycles.

import json
import re
from pathlib import Path
import os

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from chromadb_client import get_embedding_model, get_collection, ACTIVE_SKU_EXAMPLES_COLLECTION


# ─────────────────────────────────────────────────────────────
# ENV SETUP
# ─────────────────────────────────────────────────────────────

env_path = Path(__file__).resolve().parent.parent / "APIKeys.env"
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
collection = get_collection()                                    # vet_products
sku_examples_collection = get_collection(ACTIVE_SKU_EXAMPLES_COLLECTION)


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

BRAND_SEED_CSV = Path(__file__).resolve().parent.parent / "data" / "generated" / "brand_seed_list.csv"
LABELLED_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "Labelled.csv"


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
# KNOWN BRAND WORD INDEX (for Phase 1's fuzzy garbled-brand recovery)
# ─────────────────────────────────────────────────────────────
# Vet-entered queries are sometimes heavily obscured — leetspeak digit
# substitution ("Str0ngh0ld"), missing letters ("Profndr"), decorative
# symbols spliced into the brand token ("Front-lïne", "Bravct0") — which
# can break the vector search badly enough that the correct brand never
# even makes Phase 1's candidate list, so DeepSeek can't pick it no
# matter how good its reasoning is. phase1_brand.py's fuzzy_brand_match
# uses this word -> canonical-brand index to recover a candidate in that
# case via plain string comparison (no extra DB/LLM call, negligible
# cost). Built once at startup from the same brand list Phase 2 already
# treats as authoritative (BRAND_PRODUCT_IDS).

_BRAND_WORD_MIN_LEN = 5


def _build_known_brand_words(brand_product_ids: dict) -> dict:
    """Maps each significant word (>=5 letters) of every known brand name
    to that brand. First brand seen for a given word wins on collision —
    rare in practice, and this is only ever an ADDITIONAL candidate
    alongside real vector-search results, never a replacement, so a rare
    wrong pick here just gives DeepSeek one extra (ignorable) option."""
    words = {}
    for brand in brand_product_ids:
        for word in re.findall(r"[a-zA-Z]+", brand.lower()):
            if len(word) >= _BRAND_WORD_MIN_LEN:
                words.setdefault(word, brand)
    return words


KNOWN_BRAND_WORDS = _build_known_brand_words(BRAND_PRODUCT_IDS)


def brand_fallback_product_id(brand: str) -> str:
    """The known brand-level ProductId (most common if a brand has several)
    — the guarantee behind Phase 2's Rule 8: product_id is never blank
    once the brand is known, even if no historical example matched the
    variant."""
    info = BRAND_PRODUCT_IDS.get(brand)
    return str(info["primary"]) if info else ""


def id_str(v) -> str:
    """Normalizes a ChromaDB metadata id to a plain string; the sentinel 0
    (no ProductVariantId/SkuId recorded) and None both become ""."""
    if v in (None, 0, "0"):
        return ""
    return str(v)


# ─────────────────────────────────────────────────────────────
# MULTILINGUAL SPECIES HINTS (deterministic — no LLM, no API cost)
# ─────────────────────────────────────────────────────────────
# Real vet notes routinely name the animal in another language ("perro",
# "chien", "hund", "katze", "gato"...), and DeepSeek sometimes misses the
# species signal when it's non-English — the InitialTest CSV got species
# wrong on exactly these rows. A plain token-lookup catches them for free
# and injects an explicit species hint into the Phase 2 / judge prompts.
#
# Keys are ascii-lowercase (clean_query already NFKD-strips accents, so
# "kätzchen" arrives as "katzchen"). Only unambiguous animal words are
# listed — no English ones (the LLM handles those natively) and nothing
# that collides with common product/vet vocabulary.

SPECIES_LANG_TOKENS = {
    # dogs
    "perro": ("Dog", "Spanish"), "perros": ("Dog", "Spanish"),
    "chien": ("Dog", "French"), "chiens": ("Dog", "French"),
    "hund": ("Dog", "German/Scandinavian"), "hunde": ("Dog", "German"),
    "hunden": ("Dog", "German/Norwegian"),
    "cane": ("Dog", "Italian"), "cani": ("Dog", "Italian"),
    "hond": ("Dog", "Dutch"), "honden": ("Dog", "Dutch"),
    "cao": ("Dog", "Portuguese"), "cachorro": ("Dog", "Portuguese/Spanish"),
    "inu": ("Dog", "Japanese"),
    "chiot": ("Puppy", "French"), "welpe": ("Puppy", "German"),
    "cucciolo": ("Puppy", "Italian"),
    # cats
    "gato": ("Cat", "Spanish/Portuguese"), "gatos": ("Cat", "Spanish/Portuguese"),
    "gata": ("Cat", "Spanish/Portuguese"),
    "chat": ("Cat", "French"), "chats": ("Cat", "French"),
    "chatte": ("Cat", "French"),
    "katze": ("Cat", "German"), "katzen": ("Cat", "German"),
    "gatto": ("Cat", "Italian"), "gatti": ("Cat", "Italian"),
    "kat": ("Cat", "Dutch/Danish"), "katten": ("Cat", "Dutch"),
    "katt": ("Cat", "Swedish/Norwegian"),
    "neko": ("Cat", "Japanese"),
    "chaton": ("Kitten", "French"), "gatito": ("Kitten", "Spanish"),
    "katzchen": ("Kitten", "German"), "gattino": ("Kitten", "Italian"),
    # other treated species
    "lapin": ("Rabbit", "French"), "kaninchen": ("Rabbit", "German"),
    "conejo": ("Rabbit", "Spanish"), "coniglio": ("Rabbit", "Italian"),
    "konijn": ("Rabbit", "Dutch"),
    "cheval": ("Horse", "French"), "pferd": ("Horse", "German"),
    "caballo": ("Horse", "Spanish"), "cavallo": ("Horse", "Italian"),
    "paard": ("Horse", "Dutch"),
}

# Non-English PRODUCT-FORM / PACKAGING words, same idea as the species map:
# vets mix these into otherwise-English notes constantly ("caja 3",
# "comprimes 112.5 mg", "packung 6"), and a missed one can cost the model
# the pack-count or formulation signal. Values are the English term the
# word should be read as. Same ascii-normalized keys (clean_query strips
# accents: "boîte"->"boite", "comprimés"->"comprimes"). Ambiguous words
# that collide with English/product vocabulary are deliberately excluded.
FORM_LANG_TOKENS = {
    # pipette
    "pipeta": ("pipette", "Spanish/Portuguese"), "pipetas": ("pipette", "Spanish/Portuguese"),
    "pipetten": ("pipette", "German/Dutch"),
    # pack / box
    "caja": ("pack/box", "Spanish"), "cajas": ("pack/box", "Spanish"),
    "boite": ("pack/box", "French"), "boites": ("pack/box", "French"),
    "packung": ("pack/box", "German"), "packungen": ("pack/box", "German"),
    "schachtel": ("pack/box", "German"), "doos": ("pack/box", "Dutch"),
    "scatola": ("pack/box", "Italian"), "caixa": ("pack/box", "Portuguese"),
    "paquete": ("pack/box", "Spanish"), "paquet": ("pack/box", "French"),
    # tablets
    "comprime": ("tablet(s)", "French"), "comprimes": ("tablet(s)", "French"),
    "comprimido": ("tablet(s)", "Spanish/Portuguese"), "comprimidos": ("tablet(s)", "Spanish/Portuguese"),
    "tabletten": ("tablet(s)", "German/Dutch"), "tablette": ("tablet(s)", "German/French"),
    "compressa": ("tablet(s)", "Italian"), "compresse": ("tablet(s)", "Italian"),
    "tableta": ("tablet(s)", "Spanish"), "tabletas": ("tablet(s)", "Spanish"),
    # bottle
    "botella": ("bottle", "Spanish"), "bouteille": ("bottle", "French"),
    "flacon": ("bottle", "French"), "flasche": ("bottle", "German"),
    "fles": ("bottle", "Dutch"), "bottiglia": ("bottle", "Italian"),
    "flacone": ("bottle", "Italian"), "frasco": ("bottle", "Spanish/Portuguese"),
    # drops
    "gotas": ("drops", "Spanish/Portuguese"), "gouttes": ("drops", "French"),
    "tropfen": ("drops", "German"), "gocce": ("drops", "Italian"),
    # collar
    "collier": ("collar", "French"), "halsband": ("collar", "German/Dutch"),
    "collare": ("collar", "Italian"),
    # wormer
    "vermifuge": ("wormer", "French"), "wurmkur": ("wormer", "German"),
    "vermifugo": ("wormer", "Italian/Portuguese"),
    # dose
    "dosis": ("dose", "Spanish/German"), "dosi": ("dose", "Italian"),
    # paste
    "pasta": ("paste", "Spanish/Italian"), "pomada": ("paste/ointment", "Spanish/Portuguese"),
    "pommade": ("paste/ointment", "French"), "salbe": ("paste/ointment", "German"),
    # injection
    "inyeccion": ("injection", "Spanish"), "injektion": ("injection", "German"),
    "iniezione": ("injection", "Italian"), "injectie": ("injection", "Dutch"),
    # chewable
    "kautablette": ("chewable tablet", "German"), "kautabletten": ("chewable tablet", "German"),
    "masticable": ("chewable", "Spanish"), "masticables": ("chewable", "Spanish"),
    # sachet
    "beutel": ("sachet/packet", "German"), "bustina": ("sachet/packet", "Italian"),
}

_LANG_TOKEN_RE = re.compile(r"[a-z]+")


def detect_language_hints(cleaned_query: str) -> str:
    """Scans a cleaned (lowercase, accent-stripped) query for non-English
    animal AND product-form/packaging words, returning a human-readable
    hint line for the prompts, e.g.:
      'Cat (query contains "chat", French); pack/box (query contains
       "caja", Spanish)'
    Returns "" when nothing is detected — callers render that as
    "(none detected)". Pure string lookup: zero API cost, deterministic."""
    hits = []
    seen = set()

    for token in _LANG_TOKEN_RE.findall(cleaned_query or ""):
        if token in SPECIES_LANG_TOKENS:
            meaning, lang = SPECIES_LANG_TOKENS[token]
        elif token in FORM_LANG_TOKENS:
            meaning, lang = FORM_LANG_TOKENS[token]
        else:
            continue

        if meaning not in seen:
            seen.add(meaning)
            hits.append(f'{meaning} (query contains "{token}", {lang})')

    return "; ".join(hits)


# Collapses life-stage species words to the base value vet_products' own
# "species" metadata field actually uses (it says "Cat"/"Dog", never
# "Kitten"/"Puppy") — for retrieval widening, not prompt display.
_WIDENING_SPECIES_MAP = {"Cat": "Cat", "Kitten": "Cat", "Dog": "Dog", "Puppy": "Dog", "Rabbit": "Rabbit", "Horse": "Horse"}


def detect_species_for_widening(cleaned_query: str) -> str:
    """Returns just the base species token (e.g. "Cat") if a non-English
    species word was found in the query, or "" otherwise. English species
    words aren't checked here — confirmed on a real failure that plain
    "cat" retrieves cat candidates fine (11/11), it's specifically a
    foreign word like "gato" the embedding model doesn't weight strongly
    enough, which is exactly the gap this exists to patch deterministically
    in phase2_variant.py's candidate retrieval."""
    for token in _LANG_TOKEN_RE.findall(cleaned_query or ""):
        if token in SPECIES_LANG_TOKENS:
            species, _lang = SPECIES_LANG_TOKENS[token]
            if species in _WIDENING_SPECIES_MAP:
                return _WIDENING_SPECIES_MAP[species]
    return ""


# ─────────────────────────────────────────────────────────────
# HISTORICAL-ID PROBABILITY TIERS (shared by Phase 2's product/variant
# assignment and Phase 3b's SKU assignment)
# ─────────────────────────────────────────────────────────────
# sku_examples_builder.py's triplet-first rebuild attaches an occurrence
# probability to every real historical id-triplet within whatever group
# it's being compared in (a candidate variant's linked triplets for Phase
# 2; a confirmed variant's SkuIds for Phase 3b). Both phases classify that
# distribution into the same three tiers, per the exact rule specified for
# this pipeline: a clear majority is used directly, a genuine near-tie is
# surfaced as BOTH options so DeepSeek can use the query's own details to
# break it instead of the code guessing blind, and a weak/fragmented
# distribution still returns the best available rather than nothing.

DOMINANT_PROBABILITY_THRESHOLD = 0.65   # e.g. 70% vs 30% -> dominant
TIE_CONTENDER_THRESHOLD = 0.25          # e.g. 55/45 or 40/60 -> roughly tied


def classify_probability_tier(sorted_shares: list) -> str:
    """sorted_shares: [(label, probability), ...] sorted by probability
    descending. Returns "dominant" (top clears DOMINANT_PROBABILITY_THRESHOLD),
    "tied" (no dominant leader, but a real second contender exists), "low"
    (no dominant leader and nothing else close — fragmented), or "none"
    (nothing to classify)."""
    if not sorted_shares:
        return "none"

    top_p = sorted_shares[0][1]
    second_p = sorted_shares[1][1] if len(sorted_shares) > 1 else 0.0

    if top_p >= DOMINANT_PROBABILITY_THRESHOLD:
        return "dominant"
    if second_p >= TIE_CONTENDER_THRESHOLD:
        return "tied"
    return "low"


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

# LLM-as-judge (see llm_judge.py): an extra DeepSeek call per High/Medium-
# confidence primary match, roughly +50% call volume on average — real
# cost and latency, so off by default and opt-in per call, same reasoning
# as the live-research fallback above.
DEFAULT_ENABLE_LLM_JUDGE = False


# ─────────────────────────────────────────────────────────────
# SHARED DEEPSEEK CALL HELPER
# ─────────────────────────────────────────────────────────────

def deepseek_json(prompt: str, attempts: int = 3) -> dict:
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
