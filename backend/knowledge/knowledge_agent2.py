import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

# knowledge_agent2.py lives in backend/knowledge/ but chromadb_client.py
# lives in backend/ — add the parent directory to sys.path so it can be
# imported without turning this into a full installable package.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from chromadb_client import get_embedding_model, get_collection, get_all_variant_documents

# Some brand names in brand_seed_list.csv are non-Latin script (e.g.
# Japanese-market products). Windows' console defaults to cp1252, which
# can't encode them — reconfigure stdout so printing a brand name never
# crashes a run partway through a long batch.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

env_path = BACKEND_ROOT / "APIKeys.env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_2")

if not GEMINI_API_KEY:
    raise ValueError(f"Missing GEMINI_API_KEY_2 in {env_path}")

BRAND_SEED_CSV = BACKEND_ROOT / "data" / "generated" / "brand_seed_list.csv"
EXPORT_CSV = BACKEND_ROOT / "data" / "generated" / "brands_products_export.csv"
CHECKPOINT_FILE = BACKEND_ROOT / "data" / "generated" / "knowledge_research_checkpoint.json"

GEMINI_MODEL = "gemini-2.5-flash"

# Pet-pharmacy retailers by region, used as source hints for the live
# search — brands aren't necessarily UK-only, so the prompt is told to
# pick whichever region(s) actually apply to the brand being researched.
TRUSTED_RETAILERS = {
    "UK/IE": [
        "VioVet (viovet.co.uk)",
        "PetDrugsOnline (petdrugsonline.co.uk)",
        "VetUK (vetuk.co.uk)",
        "Animed Direct (animeddirect.co.uk)",
        "Amazon UK (amazon.co.uk)",
    ],
    "US": [
        "Chewy (chewy.com)",
        "1800PetMeds (1800petmeds.com)",
        "Allivet (allivet.com)",
        "Walmart Pet Pharmacy (walmart.com)",
        "Amazon US (amazon.com)",
    ],
    "Canada": [
        "VetShopCanada (vetshopcanada.com)",
        "Amazon Canada (amazon.ca)",
    ],
    "Australia/NZ": [
        "VetSupply (vetsupply.com.au)",
        "PetCircle (petcircle.com.au)",
        "Budget Pet Products (budgetpetproducts.com.au)",
        "Amazon Australia (amazon.com.au)",
    ],
    "EU": [
        "Zooplus (zooplus.de / zooplus.fr / zooplus.es etc.)",
        "Medpets (medpets.nl)",
        "regional Amazon storefront (amazon.de / amazon.fr / amazon.es / amazon.it)",
    ],
}


# ─────────────────────────────────────────────────────────────
# SETUP — Gemini (live search), ChromaDB, Embedding Model
# ─────────────────────────────────────────────────────────────

client_gemini = genai.Client(api_key=GEMINI_API_KEY)

embedding_model = get_embedding_model()

# Reuses the existing collection instead of wiping it, so re-running this
# script only adds/updates variants that are new or have changed.
collection = get_collection()


# ─────────────────────────────────────────────────────────────
# HELPERS — slugs, JSON extraction
# ─────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def variant_id(brand: str, variant_name: str, region: str = "") -> str:
    """Deterministic id so the same real-world variant maps to the same
    ChromaDB entry across weekly re-runs, regardless of call order.
    Region is folded in because the same variant_name can legitimately
    refer to different regional products (e.g. "Frontline Spot On" vs
    "Frontline Plus" are different names, but this guards against the
    rarer case of two single-region variants sharing a name and silently
    overwriting each other via upsert — the same class of bug found in
    sku_examples_builder.py's SkuId collisions)."""
    region_part = f"__{slugify(region)}" if region else ""
    return f"{slugify(brand)}__{slugify(variant_name)}{region_part}"


def extract_json(text: str) -> dict:
    """
    Pulls a JSON object out of Gemini's response text. Grounded responses
    aren't forced into strict JSON mime type (grounding tools and forced
    JSON output mode don't reliably combine), so the prompt asks for JSON
    directly and this strips markdown fences / stray text defensively.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {}


def format_retailer_block() -> str:
    lines = []

    for region, retailers in TRUSTED_RETAILERS.items():
        lines.append(f"{region}: " + ", ".join(retailers))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# STEP 1 — Ask Gemini (with live search) to Research Each Brand
# ─────────────────────────────────────────────────────────────

# Plain (non f-string) template — the golden examples and output schema
# below are full of literal JSON braces, which would otherwise all need
# escaping as {{ }} in an f-string. Substitution is done with .replace()
# on the __TOKEN__ placeholders instead.
RESEARCH_PROMPT_TEMPLATE = """
### ROLE
You are a veterinary product research assistant with live web search access.
Your job is to build an accurate, structured knowledge base of pet-care/
veterinary products — capturing every distinct regional and pack-size
variant of a brand, not just the first product line your search happens
to surface.

### CONTEXT
This research feeds a system that matches messy, real-world vet
prescription text (e.g. "FRONTLINE SPOT ON XL DOG 3PK") against a database
of known products. Vet practices worldwide use regional product names that
often differ from a brand's default/US website — e.g. a product sold as
"X Plus" in the US may be sold as "X Spot On" in the UK or Ireland, whether
as the same formulation or a genuinely different one. Missing a regional
name means real prescriptions using that name will never match correctly.

### INFO PROVIDED
- Brand to research: "__BRAND__"
- Known product lines from real historical prescription data for this
  brand: __HINT__
  -> These are REAL terms vets have actually used for this brand. Treat
  each one as a checklist item: for every term listed, either (a) find a
  matching variant and report it, or (b) only omit it if you've actively
  checked multiple regions and are confident nothing matches.
- Regional retailer reference (use these to DISCOVER variants, not just to
  verify pricing on variants you already found elsewhere):
__RETAILERS__

### RULES
1. Do not default to a single "official" brand site as your primary
   source. Brand websites usually resolve to whichever market has the
   most traffic (often the US) and will silently omit region-specific
   product names. Treat the regional retailers above as equally valid
   discovery sources, not just a pricing cross-check.
2. Actively check whether this brand is sold under DIFFERENT NAMES in
   different regions — don't assume one product name applies globally.
   If the known-product-lines hint mentions a term you haven't found yet,
   search specifically for "__BRAND__ <that term>" in each plausible
   region before concluding it doesn't exist.
3. Every distinct pack size, pipette/dose count, weight/species tier, and
   region-specific name is a SEPARATE variant entry — never collapse
   "3 pipettes" and "6 pipettes" (or "100ml" and "250ml") into one row.
4. Use ONLY information you actually found via search. Do not invent pack
   sizes, prices, or ingredient names.
5. Tag every variant with the region(s) it was found in, using the same
   short codes as the retailer reference above: UK/IE, US, Canada,
   Australia/NZ, EU. If a variant is confirmed identical across multiple
   regions, list them comma-separated (e.g. "UK/IE, EU").
6. If you cannot find reliable information for this brand at all, return
   an empty "variants" list rather than guessing.

### GOLDEN EXAMPLES

Brand: "Frontline", hint: "Spot-On, Combo, Kitten/Puppy Packs"
{
  "variant_name": "Frontline Spot On Small Dog 3 Pipette Pack",
  "active_ingredients": "Fipronil",
  "species": "Dog",
  "parasites_treated": "Fleas, ticks",
  "application_method": "Spot-on",
  "pack_size": "0.67ml pipette",
  "pack_quantity": "3",
  "dosage": "1 pipette per dog 2-10kg",
  "protection_duration": "4 weeks",
  "other_names": "Frontline Spot On Small Dog",
  "manufacturer": "Boehringer Ingelheim",
  "official_brand_url": "https://uk.frontline.com/products/frontline-spot-on-dog",
  "region": "UK/IE"
}
{
  "variant_name": "Frontline Spot On Small Dog 6 Pipette Pack",
  "active_ingredients": "Fipronil",
  "species": "Dog",
  "parasites_treated": "Fleas, ticks",
  "application_method": "Spot-on",
  "pack_size": "0.67ml pipette",
  "pack_quantity": "6",
  "dosage": "1 pipette per dog 2-10kg",
  "protection_duration": "4 weeks",
  "other_names": "Frontline Spot On Small Dog",
  "manufacturer": "Boehringer Ingelheim",
  "official_brand_url": "https://uk.frontline.com/products/frontline-spot-on-dog",
  "region": "UK/IE"
}
(same product, different pack_quantity -> two separate entries. Note the
hint term "Spot-On" was actively searched for and found — this is exactly
the case that must not be missed.)

Brand: "Frontline", region: US
{
  "variant_name": "Frontline Plus for Dogs 5-22 lbs (Orange) 3-Dose",
  "active_ingredients": "Fipronil, (S)-Methoprene",
  "species": "Dog",
  "parasites_treated": "Fleas, ticks, chewing lice, flea eggs and larvae",
  "application_method": "Spot-on",
  "pack_size": "0.67ml applicator",
  "pack_quantity": "3",
  "dosage": "1 applicator per dog 5-22 lbs",
  "protection_duration": "1 month",
  "other_names": "Frontline Plus Orange",
  "manufacturer": "Boehringer Ingelheim",
  "official_brand_url": "https://www.frontline.com/plus-for-dogs",
  "region": "US"
}
(same brand as above, different region AND different product name —
Plus, not Spot On — demonstrating regional naming divergence.)

Brand: "Frontline", multi-region
{
  "variant_name": "Frontline Spray 100ml",
  "active_ingredients": "Fipronil",
  "species": "Dog, Cat, Rabbit",
  "parasites_treated": "Fleas, ticks, biting lice",
  "application_method": "Spray",
  "pack_size": "100ml",
  "pack_quantity": "1",
  "dosage": "3-6ml per kg bodyweight",
  "protection_duration": "up to 3 months (ticks)",
  "other_names": "Frontline Spray",
  "manufacturer": "Boehringer Ingelheim",
  "official_brand_url": "https://uk.frontline.com/products/frontline-spray",
  "region": "UK/IE, EU"
}
(ml-based volume variant, sold identically across two regions -> one
entry with a comma-separated region list, not two duplicate entries.)

Brand: "Bravecto", region: US
{
  "variant_name": "Bravecto Chews for Dogs 20-40kg (44-88 lbs) 1 Tablet",
  "active_ingredients": "Fluralaner",
  "species": "Dog",
  "parasites_treated": "Fleas, ticks",
  "application_method": "Oral chew",
  "pack_size": "1000mg tablet",
  "pack_quantity": "1",
  "dosage": "1 chew per dog 20-40kg",
  "protection_duration": "12 weeks",
  "other_names": "Bravecto 1000mg, Bravecto Blue",
  "manufacturer": "MSD Animal Health",
  "official_brand_url": "https://www.bravecto.com/dogs",
  "region": "US"
}
(weight-band + tablet-count sizing instead of ml/pipettes — a different
application method entirely.)

Brand: "Seresto", region: Australia/NZ
{
  "variant_name": "Seresto Flea and Tick Collar for Cats",
  "active_ingredients": "Imidacloprid, Flumethrin",
  "species": "Cat",
  "parasites_treated": "Fleas, ticks",
  "application_method": "Collar",
  "pack_size": "1 collar",
  "pack_quantity": "1",
  "dosage": "N/A (worn continuously)",
  "protection_duration": "8 months",
  "other_names": "Seresto Cat Collar",
  "manufacturer": "Elanco",
  "official_brand_url": "https://www.seresto.com/au/cats",
  "region": "Australia/NZ"
}
(single-unit product, no pack-count concept, and a fourth distinct region.)

### OUTPUT FORMAT (STRICT JSON ONLY, no extra commentary)
{
  "brand": "__BRAND__",
  "manufacturer": "",
  "official_brand_url": "",
  "variants": [
    {
      "variant_name": "", "active_ingredients": "", "species": "",
      "parasites_treated": "", "application_method": "", "pack_size": "",
      "pack_quantity": "", "dosage": "", "protection_duration": "",
      "other_names": "", "manufacturer": "", "official_brand_url": "",
      "region": ""
    }
  ]
}

If you cannot find reliable information for this brand, return an empty
"variants" list.
"""


def research_brand_with_gemini(brand: str, known_product_lines: str = "") -> dict:
    """
    Uses live Google Search grounding to research a pet-care brand and
    return a structured list of its current product variants, each
    tagged with the region(s) it was found in.
    """
    print(f"\nResearching '{brand}' with Gemini (live search)...")

    hint_text = known_product_lines if known_product_lines else "none available — research broadly"

    prompt = (
        RESEARCH_PROMPT_TEMPLATE
        .replace("__BRAND__", brand)
        .replace("__HINT__", hint_text)
        .replace("__RETAILERS__", format_retailer_block())
    )

    response = client_gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    data = extract_json(response.text or "")

    if not data:
        print(f"Warning: could not parse Gemini's response for '{brand}' — treating as no results.")
        return {"brand": brand, "manufacturer": "", "official_brand_url": "", "variants": []}

    variants = data.get("variants", [])
    print(f"Gemini found {len(variants)} variant(s) for '{brand}'")

    return data


# ─────────────────────────────────────────────────────────────
# STEP 2 — Store Each Variant in ChromaDB (add/update only what changed)
# ─────────────────────────────────────────────────────────────

def build_variant_description(brand: str, variant: dict) -> str:
    """Turns one variant's structured fields into a focused text blob
    for embedding — one variant per document, not a whole brand blob."""

    lines = [
        f"{brand} - {variant.get('variant_name', '')}",
        f"Region: {variant.get('region', '')}",
        f"Active ingredients: {variant.get('active_ingredients', '')}",
        f"Species: {variant.get('species', '')}",
        f"Parasites treated: {variant.get('parasites_treated', '')}",
        f"Application method: {variant.get('application_method', '')}",
        f"Pack size: {variant.get('pack_size', '')}",
        f"Pack quantity: {variant.get('pack_quantity', '')}",
        f"Dosage: {variant.get('dosage', '')}",
        f"Protection duration: {variant.get('protection_duration', '')}",
        f"Also known as: {variant.get('other_names', '')}",
        f"Manufacturer: {variant.get('manufacturer', '')}",
    ]

    return "\n".join(line for line in lines if line.split(": ", 1)[-1].strip())


def store_variant_in_chromadb(brand: str, variant: dict, existing_docs: dict) -> bool:
    """
    Upserts one product variant into ChromaDB under a deterministic id.
    Returns True if the entry was written (new or changed), False if it
    was left untouched because nothing changed since the last run.

    `existing_docs` is a {id: document} dict bulk-fetched once per run
    (see chromadb_client.get_all_variant_documents) rather than looked
    up with a collection.get(ids=[vid]) round trip per variant — cheaper
    as the variant count grows. It's updated in place after every write
    so later variants in the same run see the latest state.
    """
    variant_name = str(variant.get("variant_name", "")).strip()

    if not variant_name:
        return False

    region = str(variant.get("region", "")).strip()
    vid = variant_id(brand, variant_name, region)
    description = build_variant_description(brand, variant)

    if existing_docs.get(vid) == description:
        return False

    embedding = embedding_model.encode(description, normalize_embeddings=True).tolist()

    collection.upsert(
        documents=[description],
        embeddings=[embedding],
        ids=[vid],
        metadatas=[{
            "brand": brand,
            "variant_name": variant_name,
            "region": region,
            "active_ingredients": str(variant.get("active_ingredients", "")),
            "species": str(variant.get("species", "")),
            "pack_size": str(variant.get("pack_size", "")),
            "pack_quantity": str(variant.get("pack_quantity", "")),
            "official_brand_url": str(variant.get("official_brand_url", "")),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }]
    )

    existing_docs[vid] = description

    return True


# ─────────────────────────────────────────────────────────────
# CHECKPOINT — resume without re-spending Gemini credits on a crash
# ─────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Returns {brand: research_data_dict} for brands already researched
    in a previous (possibly crashed/credit-exhausted) run."""
    if not CHECKPOINT_FILE.exists():
        return {}
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_checkpoint(checkpoint: dict):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


# ─────────────────────────────────────────────────────────────
# STEP 3 — Export a flat Brand / Product / Official URL CSV
# ─────────────────────────────────────────────────────────────

def export_brand_product_csv(all_results: list):
    rows = []

    for data in all_results:
        brand = data.get("brand", "")
        brand_url = data.get("official_brand_url", "")

        for variant in data.get("variants", []):
            rows.append({
                "Brand": brand,
                "Product": variant.get("variant_name", ""),
                "OfficialBrandURL": variant.get("official_brand_url") or brand_url,
            })

    try:
        pd.DataFrame(rows).to_csv(EXPORT_CSV, index=False)
        output_path = EXPORT_CSV
    except PermissionError:
        fallback_path = EXPORT_CSV.with_stem(EXPORT_CSV.stem + "_new")
        pd.DataFrame(rows).to_csv(fallback_path, index=False)
        output_path = fallback_path
        print(
            f"\nWARNING: {EXPORT_CSV} is open in another program and could not be "
            f"overwritten. Wrote to {fallback_path} instead — close the original "
            f"file and rename/replace it manually."
        )

    print(f"Exported brand/product/URL list to {output_path} ({len(rows)} row(s))")


# ─────────────────────────────────────────────────────────────
# STEP 4 — Run it for every brand in the seed list
# ─────────────────────────────────────────────────────────────

def load_brand_seed_list() -> pd.DataFrame:
    if not BRAND_SEED_CSV.exists():
        raise FileNotFoundError(
            f"{BRAND_SEED_CSV} not found — run extract_brands.py first "
            "to generate it from Labelled.csv."
        )

    return pd.read_csv(BRAND_SEED_CSV)


if __name__ == "__main__":

    print("=" * 60)
    print("KNOWLEDGE BUILDER — Gemini (live search) + ChromaDB")
    print("=" * 60)

    seed_df = load_brand_seed_list()

    # One bulk fetch of everything already stored, instead of a
    # collection.get(ids=[vid]) round trip per variant.
    existing_docs = get_all_variant_documents()
    print(f"Prefetched {len(existing_docs)} existing variant document(s) for diffing.\n")

    # Resumable checkpoint: {brand: research_data_dict}, saved to disk after
    # every single brand (not batched) so a crash or a Gemini credit
    # exhaustion partway through the ~146-brand loop never loses more than
    # the one in-flight call, and a restart skips brands already researched
    # instead of re-spending API credits on them.
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"Resuming from checkpoint — {len(checkpoint)} brand(s) already researched.\n")

    for _, row in seed_df.iterrows():

        brand = str(row["Brand"]).strip()

        if brand in checkpoint:
            continue

        known_lines = str(row.get("KnownProductLines", "") or "")

        data = research_brand_with_gemini(brand, known_lines)

        variants = data.get("variants", [])
        updated = sum(store_variant_in_chromadb(brand, v, existing_docs) for v in variants)

        print(f"'{brand}': {updated} variant(s) added/updated, {len(variants) - updated} unchanged.")

        checkpoint[brand] = data
        save_checkpoint(checkpoint)

    export_brand_product_csv(list(checkpoint.values()))

    print("\n" + "=" * 60)
    print(f"Done! ChromaDB now has {collection.count()} variant(s) total.")
    print("Run query_agent.py next to test it.")
    print("=" * 60)
