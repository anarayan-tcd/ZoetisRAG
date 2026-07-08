"""
knowledge_builder.py
─────────────────────────────────────────────────────────────
Builds a ChromaDB knowledge base for 45 veterinary products.

FIRST RUN BEHAVIOUR:
  Pass reset=True to wipe ChromaDB and rebuild everything cleanly.
  python knowledge_builder.py  (runs with reset=True by default first time)

FUTURE RUN BEHAVIOUR:
  Pass reset=False (default after first run).
  Only researches products NOT already in ChromaDB.
  Existing products are skipped — no duplicates, no overwrites.
  To add a brand new product: just add it to PRODUCTS dict and run.
  To update an existing product: pass force=["Product Name"].

PHASE 1: ProductId level (this file)
PHASE 2: SKU level (sku_builder.py)
─────────────────────────────────────────────────────────────
"""

import os
import time
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# LOAD API KEY
# ─────────────────────────────────────────────────────────────

env_path = Path(__file__).resolve().parent / "APIKeys.env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        f"Missing GEMINI_API_KEY in {env_path}\n"
        "Add this line to your APIKeys.env file:\n"
        "GEMINI_API_KEY=your_actual_key_here"
    )

# ─────────────────────────────────────────────────────────────
# ALL 45 PRODUCTS
# ─────────────────────────────────────────────────────────────

PRODUCTS = {

    # ── Flea & Tick Spot-On Treatments ──────────────────────
    "Frontline": {
        "product_id":   32,
        "official_url": "https://uk.frontline.com/products",
        "amazon_url":   "https://www.amazon.co.uk/s?k=frontline+flea+treatment",
        "manufacturer": "Boehringer Ingelheim"
    },
    "Frontline Plus": {
        "product_id":   37,
        "official_url": "https://uk.frontline.com/products",
        "amazon_url":   "https://www.amazon.co.uk/s?k=frontline+plus+flea+treatment",
        "manufacturer": "Boehringer Ingelheim"
    },
    "Advocate": {
        "product_id":   1,
        "official_url": "https://www.elanco.co.uk/products-services/dogs/advocate",
        "amazon_url":   "https://www.amazon.co.uk/s?k=advocate+flea+treatment+dog",
        "manufacturer": "Elanco"
    },
    "Advantage": {
        "product_id":   5,
        "official_url": "https://www.elanco.co.uk/products-services/dogs/advantage",
        "amazon_url":   "https://www.amazon.co.uk/s?k=advantage+flea+treatment",
        "manufacturer": "Elanco"
    },
    "Advantix": {
        "product_id":   2,
        "official_url": "https://www.elanco.co.uk/products-services/dogs/advantix",
        "amazon_url":   "https://www.amazon.co.uk/s?k=advantix+dog+flea+tick",
        "manufacturer": "Elanco"
    },
    "Stronghold": {
        "product_id":   68,
        "official_url": "https://www.zoetis.co.uk/products/dogs/stronghold.aspx",
        "amazon_url":   "https://www.amazon.co.uk/s?k=stronghold+spot+on+dog",
        "manufacturer": "Zoetis"
    },
    "Effipro": {
        "product_id":   40,
        "official_url": "https://www.virbac.com/home/our-range/companion-animal/effipro.html",
        "amazon_url":   "https://www.amazon.co.uk/s?k=effipro+spot+on+dog+cat",
        "manufacturer": "Virbac"
    },
    "Revolution": {
        "product_id":   71,
        "official_url": "https://www.zoetis.co.uk/products/dogs/revolution.aspx",
        "amazon_url":   "https://www.amazon.co.uk/s?k=revolution+spot+on+dog",
        "manufacturer": "Zoetis"
    },
    "Prinovox": {
        "product_id":   45,
        "official_url": "https://www.vetuk.co.uk/dogs-flea-tick-treatments-c-1-c-59/prinovox-p-3543",
        "amazon_url":   "https://www.amazon.co.uk/s?k=prinovox+spot+on+dog",
        "manufacturer": "Elanco"
    },
    "Broadline": {
        "product_id":   34,
        "official_url": "https://www.boehringer-ingelheim.com/animal-health/broadline",
        "amazon_url":   "https://www.amazon.co.uk/s?k=broadline+spot+on+cat",
        "manufacturer": "Boehringer Ingelheim"
    },
    "Activyl": {
        "product_id":   24,
        "official_url": "https://www.msd-animal-health.co.uk/products/activyl/",
        "amazon_url":   "https://www.amazon.co.uk/s?k=activyl+spot+on+dog+cat",
        "manufacturer": "MSD Animal Health"
    },
    "Practic": {
        "product_id":   14,
        "official_url": "https://www.petdrugsonline.co.uk/s?q=practic+spot+on",
        "amazon_url":   "https://www.amazon.co.uk/s?k=practic+spot+on+dog",
        "manufacturer": "Elanco"
    },
    "Xeno 450": {
        "product_id":   62,
        "official_url": "https://www.virbac.com/home/our-range/companion-animal/xeno-450.html",
        "amazon_url":   "https://www.amazon.co.uk/s?k=xeno+450+spot+on+rabbit",
        "manufacturer": "Virbac"
    },
    "Promeris Duo": {
        "product_id":   70,
        "official_url": "https://www.petdrugsonline.co.uk/s?q=promeris+duo",
        "amazon_url":   "https://www.amazon.co.uk/s?k=promeris+duo+spot+on+dog",
        "manufacturer": "Fort Dodge"
    },
    "Bravecto Plus Cat": {
        "product_id":   165,
        "official_url": "https://www.bravecto.co.uk/cats",
        "amazon_url":   "https://www.amazon.co.uk/s?k=bravecto+plus+spot+on+cat",
        "manufacturer": "MSD Animal Health"
    },
    "Vectra 3D": {
        "product_id":   52,
        "official_url": "https://www.ceva.co.uk/Products/Companion-animal/Anti-parasitic/VECTRA-3D",
        "amazon_url":   "https://www.amazon.co.uk/s?k=vectra+3d+dog",
        "manufacturer": "Ceva"
    },
    "Xenex Spray": {
        "product_id":   61,
        "official_url": "https://www.virbac.com",
        "amazon_url":   "https://www.amazon.co.uk/s?k=xenex+spray+dog+cat",
        "manufacturer": "Virbac"
    },

    # ── Oral / Chewable Flea & Tick Treatments ───────────────
    "Bravecto": {
        "product_id":   22,
        "official_url": "https://www.bravecto.co.uk",
        "amazon_url":   "https://www.amazon.co.uk/s?k=bravecto+chewable+tablet+dog",
        "manufacturer": "MSD Animal Health"
    },
    "NexGard": {
        "product_id":   35,
        "official_url": "https://www.nexgard.co.uk",
        "amazon_url":   "https://www.amazon.co.uk/s?k=nexgard+chewable+dog",
        "manufacturer": "Boehringer Ingelheim"
    },
    "NexGard Spectra": {
        "product_id":   36,
        "official_url": "https://www.nexgard.co.uk/nexgard-spectra",
        "amazon_url":   "https://www.amazon.co.uk/s?k=nexgard+spectra+dog",
        "manufacturer": "Boehringer Ingelheim"
    },
    "NexGard Combo": {
        "product_id":   218,
        "official_url": "https://www.nexgard.co.uk/nexgard-combo",
        "amazon_url":   "https://www.amazon.co.uk/s?k=nexgard+combo+cat",
        "manufacturer": "Boehringer Ingelheim"
    },
    "Simparica": {
        "product_id":   74,
        "official_url": "https://www.zoetis.co.uk/products/dogs/simparica.aspx",
        "amazon_url":   "https://www.amazon.co.uk/s?k=simparica+chewable+dog",
        "manufacturer": "Zoetis"
    },
    "Simparica Trio": {
        "product_id":   199,
        "official_url": "https://www.zoetis.co.uk/products/dogs/simparica-trio.aspx",
        "amazon_url":   "https://www.amazon.co.uk/s?k=simparica+trio+dog",
        "manufacturer": "Zoetis"
    },
    "Comfortis": {
        "product_id":   10,
        "official_url": "https://www.elanco.co.uk/products-services/dogs/comfortis",
        "amazon_url":   "https://www.amazon.co.uk/s?k=comfortis+chewable+dog+cat",
        "manufacturer": "Elanco"
    },
    "Credelio": {
        "product_id":   141,
        "official_url": "https://www.elanco.co.uk/products-services/dogs/credelio",
        "amazon_url":   "https://www.amazon.co.uk/s?k=credelio+chewable+dog+cat",
        "manufacturer": "Elanco"
    },
    "Capstar": {
        "product_id":   15,
        "official_url": "https://www.elanco.co.uk/products-services/dogs/capstar",
        "amazon_url":   "https://www.amazon.co.uk/s?k=capstar+flea+tablets+dog+cat",
        "manufacturer": "Elanco"
    },

    # ── Wormers ──────────────────────────────────────────────
    "Panacur": {
        "product_id":   23,
        "official_url": "https://www.msd-animal-health.co.uk/products/panacur/",
        "amazon_url":   "https://www.amazon.co.uk/s?k=panacur+wormer+dog+cat",
        "manufacturer": "MSD Animal Health"
    },
    "Drontal": {
        "product_id":   3,
        "official_url": "https://www.elanco.co.uk/products-services/dogs/drontal",
        "amazon_url":   "https://www.amazon.co.uk/s?k=drontal+plus+wormer+dog+cat",
        "manufacturer": "Elanco"
    },
    "Milbemax": {
        "product_id":   13,
        "official_url": "https://www.elanco.co.uk/products-services/dogs/milbemax",
        "amazon_url":   "https://www.amazon.co.uk/s?k=milbemax+wormer+dog+cat",
        "manufacturer": "Elanco"
    },
    "Milbemax Worming Program": {
        "product_id":   211,
        "official_url": "https://www.elanco.co.uk/products-services/dogs/milbemax",
        "amazon_url":   "https://www.amazon.co.uk/s?k=milbemax+wormer",
        "manufacturer": "Elanco"
    },
    "Milpro": {
        "product_id":   44,
        "official_url": "https://www.virbac.com/home/our-range/companion-animal/milpro.html",
        "amazon_url":   "https://www.amazon.co.uk/s?k=milpro+wormer+cat+dog",
        "manufacturer": "Virbac"
    },
    "Milprazon": {
        "product_id":   80,
        "official_url": "https://www.vmd.defra.gov.uk/ProductInformationDatabase/",
        "amazon_url":   "https://www.amazon.co.uk/s?k=milprazon+cat+dog+wormer",
        "manufacturer": "KRKA"
    },
    "Profender": {
        "product_id":   7,
        "official_url": "https://www.elanco.co.uk/products-services/cats/profender",
        "amazon_url":   "https://www.amazon.co.uk/s?k=profender+spot+on+cat+wormer",
        "manufacturer": "Elanco"
    },
    "Program": {
        "product_id":   12,
        "official_url": "https://www.elanco.co.uk/products-services/dogs/program",
        "amazon_url":   "https://www.amazon.co.uk/s?k=program+flea+tablet+dog+cat",
        "manufacturer": "Elanco"
    },
    "Droncit": {
        "product_id":   4,
        "official_url": "https://www.elanco.co.uk/products-services/dogs/droncit",
        "amazon_url":   "https://www.amazon.co.uk/s?k=droncit+tapeworm+dog+cat",
        "manufacturer": "Elanco"
    },
    "Proheart": {
        "product_id":   73,
        "official_url": "https://www.zoetis.co.uk/products/dogs/proheart.aspx",
        "amazon_url":   "https://www.amazon.co.uk/s?k=proheart+injection+dog",
        "manufacturer": "Zoetis"
    },
    "Interceptor": {
        "product_id":   20,
        "official_url": "https://www.elanco.co.uk/products-services/dogs/interceptor",
        "amazon_url":   "https://www.amazon.co.uk/s?k=interceptor+wormer+dog",
        "manufacturer": "Elanco"
    },
    "Endogard": {
        "product_id":   41,
        "official_url": "https://www.virbac.com",
        "amazon_url":   "https://www.amazon.co.uk/s?k=endogard+wormer+dog",
        "manufacturer": "Virbac"
    },
    "Strongid P": {
        "product_id":   17,
        "official_url": "https://www.zoetis.co.uk",
        "amazon_url":   "https://www.amazon.co.uk/s?k=strongid+p+wormer+horse",
        "manufacturer": "Zoetis"
    },

    # ── Collars ──────────────────────────────────────────────
    "Seresto": {
        "product_id":   6,
        "official_url": "https://www.seresto.co.uk",
        "amazon_url":   "https://www.amazon.co.uk/s?k=seresto+flea+tick+collar+dog+cat",
        "manufacturer": "Elanco"
    },
    "Scalibor": {
        "product_id":   27,
        "official_url": "https://www.msd-animal-health.co.uk/products/scalibor/",
        "amazon_url":   "https://www.amazon.co.uk/s?k=scalibor+collar+dog",
        "manufacturer": "MSD Animal Health"
    },

    # ── Household / Environmental ────────────────────────────
    "Indorex": {
        "product_id":   43,
        "official_url": "https://www.indorex.co.uk",
        "amazon_url":   "https://www.amazon.co.uk/s?k=indorex+household+flea+spray",
        "manufacturer": "Virbac"
    },
    "Rearguard": {
        "product_id":   16,
        "official_url": "https://www.norbrook.co.uk",
        "amazon_url":   "https://www.amazon.co.uk/s?k=rearguard+rabbit+fly+strike",
        "manufacturer": "Norbrook"
    },
    "Sentinel": {
        "product_id":   18,
        "official_url": "https://www.petdrugsonline.co.uk/s?q=sentinel+spectrum",
        "amazon_url":   "https://www.amazon.co.uk/s?k=sentinel+spectrum+dog",
        "manufacturer": "Elanco"
    },
    "Odour Eliminator": {
        "product_id":   75,
        "official_url": "https://www.petdrugsonline.co.uk/s?q=odour+eliminator",
        "amazon_url":   "https://www.amazon.co.uk/s?k=pet+odour+eliminator+spray",
        "manufacturer": "Various"
    },
}

# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash-lite")

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")

client = chromadb.PersistentClient(path="./chroma_db")


def get_or_create_collection(reset: bool = False):
    """
    Gets existing collection or creates a new one.
    If reset=True, wipes everything first for a clean rebuild.
    """
    if reset:
        try:
            client.delete_collection("vet_products")
            print("✅ Wiped existing collection — starting completely fresh")
        except Exception:
            print("No existing collection to wipe — creating fresh")

        col = client.create_collection(
            name="vet_products",
            metadata={"hnsw:space": "cosine"}
        )
        print("✅ Created clean ChromaDB collection — vet_products")
        return col

    else:
        try:
            col = client.get_collection("vet_products")
            print(f"Connected to existing collection — {col.count()} entries stored")
            return col
        except Exception:
            col = client.create_collection(
                name="vet_products",
                metadata={"hnsw:space": "cosine"}
            )
            print("Created new ChromaDB collection — vet_products")
            return col


def get_existing_products(col) -> set:
    """
    Returns a set of product names already stored in ChromaDB.
    Used to skip products that are already built.
    """
    try:
        all_data = col.get(include=["metadatas"])
        existing = set()
        for meta in all_data["metadatas"]:
            name = meta.get("product_name")
            if name:
                existing.add(name)
        return existing
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────────
# STEP 1 — GEMINI RESEARCH
# ─────────────────────────────────────────────────────────────

def research_product_with_gemini(product_name: str, info: dict, col) -> str:
    """
    Asks Gemini to research the product and return a structured table.
    Includes dedicated Vet Abbreviations column so rows match
    how vets actually write product names in practice systems.
    """
    print(f"\n  Researching with Gemini...")

    prompt = f"""
You are a veterinary product data extraction assistant.

Search the web for this veterinary product:
Product Name: {product_name}
Manufacturer: {info['manufacturer']}
Official Website: {info['official_url']}
Amazon UK: {info['amazon_url']}

Also check: VioVet, PetDrugsOnline, VetUK, NoahCompendium, VMD Product Database.

CRITICAL INSTRUCTIONS — FOLLOW EXACTLY:
- Return ONLY a pipe-separated table
- NO paragraphs before or after the table
- NO introduction text
- NO conclusion text
- NO markdown formatting (no **, no ##, no ```)
- ONLY the table, nothing else

The table MUST have exactly these columns in this exact order:
Product Name | Size | Active Ingredients | Animals | Weight Range | Parasites | Application Method | Pack Quantity | Protection Duration | Age Minimum | Vet Abbreviations | Extra Notes

Table rules:
- First row = header row exactly as above
- One row per unique product variant
- Different pack sizes = different rows (3 pipettes vs 6 pipettes = 2 rows)
- Different weight ranges = different rows (small dog vs large dog = 2 rows)
- Different sizes = different rows (100ml vs 250ml = 2 rows)
- Different species = different rows (dog vs cat = 2 rows)

IMPORTANT — Vet Abbreviations column MUST include ALL of these where relevant:
- Common abbreviations vets use for this specific product variant
- For spot-on products: "spot on", "S/ON", "spot-on", "pipette", "pip", "pips"
- For sprays: "spray", "spr"
- For tablets: "tab", "tabs", "tablet", "tablets", "chew", "chews"
- For injections: "inj", "injection", "jab"
- BOTH the ml volume AND pipette count where applicable
  e.g. for 1.0ml spot-on in 3-pack write: "3 pipettes, 3 pip, 3 x 1.0ml, 1.0ml pip"
- Size abbreviations: S, M, L, XL, sml, med, lge, x-lge
- Weight as vets write it: "4-10kg", "4-10K", "4-10 kg", "10-25kg", "10-25K"
- Any other common abbreviations specific to this product

Include ALL chemical names for active ingredients.
If a field is unknown write Unknown.
Do not include any text before or after the table.

REMEMBER: Return ONLY the pipe-separated table. Nothing before. Nothing after.
"""

    response    = gemini.generate_content(prompt)
    raw_text    = response.text.strip()
    raw_text    = raw_text.replace("```", "").replace("**", "").strip()
    lines       = raw_text.split("\n")
    table_lines = [line.strip() for line in lines if "|" in line]
    description = "\n".join(table_lines).strip()

    print(f"  Gemini returned {len(table_lines)} rows")
    if table_lines:
        print(f"  Preview: {table_lines[0][:120]}...")
        if len(table_lines) > 1:
            print(f"           {table_lines[1][:120]}...")

    return description


# ─────────────────────────────────────────────────────────────
# STEP 2 — PARSE TABLE ROWS
# ─────────────────────────────────────────────────────────────

def parse_table_rows(table_text: str) -> list:

    rows  = []
    lines = table_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("product name"):
            continue
        cleaned = line.replace("|", "").replace("-", "").replace(" ", "")
        if not cleaned:
            continue
        if "|" in line:
            rows.append(line)

    return rows


# ─────────────────────────────────────────────────────────────
# STEP 3 — STORE ROWS
# ─────────────────────────────────────────────────────────────

def store_product_rows(product_name: str, product_id: int, rows: list, col):

    stored_count = 0

    for i, row in enumerate(rows):

        safe_name = (
            product_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        entry_id = f"{safe_name}_row_{i}"

        embedding = embedding_model.encode(
            row,
            normalize_embeddings=True
        ).tolist()

        col.upsert(
            documents=[row],
            embeddings=[embedding],
            ids=[entry_id],
            metadatas=[{
                "product_name": product_name,
                "product_id":   product_id,
                "sku_id":       "",
                "row_index":    i,
                "phase":        1
            }]
        )

        stored_count += 1

    print(f"  Stored {stored_count} rows for {product_name}")
    return stored_count


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def build_knowledge_base(
    products_to_run=None,
    reset=False,
    force=None
):
    """
    Builds the ChromaDB knowledge base.

    products_to_run : list of product names to run, or None for all 45
    reset           : True = wipe ChromaDB and rebuild everything fresh
                      False = only add NEW products not already in ChromaDB
    force           : list of product names to force-update even if already stored
                      e.g. force=["Advocate"] to re-research just Advocate
    """

    # Setup collection
    col = get_or_create_collection(reset=reset)

    # Get what's already stored
    existing_products = get_existing_products(col)
    force             = force or []

    if existing_products and not reset:
        print(f"\nAlready stored: {sorted(existing_products)}")
        print(f"These will be SKIPPED unless in force list")

    # Determine which products to run
    if products_to_run is None:
        products_to_run = list(PRODUCTS.keys())

    total_products  = len(products_to_run)
    total_rows      = 0
    skipped         = 0
    processed       = 0

    print("\n" + "=" * 65)
    print(f"KNOWLEDGE BUILDER — {total_products} products to check")
    print(f"ChromaDB currently has {col.count()} entries")
    print("=" * 65)

    for idx, product_name in enumerate(products_to_run, 1):

        if product_name not in PRODUCTS:
            print(f"\n[{idx}/{total_products}] ⚠️  '{product_name}' not in PRODUCTS — skipping")
            continue

        # Skip if already stored AND not in force list
        if product_name in existing_products and product_name not in force:
            print(f"\n[{idx}/{total_products}] ⏭️  {product_name} — already in ChromaDB, skipping")
            skipped += 1
            continue

        info = PRODUCTS[product_name]

        if product_name in force:
            print(f"\n[{idx}/{total_products}] 🔄 {product_name} (ProductId: {info['product_id']}) — FORCED UPDATE")
        else:
            print(f"\n[{idx}/{total_products}] ✨ {product_name} (ProductId: {info['product_id']}) — NEW")

        print(f"  Official : {info['official_url']}")

        try:
            table_text = research_product_with_gemini(product_name, info, col)
            rows       = parse_table_rows(table_text)

            print(f"  Parsed {len(rows)} variant rows")

            if not rows:
                print(f"  ⚠️  No rows parsed — Gemini returned unexpected format")
                print(f"  Raw: {table_text[:300]}")
                continue

            count       = store_product_rows(product_name, info["product_id"], rows, col)
            total_rows += count
            processed  += 1

            print(f"  ✅ Done — ChromaDB now has {col.count()} total entries")

            if idx < total_products:
                time.sleep(7)

        except Exception as e:
            print(f"  ❌ Error processing {product_name}: {e}")
            continue

    print("\n" + "=" * 65)
    print("KNOWLEDGE BASE BUILD COMPLETE")
    print(f"  Products processed : {processed}")
    print(f"  Products skipped   : {skipped} (already in ChromaDB)")
    print(f"  Rows stored        : {total_rows}")
    print(f"  ChromaDB total     : {col.count()} entries")
    print("=" * 65)
    print("\nNext step: run sku_builder.py to add SKU layer")


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── FIRST RUN — wipe everything and rebuild cleanly ──────
    build_knowledge_base(reset=False)

    # ── FUTURE RUNS — only add new products ──────────────────
    # build_knowledge_base(reset=False)

    # ── FORCE UPDATE specific products ───────────────────────
    # build_knowledge_base(reset=False, force=["Advocate", "Milbemax"])

    # ── ADD specific new products only ───────────────────────
    # build_knowledge_base(products_to_run=["NexGard Combo", "Milprazon"], reset=False)
