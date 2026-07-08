"""
knowledge_builder.py
─────────────────────────────────────────────────────────────
Builds a ChromaDB knowledge base for 35 veterinary products.

HOW IT WORKS:
1. Gemini Flash 2.5 researches each product from official sources
2. Returns a structured table — one row per product variant
3. Each row is embedded SEPARATELY using MiniLM
4. Stored in ChromaDB with ProductId label
5. Upsert logic — never wipes, knowledge only grows
6. Re-running updates existing rows and adds new ones

PHASE 1: ProductId level (this file)
PHASE 2: SKU level (sku_builder.py — coming later)
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
# LOAD API KEY FROM ENV FILE
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
# ALL 35 PRODUCTS — Verified from Labelled.csv
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

    # ── Household Flea Spray ─────────────────────────────────
    "Indorex": {
        "product_id":   43,
        "official_url": "https://www.indorex.co.uk",
        "amazon_url":   "https://www.amazon.co.uk/s?k=indorex+household+flea+spray",
        "manufacturer": "Virbac"
    },
}

# ─────────────────────────────────────────────────────────────
# SETUP — Gemini, ChromaDB, Embedding Model
# ─────────────────────────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")

client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = client.get_collection("vet_products")
    print(f"Connected to existing collection — {collection.count()} entries already stored")
except Exception:
    collection = client.create_collection(
        name="vet_products",
        metadata={"hnsw:space": "cosine"}
    )
    print("Created new ChromaDB collection — vet_products")


# ─────────────────────────────────────────────────────────────
# STEP 1 — Ask Gemini to Research the Product
# ─────────────────────────────────────────────────────────────

def research_product_with_gemini(product_name: str, info: dict) -> str:
    """
    Asks Gemini to search the web and return a structured product table.
    Returns clean pipe-separated rows only — no paragraphs.
    """
    print(f"\n  Researching with Gemini...")

    prompt = f"""
You are a veterinary product data extraction assistant.

Search the web for this veterinary product:
Product Name: {product_name}
Manufacturer: {info['manufacturer']}
Official Website: {info['official_url']}
Amazon UK: {info['amazon_url']}

Also check UK vet retailer sites like VioVet, PetDrugsOnline, VetUK, and NoahCompendium.

CRITICAL INSTRUCTIONS — FOLLOW EXACTLY:
- Return ONLY a pipe-separated table
- NO paragraphs before or after the table
- NO introduction text
- NO conclusion text  
- NO markdown formatting (no **, no ##, no ```)
- NO extra commentary of any kind
- ONLY the table, nothing else

The table MUST have exactly these columns in this exact order:
Product Name | Size | Active Ingredients | Animals | Weight Range | Parasites | Application Method | Pack Quantity | Protection Duration | Age Minimum | Extra Notes

Table rules:
- First row = header row exactly as above
- One row per unique product variant
- Different pack sizes = different rows (3 pipettes vs 6 pipettes = 2 rows)
- Different weight ranges = different rows (small dog vs large dog = 2 rows)
- Different sizes = different rows (100ml vs 250ml = 2 rows)
- Different species = different rows (dog vs cat = 2 rows)
- Include ALL chemical names for active ingredients (e.g. Fipronil AND (S)-methoprene)
- Unknown fields = write Unknown
- Do NOT merge variants into one row

REMEMBER: Return ONLY the pipe-separated table. Nothing before it. Nothing after it.
"""

    response = gemini.generate_content(prompt)
    raw_text = response.text.strip()

    # Remove any markdown formatting Gemini might add
    raw_text = raw_text.replace("```", "").replace("**", "").strip()

    # Extract ONLY lines containing pipe separators
    # This strips out any paragraph text Gemini adds before or after
    lines      = raw_text.split("\n")
    table_lines = [line.strip() for line in lines if "|" in line]
    description = "\n".join(table_lines).strip()

    print(f"  Gemini returned {len(table_lines)} rows")
    if table_lines:
        print(f"  Preview: {table_lines[0][:100]}...")
        if len(table_lines) > 1:
            print(f"           {table_lines[1][:100]}...")

    return description


# ─────────────────────────────────────────────────────────────
# STEP 2 — Parse Table into Individual Rows
# ─────────────────────────────────────────────────────────────

def parse_table_rows(table_text: str) -> list:
    """
    Splits the pipe-separated table into individual data rows.
    Skips header row, separator lines, and empty lines.
    Returns a list of clean row strings.
    """
    rows  = []
    lines = table_text.strip().split("\n")

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip header row
        if line.lower().startswith("product name"):
            continue

        # Skip separator lines like |---|---|
        cleaned = line.replace("|", "").replace("-", "").replace(" ", "")
        if not cleaned:
            continue

        # Must contain pipe separators to be a valid data row
        if "|" in line:
            rows.append(line)

    return rows


# ─────────────────────────────────────────────────────────────
# STEP 3 — Store Each Row as a Separate Embedding
# ─────────────────────────────────────────────────────────────

def store_product_rows(product_name: str, product_id: int, rows: list):
    """
    Embeds each row individually and stores in ChromaDB.

    NEVER WIPE logic:
    - Uses upsert — updates a row if ID exists, inserts if new
    - Existing rows are NEVER deleted
    - Knowledge base only ever grows
    - Running twice = no duplicates, just updates
    - New variants get added alongside existing ones
    """

    stored_count = 0

    for i, row in enumerate(rows):

        # Deterministic ID — same product + same row index = same ID always
        safe_name = (
            product_name.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        entry_id = f"{safe_name}_row_{i}"

        # Embed this specific row
        embedding = embedding_model.encode(
            row,
            normalize_embeddings=True
        ).tolist()

        # Upsert — update if exists, insert if new
        collection.upsert(
            documents=[row],
            embeddings=[embedding],
            ids=[entry_id],
            metadatas=[{
                "product_name": product_name,
                "product_id":   product_id,
                "sku_id":       "",    # empty now — Phase 2 will fill this
                "row_index":    i,
                "phase":        1
            }]
        )

        stored_count += 1

    print(f"  Stored/updated {stored_count} rows for {product_name}")
    return stored_count


# ─────────────────────────────────────────────────────────────
# STEP 4 — Main Loop
# ─────────────────────────────────────────────────────────────

def build_knowledge_base(products_to_run=None):
    """
    Main function — researches each product with Gemini and
    stores structured variant rows in ChromaDB.

    products_to_run: optional list of product names
                     If None — runs ALL 35 products
    """

    if products_to_run is None:
        products_to_run = list(PRODUCTS.keys())

    total_products = len(products_to_run)
    total_rows     = 0

    print("\n" + "=" * 65)
    print(f"KNOWLEDGE BUILDER — {total_products} products to research")
    print(f"ChromaDB currently has {collection.count()} entries")
    print("=" * 65)

    for idx, product_name in enumerate(products_to_run, 1):

        if product_name not in PRODUCTS:
            print(f"\n[{idx}/{total_products}] ⚠️  '{product_name}' not in PRODUCTS — skipping")
            continue

        info = PRODUCTS[product_name]

        print(f"\n[{idx}/{total_products}] {product_name}  (ProductId: {info['product_id']})")
        print(f"  Official : {info['official_url']}")
        print(f"  Amazon   : {info['amazon_url']}")

        try:
            # Step 1 — Research with Gemini
            table_text = research_product_with_gemini(product_name, info)

            # Step 2 — Parse into individual rows
            rows = parse_table_rows(table_text)
            print(f"  Parsed {len(rows)} variant rows")

            if not rows:
                print(f"  ⚠️  No rows parsed — Gemini returned unexpected format")
                print(f"  Raw output snippet: {table_text[:300]}")
                continue

            # Step 3 — Store each row as separate embedding
            count       = store_product_rows(product_name, info["product_id"], rows)
            total_rows += count

            print(f"  ✅ Done — ChromaDB now has {collection.count()} total entries")

            # Pause between products — respect Gemini rate limits
            if idx < total_products:
                time.sleep(2)

        except Exception as e:
            print(f"  ❌ Error processing {product_name}: {e}")
            continue

    print("\n" + "=" * 65)
    print("KNOWLEDGE BASE BUILD COMPLETE")
    print(f"  Products processed : {total_products}")
    print(f"  Rows stored        : {total_rows}")
    print(f"  ChromaDB total     : {collection.count()} entries")
    print("=" * 65)
    print("\nNext step: run query_pipeline.py to test")


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Option A — Run ALL 35 products
    build_knowledge_base()

    # Option B — Run specific products only
    # build_knowledge_base(["Frontline", "Advocate", "Bravecto"])

    # Option C — Run a single product (good for testing)
    # build_knowledge_base(["Frontline"])
