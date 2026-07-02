import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AQ.Ab8RN6ICdxKK0R7Y0H8cMMLxxI-j-nQigjbWfo404lA2IHmo7Q"  # ← paste your key here

# Known products and their SKUs
# Add more products here later when you scale to all 10
PRODUCTS = {
    "Frontline" : 229
}

# ─────────────────────────────────────────────────────────────
# SETUP — Gemini, ChromaDB, Embedding Model
# ─────────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")

# Connect to ChromaDB — creates chroma_db folder if it doesn't exist
client = chromadb.PersistentClient(path="./chroma_db")

# Delete existing collection and start fresh
try:
    client.delete_collection("vet_products")
    print("Cleared existing collection")
except:
    pass

collection = client.create_collection(
    name="vet_products",
    metadata={"hnsw:space": "cosine"}
)

# ─────────────────────────────────────────────────────────────
# STEP 1 — Ask Gemini to Research Each Product on the Web
# ─────────────────────────────────────────────────────────────

def research_product_with_gemini(product_name: str, sku: int) -> str:
    """
    Asks Gemini to search the web and return rich product knowledge
    for a given veterinary product.
    """
    print(f"\nResearching '{product_name}' with Gemini...")

    prompt = f"""
    You are a veterinary product research assistant.
    
    Search the web and find detailed information about this veterinary product:
    Product: {product_name}
    
    Look at sources like:
    - The official product website: https://uk.frontline.com/products
    - Amazon UK product listing: https://www.amazon.co.uk/stores/page/8F24B330-051E-41C4-87D7-F8353AC9669B/search?terms=frontline
    - UK vet retailer sites like VioVet, PetDrugsOnline, VetUK
    
    Return a comprehensive product description that includes ALL of the following, for all products found from the specific brand in {product_name}:
    - Full product name and any alternative names it is known by
    - Active ingredients and chemical names (e.g. Fipronil)
    - What animals it is used for (cats, dogs, rabbits etc)
    - What parasites it treats (fleas, ticks, lice, mites etc)
    - Pack size (e.g. 100ml, 250ml)
    - How it is applied (spray, spot-on, tablet etc)
    - Dosage information
    - How long it protects for
    - Any other names vets or pet owners commonly use for this product
    - The manufacturer name
    
    Write this as a rich, detailed paragraph — include as many relevant 
    keywords as possible so it can be used for semantic search matching.
    The more detail the better. try to identify all variants of the product 
    and include them in the description, at the start of the paragraph. 
    If you can't find information, say "No information found".
    after that, create a text based | seperated table labeled "Product Information: {product_name}" for the product with the following columns:
    Product Name | Size | Active Ingredients | Animals | Parasites | Application Method | extra relevant unique info based on product in entry
    and create a new entry for every unique combination found on any of the reference sources, 
    including defferent sizes, i.e. 3 pipettes get a different row that 6 pipettes, 
    and the same for different tablets or bottle sizes.
    include active ingredients by all names found in the text paragraph.
    then delete the paragraph and only return the table, with no extra text or commentary.
    """

    response = gemini.generate_content(prompt)
    description = response.text.strip()

    print(f"Gemini returned {len(description)} characters of product knowledge")
    print(f"\nSample of what Gemini found:")
    print(description[:300] + "...")

    return description


# ─────────────────────────────────────────────────────────────
# STEP 2 — Store the Knowledge in ChromaDB
# ─────────────────────────────────────────────────────────────

def store_in_chromadb(product_name: str, description: str, sku: int):
    """
    Converts Gemini's product description into an embedding
    and stores it in ChromaDB with the SKU label.
    """
    print(f"\nStoring '{product_name}' in ChromaDB...")

    # Convert the rich description to a vector
    embedding = embedding_model.encode(description).tolist()

    # Store in ChromaDB
    collection.add(
        documents=[description],
        embeddings=[embedding],
        ids=[f"sku_{sku}"],
        metadatas=[{
            "product_name": product_name,
            "sku_id": sku
        }]
    )

    print(f"Stored successfully! ChromaDB now has {collection.count()} entries")


# ─────────────────────────────────────────────────────────────
# STEP 3 — Run it for all products
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("KNOWLEDGE BUILDER — Gemini + ChromaDB")
print("=" * 60)

for product_name, sku in PRODUCTS.items():
    # Ask Gemini to research the product
    description = research_product_with_gemini(product_name, sku)

    # Store in ChromaDB
    store_in_chromadb(product_name, description, sku)

print("\n" + "=" * 60)
print("Done! ChromaDB is now populated with Gemini's product knowledge.")
print("Run query_agent.py next to test it.")
print("=" * 60)
