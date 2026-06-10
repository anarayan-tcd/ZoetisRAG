import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────
# STEP 1 — Load the cleaned data
# ─────────────────────────────────────────────────────────────
# Change this path to wherever your cleaned file lives on your laptop
df = pd.read_excel('Group_1_Data_Cleaned.xlsx')

print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print()

# ─────────────────────────────────────────────────────────────
# STEP 2 — Load the embedding model
# ─────────────────────────────────────────────────────────────
# This downloads the model the FIRST time (~90MB)
# After that it loads from your local cache — very fast
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")
print()

# ─────────────────────────────────────────────────────────────
# STEP 3 — Get the unique, clean product names to embed
# ─────────────────────────────────────────────────────────────
# We don't need to embed ALL 2037 rows — many are duplicates
# We just need the UNIQUE cleaned names with their SKU
# This is what goes into our vector database as the "reference"

reference = df[['Cleaned_Notes', 'SkuId', 'ProductId']].drop_duplicates(subset='Cleaned_Notes')
reference = reference.dropna(subset=['Cleaned_Notes'])

print(f"Unique product name variations to embed: {len(reference)}")
print()
print("Sample of what we're embedding:")
print(reference.head(10).to_string())
print()

# ─────────────────────────────────────────────────────────────
# STEP 4 — Generate the embeddings
# ─────────────────────────────────────────────────────────────
# This is the magic line — it converts each product name
# into a list of 384 numbers that captures its meaning

print("Generating embeddings... (this takes 10-30 seconds)")

product_names = reference['Cleaned_Notes'].tolist()
embeddings = model.encode(product_names, show_progress_bar=True)

print()
print(f"Done! Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions (numbers)")
print()

# Let's peek at what an embedding actually looks like
print("What does one embedding look like?")
print(f"Product name : '{product_names[0]}'")
print(f"Embedding    : {embeddings[0][:10].tolist()} ... (first 10 of 384 numbers)")
print()

# ─────────────────────────────────────────────────────────────
# STEP 5 — Store embeddings in ChromaDB
# ─────────────────────────────────────────────────────────────
# ChromaDB is our vector database — it stores the embeddings
# so we can search them later by similarity

print("Setting up ChromaDB...")

# This creates a local database folder called 'chroma_db' 
# in whatever folder you run this script from
client = chromadb.PersistentClient(path="./chroma_db")

# Delete existing collection if it exists (so we start fresh)
try:
    client.delete_collection("group1_products")
    print("Cleared existing collection")
except:
    pass

# Create a fresh collection
collection = client.create_collection(
    name="group1_products",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity for search
)

print("Adding embeddings to ChromaDB...")

# Add everything to the database in one go
collection.add(
    documents=product_names,                              # the text
    embeddings=embeddings.tolist(),                       # the vectors
    ids=[str(i) for i in range(len(product_names))],     # unique ID for each
    metadatas=[                                           # extra info to store
        {
            "sku_id": int(reference.iloc[i]['SkuId']),
            "product_id": int(reference.iloc[i]['ProductId'])
        }
        for i in range(len(product_names))
    ]
)

print(f"Stored {collection.count()} entries in ChromaDB")
print()

# ─────────────────────────────────────────────────────────────
# STEP 6 — Test it! Search with messy product names
# ─────────────────────────────────────────────────────────────
# This is the exciting part — let's see if it actually works
# We'll query with messy names that are NOT in the database

test_queries = [
    "Tick Spray: Frontline",            # completely different phrasing
    "FRONTLINE FLEA TREATMENT 100ML",   # slightly different wording
    "frontline spray cat",              # vague / incomplete
    "apply frontline spray to dog",     # action sentence, not a product name
    "fipronil spray 100ml",             # uses the chemical name instead
]

print("=" * 60)
print("TESTING THE VECTOR SEARCH")
print("=" * 60)

for query in test_queries:
    results = collection.query(
        query_texts=[query],
        n_results=3   # return top 3 closest matches
    )

    print(f"\nQuery : '{query}'")
    print(f"Top matches:")
    for j in range(3):
        doc      = results['documents'][0][j]
        distance = results['distances'][0][j]
        sku      = results['metadatas'][0][j]['sku_id']
        similarity = round(1 - distance, 3)   # convert distance to similarity score
        print(f"  {j+1}. '{doc}'")
        print(f"     Similarity: {similarity}  |  SKU: {sku}")

print()
print("=" * 60)
print("All done! Your vector database is saved in ./chroma_db")
print("You can query it any time without re-running the embeddings.")
print("=" * 60)
