import chromadb

# ─────────────────────────────────────────────────────────────
# Connect to your existing ChromaDB
# ─────────────────────────────────────────────────────────────
client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("vet_products")

# ─────────────────────────────────────────────────────────────
# Get EVERYTHING currently stored — documents, metadata, ids
# ─────────────────────────────────────────────────────────────
all_data = collection.get(
    include=["documents", "metadatas"]
)

total = len(all_data["ids"])

print("=" * 70)
print(f"CHROMADB CONTENTS — {total} entries currently stored")
print("=" * 70)

for i in range(total):
    entry_id = all_data["ids"][i]
    document = all_data["documents"][i]
    metadata = all_data["metadatas"][i]

    print(f"\n{'─' * 70}")
    print(f"ENTRY {i + 1}")
    print(f"{'─' * 70}")
    print(f"ID           : {entry_id}")
    print(f"Product Name : {metadata.get('product_name', 'N/A')}")
    print(f"SKU          : {metadata.get('sku_id', 'N/A')}")
    print(f"\nFull Description / Knowledge stored:")
    print(document)

print("\n" + "=" * 70)
print(f"Total entries in ChromaDB: {total}")
print("=" * 70)
