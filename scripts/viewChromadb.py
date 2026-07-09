# viewChromadb.py — dumps the full contents of a ChromaDB collection to
# the console for debugging.
#
# Usage:
#   py -3 scripts/viewChromadb.py                # vet_products (default)
#   py -3 scripts/viewChromadb.py sku_examples    # sku_examples

import sys
from pathlib import Path

# This script lives in scripts/, but chromadb_client.py lives in backend/ —
# add it to sys.path so it can be imported without turning this into a
# full installable package (same pattern used throughout backend/knowledge/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

# Brand names can be non-Latin script (e.g. Japanese-market products);
# Windows' console defaults to cp1252, which can't encode them.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from chromadb_client import get_collection, VET_PRODUCTS_COLLECTION, SKU_EXAMPLES_COLLECTION

collection_name = sys.argv[1] if len(sys.argv) > 1 else VET_PRODUCTS_COLLECTION

if collection_name not in (VET_PRODUCTS_COLLECTION, SKU_EXAMPLES_COLLECTION):
    print(f"Unknown collection '{collection_name}'.")
    print(f"Use '{VET_PRODUCTS_COLLECTION}' or '{SKU_EXAMPLES_COLLECTION}'.")
    sys.exit(1)

collection = get_collection(collection_name)

all_data = collection.get(include=["documents", "metadatas"])

total = len(all_data["ids"])

print("=" * 70)
print(f"CHROMADB CONTENTS — '{collection_name}' — {total} entries currently stored")
print("=" * 70)

for i in range(total):
    entry_id = all_data["ids"][i]
    document = all_data["documents"][i]
    metadata = all_data["metadatas"][i]

    print(f"\n{'─' * 70}")
    print(f"ENTRY {i + 1}")
    print(f"{'─' * 70}")
    print(f"ID           : {entry_id}")
    print(f"Brand        : {metadata.get('brand', 'N/A')}")

    if collection_name == VET_PRODUCTS_COLLECTION:
        print(f"Variant      : {metadata.get('variant_name', 'N/A')}")
        print(f"Region       : {metadata.get('region', 'N/A')}")
        print(f"Pack Size    : {metadata.get('pack_size', 'N/A')}")
        print(f"Last Updated : {metadata.get('last_updated', 'N/A')}")
    else:
        print(f"Product ID   : {metadata.get('product_id', 'N/A')}")
        print(f"SKU ID       : {metadata.get('sku_id', 'N/A')}")
        print(f"Variant ID   : {metadata.get('product_variant_id', 'N/A')}")

    print("\nFull Document:")
    print(document)

print("\n" + "=" * 70)
print(f"Total entries: {total}")
print("=" * 70)
