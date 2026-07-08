"""
sku_builder.py
─────────────────────────────────────────────────────────────
Builds a second ChromaDB collection — sku_knowledge — for
SKU-level identification within each product family.

HOW IT WORKS:
1. Reads Labelled.csv (your ground truth)
2. For each product already in vet_products ChromaDB:
   - Finds all unique SKUs under that ProductId
   - Gets the most representative Notes examples per SKU
   - Embeds each example
   - Stores in sku_knowledge collection with SKU label
3. Phase 2 of the pipeline:
   - Phase 1 (vet_products) identifies ProductId
   - Phase 2 (sku_knowledge) narrows down to exact SKU

NEVER WIPES — upsert logic, knowledge only grows
─────────────────────────────────────────────────────────────
"""

import os
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Path to your labelled CSV
CSV_PATH = "Labelled.csv"

# How many example Notes to store per SKU
# More examples = better matching but larger database
EXAMPLES_PER_SKU = 5

# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")

client = chromadb.PersistentClient(path="./chroma_db")

# Get existing vet_products collection to know which products are stored
try:
    vet_collection = client.get_collection("vet_products")
    print(f"Connected to vet_products — {vet_collection.count()} entries")
except Exception:
    raise ValueError("vet_products collection not found — run knowledge_builder.py first!")

# Get or create sku_knowledge collection
try:
    sku_collection = client.get_collection("sku_knowledge")
    print(f"Connected to existing sku_knowledge — {sku_collection.count()} entries")
except Exception:
    sku_collection = client.create_collection(
        name="sku_knowledge",
        metadata={"hnsw:space": "cosine"}
    )
    print("Created new sku_knowledge collection")


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD LABELLED CSV
# ─────────────────────────────────────────────────────────────

print(f"\nLoading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows")
print(f"Columns: {df.columns.tolist()}")

# Keep only rows with both Notes and SkuId filled in
df_clean = df.dropna(subset=["Notes", "SkuId", "ProductId"]).copy()
df_clean["SkuId"]     = df_clean["SkuId"].astype(int)
df_clean["ProductId"] = df_clean["ProductId"].astype(int)
print(f"Rows with Notes + ProductId + SkuId: {len(df_clean):,}")


# ─────────────────────────────────────────────────────────────
# STEP 2 — GET WHICH PRODUCTS ARE IN VET_PRODUCTS
# ─────────────────────────────────────────────────────────────

# Get all product_ids already in vet_products collection
vet_data     = vet_collection.get(include=["metadatas"])
stored_pids  = set()
pid_to_name  = {}

for meta in vet_data["metadatas"]:
    pid  = meta.get("product_id")
    name = meta.get("product_name")
    if pid:
        stored_pids.add(int(pid))
        pid_to_name[int(pid)] = name

print(f"\nProductIds in vet_products: {sorted(stored_pids)}")
print(f"Product names: {[pid_to_name[p] for p in sorted(stored_pids)]}")


# ─────────────────────────────────────────────────────────────
# STEP 3 — FILTER CSV TO ONLY STORED PRODUCTS
# ─────────────────────────────────────────────────────────────

df_filtered = df_clean[df_clean["ProductId"].isin(stored_pids)].copy()
print(f"\nRows matching stored products: {len(df_filtered):,}")
print(f"Unique ProductIds: {df_filtered['ProductId'].nunique()}")
print(f"Unique SKUs: {df_filtered['SkuId'].nunique()}")


# ─────────────────────────────────────────────────────────────
# STEP 4 — GET EXISTING SKUs IN SKU_KNOWLEDGE
# ─────────────────────────────────────────────────────────────

def get_existing_skus() -> set:
    """Returns set of SKU IDs already stored in sku_knowledge."""
    try:
        data = sku_collection.get(include=["metadatas"])
        return set(
            str(m.get("sku_id"))
            for m in data["metadatas"]
            if m.get("sku_id")
        )
    except Exception:
        return set()

existing_skus = get_existing_skus()
print(f"\nSKUs already in sku_knowledge: {len(existing_skus)}")


# ─────────────────────────────────────────────────────────────
# STEP 5 — BUILD SKU KNOWLEDGE
# ─────────────────────────────────────────────────────────────

def get_representative_notes(sku_df: pd.DataFrame, n: int = EXAMPLES_PER_SKU) -> list:
    """
    Gets the most representative Notes examples for a SKU.
    Strategy:
    - Take the most frequent Notes (most commonly how this SKU is written)
    - Filter out very short or very long entries
    - Deduplicate cleaned versions
    """
    notes = sku_df["Notes"].dropna().astype(str)

    # Filter reasonable length entries
    notes = notes[notes.str.len().between(5, 100)]

    if len(notes) == 0:
        return []

    # Get most frequent entries
    top_notes = notes.value_counts().head(n * 2).index.tolist()

    # Deduplicate by lowercase
    seen    = set()
    results = []
    for note in top_notes:
        key = note.lower().strip()
        if key not in seen:
            seen.add(key)
            results.append(note.strip())
        if len(results) >= n:
            break

    return results


def store_sku(product_name: str, product_id: int, sku_id: int,
              notes_examples: list, variant_id=None):
    """
    Embeds each Notes example for a SKU and stores in sku_knowledge.
    Uses upsert — never wipes existing entries.
    """
    stored = 0
    safe_name = (
        product_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )

    for i, note in enumerate(notes_examples):
        entry_id  = f"sku_{sku_id}_example_{i}"
        embedding = embedding_model.encode(
            note,
            normalize_embeddings=True
        ).tolist()

        sku_collection.upsert(
            documents=[note],
            embeddings=[embedding],
            ids=[entry_id],
            metadatas=[{
                "product_name":       str(product_name),
                "product_id":         int(product_id),
                "sku_id":             int(sku_id),
                "product_variant_id": int(variant_id) if variant_id is not None else 0,
                "example_index":      int(i),
                "phase":              2
            }]
        )
        stored += 1

    return stored


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SKU BUILDER — Building Phase 2 knowledge")
print("=" * 65)

total_skus   = 0
total_rows   = 0
skipped_skus = 0

# Process each ProductId
for pid in sorted(stored_pids):

    product_name = pid_to_name.get(pid, f"ProductId_{pid}")
    product_df   = df_filtered[df_filtered["ProductId"] == pid]
    unique_skus  = product_df["SkuId"].unique()

    print(f"\n{product_name} (ProductId: {pid})")
    print(f"  Unique SKUs: {len(unique_skus)}")

    for sku_id in sorted(unique_skus):

        # Skip if already stored
        if str(sku_id) in existing_skus:
            skipped_skus += 1
            continue

        sku_df     = product_df[product_df["SkuId"] == sku_id]
        variant_id = sku_df["ProductVariantId"].iloc[0] if "ProductVariantId" in sku_df.columns else None
        count      = len(sku_df)

        # Get representative Notes examples
        notes = get_representative_notes(sku_df)

        if not notes:
            print(f"  SKU {sku_id}: no valid notes — skipping")
            continue

        # Store in ChromaDB
        stored = store_sku(
            product_name=product_name,
            product_id=pid,
            sku_id=sku_id,
            notes_examples=notes,
            variant_id=variant_id
        )

        print(f"  SKU {sku_id:6d}: {count:5d} rows in CSV → stored {stored} examples")

        total_skus += 1
        total_rows += stored


print("\n" + "=" * 65)
print("SKU KNOWLEDGE BUILD COMPLETE")
print(f"  SKUs processed  : {total_skus}")
print(f"  SKUs skipped    : {skipped_skus} (already stored)")
print(f"  Examples stored : {total_rows}")
print(f"  sku_knowledge   : {sku_collection.count()} total entries")
print("=" * 65)
print("\nNext step: update query_pipeline.py to use Phase 2 SKU lookup")
