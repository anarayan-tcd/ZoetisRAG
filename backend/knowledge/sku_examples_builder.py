# sku_examples_builder.py
#
# Builds a second ChromaDB collection ("sku_examples") of REAL historical
# vet-entered `Notes` strings from Labelled.csv, grouped by actual
# historical SkuId within each canonical brand from brand_seed_list.csv.
#
# Why a separate collection: knowledge_agent2.py's `vet_products` documents
# are Gemini-researched, structured descriptions — accurate, but phrased
# nothing like how a vet practice actually types a prescription line. This
# collection gives Query_agent2.py's Phase 2 a few authentic, real-world
# phrasing examples per brand to use as auxiliary context, which should
# help disambiguate messy queries in a way a synthesized description can't.
#
# Never wipes — skip-if-already-embedded. Labelled.csv is a fixed
# historical snapshot, so unlike the live-researched vet_products
# collection there's nothing to re-diff on a later run.

import re
import sys
from pathlib import Path

import pandas as pd

# sku_examples_builder.py lives in backend/knowledge/ but chromadb_client.py
# lives in backend/ — add the parent directory to sys.path so it can be
# imported without turning this into a full installable package.
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from chromadb_client import get_embedding_model, get_collection, SKU_EXAMPLES_COLLECTION

# Some brand names in brand_seed_list.csv are non-Latin script (e.g.
# Japanese-market products). Windows' console defaults to cp1252, which
# can't encode them — reconfigure stdout so printing a brand name never
# crashes the run partway through.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LABELLED_CSV = BACKEND_ROOT / "data" / "raw" / "Labelled.csv"
BRAND_SEED_CSV = BACKEND_ROOT / "data" / "generated" / "brand_seed_list.csv"

# How many example Notes to store per (brand, SkuId) pair.
EXAMPLES_PER_SKU = 5


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def parse_id_list(raw: str) -> list:
    if not raw or pd.isna(raw):
        return []
    return [int(x) for x in str(raw).split(";") if x.strip().isdigit()]


def get_representative_notes(sku_df: pd.DataFrame, n: int = EXAMPLES_PER_SKU) -> list:
    """
    Picks the most representative Notes examples for a SKU: the most
    frequent entries (most common way vets actually write this SKU),
    filtered to a reasonable length and deduplicated case-insensitively.
    """
    notes = sku_df["Notes"].dropna().astype(str)
    notes = notes[notes.str.len().between(5, 120)]

    if notes.empty:
        return []

    top_notes = notes.value_counts().head(n * 2).index.tolist()

    seen = set()
    results = []

    for note in top_notes:
        key = note.lower().strip()
        if key not in seen:
            seen.add(key)
            results.append(note.strip())
        if len(results) >= n:
            break

    return results


def build_sku_examples():

    print("=" * 60)
    print("SKU EXAMPLES BUILDER — Labelled.csv -> sku_examples collection")
    print("=" * 60)

    if not BRAND_SEED_CSV.exists():
        raise FileNotFoundError(f"{BRAND_SEED_CSV} not found — run extract_brands.py first.")

    embedding_model = get_embedding_model()
    collection = get_collection(SKU_EXAMPLES_COLLECTION)

    existing_ids = set(collection.get(include=[])["ids"]) if collection.count() else set()
    print(f"{len(existing_ids)} example(s) already stored — these will be skipped.\n")

    seed_df = pd.read_csv(BRAND_SEED_CSV)

    df = pd.read_csv(LABELLED_CSV)
    df["Notes"] = df["Notes"].astype(str)

    total_stored = 0
    total_skipped = 0

    for _, row in seed_df.iterrows():

        brand = str(row["Brand"]).strip()
        product_ids = parse_id_list(row.get("SourceProductIds"))

        if not product_ids:
            continue

        brand_df = df[df["ProductId"].isin(product_ids)]

        if brand_df.empty:
            continue

        def store_examples(group_df: pd.DataFrame, sku_id: int, variant_id, id_prefix: str):
            """Stores up to EXAMPLES_PER_SKU representative Notes for one
            (brand, sku_id-or-variant) group. Shared by both the main
            SkuId-grouped pass and the missing-SkuId fallback pass below."""
            nonlocal total_stored, total_skipped

            examples = get_representative_notes(group_df)

            for i, note in enumerate(examples):

                # SkuId is NOT globally unique in Labelled.csv — 84 SkuIds
                # (e.g. 7, which is both Advocate's and Frontline's) are
                # reused across different ProductIds/brands. Namespacing
                # by brand as well as sku_id/variant avoids cross-brand id
                # collisions silently overwriting or skipping real examples.
                entry_id = f"example__{slugify(brand)}__{id_prefix}__{i}"

                if entry_id in existing_ids:
                    total_skipped += 1
                    continue

                embedding = embedding_model.encode(note, normalize_embeddings=True).tolist()

                collection.upsert(
                    documents=[note],
                    embeddings=[embedding],
                    ids=[entry_id],
                    metadatas=[{
                        "brand": brand,
                        "product_id": int(group_df["ProductId"].iloc[0]),
                        "sku_id": sku_id,
                        "product_variant_id": variant_id if variant_id is not None else 0,
                    }]
                )

                existing_ids.add(entry_id)
                total_stored += 1

        sku_ids = sorted(brand_df["SkuId"].dropna().unique())

        if sku_ids:
            print(f"{brand}: {len(sku_ids)} historical SKU(s)")

        reached_variant_ids = set()

        for sku_id in sku_ids:

            sku_id = int(sku_id)
            sku_df = brand_df[brand_df["SkuId"] == sku_id]

            variant_id = (
                int(sku_df["ProductVariantId"].dropna().iloc[0])
                if sku_df["ProductVariantId"].notna().any() else None
            )

            if variant_id is not None:
                reached_variant_ids.add(variant_id)

            store_examples(sku_df, sku_id, variant_id, id_prefix=f"sku_{sku_id}")

        # Some ProductVariantIds have NO row with a SkuId recorded at all —
        # the SkuId-driven loop above never sees them, since it only groups
        # rows that have a non-null SkuId. Without this pass those variants
        # would be silently invisible to Phase 2's real-phrasing lookup and
        # id-validation set, even though they're real, valid variants.
        # sku_id is stored as 0 (the same "not available" sentinel already
        # used for a missing product_variant_id) since no historical SkuId
        # exists to attach.
        all_variant_ids = set(int(v) for v in brand_df["ProductVariantId"].dropna().unique())
        missing_variant_ids = sorted(all_variant_ids - reached_variant_ids)

        if missing_variant_ids:
            print(f"{brand}: {len(missing_variant_ids)} variant(s) with no recorded SkuId — adding fallback example(s)")

        for variant_id in missing_variant_ids:
            variant_df = brand_df[brand_df["ProductVariantId"] == variant_id]
            store_examples(variant_df, sku_id=0, variant_id=variant_id, id_prefix=f"variant_{variant_id}")

    print("\n" + "=" * 60)
    print(f"Done! {total_stored} example(s) stored, {total_skipped} already present and skipped.")
    print(f"sku_examples collection now has {collection.count()} entries total.")
    print("=" * 60)


if __name__ == "__main__":
    build_sku_examples()
