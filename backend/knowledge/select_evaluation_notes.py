# select_evaluation_notes.py — one-off evaluation-set builder, not part of
# the live app or the knowledge-base build pipeline.
#
# For every brand already completed in knowledge_agent2.py's checkpoint,
# selects up to 5 real historical Notes rows from Labelled.csv that have a
# complete (ProductId, ProductVariantId, SkuId) triplet — these become
# ground truth for evaluating query_agent.py's matching accuracy. Reuses
# the same Brand -> SourceProductIds mapping sku_examples_builder.py uses.

import json
import re
import sys
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LABELLED_CSV = BACKEND_ROOT / "data" / "raw" / "Labelled.csv"
BRAND_SEED_CSV = BACKEND_ROOT / "data" / "generated" / "brand_seed_list.csv"
CHECKPOINT_FILE = BACKEND_ROOT / "data" / "generated" / "knowledge_research_checkpoint.json"
OUTPUT_CSV = BACKEND_ROOT / "data" / "generated" / "evaluation_notes_300.csv"

NOTES_PER_BRAND = 5


def parse_id_list(raw: str) -> list:
    if not raw or pd.isna(raw):
        return []
    return [int(x) for x in str(raw).split(";") if x.strip().isdigit()]


def select_representative_rows(brand_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Picks up to n rows with distinct, representative Notes text — most
    frequent phrasing first, deduplicated case-insensitively — mirroring
    sku_examples_builder.py's get_representative_notes."""
    counts = brand_df["Notes"].str.strip().str.lower().value_counts()
    ranked_keys = counts.head(n * 3).index.tolist()

    seen = set()
    chosen_indices = []

    for key in ranked_keys:
        if key in seen:
            continue
        seen.add(key)
        match = brand_df[brand_df["Notes"].str.strip().str.lower() == key].iloc[0]
        chosen_indices.append(match.name)
        if len(chosen_indices) >= n:
            break

    return brand_df.loc[chosen_indices]


def main():
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(f"{CHECKPOINT_FILE} not found — run knowledge_agent2.py first.")

    completed_brands = list(json.load(open(CHECKPOINT_FILE, encoding="utf-8")).keys())
    print(f"{len(completed_brands)} completed brand(s) in the checkpoint.\n")

    seed_df = pd.read_csv(BRAND_SEED_CSV)
    seed_by_brand = {str(row["Brand"]).strip(): row for _, row in seed_df.iterrows()}

    labelled = pd.read_csv(LABELLED_CSV)
    labelled["Notes"] = labelled["Notes"].astype(str)

    # Only rows with a COMPLETE id triplet are eligible.
    labelled = labelled.dropna(subset=["ProductId", "ProductVariantId", "SkuId"])

    selected_rows = []

    for brand in completed_brands:
        seed_row = seed_by_brand.get(brand)

        if seed_row is None:
            print(f"'{brand}': not found in brand_seed_list.csv — skipping.")
            continue

        product_ids = parse_id_list(seed_row.get("SourceProductIds"))

        if not product_ids:
            print(f"'{brand}': no SourceProductIds — skipping.")
            continue

        brand_df = labelled[labelled["ProductId"].isin(product_ids)]

        if brand_df.empty:
            print(f"'{brand}': no Labelled.csv rows with a complete id triplet — skipping.")
            continue

        chosen = select_representative_rows(brand_df, NOTES_PER_BRAND)
        print(f"'{brand}': {len(chosen)} row(s) selected (of {len(brand_df)} eligible).")

        for _, row in chosen.iterrows():
            selected_rows.append({
                "Notes": row["Notes"].strip(),
                "Brand": brand,
                "ProductId": int(row["ProductId"]),
                "ProductVariantId": int(row["ProductVariantId"]),
                "SkuId": int(row["SkuId"]),
            })

    out_df = pd.DataFrame(selected_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nTotal rows selected: {len(out_df)}")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
