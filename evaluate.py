"""
evaluate.py
─────────────────────────────────────────────────────────────
Evaluates the two-phase pipeline across four data quality buckets:

Bucket 1 — Fully labelled (Notes + ProductId + SkuId all present)
Bucket 3 — Over-classified (very short/vague notes but fully labelled)
Bucket 4 — Under-classified (complete notes but ProductId/SkuId missing)

Results saved to: evaluation_results.csv
Summary saved to: evaluation_summary.txt
─────────────────────────────────────────────────────────────
"""

import pandas as pd
import time
import os
from datetime import datetime

from query_pipeline import identify_product

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

CSV_PATH           = "Labelled.csv"
RESULTS_FILE       = "evaluation_results.csv"
SUMMARY_FILE       = "evaluation_summary.txt"
SAMPLES_PER_BUCKET = 50
DELAY_SECONDS      = 3
SHORT_NOTE_THRESHOLD = 15

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD AND CLASSIFY INTO BUCKETS
# ─────────────────────────────────────────────────────────────

print("Loading Labelled.csv...")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows")

df["Notes"]     = df["Notes"].astype(str).str.strip()
df["has_notes"] = df["Notes"].notna() & (df["Notes"] != "") & (df["Notes"] != "nan")
df["has_pid"]   = df["ProductId"].notna()
df["has_sku"]   = df["SkuId"].notna()
df["note_len"]  = df["Notes"].str.len()


def classify_bucket(row) -> int:
    has_notes = row["has_notes"]
    has_pid   = row["has_pid"]
    has_sku   = row["has_sku"]
    note_len  = row["note_len"]

    if not has_notes:
        return 0

    if has_notes and not has_pid and not has_sku:
        return 4

    if has_notes and has_pid and has_sku:
        if note_len < SHORT_NOTE_THRESHOLD:
            return 3
        return 1

    if has_notes and (not has_pid or not has_sku):
        return 4

    return 0


df["bucket"] = df.apply(classify_bucket, axis=1)

print("\nBucket distribution:")
bucket_names = {
    1: "Fully Labelled",
    3: "Over-Classified (vague notes)",
    4: "Under-Classified (missing labels)"
}
for b in [1, 3, 4]:
    count = (df["bucket"] == b).sum()
    print(f"  Bucket {b} — {bucket_names[b]}: {count:,} rows")


# ─────────────────────────────────────────────────────────────
# STEP 2 — SAMPLE FROM EACH BUCKET
# ─────────────────────────────────────────────────────────────

def get_stored_pids() -> set:
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        col  = client.get_collection("vet_products")
        data = col.get(include=["metadatas"])
        return set(
            int(m["product_id"])
            for m in data["metadatas"]
            if m.get("product_id") and str(m.get("product_id")).lstrip("-").isdigit()
        )
    except Exception:
        return set()


def sample_bucket(bucket_num: int, n: int) -> pd.DataFrame:
    stored_pids = get_stored_pids()
    bucket_df   = df[df["bucket"] == bucket_num].copy()

    if bucket_num in [1, 3] and stored_pids:
        bucket_df = bucket_df[
            bucket_df["ProductId"].notna() &
            bucket_df["ProductId"].astype(float).astype(int).isin(stored_pids)
        ]

    if len(bucket_df) == 0:
        print(f"  No rows found for bucket {bucket_num}")
        return pd.DataFrame()

    return bucket_df.sample(
        n=min(n, len(bucket_df)),
        random_state=42
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# STEP 3 — EVALUATE ONE ROW
# ─────────────────────────────────────────────────────────────

def evaluate_row(row, bucket_num: int) -> dict:
    notes    = str(row.get("Notes", ""))
    true_pid = row.get("ProductId")
    true_sku = row.get("SkuId")

    try:
        result = identify_product(notes)
    except Exception as e:
        return {
            "bucket":          bucket_num,
            "notes":           notes,
            "note_length":     len(notes),
            "true_product_id": true_pid,
            "true_sku_id":     true_sku,
            "pred_product":    "ERROR",
            "pred_product_id": None,
            "pred_sku_id":     None,
            "p1_confidence":   "Error",
            "p2_confidence":   None,
            "p1_reason":       "",
            "p2_reason":       "",
            "matched_example": "",
            "p1_correct":      False,
            "sku_correct":     False,
            "flagged_for_review": False,
            "error":           str(e)
        }

    pred_pid = result.get("product_id")
    pred_sku = result.get("sku_id")
    p1_conf  = result.get("p1_confidence", "")

    p1_correct  = False
    sku_correct = False

    try:
        if pred_pid and true_pid:
            p1_correct = int(float(str(pred_pid))) == int(float(str(true_pid)))
    except Exception:
        pass

    try:
        if pred_sku and true_sku:
            sku_correct = int(float(str(pred_sku))) == int(float(str(true_sku)))
    except Exception:
        pass

    # Flag rows where pipeline disagrees at High confidence
    # These may indicate labelling errors in the source data
    flagged_for_review = (
        bucket_num in [1, 3] and
        not p1_correct and
        p1_conf == "High" and
        true_pid is not None
    )

    return {
        "bucket":             bucket_num,
        "notes":              notes,
        "note_length":        len(notes),
        "true_product_id":    true_pid,
        "true_sku_id":        true_sku,
        "pred_product":       result.get("product_family", ""),
        "pred_product_id":    pred_pid,
        "pred_sku_id":        pred_sku,
        "p1_confidence":      p1_conf,
        "p2_confidence":      result.get("p2_confidence", ""),
        "p1_reason":          result.get("p1_reason", ""),
        "p2_reason":          result.get("p2_reason", ""),
        "matched_example":    result.get("matched_example", ""),
        "p1_correct":         p1_correct,
        "sku_correct":        sku_correct,
        "flagged_for_review": flagged_for_review,
        "error":              ""
    }


# ─────────────────────────────────────────────────────────────
# STEP 4 — MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────

# Resume from existing results if available
existing_results = []
processed_keys   = set()

if os.path.exists(RESULTS_FILE):
    existing_df      = pd.read_csv(RESULTS_FILE)
    existing_results = existing_df.to_dict("records")
    processed_keys   = set(zip(existing_df["bucket"], existing_df["notes"]))
    print(f"\nResuming — {len(existing_results)} results already saved")

all_results = list(existing_results)

print("\n" + "=" * 65)
print(f"EVALUATION — {SAMPLES_PER_BUCKET} samples per bucket")
print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
print("=" * 65)

for bucket_num in [1, 3, 4]:

    print(f"\n{'─' * 65}")
    print(f"BUCKET {bucket_num} — {bucket_names[bucket_num]}")
    print(f"{'─' * 65}")

    sample = sample_bucket(bucket_num, SAMPLES_PER_BUCKET)

    if sample.empty:
        print(f"  No samples available — skipping")
        continue

    print(f"  Sampled {len(sample)} rows")

    for idx, row in sample.iterrows():
        notes = str(row.get("Notes", ""))

        if (bucket_num, notes) in processed_keys:
            continue

        print(f"  [{idx+1}/{len(sample)}] {notes[:60]}...")

        result = evaluate_row(row, bucket_num)
        all_results.append(result)
        processed_keys.add((bucket_num, notes))

        pid_mark = "✅" if result["p1_correct"] else "❌"
        sku_mark = "✅" if result["sku_correct"] else "❌"
        flag     = " 🚩 flagged" if result["flagged_for_review"] else ""
        print(f"         ProductId: {pid_mark} | SKU: {sku_mark} | "
              f"Confidence: {result['p1_confidence']}{flag}")

        # Save after every row
        pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)

        time.sleep(DELAY_SECONDS)


# ─────────────────────────────────────────────────────────────
# STEP 5 — SUMMARY REPORT
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("GENERATING SUMMARY")
print("=" * 65)

results_df    = pd.DataFrame(all_results)
summary_lines = []

summary_lines.append("=" * 65)
summary_lines.append("VETERINARY PRODUCT IDENTIFICATION — EVALUATION REPORT")
summary_lines.append(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary_lines.append(f"Pipeline  : ChromaDB ({len(get_stored_pids())} products) + DeepSeek Reasoner")
summary_lines.append("=" * 65)

for bucket_num in [1, 3, 4]:

    b_df = results_df[results_df["bucket"] == bucket_num]
    if b_df.empty:
        continue

    total       = len(b_df)
    p1_correct  = b_df["p1_correct"].sum()
    sku_correct = b_df["sku_correct"].sum()
    errors      = (b_df["error"] != "").sum()
    flagged     = b_df["flagged_for_review"].sum()

    p1_acc  = round(p1_correct / total * 100, 1) if total > 0 else 0
    sku_acc = round(sku_correct / total * 100, 1) if total > 0 else 0

    conf_counts = b_df["p1_confidence"].value_counts().to_dict()
    high   = conf_counts.get("High", 0)
    medium = conf_counts.get("Medium", 0)
    low    = conf_counts.get("Low", 0) + conf_counts.get("Very Low", 0)

    summary_lines.append(f"\nBUCKET {bucket_num} — {bucket_names[bucket_num]}")
    summary_lines.append(f"{'─' * 50}")
    summary_lines.append(f"  Samples tested     : {total}")
    summary_lines.append(f"  ProductId accuracy : {p1_correct}/{total} ({p1_acc}%)")
    summary_lines.append(f"  SKU accuracy       : {sku_correct}/{total} ({sku_acc}%)")
    summary_lines.append(f"  Confidence         : High={high} Medium={medium} Low={low}")

    if bucket_num in [1, 3] and flagged > 0:
        summary_lines.append(
            f"  Flagged for review : {flagged} "
            f"(system disagrees at High confidence — possible labelling error)"
        )

    if errors > 0:
        summary_lines.append(f"  Errors             : {errors}")

total_all  = len(results_df)
p1_all     = results_df["p1_correct"].sum()
sku_all    = results_df["sku_correct"].sum()

summary_lines.append(f"\n{'=' * 65}")
summary_lines.append("OVERALL")
summary_lines.append(f"{'─' * 50}")
summary_lines.append(f"  Total tested       : {total_all}")
summary_lines.append(f"  ProductId accuracy : {p1_all}/{total_all} "
                     f"({round(p1_all/total_all*100,1) if total_all>0 else 0}%)")
summary_lines.append(f"  SKU accuracy       : {sku_all}/{total_all} "
                     f"({round(sku_all/total_all*100,1) if total_all>0 else 0}%)")
summary_lines.append("=" * 65)

summary_text = "\n".join(summary_lines)
print(summary_text)

with open(SUMMARY_FILE, "w") as f:
    f.write(summary_text)

print(f"\nDetailed results : {RESULTS_FILE}")
print(f"Summary          : {SUMMARY_FILE}")
print(f"Finished         : {datetime.now().strftime('%H:%M:%S')}")
