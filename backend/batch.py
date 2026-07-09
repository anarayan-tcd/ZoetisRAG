# batch.py — in-memory job store + background execution for batch file
# processing. Simple dict-based job tracking, not a real task queue —
# this is a local single-user demo, so a process-lifetime in-memory
# store is enough; nothing here needs to survive a restart.

import time
import uuid
from io import BytesIO

import pandas as pd

import query_agent

_jobs = {}

# Pretty, unambiguous export column names — used for both the primary match
# (no prefix) and alternatives (prefixed "Alternative N "). Keeps CSV/Excel
# headers from colliding with each other (e.g. "Brand" vs "Products") the
# way the old generic "variant"/"exact_variant" names used to.
COLUMN_LABELS = {
    "brand": "Brand",
    "products": "Products",
    "full_product_name": "Full_Product_Name",
    "product_id": "Product ID",
    "sku_id": "SKU ID",
    "product_variant_id": "Product Variant ID",
    "region": "Region",
    "species": "Species",
    "animal_size": "Animal Size",
    "pack_size": "Pack Size",
    "pack_quantity": "Pack Quantity",
    "animal_age": "Animal Age",
    "confidence": "Confidence",
    "reason": "Reason",
}


def read_uploaded_content(filename: str, content: bytes) -> list:
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
        return df.iloc[:, 0].dropna().astype(str).tolist()

    text = content.decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def create_job(total_rows: int) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "completed": 0,
        "total": 0,
        "total_rows": total_rows,
        "rows": None,
        "stats": None,
        "error": None,
    }
    return job_id


def get_job(job_id: str):
    return _jobs.get(job_id)


def _flatten_result(query: str, result: dict) -> dict:
    """
    Primary match columns keep their plain names (Brand, Products, ...) with
    no prefix — they're the main result. Alternatives get an "Alternative N "
    prefix so they don't collide with the primary columns in the exported
    table.
    """
    primary = result.get("primary_match", {})
    alternatives = result.get("alternative_matches", [])

    row = {"Input": query}
    row.update({COLUMN_LABELS.get(k, k): v for k, v in primary.items()})

    for i, alt in enumerate(alternatives[:2], start=1):
        row.update({f"Alternative {i} {COLUMN_LABELS.get(k, k)}": v for k, v in alt.items()})

    return row


def run_batch_job(job_id: str, queries: list, enable_live_fallback: bool):

    job = _jobs[job_id]
    job["status"] = "running"

    def on_progress(done, total):
        job["completed"] = done
        job["total"] = total

    try:
        start = time.time()

        raw_results, stats = query_agent.identify_products_batch(
            queries,
            progress_callback=on_progress,
            enable_live_fallback=enable_live_fallback
        )

        job["rows"] = [
            _flatten_result(query, result)
            for query, result in zip(queries, raw_results)
        ]
        job["stats"] = {**stats, "elapsed": round(time.time() - start, 2)}
        job["status"] = "done"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
