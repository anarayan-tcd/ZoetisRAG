# batch.py — in-memory job store + background execution for batch file
# processing. Simple dict-based job tracking, not a real task queue —
# this is a local single-user demo, so a process-lifetime in-memory
# store is enough; nothing here needs to survive a restart.

import time
import uuid
from io import BytesIO

import pandas as pd

from pipeline import query_agent
from export import FIELD_ORDER, COLUMN_LABELS

_jobs = {}


def read_uploaded_content(filename: str, content: bytes) -> list:
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
        return df.iloc[:, 0].dropna().astype(str).tolist()

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("utf-8", errors="replace")
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
        "cancel_requested": False,
    }
    return job_id


def get_job(job_id: str):
    return _jobs.get(job_id)


def cancel_job(job_id: str) -> bool:
    """Requests cancellation of a pending/running job. Already-running
    DeepSeek calls (bounded by MAX_WORKERS) still finish, but no further
    calls are started. Returns False if the job doesn't exist or has
    already finished."""
    job = _jobs.get(job_id)

    if not job or job["status"] not in ("pending", "running"):
        return False

    job["cancel_requested"] = True
    return True


def _flatten_result(query: str, result: dict) -> dict:
    """
    Primary match columns keep their plain names (Brand, Products, ...) with
    no prefix — they're the main result. Alternatives get an "Alternative N "
    prefix so they don't collide with the primary columns in the exported
    table. Columns are always built in export.FIELD_ORDER (not whatever
    order the LLM/pipeline happened to construct the match dict in), so
    CSV/JSON/Excel all get the same deterministic, logically-grouped
    column layout.
    """
    primary = result.get("primary_match", {})
    alternatives = result.get("alternative_matches", [])

    row = {"Input": query}

    for field in FIELD_ORDER:
        row[COLUMN_LABELS[field]] = primary.get(field, "")

    for i, alt in enumerate(alternatives[:2], start=1):
        for field in FIELD_ORDER:
            row[f"Alternative {i} {COLUMN_LABELS[field]}"] = alt.get(field, "")

    return row


def run_batch_job(job_id: str, queries: list, enable_live_fallback: bool, enable_llm_judge: bool = False):

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
            enable_live_fallback=enable_live_fallback,
            enable_llm_judge=enable_llm_judge,
            cancel_check=lambda: job.get("cancel_requested", False)
        )

        job["rows"] = [
            _flatten_result(query, result)
            for query, result in zip(queries, raw_results)
        ]
        job["stats"] = {**stats, "elapsed": round(time.time() - start, 2)}
        job["status"] = "cancelled" if job.get("cancel_requested") else "done"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
