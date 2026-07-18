# batch.py — in-memory job store + background execution for batch file
# processing. Simple dict-based job tracking, not a real task queue —
# this is a local single-user demo, so a process-lifetime in-memory
# store is enough; nothing here needs to survive a restart.

import time
import uuid
from io import BytesIO

import pandas as pd

from pipeline.raw import query_agent
from .export import (
    FIELD_ORDER, COLUMN_LABELS, ID_COUNT_COLUMN, POSSIBLE_IDS_COLUMN,
    ID_REASON_COLUMN, format_combination,
)

_jobs = {}


# Common names test files have used for the free-text query column across
# this project's various challenge/eval spreadsheets — checked before
# falling back to a heuristic, since column position isn't reliable (some
# files lead with a "Row" index column instead).
_NOTES_COLUMN_NAMES = ("Notes", "Test_Name", "Query", "Text", "Description")


def _pick_notes_column(df: pd.DataFrame) -> str:
    for name in _NOTES_COLUMN_NAMES:
        if name in df.columns:
            return name

    text_columns = [c for c in df.columns if df[c].dtype == object]
    if not text_columns:
        return df.columns[0]

    # No recognized name — assume it's whichever text column holds the
    # longest strings on average (an index/id column stays short and
    # numeric-like; free-text prescription notes run much longer).
    return max(text_columns, key=lambda c: df[c].dropna().astype(str).str.len().mean())


def read_uploaded_content(filename: str, content: bytes) -> list:
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
        return df[_pick_notes_column(df)].dropna().astype(str).tolist()

    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(BytesIO(content))
        return df[_pick_notes_column(df)].dropna().astype(str).tolist()

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
    One flat row per query: the product's own fields (always built in
    export.FIELD_ORDER, not whatever order the LLM/pipeline happened to
    construct the dict in, so CSV/JSON/Excel share one deterministic
    layout), plus a summary of the id combinations.

    The combinations themselves are carried under the private
    "_id_combinations" key rather than flattened into columns: their count
    varies per row (one product can map to five duplicate catalog
    listings), so the exporters expand them into their own sheet/array
    instead. export._public_row strips this before writing the flat sheet.
    """
    match = result.get("product_match", {})
    combinations = result.get("id_combinations", []) or []

    row = {"Input": query}

    for field in FIELD_ORDER:
        row[COLUMN_LABELS[field]] = match.get(field, "")

    row[ID_COUNT_COLUMN] = len(combinations)
    row[POSSIBLE_IDS_COLUMN] = "; ".join(format_combination(c) for c in combinations)
    row[ID_REASON_COLUMN] = result.get("id_combinations_reason", "")
    row["_id_combinations"] = combinations

    return row


def run_batch_job(job_id: str, queries: list, enable_live_fallback: bool,
                  enable_llm_judge: bool = False, mode: str = "raw"):

    job = _jobs[job_id]
    job["status"] = "running"

    def on_progress(done, total):
        job["completed"] = done
        job["total"] = total

    # "structured" routes to the catalog-backed pipeline (which accepts and
    # ignores the two enable_* flags — they patch raw-pipeline knowledge
    # gaps the authoritative catalog doesn't have). Anything else is "raw".
    if mode == "structured":
        from pipeline.structured import query_agent_structured as engine
    else:
        engine = query_agent

    try:
        start = time.time()

        raw_results, stats = engine.identify_products_batch(
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

        elapsed = round(time.time() - start, 2)

        job["stats"] = {
            **stats,
            "elapsed": elapsed,
            # Only rows actually reasoned over by DeepSeek this run (not
            # cache hits) count toward the per-row rate — cache hits are
            # ~free and would otherwise water down the figure.
            "avg_seconds_per_resolved_row": (
                round(elapsed / stats["llm_calls"], 3) if stats.get("llm_calls") else 0.0
            ),
        }
        job["status"] = "cancelled" if job.get("cancel_requested") else "done"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
