# ui_helpers.py
#
# DEPRECATED — retired in favor of the React frontend (frontend/) + FastAPI
# backend (backend/). Kept here for reference only; not part of the active
# app. If you need to run it, it still works from this location.

import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from query_agent import (
    identify_product,
    identify_products_batch,
    clean_query,
    cache_size,
    clear_cache,
)


# ============================================================
# CONSTANTS
# ============================================================

MATCH_FIELDS = [
    "product_family",
    "variant",
    "exact_variant",
    "sku",
    "species",
    "animal_size",
    "pack_size",
    "pack_quantity",
    "animal_age",
    "confidence",
    "reason"
]


# ============================================================
# BASIC HELPERS
# ============================================================

def safe(value):
    """
    Converts None values into empty strings
    """
    if value is None:
        return ""

    return str(value)



def unpack_match(match):
    """
    Converts query_agent output into UI friendly format
    """

    if not match:
        match = {}

    return {
        "product_family": safe(match.get("product_family")),
        "variant": safe(match.get("variant")),
        "exact_variant": safe(match.get("exact_variant")),
        "sku": safe(match.get("sku")),
        "species": safe(match.get("species")),
        "animal_size": safe(match.get("animal_size")),
        "pack_size": safe(match.get("pack_size")),
        "pack_quantity": safe(match.get("pack_quantity")),
        "animal_age": safe(match.get("animal_age")),
        "confidence": safe(match.get("confidence")),
        "reason": safe(match.get("reason"))
    }



def flatten(prefix, data):
    """
    Converts nested match dictionaries into CSV columns
    """

    return {
        f"{prefix}_{key}": value
        for key, value in data.items()
    }



def split_result(result):
    """
    Turns a raw identify_product()-style dict (or an {"error": ...}
    dict) into (primary, alt1, alt2) UI-friendly match dicts.
    """

    if "error" in result:
        error_match = unpack_match({})
        error_match["product_family"] = "Error"
        error_match["reason"] = result["error"]
        return error_match, unpack_match({}), unpack_match({})

    primary = unpack_match(result.get("primary_match", {}))
    alternatives = result.get("alternative_matches", [])

    alt1 = unpack_match(alternatives[0]) if len(alternatives) > 0 else unpack_match({})
    alt2 = unpack_match(alternatives[1]) if len(alternatives) > 1 else unpack_match({})

    return primary, alt1, alt2



# ============================================================
# INPUT VALIDATION / FEEDBACK
# ============================================================

def reset_cache():
    """
    Clears the DeepSeek result cache so subsequent queries are
    re-reasoned instead of served from prior results.
    """

    cleared = cache_size()
    clear_cache()

    gr.Info(f"Cache cleared ({cleared} cached result(s) removed).")

    return gr.update(value=f"🗑️ Cache cleared — {cleared} entr{'y' if cleared == 1 else 'ies'} removed.")



def validate_input(query, file):
    """
    Returns an error message string if nothing usable was provided,
    otherwise None. Lets the UI give immediate feedback before
    spending an API call.
    """

    has_query = bool((query or "").strip())
    has_file = file is not None

    if not has_query and not has_file:
        return "Enter a product query or upload a file before running."

    return None



# ============================================================
# CONFIDENCE DISPLAY
# ============================================================

def confidence_html(confidence):

    confidence = safe(confidence)

    c = confidence.lower()

    colour = "#dc2626"  # red

    if "high" in c:
        colour = "#16a34a"  # green

    elif "medium" in c or "mid" in c:
        colour = "#d97706"  # orange

    label = confidence if confidence else "—"

    return f"""
    <div style="
        font-size:15px;
        font-weight:600;
        display:flex;
        align-items:center;
        gap:6px;
    ">
        <span style="
            color:{colour};
            font-size:20px;
            line-height:1;
        ">
        ●
        </span>
        {label}
    </div>
    """



# ============================================================
# QUERY PROCESSING — SINGLE
# ============================================================

def process_single_query(query, enable_live_fallback=False):

    result = identify_product(query, enable_live_fallback=enable_live_fallback)

    return split_result(result)



# ============================================================
# QUERY PROCESSING — BATCH
# ============================================================

def read_uploaded_file(file):

    if file.name.endswith(".csv"):

        df = pd.read_csv(file.name)

        queries = (
            df.iloc[:,0]
            .dropna()
            .astype(str)
            .tolist()
        )

    else:

        with open(
            file.name,
            "r",
            encoding="utf-8"
        ) as f:

            queries = [
                line.strip()
                for line in f
                if line.strip()
            ]


    return queries



def process_batch(file, progress=None, enable_live_fallback=False):
    """
    Runs a full file through the classifier using the batched,
    cached, parallel pipeline in query_agent, and reports live
    progress via the optional `progress` callback (e.g. gr.Progress).
    """

    queries = read_uploaded_file(file)

    if not queries:
        gr.Warning("No readable rows found in the uploaded file.")
        return [], 0.0, {}

    start = time.time()

    def on_progress(done, of_total):
        if progress:
            progress(
                done / of_total if of_total else 1,
                desc=f"Reasoning over unique products ({done}/{of_total})"
            )

    raw_results, stats = identify_products_batch(
        queries,
        progress_callback=on_progress,
        enable_live_fallback=enable_live_fallback
    )

    rows = []

    for query, result in zip(queries, raw_results):

        primary, alt1, alt2 = split_result(result)

        row = {"input": query}

        row.update(flatten("primary", primary))
        row.update(flatten("alternative_1", alt1))
        row.update(flatten("alternative_2", alt2))

        rows.append(row)

    elapsed = round(time.time() - start, 2)

    gr.Info(
        f"Processed {stats['total_rows']} rows "
        f"({stats['unique_queries']} unique, {stats['cache_hits']} served from cache, "
        f"{stats['llm_calls']} reasoned) in {elapsed}s"
    )

    return rows, elapsed, stats



# ============================================================
# DOWNLOAD FUNCTIONS
# ============================================================


def create_csv(results):

    if not results:
        return None


    path = (
        Path(tempfile.gettempdir())
        /
        "veterinary_product_results.csv"
    )


    pd.DataFrame(results).to_csv(
        path,
        index=False
    )


    gr.Info(
        "CSV file created successfully"
    )


    return str(path)



def create_excel(results):

    if not results:
        return None


    path = (
        Path(tempfile.gettempdir())
        /
        "veterinary_product_results.xlsx"
    )


    pd.DataFrame(results).to_excel(
        path,
        index=False
    )


    gr.Info(
        "Excel file created successfully"
    )


    return str(path)



def create_json(results):

    if not results:
        return None


    path = (
        Path(tempfile.gettempdir())
        /
        "veterinary_product_results.json"
    )


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


    gr.Info(
        "JSON file created successfully"
    )


    return str(path)
