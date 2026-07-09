# DEPRECATED — retired in favor of the React frontend (frontend/) + FastAPI
# backend (backend/). Kept here for reference only; not part of the active
# app. If you need to run it, it still works from this location.

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from query_agent import identify_product, clean_query
from ui_helpers import (
    split_result,
    flatten,
    confidence_html,
    validate_input,
    process_batch,
    read_uploaded_file,
    create_csv,
    create_excel,
    create_json,
    reset_cache,
)

# ─────────────────────────────────────────────────────────────
# LAYOUT CONSTANTS
# ─────────────────────────────────────────────────────────────

FIELD_ORDER = [
    "product_family", "variant", "exact_variant", "sku", "species",
    "animal_size", "pack_size", "pack_quantity", "animal_age",
]

EMPTY_MATCH = {**{f: "" for f in FIELD_ORDER}, "confidence": "", "reason": ""}


def expand_match(match):
    """(product_family, ..., animal_age, confidence_html, reason) tuple."""
    values = tuple(match.get(f, "") for f in FIELD_ORDER)
    return values + (confidence_html(match.get("confidence")), match.get("reason", ""))


EMPTY_FIELDS = expand_match(EMPTY_MATCH) * 3

BTN_IDLE = ["btn-idle"]
BTN_PROCESSING = ["btn-processing"]
BTN_SUCCESS = ["btn-success"]
BTN_ERROR = ["btn-error"]


CSS = """
#run-btn { transition: background-color 0.2s ease, color 0.2s ease; font-weight: 600; }
#run-btn.btn-idle { background: linear-gradient(90deg,#f97316,#fb923c) !important; color: white !important; border: none !important; }
#run-btn.btn-processing { background: #64748b !important; color: white !important; border: none !important; cursor: progress !important; }
#run-btn.btn-success { background: #16a34a !important; color: white !important; border: none !important; }
#run-btn.btn-error { background: #dc2626 !important; color: white !important; border: none !important; }
"""


# ─────────────────────────────────────────────────────────────
# FEEDBACK HELPERS
# ─────────────────────────────────────────────────────────────

def preview_query(query):
    """Live preview of the normalized query as the user types."""

    cleaned = clean_query(query)

    if not cleaned:
        return gr.update(value="")

    return gr.update(value=f"🧹 Will be matched as: **{cleaned}**")


def preview_file(file):
    """Immediate row-count feedback when a file is uploaded."""

    if file is None:
        return gr.update(value="")

    try:
        queries = read_uploaded_file(file)
        return gr.update(value=f"📄 {len(queries)} row(s) detected in file.")
    except Exception as e:
        return gr.update(value=f"⚠️ Could not read file: {e}")


def reset_button():
    return gr.Button(value="▶ Run", elem_classes=BTN_IDLE, interactive=True)


# ─────────────────────────────────────────────────────────────
# MAIN ROUTER (drives button state + progress bar + results)
# ─────────────────────────────────────────────────────────────

def ui_router(query, file, enable_live_fallback, progress=gr.Progress()):

    error = validate_input(query, file)

    if error:
        gr.Warning(error)
        yield (
            gr.Button(value="▶ Run", elem_classes=BTN_IDLE, interactive=True),
            *EMPTY_FIELDS,
            [],
            gr.update(value=f"⚠️ {error}"),
        )
        return

    # Immediately reflect "working" state before any network calls
    yield (
        gr.Button(value="⏳ Working...", elem_classes=BTN_PROCESSING, interactive=False),
        *EMPTY_FIELDS,
        gr.update(),
        gr.update(value="Working..."),
    )

    try:
        if file is not None:
            progress(0, desc="Reading file...")
            rows, elapsed, stats = process_batch(file, progress=progress, enable_live_fallback=enable_live_fallback)

            status = (
                f"**Done** — {stats.get('total_rows', 0)} row(s), "
                f"{stats.get('unique_queries', 0)} unique, "
                f"{stats.get('cache_hits', 0)} served from cache, "
                f"{stats.get('llm_calls', 0)} reasoned by DeepSeek — {elapsed}s"
            )

            yield (
                gr.Button(value="✅ Done", elem_classes=BTN_SUCCESS, interactive=True),
                *EMPTY_FIELDS,
                rows,
                gr.update(value=status),
            )
            return

        progress(0.2, desc="Searching product database...")
        result = identify_product(query, enable_live_fallback=enable_live_fallback)
        progress(0.9, desc="Finalizing result...")

        primary, alt1, alt2 = split_result(result)

        row = {"input": query}
        row.update(flatten("primary", primary))
        row.update(flatten("alternative_1", alt1))
        row.update(flatten("alternative_2", alt2))

        status = f"**Done** — matched \"{query.strip()}\""

        yield (
            gr.Button(value="✅ Done", elem_classes=BTN_SUCCESS, interactive=True),
            *expand_match(primary),
            *expand_match(alt1),
            *expand_match(alt2),
            [row],
            gr.update(value=status),
        )

    except Exception as e:
        gr.Warning(f"Something went wrong: {e}")
        yield (
            gr.Button(value="⚠ Error — Retry", elem_classes=BTN_ERROR, interactive=True),
            *EMPTY_FIELDS,
            [],
            gr.update(value=f"⚠️ **Error:** {e}"),
        )


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Veterinary Product AI Agent") as app:
    download_state = gr.State([])

    gr.Markdown("## Veterinary Product Matching System")

    with gr.Row():

        # ───────── INPUT ─────────
        with gr.Column(scale=2):

            input_box = gr.Textbox(
                label="Product Query",
                placeholder="e.g. Frontline Spray 100ml dog",
                lines=2
            )
            query_preview = gr.Markdown()

            file_upload = gr.File(label="Upload CSV or TXT")
            file_preview = gr.Markdown()

            live_fallback_checkbox = gr.Checkbox(
                label="Enable live web research fallback",
                value=False,
                info="When a brand can't be confidently matched, research it live via Gemini "
                     "search and add it to the knowledge base on the spot. Uses extra API calls "
                     "and adds latency — off by default."
            )

            with gr.Row():
                run_btn = gr.Button("▶ Run", elem_id="run-btn", elem_classes=BTN_IDLE, scale=3)
                cache_btn = gr.Button("🗑️ Reset Cache", variant="secondary", size="sm", scale=1)

            status_md = gr.Markdown()
            cache_status_md = gr.Markdown()

        # ───────── OUTPUT ─────────
        with gr.Column(scale=3):

            # PRIMARY
            with gr.Accordion("Primary Match", open=True):
                p_product_family = gr.Textbox(label="Product Family", interactive=False)
                p_variant = gr.Textbox(label="Variant", interactive=False)
                p_exact = gr.Textbox(label="Exact Variant", interactive=False)
                p_sku = gr.Textbox(label="SKU", interactive=False)
                p_species = gr.Textbox(label="Species", interactive=False)
                p_size = gr.Textbox(label="Animal Size", interactive=False)
                p_pack_size = gr.Textbox(label="Pack Size", interactive=False)
                p_pack_qty = gr.Textbox(label="Pack Quantity", interactive=False)
                p_age = gr.Textbox(label="Animal Age", interactive=False)
                p_conf = gr.HTML(label="Confidence")
                p_reason = gr.Textbox(label="Reason", interactive=False, lines=3)

            # ALT 1
            with gr.Accordion("Alternative Match 1", open=False):
                a1_product_family = gr.Textbox(label="Product Family", interactive=False)
                a1_variant = gr.Textbox(label="Variant", interactive=False)
                a1_exact = gr.Textbox(label="Exact Variant", interactive=False)
                a1_sku = gr.Textbox(label="SKU", interactive=False)
                a1_species = gr.Textbox(label="Species", interactive=False)
                a1_size = gr.Textbox(label="Animal Size", interactive=False)
                a1_pack_size = gr.Textbox(label="Pack Size", interactive=False)
                a1_pack_qty = gr.Textbox(label="Pack Quantity", interactive=False)
                a1_age = gr.Textbox(label="Animal Age", interactive=False)
                a1_conf = gr.HTML(label="Confidence")
                a1_reason = gr.Textbox(label="Reason", interactive=False, lines=3)

            # ALT 2
            with gr.Accordion("Alternative Match 2", open=False):
                a2_product_family = gr.Textbox(label="Product Family", interactive=False)
                a2_variant = gr.Textbox(label="Variant", interactive=False)
                a2_exact = gr.Textbox(label="Exact Variant", interactive=False)
                a2_sku = gr.Textbox(label="SKU", interactive=False)
                a2_species = gr.Textbox(label="Species", interactive=False)
                a2_size = gr.Textbox(label="Animal Size", interactive=False)
                a2_pack_size = gr.Textbox(label="Pack Size", interactive=False)
                a2_pack_qty = gr.Textbox(label="Pack Quantity", interactive=False)
                a2_age = gr.Textbox(label="Animal Age", interactive=False)
                a2_conf = gr.HTML(label="Confidence")
                a2_reason = gr.Textbox(label="Reason", interactive=False, lines=3)

    OUTPUTS = [
        run_btn,

        p_product_family, p_variant, p_exact, p_sku,
        p_species, p_size, p_pack_size, p_pack_qty,
        p_age, p_conf, p_reason,

        a1_product_family, a1_variant, a1_exact, a1_sku,
        a1_species, a1_size, a1_pack_size, a1_pack_qty,
        a1_age, a1_conf, a1_reason,

        a2_product_family, a2_variant, a2_exact, a2_sku,
        a2_species, a2_size, a2_pack_size, a2_pack_qty,
        a2_age, a2_conf, a2_reason,

        download_state,
        status_md,
    ]

    # live feedback as the user types / uploads
    input_box.change(preview_query, inputs=input_box, outputs=query_preview)
    input_box.change(reset_button, outputs=run_btn)
    file_upload.change(preview_file, inputs=file_upload, outputs=file_preview)
    file_upload.change(reset_button, outputs=run_btn)

    run_btn.click(fn=ui_router, inputs=[input_box, file_upload, live_fallback_checkbox], outputs=OUTPUTS)
    cache_btn.click(fn=reset_cache, outputs=cache_status_md)

    gr.Markdown("### Download Results")

    with gr.Row():
        csv_btn = gr.Button("Download CSV")
        xlsx_btn = gr.Button("Download Excel")
        json_btn = gr.Button("Download JSON")

    download_file = gr.File(label="Generated File")

    csv_btn.click(create_csv, inputs=download_state, outputs=download_file)
    xlsx_btn.click(create_excel, inputs=download_state, outputs=download_file)
    json_btn.click(create_json, inputs=download_state, outputs=download_file)


if __name__ == "__main__":
    app.launch(css=CSS)
