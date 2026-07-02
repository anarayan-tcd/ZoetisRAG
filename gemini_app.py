import gradio as gr
import pandas as pd
from Query_agent2 import identify_product


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def safe(x):
    return "" if x is None else str(x)


def format_confidence(conf):

    if conf is None:
        return "Low ●"

    # convert everything to string safely
    conf_str = str(conf).lower()

    if conf_str in ["high", "high ●"] or "0.8" in conf_str or "0.9" in conf_str:
        return f"{conf} ●"
    elif conf_str in ["medium", "mid"] or "0.5" in conf_str or "0.6" in conf_str:
        return f"{conf} ●"
    elif conf_str.replace(".", "", 1).isdigit():
        # numeric confidence like 0.82
        val = float(conf)

        if val >= 0.75:
            return f"High ({val}) ●"
        elif val >= 0.5:
            return f"Medium ({val}) ●"
        else:
            return f"Low ({val}) ●"

    return f"{conf} ●"


def unpack_match(match):
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
        "confidence": format_confidence(match.get("confidence")),
        "reason": safe(match.get("reason")),
    }


# ─────────────────────────────────────────────────────────────
# CORE PROCESSING
# ─────────────────────────────────────────────────────────────

def run_query(query):

    if not query.strip():
        empty = {k: "" for k in [
            "product_family","variant","exact_variant","sku","species",
            "animal_size","pack_size","pack_quantity","animal_age",
            "confidence","reason"
        ]}
        return empty, empty, empty

    result = identify_product(query)

    primary = unpack_match(result.get("primary_match", {}))
    alts = result.get("alternative_matches", [])

    alt1 = unpack_match(alts[0]) if len(alts) > 0 else unpack_match({})
    alt2 = unpack_match(alts[1]) if len(alts) > 1 else unpack_match({})

    return primary, alt1, alt2


# ─────────────────────────────────────────────────────────────
# FILE PROCESSING
# ─────────────────────────────────────────────────────────────

def process_file(file):

    if file is None:
        return pd.DataFrame()

    path = file.name

    if path.endswith(".csv"):
        df = pd.read_csv(path)
        col = df.columns[0]
        queries = df[col].dropna().tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            queries = [l.strip() for l in f if l.strip()]

    rows = []

    for q in queries:

        result = identify_product(q)
        primary = unpack_match(result.get("primary_match", {}))

        rows.append({
            "input": q,
            **primary
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Veterinary Product AI Agent") as app:

    gr.Markdown("Veterinary Product Matching System")

    with gr.Row():

        # ───────── INPUT ─────────
        with gr.Column(scale=2):

            input_box = gr.Textbox(
                label="Product Query",
                placeholder="e.g. Frontline Spray 100ml dog",
                lines=2
            )

            file_upload = gr.File(label="Upload CSV or TXT")

            run_btn = gr.Button("Run")

        # ───────── OUTPUT ─────────
        with gr.Column(scale=3):

            # PRIMARY
            with gr.Accordion("Primary Match", open=True) as primary_acc:
                p_product_family = gr.Textbox(label="Product Family", interactive=False)
                p_variant = gr.Textbox(label="Variant", interactive=False)
                p_exact = gr.Textbox(label="Exact Variant", interactive=False)
                p_sku = gr.Textbox(label="SKU", interactive=False)
                p_species = gr.Textbox(label="Species", interactive=False)
                p_size = gr.Textbox(label="Animal Size", interactive=False)
                p_pack_size = gr.Textbox(label="Pack Size", interactive=False)
                p_pack_qty = gr.Textbox(label="Pack Quantity", interactive=False)
                p_age = gr.Textbox(label="Animal Age", interactive=False)
                p_conf = gr.Textbox(label="Confidence", interactive=False)
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
                a1_conf = gr.Textbox(label="Confidence", interactive=False)
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
                a2_conf = gr.Textbox(label="Confidence", interactive=False)
                a2_reason = gr.Textbox(label="Reason", interactive=False, lines=3)

    # ─────────────────────────────────────────────────────────────
    # ROUTER (single query only for UI)
    # ─────────────────────────────────────────────────────────────

    def ui_router(query, file):

        if file is not None:
            return process_file(file)

        result = identify_product(query)

        primary = unpack_match(result.get("primary_match", {}))
        alts = result.get("alternative_matches", [])

        alt1 = unpack_match(alts[0]) if len(alts) > 0 else unpack_match({})
        alt2 = unpack_match(alts[1]) if len(alts) > 1 else unpack_match({})

        return (

            # ───────── PRIMARY (11) ─────────
            primary["product_family"],
            primary["variant"],
            primary["exact_variant"],
            primary["sku"],
            primary["species"],
            primary["animal_size"],
            primary["pack_size"],
            primary["pack_quantity"],
            primary["animal_age"],
            primary["confidence"],
            primary["reason"],

            # ───────── ALT 1 (11) ─────────
            alt1["product_family"],
            alt1["variant"],
            alt1["exact_variant"],
            alt1["sku"],
            alt1["species"],
            alt1["animal_size"],
            alt1["pack_size"],
            alt1["pack_quantity"],
            alt1["animal_age"],
            alt1["confidence"],
            alt1["reason"],

            # ───────── ALT 2 (11) ─────────
            alt2["product_family"],
            alt2["variant"],
            alt2["exact_variant"],
            alt2["sku"],
            alt2["species"],
            alt2["animal_size"],
            alt2["pack_size"],
            alt2["pack_quantity"],
            alt2["animal_age"],
            alt2["confidence"],
            alt2["reason"],
        )
    run_btn.click(
        fn=ui_router,
        inputs=[input_box, file_upload],
        outputs=[

            # ───────── PRIMARY ─────────
            p_product_family, p_variant, p_exact, p_sku,
            p_species, p_size, p_pack_size, p_pack_qty,
            p_age, p_conf, p_reason,

            # ───────── ALT 1 ─────────
            a1_product_family, a1_variant, a1_exact, a1_sku,
            a1_species, a1_size, a1_pack_size, a1_pack_qty,
            a1_age, a1_conf, a1_reason,

            # ───────── ALT 2 ─────────
            a2_product_family, a2_variant, a2_exact, a2_sku,
            a2_species, a2_size, a2_pack_size, a2_pack_qty,
            a2_age, a2_conf, a2_reason,
        ]
    )

app.launch()