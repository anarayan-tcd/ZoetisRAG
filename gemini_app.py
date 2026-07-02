import gradio as gr
from query_agent import identify_product

# ─────────────────────────────────────────────────────────────
# Gradio wrapper function
# ─────────────────────────────────────────────────────────────
def run_identification(messy_name: str):
    if not messy_name.strip():
        return "", "", "", "", ""

    result = identify_product(messy_name)

    # Format SKU
    sku = result["sku"]
    if sku == "UNCERTAIN":
        sku_display = "⚠️ UNCERTAIN"
    else:
        sku_display = f"✅ SKU {sku}"

    # Format confidence
    confidence = result["confidence"]
    if "high" in confidence.lower():
        confidence_display = "✅ High"
    elif "medium" in confidence.lower():
        confidence_display = "⚠️ Medium"
    else:
        confidence_display = "❌ Low"

    # Format method
    method = result.get("method", "")
    similarity = result.get("similarity", 0)
    method_display = f"{method}  |  Top similarity: {similarity}"

    return (
        result["product"],
        sku_display,
        confidence_display,
        result["reason"],
        method_display
    )


# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────
with gr.Blocks(title="Vet Product AI Agent") as app:

    gr.Markdown("""
    # 🐾 Veterinary Product AI Agent
    ### Powered by Gemini + ChromaDB
    Knowledge sourced directly from official websites and Amazon by Gemini.
    """)

    with gr.Row():

        # ── Left — Input ─────────────────────────────────────
        with gr.Column(scale=2):
            input_box = gr.Textbox(
                label="Enter a product name",
                placeholder="e.g. Tick Spray: Frontline",
                lines=2
            )
            identify_btn = gr.Button(
                "🤖 Identify Product",
                variant="primary",
                size="lg"
            )

            gr.Examples(
                examples=[
                    ["Tick Spray: Frontline"],
                    ["FRONTLINE FLEA TREATMENT 100ML"],
                    ["apply frontline spray to dog"],
                    ["fipronil spray 100ml"],
                    ["flea treatment spray"],
                    ["random product xyz"],
                ],
                inputs=input_box,
                label="Try these examples"
            )

            gr.Markdown("""
            **How it works:**
            1. Gemini researched Frontline from official sites + Amazon
            2. That knowledge is stored in ChromaDB as vectors
            3. Your input is matched against that knowledge
            4. If confident → instant answer from ChromaDB
            5. If uncertain → Gemini reasons over the candidates
            """)

        # ── Right — Output ────────────────────────────────────
        with gr.Column(scale=2):
            product_output    = gr.Textbox(label="Product Name",     interactive=False)
            sku_output        = gr.Textbox(label="Identified SKU",   interactive=False)
            confidence_output = gr.Textbox(label="Confidence Level", interactive=False)
            reason_output     = gr.Textbox(label="Reason",           interactive=False, lines=3)
            method_output     = gr.Textbox(label="Method Used",      interactive=False)

    identify_btn.click(
        fn=run_identification,
        inputs=input_box,
        outputs=[
            product_output,
            sku_output,
            confidence_output,
            reason_output,
            method_output
        ]
    )

app.launch(share=False)
