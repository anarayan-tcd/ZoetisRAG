import gradio as gr
import chromadb
import ollama

# ─────────────────────────────────────────────────────────────
# SETUP — Connect to ChromaDB (runs once when app starts)
# ─────────────────────────────────────────────────────────────
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("group1_products")
print(f"Ready! {collection.count()} products in database.")


# ─────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────

def build_prompt(messy_name, candidates):
    candidate_text = ""
    for i, c in enumerate(candidates):
        candidate_text += f"  {i+1}. '{c['name']}' (Similarity: {c['similarity']}, SKU: {c['sku_id']})\n"

    return f"""You are a veterinary product classification expert.

A vet practice entered this product name: "{messy_name}"

Based on a vector similarity search, these are the closest matching products in our database:
{candidate_text}
Your task:
1. Identify which SKU this entry refers to.
2. If you are confident, return that SKU.
3. If the entry is too vague or could mean multiple products, say "UNCERTAIN".

Rules:
- Only return a SKU from the candidates list above.
- Do not guess a SKU that is not in the list.
- Be concise — one short paragraph maximum.

Reply in this exact format:
SKU: [number]
Confidence: [High / Medium / Low]
Reason: [one sentence explaining why]"""


def identify_product(messy_name, similarity_threshold=0.5):
    if not messy_name.strip():
        return "", "", "", "", ""

    # Step 1 — Vector Search
    results = collection.query(
        query_texts=[messy_name],
        n_results=3
    )

    candidates = []
    for j in range(len(results['documents'][0])):
        similarity = round(1 - results['distances'][0][j], 3)
        candidates.append({
            'name':       results['documents'][0][j],
            'similarity': similarity,
            'sku_id':     results['metadatas'][0][j]['sku_id']
        })

    # Format top matches for display
    top_matches_text = "\n".join([
        f"{i+1}.  '{c['name']}'\n    Similarity: {c['similarity']}  |  SKU: {c['sku_id']}"
        for i, c in enumerate(candidates)
    ])

    # Step 2 — Too low similarity — flag as uncertain
    if candidates[0]['similarity'] < similarity_threshold:
        return (
            "Unknown",
            "⚠️ UNCERTAIN",
            "Low",
            f"Best similarity was only {candidates[0]['similarity']} — too low to classify reliably.",
            top_matches_text
        )

    # Step 3 — Near perfect match — skip LLM
    if candidates[0]['similarity'] >= 0.98:
        return (
            candidates[0]['name'].title(),
            f"✅ SKU {candidates[0]['sku_id']}",
            "High",
            f"Near-perfect vector match to '{candidates[0]['name']}'.",
            top_matches_text
        )

    # Step 4 — LLM Reasoning
    prompt = build_prompt(messy_name, candidates)
    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': prompt}]
    )
    raw_reply = response['message']['content'].strip()

    # Parse LLM reply
    sku          = '⚠️ UNCERTAIN'
    confidence   = 'Unknown'
    reason       = raw_reply
    product_name = candidates[0]['name'].title()

    for line in raw_reply.split('\n'):
        line = line.strip()
        if line.startswith('SKU:'):
            sku_part = line.replace('SKU:', '').strip()
            if sku_part.isdigit():
                sku = f"✅ SKU {sku_part}"
        elif line.startswith('Confidence:'):
            confidence = line.replace('Confidence:', '').strip()
        elif line.startswith('Reason:'):
            reason = line.replace('Reason:', '').strip()

    return product_name, sku, confidence, reason, top_matches_text


# ─────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Vet Product Identifier") as app:

    gr.Markdown("""
    # 🐾 Veterinary Product Identifier
    ### Powered by Embeddings + RAG (Retrieval Augmented Generation)
    """)

    with gr.Row():

        # ── Left Column — Input ──────────────────────────────
        with gr.Column(scale=2):
            input_box = gr.Textbox(
                label="Enter a product name",
                placeholder="e.g. Tick Spray: Frontline",
                lines=2
            )
            identify_btn = gr.Button("🔍 Identify Product", variant="primary", size="lg")

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

        # ── Right Column — Output ────────────────────────────
        with gr.Column(scale=2):
            product_name_output = gr.Textbox(label="Product Name", interactive=False)
            sku_output          = gr.Textbox(label="Identified SKU", interactive=False)
            confidence_output   = gr.Textbox(label="Confidence Level", interactive=False)
            reason_output       = gr.Textbox(label="Reason", interactive=False, lines=3)
            matches_output      = gr.Textbox(
                label="Top Vector Matches from Database",
                interactive=False,
                lines=6
            )

    identify_btn.click(
        fn=identify_product,
        inputs=input_box,
        outputs=[product_name_output, sku_output, confidence_output, reason_output, matches_output]
    )

    gr.Markdown("""
    ---
    **How it works:**
    1. Your input is converted into a 384-dimensional embedding vector
    2. ChromaDB searches for the most similar vectors in the database
    3. If similarity is very high (≥0.98), the answer is returned immediately
    4. Otherwise, Mistral (running locally via Ollama) reasons over the top matches and decides
    """)

app.launch(share=False)