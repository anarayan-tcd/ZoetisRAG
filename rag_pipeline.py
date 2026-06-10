import chromadb
import ollama

# ─────────────────────────────────────────────────────────────
# STEP 1 — Connect to your existing ChromaDB
# (the one you already built in embeddings.py)
# ─────────────────────────────────────────────────────────────
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("group1_products")
print(f"Connected! Database has {collection.count()} entries")
print()

# ─────────────────────────────────────────────────────────────
# STEP 2 — The Prompt Template
# This is what we send to the LLM.
# Notice how structured it is — we tell it exactly:
#   - What role it plays
#   - What the task is
#   - What the candidates are
#   - What format to reply in
# ─────────────────────────────────────────────────────────────

def build_prompt(messy_name, candidates):
    """
    Builds a clear, structured prompt for the LLM.
    
    messy_name  : the raw product name we want to identify
    candidates  : list of dicts with 'name', 'similarity', 'sku_id'
    """
    
    candidate_text = ""
    for i, c in enumerate(candidates):
        candidate_text += f"  {i+1}. '{c['name']}' (Similarity: {c['similarity']}, SKU: {c['sku_id']})\n"

    prompt = f"""You are a veterinary product classification expert.

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

    return prompt


# ─────────────────────────────────────────────────────────────
# STEP 3 — The RAG Function
# This is the full pipeline in one function:
#   1. Search ChromaDB for similar entries
#   2. Build a prompt with the results
#   3. Send to Mistral via Ollama
#   4. Return the final answer
# ─────────────────────────────────────────────────────────────

def identify_product(messy_name, n_candidates=3, similarity_threshold=0.5):
    """
    Takes a messy product name and returns the most likely SKU.
    
    messy_name           : the raw text from the vet practice
    n_candidates         : how many similar entries to retrieve from ChromaDB
    similarity_threshold : if best match is below this, flag as uncertain
    """

    # --- Part A: Vector Search ---
    results = collection.query(
        query_texts=[messy_name],
        n_results=n_candidates
    )

    # Parse the results into a clean list
    candidates = []
    for j in range(len(results['documents'][0])):
        similarity = round(1 - results['distances'][0][j], 3)
        candidates.append({
            'name':       results['documents'][0][j],
            'similarity': similarity,
            'sku_id':     results['metadatas'][0][j]['sku_id']
        })

    # If top match is very low similarity, flag immediately
    if candidates[0]['similarity'] < similarity_threshold:
        return {
            'input':       messy_name,
            'sku':         'UNCERTAIN',
            'confidence':  'Low',
            'reason':      f"Best similarity was only {candidates[0]['similarity']} — too low to classify reliably.",
            'candidates':  candidates
        }

    # If top match is extremely high, we can skip the LLM entirely
    # No point asking the LLM when it's already 99% certain
    if candidates[0]['similarity'] >= 0.98:
        return {
            'input':       messy_name,
            'sku':         candidates[0]['sku_id'],
            'confidence':  'High',
            'reason':      f"Near-perfect vector match to '{candidates[0]['name']}'.",
            'candidates':  candidates
        }

    # --- Part B: LLM Reasoning ---
    prompt = build_prompt(messy_name, candidates)

    response = ollama.chat(
        model='mistral',
        messages=[{'role': 'user', 'content': prompt}]
    )

    raw_reply = response['message']['content'].strip()

    # --- Part C: Parse the LLM reply ---
    sku        = 'UNCERTAIN'
    confidence = 'Unknown'
    reason     = raw_reply  # fallback if parsing fails

    for line in raw_reply.split('\n'):
        line = line.strip()
        if line.startswith('SKU:'):
            sku_part = line.replace('SKU:', '').strip()
            if sku_part.isdigit():
                sku = int(sku_part)
        elif line.startswith('Confidence:'):
            confidence = line.replace('Confidence:', '').strip()
        elif line.startswith('Reason:'):
            reason = line.replace('Reason:', '').strip()

    return {
        'input':      messy_name,
        'sku':        sku,
        'confidence': confidence,
        'reason':     reason,
        'candidates': candidates
    }


# ─────────────────────────────────────────────────────────────
# STEP 4 — Test it with real messy examples
# ─────────────────────────────────────────────────────────────

test_queries = [
    "Tick Spray: Frontline",
    "FRONTLINE FLEA TREATMENT 100ML",
    "apply frontline spray to dog",
    "fipronil spray 100ml",
    "frontline spray cat",
    "flea treatment spray",               # vague — no brand name
    "random product xyz",                 # completely wrong — should be UNCERTAIN
]

print("=" * 65)
print("FULL RAG PIPELINE — VECTOR SEARCH + LLM")
print("=" * 65)

for query in test_queries:
    print(f"\nInput    : '{query}'")
    
    result = identify_product(query)
    
    print(f"SKU      : {result['sku']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Reason   : {result['reason']}")
    print(f"Top vector match: '{result['candidates'][0]['name']}' (similarity: {result['candidates'][0]['similarity']})")
    print("-" * 65)

print()
print("Pipeline complete!")
