import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AQ.Ab8RN6I-EjdKW3GReENKjtJj8WSUD3e1msquBCts1zbS5Hii0w"  # ← paste your key here

PRODUCTS = {
    "Frontline Spray 100ml": 229
}

# Similarity threshold — above this we are confident enough
HIGH_CONFIDENCE_THRESHOLD  = 0.75
# Below this we are not confident at all
LOW_CONFIDENCE_THRESHOLD   = 0.45

# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("vet_products")

print(f"Connected! ChromaDB has {collection.count()} products stored.")
print()


# ─────────────────────────────────────────────────────────────
# STEP 1 — Search ChromaDB
# ─────────────────────────────────────────────────────────────

def search_chromadb(messy_name: str) -> list:
    """
    Converts the messy name to a vector and searches ChromaDB
    for the most similar stored product knowledge.
    Returns a list of candidates with similarity scores.
    """
    embedding = embedding_model.encode(messy_name).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(3, collection.count())
    )

    candidates = []
    for j in range(len(results["documents"][0])):
        similarity = round(1 - results["distances"][0][j], 3)
        candidates.append({
            "description": results["documents"][0][j][:200],
            "similarity":  similarity,
            "sku_id":      results["metadatas"][0][j]["sku_id"],
            "product_name":results["metadatas"][0][j]["product_name"]
        })

    return candidates


# ─────────────────────────────────────────────────────────────
# STEP 2 — Ask Gemini to Reason Over the Candidates
# ─────────────────────────────────────────────────────────────

def ask_gemini_to_identify(messy_name: str, candidates: list) -> dict:
    """
    Passes the messy name and ChromaDB candidates to Gemini.
    Gemini reasons over them and returns the final answer.
    """
    candidate_text = ""
    for i, c in enumerate(candidates):
        candidate_text += f"""
  {i+1}. Product: {c['product_name']}
     SKU: {c['sku_id']}
     Similarity Score: {c['similarity']}
     Description snippet: {c['description']}...
"""

    prompt = f"""
You are an expert veterinary product identification assistant.

A vet practice entered this product name: "{messy_name}"

I searched a vector database of known veterinary products and found these candidates:
{candidate_text}

Known products and their SKUs:
{chr(10).join([f"  - {name}: SKU {sku}" for name, sku in PRODUCTS.items()])}

Your task:
1. Look at the messy product name and the candidates
2. Decide which product this entry refers to
3. If you are confident, return that SKU
4. If too vague or unclear, return UNCERTAIN

Reply in EXACTLY this format — no extra text:
Product: [full product name]
SKU: [number or UNCERTAIN]
Confidence: [High / Medium / Low]
Reason: [one sentence]
"""

    response  = gemini.generate_content(prompt)
    raw_reply = response.text.strip()

    # Parse the reply
    product    = ""
    sku        = "UNCERTAIN"
    confidence = "Low"
    reason     = raw_reply

    for line in raw_reply.split("\n"):
        line = line.strip()
        if line.startswith("Product:"):
            product = line.replace("Product:", "").strip()
        elif line.startswith("SKU:"):
            sku_part = line.replace("SKU:", "").strip()
            if sku_part.isdigit():
                sku = int(sku_part)
            else:
                sku = "UNCERTAIN"
        elif line.startswith("Confidence:"):
            confidence = line.replace("Confidence:", "").strip()
        elif line.startswith("Reason:"):
            reason = line.replace("Reason:", "").strip()

    return {
        "product":    product,
        "sku":        sku,
        "confidence": confidence,
        "reason":     reason
    }


# ─────────────────────────────────────────────────────────────
# STEP 3 — The Main Function
# This is what the Gradio UI calls
# ─────────────────────────────────────────────────────────────

def identify_product(messy_name: str) -> dict:
    """
    Full pipeline:
    1. Search ChromaDB for similar products
    2. If very high similarity — return immediately
    3. If medium — ask Gemini to reason over candidates
    4. If very low — flag as uncertain
    """
    if not messy_name.strip():
        return {
            "product":    "",
            "sku":        "",
            "confidence": "",
            "reason":     "",
            "similarity": 0,
            "method":     ""
        }

    # Step 1 — ChromaDB search
    candidates = search_chromadb(messy_name)

    if not candidates:
        return {
            "product":    "Unknown",
            "sku":        "UNCERTAIN",
            "confidence": "Low",
            "reason":     "No products found in database.",
            "similarity": 0,
            "method":     "ChromaDB — empty"
        }

    top = candidates[0]

    # Step 2 — Very high similarity — skip Gemini entirely
    if top["similarity"] >= HIGH_CONFIDENCE_THRESHOLD:
        return {
            "product":    top["product_name"],
            "sku":        top["sku_id"],
            "confidence": "High",
            "reason":     f"Strong vector match (similarity: {top['similarity']}) to known product description.",
            "similarity": top["similarity"],
            "method":     "ChromaDB only — high confidence"
        }

    # Step 3 — Very low similarity — flag uncertain immediately
    if top["similarity"] < LOW_CONFIDENCE_THRESHOLD:
        return {
            "product":    "Unknown",
            "sku":        "UNCERTAIN",
            "confidence": "Low",
            "reason":     f"Best similarity was only {top['similarity']} — too low to classify.",
            "similarity": top["similarity"],
            "method":     "ChromaDB — too low similarity"
        }

    # Step 4 — Medium similarity — ask Gemini to reason
    result          = ask_gemini_to_identify(messy_name, candidates)
    result["similarity"] = top["similarity"]
    result["method"]     = "ChromaDB + Gemini reasoning"

    return result


# ─────────────────────────────────────────────────────────────
# Test it directly
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "Tick Spray: Frontline",
        "FRONTLINE FLEA TREATMENT 100ML",
        "apply frontline spray to dog",
        "fipronil spray 100ml",
        "flea treatment spray",
        "random product xyz",
    ]

    print("=" * 60)
    print("QUERY AGENT TEST")
    print("=" * 60)

    for query in test_queries:
        print(f"\nInput      : '{query}'")
        result = identify_product(query)
        print(f"Product    : {result['product']}")
        print(f"SKU        : {result['sku']}")
        print(f"Confidence : {result['confidence']}")
        print(f"Reason     : {result['reason']}")
        print(f"Similarity : {result['similarity']}")
        print(f"Method     : {result['method']}")
        print("-" * 60)
