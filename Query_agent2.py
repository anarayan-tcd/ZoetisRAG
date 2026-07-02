from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import json

import chromadb
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────
# ENV SETUP
# ─────────────────────────────────────────────────────────────

env_path = Path(__file__).resolve().parent / "APIKeys.env"
load_dotenv(dotenv_path=env_path)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError(f"Missing DEEPSEEK_API_KEY in {env_path}")


# ─────────────────────────────────────────────────────────────
# LLM CLIENT (DEEPSEEK)
# ─────────────────────────────────────────────────────────────

client_llm = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


# ─────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ─────────────────────────────────────────────────────────────

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ─────────────────────────────────────────────────────────────
# CHROMADB
# ─────────────────────────────────────────────────────────────

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("vet_products")

print(f"Connected to ChromaDB. Total entries: {collection.count()}")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

VECTOR_THRESHOLD = 0.30   # include all relevant product families


# ─────────────────────────────────────────────────────────────
# STEP 1 — VECTOR SEARCH (RETRIEVE PRODUCT FAMILIES)
# ─────────────────────────────────────────────────────────────

def search_chromadb(query: str):

    embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=collection.count()
    )

    candidates = []

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        similarity = 1 - distance

        if similarity >= VECTOR_THRESHOLD:

            candidates.append({
                "product_name": meta.get("product_name"),
                "sku": meta.get("sku_id"),
                "description": doc,
                "similarity": round(similarity, 3)
            })

    candidates.sort(key=lambda x: x["similarity"], reverse=True)

    return candidates


# ─────────────────────────────────────────────────────────────
# STEP 2 — FORMAT CONTEXT FOR LLM
# ─────────────────────────────────────────────────────────────

def build_context(candidates):

    context = ""

    for i, c in enumerate(candidates):

        context += f"""
────────────────────────────────────────────
CANDIDATE {i+1}
PRODUCT FAMILY: {c['product_name']}
SKU: {c['sku']}
SIMILARITY: {c['similarity']}

FULL PRODUCT TABLE:
{c['description']}
"""

    return context


# ─────────────────────────────────────────────────────────────
# STEP 3 — DEEPSEEK REASONING
# ─────────────────────────────────────────────────────────────

def ask_deepseek(query: str, context: str):

    prompt = f"""
You are a veterinary product matching system.

You will be given:
- a messy user query
- a database of veterinary product families and variants

Your task:
1. Identify the BEST matching product variant
2. Also identify TWO additional plausible alternatives
3. Rank them by likelihood
4. Use ONLY provided data (do not invent products)

────────────────────────────────────────────
RULES:
- Primary match = highest confidence
- Alternatives = lower or equal confidence
- All 3 must be different variants if possible
- If only 1 real match exists, repeat it as alternatives with LOWER confidence
- "variant" generates the full name given in Product name column of the database table eg: Spot On for FRONTLINE Spot On Dog Small (2-10kg)
- "exact_variant" generates the products specific name, eg: Spot On for FRONTLINE Spot On Dog Small (2-10kg), or Plus, or Spray etc.
- "animal_size" should also include tags like S, M, L, XL if known, in addition to the range of weights in the product name, eg: 2-10kg, 10-20kg, 20-40kg etc.

────────────────────────────────────────────
OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "primary_match": {{
    "product_family": "",
    "variant": "",
    "exact_variant": "",
    "sku": "",
    "species": "",
    "animal_size": "",
    "pack_size": "",
    "pack_quantity": "",
    "animal_age": "",
    "confidence": "",
    "reason": ""
  }},
  "alternative_matches": [
    {{
        "product_family": "",
        "variant": "",
        "exact_variant": "",
        "sku": "",
        "species": "",
        "animal_size": "",
        "pack_size": "",
        "pack_quantity": "",
        "animal_age": "",
        "confidence": "",
        "reason": ""
    }},
    {{
        "product_family": "",
        "variant": "",
        "exact_variant": "",
        "sku": "",
        "species": "",
        "animal_size": "",
        "pack_size": "",
        "pack_quantity": "",
        "animal_age": "",
        "confidence": "",
        "reason": ""
    }}
  ]
}}

────────────────────────────────────────────
USER QUERY:
{query}

────────────────────────────────────────────
DATABASE:
{context}
"""

    response = client_llm.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content.strip()

    return json.loads(raw)


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def identify_product(query: str):

    if not query.strip():
        return {"error": "Empty query"}

    # Step 1: retrieve relevant product families
    candidates = search_chromadb(query)

    if not candidates:
        return {
            "product_family": "Unknown",
            "alternative_matches": [],
            "confidence": "Low",
            "reason": "No matching product families found in ChromaDB."
        }

    # Step 2: build context
    context = build_context(candidates)

    # DEBUG: show what is being sent to LLM
    #print("\n================ CONTEXT SENT TO DEEPSEEK ================\n")
    # print(context[:3000])  # avoid flooding terminal

    # Step 3: reasoning
    result = ask_deepseek(query, context)

    return result


# ─────────────────────────────────────────────────────────────
# TESTING
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_queries = [
        "Frontline Spray 100ml for dogs",
        "Frontline spot on large dog",
        "fleas treatment cat 3 pipettes",
        "frontline 250ml spray",
        "tick treatment small dog frontline",
        "frontline spot on dog xl 6",
        "frontline plus dog 3 pack"
    ]

    print("=" * 60)
    print("DEEPSEEK VETERINARY PRODUCT CLASSIFIER")
    print("=" * 60)

    for q in test_queries:

        print(f"\nQUERY: {q}")

        result = identify_product(q)

        print(json.dumps(result, indent=2))
        print("-" * 60)