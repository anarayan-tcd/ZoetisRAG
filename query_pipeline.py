"""
query_pipeline.py
─────────────────────────────────────────────────────────────
Identifies veterinary products from messy vet entries.

HOW IT WORKS:
1. Search ChromaDB for similar product variants
2. If confident enough → pass to DeepSeek for final answer
3. If NOT confident → Gemini researches the product live
   - Gemini decides if it matches an existing ProductId
   - If yes → merges new knowledge into ChromaDB
   - If no  → stores as new unknown product
   - Re-searches ChromaDB with enriched knowledge
   - Passes to DeepSeek for final answer

NEVER WIPES — knowledge base only ever grows
─────────────────────────────────────────────────────────────
"""

import os
import json
import time
import google.generativeai as genai
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# LOAD API KEYS
# ─────────────────────────────────────────────────────────────

env_path = Path(__file__).resolve().parent / "APIKeys.env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in APIKeys.env")

if not DEEPSEEK_API_KEY:
    raise ValueError("Missing DEEPSEEK_API_KEY in APIKeys.env")

# ─────────────────────────────────────────────────────────────
# KNOWN PRODUCTS — used by Gemini to match new queries
# ─────────────────────────────────────────────────────────────

KNOWN_PRODUCTS = {
    "Frontline":                32,
    "Frontline Plus":           37,
    "Advocate":                 1,
    "Advantage":                5,
    "Advantix":                 2,
    "Stronghold":               68,
    "Effipro":                  40,
    "Revolution":               71,
    "Prinovox":                 45,
    "Broadline":                34,
    "Activyl":                  24,
    "Practic":                  14,
    "Xeno 450":                 62,
    "Promeris Duo":             70,
    "Bravecto Plus Cat":        165,
    "Bravecto":                 22,
    "NexGard":                  35,
    "NexGard Spectra":          36,
    "Simparica":                74,
    "Simparica Trio":           199,
    "Comfortis":                10,
    "Credelio":                 141,
    "Capstar":                  15,
    "Panacur":                  23,
    "Drontal":                  3,
    "Milbemax":                 13,
    "Milbemax Worming Program":  211,
    "Milpro":                   44,
    "Profender":                7,
    "Program":                  12,
    "Droncit":                  4,
    "Proheart":                 73,
    "Seresto":                  6,
    "Scalibor":                 27,
    "Indorex":                  43,
}

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Below this threshold → trigger live Gemini research
# Raised to 0.55 so edge cases like "Bravecto Plus Dog" get researched
CONFIDENCE_THRESHOLD = 0.55

# How many ChromaDB candidates to pass to DeepSeek
MAX_CANDIDATES = 10

# ─────────────────────────────────────────────────────────────
# SETUP — Gemini, DeepSeek, ChromaDB, Embeddings
# ─────────────────────────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

deepseek = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("vet_products")

print(f"Connected to ChromaDB — {collection.count()} entries loaded")


# ─────────────────────────────────────────────────────────────
# STEP 1 — VECTOR SEARCH
# ─────────────────────────────────────────────────────────────

def search_chromadb(query: str, n_results: int = MAX_CANDIDATES) -> list:
    """
    Converts the query to a vector and searches ChromaDB.
    Returns a list of candidate matches with similarity scores.
    """
    embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    n_results = min(n_results, collection.count())

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )

    candidates = []

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        similarity = round(1 - distance, 3)

        candidates.append({
            "product_name": meta.get("product_name"),
            "product_id":   meta.get("product_id"),
            "sku_id":       meta.get("sku_id", ""),
            "description":  doc,
            "similarity":   similarity
        })

    candidates.sort(key=lambda x: x["similarity"], reverse=True)

    return candidates


# ─────────────────────────────────────────────────────────────
# STEP 2 — LIVE GEMINI RESEARCH (fallback for unknown products)
# ─────────────────────────────────────────────────────────────

def research_unknown_with_gemini(query: str) -> dict:
    """
    Called when ChromaDB confidence is too low.
    Gemini researches the product on the web and decides:
    - Does it match an existing known product? → return that ProductId
    - Is it genuinely new? → return product_id = "new_unknown"
    """
    print(f"\n  Low confidence — triggering live Gemini research...")

    known_list = "\n".join([
        f"  - {name} (ProductId: {pid})"
        for name, pid in KNOWN_PRODUCTS.items()
    ])

    prompt = f"""
You are a veterinary product research assistant.

A vet practice entered this product name: "{query}"

TASK 1 — IDENTIFY THE PRODUCT:
Search the web to find out what veterinary product this refers to.
Look at official manufacturer websites, Amazon UK, VioVet, PetDrugsOnline, VetUK.

TASK 2 — MATCH TO KNOWN PRODUCTS:
Here is a list of known products in our database:
{known_list}

Does this query refer to one of the above known products?
- If YES → return the exact product name and ProductId from the list above
- If NO  → return product_name as the full product name you found, and product_id as "new_unknown"

TASK 3 — BUILD PRODUCT TABLE:
Return a pipe-separated table of ALL variants of this product.

Table columns (exactly):
Product Name | Size | Active Ingredients | Animals | Weight Range | Parasites | Application Method | Pack Quantity | Protection Duration | Age Minimum | Extra Notes

Table rules:
- First row = header
- One row per unique variant
- Different sizes/weights/pack quantities = different rows
- NO paragraphs, NO markdown, NO commentary

Return your response in this EXACT format:

MATCHED_PRODUCT_NAME: [exact name from known list, or full product name if new]
MATCHED_PRODUCT_ID: [ProductId number from known list, or new_unknown]

TABLE:
[pipe-separated table here]
"""

    response = gemini.generate_content(prompt)
    raw      = response.text.strip()

    product_name = "Unknown"
    product_id   = "new_unknown"
    table_text   = ""

    lines    = raw.split("\n")
    in_table = False

    for line in lines:
        line = line.strip()

        if line.startswith("MATCHED_PRODUCT_NAME:"):
            product_name = line.replace("MATCHED_PRODUCT_NAME:", "").strip()

        elif line.startswith("MATCHED_PRODUCT_ID:"):
            pid_raw = line.replace("MATCHED_PRODUCT_ID:", "").strip()
            if pid_raw.isdigit():
                product_id = int(pid_raw)
            else:
                product_id = "new_unknown"

        elif line == "TABLE:":
            in_table = True

        elif in_table and "|" in line:
            table_text += line + "\n"

    print(f"  Gemini identified: {product_name} (ProductId: {product_id})")

    return {
        "product_name": product_name,
        "product_id":   product_id,
        "table_text":   table_text.strip()
    }


# ─────────────────────────────────────────────────────────────
# STEP 3 — MERGE NEW KNOWLEDGE INTO CHROMADB
# ─────────────────────────────────────────────────────────────

def merge_into_chromadb(product_name: str, product_id, table_text: str):
    """
    Merges Gemini's new research into ChromaDB.
    NEVER WIPES — uses upsert, knowledge only grows.
    New rows get unique IDs so they don't overwrite existing ones.
    """

    if not table_text:
        print("  No table to store — skipping merge")
        return 0

    rows = []
    for line in table_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("product name"):
            continue
        cleaned = line.replace("|", "").replace("-", "").replace(" ", "")
        if not cleaned:
            continue
        if "|" in line:
            rows.append(line)

    if not rows:
        print("  No valid rows found in table")
        return 0

    safe_name = (
        product_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )

    try:
        existing    = collection.get(where={"product_name": product_name})
        start_index = len(existing["ids"])
    except Exception:
        start_index = 0

    stored = 0

    for i, row in enumerate(rows):
        entry_id  = f"{safe_name}_live_{start_index + i}"
        embedding = embedding_model.encode(
            row,
            normalize_embeddings=True
        ).tolist()

        collection.upsert(
            documents=[row],
            embeddings=[embedding],
            ids=[entry_id],
            metadatas=[{
                "product_name": product_name,
                "product_id":   product_id,
                "sku_id":       "",
                "row_index":    start_index + i,
                "phase":        1,
                "source":       "live_research"
            }]
        )
        stored += 1

    print(f"  Merged {stored} new rows into ChromaDB for '{product_name}'")
    print(f"  ChromaDB now has {collection.count()} total entries")
    return stored


# ─────────────────────────────────────────────────────────────
# STEP 4 — DEEPSEEK REASONING
# ─────────────────────────────────────────────────────────────

def build_context(candidates: list) -> str:
    """Formats ChromaDB candidates into context for DeepSeek."""

    context = ""
    for i, c in enumerate(candidates):
        context += f"""
────────────────────────────────────────────
CANDIDATE {i+1}
PRODUCT FAMILY : {c['product_name']}
PRODUCT ID     : {c['product_id']}
SIMILARITY     : {c['similarity']}

VARIANT ROW:
{c['description']}
"""
    return context


def ask_deepseek(query: str, context: str) -> dict:
    """
    Passes the query and ChromaDB candidates to DeepSeek.
    Returns structured JSON with primary match + 2 alternatives.
    """

    prompt = f"""
You are a veterinary product matching system.

You will be given:
- A messy product name entered by a vet practice
- A database of veterinary product variants from ChromaDB

Your task:
1. Identify the BEST matching product variant
2. Identify TWO additional plausible alternatives
3. Rank them by likelihood
4. Use ONLY data provided — do not invent products

────────────────────────────────────────────
CRITICAL RULES — FOLLOW EXACTLY:

RULE 1 — BRAND NAME TAKES ABSOLUTE PRIORITY:
If the query contains an explicit brand name (e.g. "Advocate", "Frontline",
"Bravecto", "NexGard", "Milbemax", "Simparica" etc.), that product family
MUST be the primary match — even if another product scores higher similarity.
Never return a different brand as primary when a brand is explicitly named in the query.
Only ignore the brand name if it genuinely does not exist in ANY candidate.

RULE 2 — PRODUCT VARIANT MATTERS:
"NexGard Spectra" and "NexGard" are different products.
"Frontline" and "Frontline Plus" are different products.
"Bravecto" and "Bravecto Plus" are different products.
Match the most specific variant mentioned in the query.

RULE 3 — CONFIDENCE SCORING:
- High = brand name matches AND weight/size/pack matches
- Medium = brand name matches but size/pack uncertain
- Low = brand name matches but other details don't match or species mismatch
- Very Low = no brand name match, just similar product type

RULE 4 — OUTPUT FORMAT:
- "variant" = full product name from the database row
- "exact_variant" = specific type e.g. Spray, Spot On, Plus, Combo, Chewable
- "animal_size" = include weight range AND size tag (S/M/L/XL) if known
- "confidence" = use words: High, Medium, Low, Very Low (not numbers or percentages)

────────────────────────────────────────────
OUTPUT FORMAT (STRICT JSON ONLY):

{{
  "primary_match": {{
    "product_family": "",
    "product_id": "",
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
      "product_id": "",
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
      "product_id": "",
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

    response = deepseek.chat.completions.create(
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

def identify_product(query: str) -> dict:
    """
    Full pipeline:
    1. Search ChromaDB
    2. If confident → DeepSeek → answer
    3. If not confident → Gemini researches live
                       → merges into ChromaDB
                       → re-searches ChromaDB
                       → DeepSeek → answer
    """

    if not query.strip():
        return {"error": "Empty query"}

    print(f"\nQuery: '{query}'")

    # ── Step 1: Vector Search ──────────────────────────────
    candidates = search_chromadb(query)

    if not candidates:
        return {
            "error": "ChromaDB is empty",
            "primary_match": {
                "product_family": "Unknown",
                "confidence":     "Low",
                "reason":         "No products in knowledge base"
            }
        }

    top_similarity = candidates[0]["similarity"]
    print(f"  Top similarity: {top_similarity} ({candidates[0]['product_name']})")

    # ── Step 2: Check confidence ───────────────────────────
    if top_similarity < CONFIDENCE_THRESHOLD:

        print(f"  Similarity {top_similarity} below threshold {CONFIDENCE_THRESHOLD}")

        research = research_unknown_with_gemini(query)

        if research["table_text"]:
            merge_into_chromadb(
                research["product_name"],
                research["product_id"],
                research["table_text"]
            )

            print(f"  Re-searching ChromaDB with enriched knowledge...")
            candidates     = search_chromadb(query)
            top_similarity = candidates[0]["similarity"]
            print(f"  New top similarity: {top_similarity}")

        else:
            print("  Gemini returned no table — using original candidates")

    # ── Step 3: DeepSeek Reasoning ─────────────────────────
    print(f"  Passing {len(candidates)} candidates to DeepSeek...")
    context = build_context(candidates)
    result  = ask_deepseek(query, context)

    return result


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_queries = [
        "Frontline Spray 100ml for dogs",
        "FRONTLINE S/ON LGE 20-40K 6 PIP",
        "Advocate medium dog 3 pipettes",
        "NexGard Spectra 3.5-7.5kg 3 pack",
        "Bravecto chew large dog 20-40kg",
        "fipronil spray cat",
        "flea tick treatment small dog spot on",
        "milbemax wormer cat each",
        "Bravecto Plus Dog Large",
    ]

    print("=" * 65)
    print("QUERY PIPELINE TEST")
    print("=" * 65)

    for q in test_queries:
        print(f"\n{'─' * 65}")
        result = identify_product(q)
        print(json.dumps(result, indent=2))
