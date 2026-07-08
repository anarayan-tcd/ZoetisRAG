"""
query_pipeline.py
─────────────────────────────────────────────────────────────
Two-phase veterinary product identification pipeline.

PHASE 1 — Product Identification (vet_products collection)
  Searches Gemini-researched knowledge base
  Returns: product_name + product_id

PHASE 2 — SKU Identification (sku_knowledge collection)
  Searches real labelled data examples filtered to product_id
  Returns: exact sku_id

FALLBACK — Live Gemini Research
  If Phase 1 confidence too low → Gemini researches live
  Merges new knowledge → re-searches → continues pipeline
─────────────────────────────────────────────────────────────
"""

import os
import json
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
# KNOWN PRODUCTS
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
    "NexGard Combo":            218,
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
    "Milprazon":                80,
    "Profender":                7,
    "Program":                  12,
    "Droncit":                  4,
    "Proheart":                 73,
    "Seresto":                  6,
    "Scalibor":                 27,
    "Indorex":                  43,
    "Rearguard":                16,
    "Sentinel":                 18,
    "Interceptor":              20,
    "Endogard":                 41,
    "Strongid P":               17,
    "Vectra 3D":                52,
    "Xenex Spray":              61,
    "Odour Eliminator":         75,
}

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.55   # below → trigger live Gemini research
MAX_CANDIDATES       = 10     # candidates passed to DeepSeek
SKU_CANDIDATES       = 5      # top SKU matches to consider

# ─────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────

genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

deepseek = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

# Phase 1 collection
try:
    vet_collection = client.get_collection("vet_products")
    print(f"vet_products    : {vet_collection.count()} entries")
except Exception as e:
    raise ValueError(f"vet_products collection not found: {e}")

# Phase 2 collection
try:
    sku_collection = client.get_collection("sku_knowledge")
    print(f"sku_knowledge   : {sku_collection.count()} entries")
except Exception as e:
    raise ValueError(f"sku_knowledge collection not found: {e}")


# ─────────────────────────────────────────────────────────────
# PHASE 1 — VECTOR SEARCH (vet_products)
# ─────────────────────────────────────────────────────────────

def search_vet_products(query: str) -> list:
    """
    Searches the Gemini-built product knowledge base.
    Returns top candidates with product_name and product_id.
    """
    embedding = embedding_model.encode(
        query, normalize_embeddings=True
    ).tolist()

    n = min(MAX_CANDIDATES, vet_collection.count())

    results = vet_collection.query(
        query_embeddings=[embedding],
        n_results=n
    )

    candidates = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        candidates.append({
            "product_name": str(meta.get("product_name", "")),
            "product_id":   meta.get("product_id"),
            "description":  str(doc),
            "similarity":   round(1 - distance, 3)
        })

    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return candidates


# ─────────────────────────────────────────────────────────────
# PHASE 2 — SKU SEARCH (sku_knowledge)
# ─────────────────────────────────────────────────────────────

def search_sku(query: str, product_id: int) -> list:
    """
    Searches the SKU knowledge base filtered to a specific ProductId.
    Returns top SKU candidates from real labelled data.
    """
    try:
        embedding = embedding_model.encode(
            query, normalize_embeddings=True
        ).tolist()

        # Get all entries for this product_id
        all_entries = sku_collection.get(
            where={"product_id": int(product_id)},
            include=["documents", "metadatas", "embeddings"]
        )

        if not all_entries["ids"]:
            return []

        # Score each entry by cosine similarity manually
        import numpy as np
        query_vec = np.array(embedding)
        scored    = []

        for i, (doc, meta, emb) in enumerate(zip(
            all_entries["documents"],
            all_entries["metadatas"],
            all_entries["embeddings"]
        )):
            stored_vec = np.array(emb)
            similarity = float(
                np.dot(query_vec, stored_vec) /
                (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-10)
            )
            scored.append({
                "sku_id":             int(meta.get("sku_id", 0)),
                "product_variant_id": int(meta.get("product_variant_id", 0)),
                "product_name":       str(meta.get("product_name", "")),
                "example_note":       str(doc),
                "similarity":         round(similarity, 3)
            })

        # Sort and deduplicate by SKU
        scored.sort(key=lambda x: x["similarity"], reverse=True)

        seen     = set()
        top_skus = []
        for entry in scored:
            sku = entry["sku_id"]
            if sku not in seen:
                seen.add(sku)
                top_skus.append(entry)
            if len(top_skus) >= SKU_CANDIDATES:
                break

        return top_skus

    except Exception as e:
        print(f"  SKU search error: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# LIVE GEMINI RESEARCH (fallback)
# ─────────────────────────────────────────────────────────────

def research_with_gemini(query: str) -> dict:
    """Triggered when Phase 1 confidence is too low."""
    print(f"\n  Low confidence — triggering live Gemini research...")

    known_list = "\n".join([
        f"  - {name} (ProductId: {pid})"
        for name, pid in KNOWN_PRODUCTS.items()
    ])

    prompt = f"""
You are a veterinary product research assistant.
A vet practice entered: "{query}"

Search the web for this product. Then check if it matches any of these:
{known_list}

Reply in EXACTLY this format:

MATCHED_PRODUCT_NAME: [name from list above, or full product name if new]
MATCHED_PRODUCT_ID: [ProductId number, or new_unknown]

TABLE:
[pipe-separated table with columns:
Product Name | Size | Active Ingredients | Animals | Weight Range | Parasites | Application Method | Pack Quantity | Protection Duration | Age Minimum | Vet Abbreviations | Extra Notes]
"""

    try:
        response     = gemini.generate_content(prompt)
        raw          = response.text.strip()
        product_name = "Unknown"
        product_id   = "new_unknown"
        table_text   = ""
        in_table     = False

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("MATCHED_PRODUCT_NAME:"):
                product_name = line.replace("MATCHED_PRODUCT_NAME:", "").strip()
            elif line.startswith("MATCHED_PRODUCT_ID:"):
                pid_raw = line.replace("MATCHED_PRODUCT_ID:", "").strip()
                product_id = int(pid_raw) if pid_raw.isdigit() else "new_unknown"
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

    except Exception as e:
        print(f"  Gemini research failed: {e}")
        return {"product_name": "Unknown", "product_id": "new_unknown", "table_text": ""}


def merge_into_chromadb(product_name: str, product_id, table_text: str):
    """Merges Gemini research into vet_products. Never wipes."""
    if not table_text:
        return 0

    rows = [
        line.strip() for line in table_text.split("\n")
        if "|" in line.strip()
        and not line.strip().lower().startswith("product name")
        and line.strip().replace("|", "").replace("-", "").replace(" ", "")
    ]

    if not rows:
        return 0

    safe_name = (
        product_name.lower()
        .replace(" ", "_").replace("/", "_")
        .replace("(", "").replace(")", "")
    )

    try:
        existing    = vet_collection.get(where={"product_name": product_name})
        start_index = len(existing["ids"])
    except Exception:
        start_index = 0

    stored = 0
    for i, row in enumerate(rows):
        entry_id  = f"{safe_name}_live_{start_index + i}"
        embedding = embedding_model.encode(
            row, normalize_embeddings=True
        ).tolist()

        vet_collection.upsert(
            documents=[row],
            embeddings=[embedding],
            ids=[entry_id],
            metadatas=[{
                "product_name": str(product_name),
                "product_id":   int(product_id) if str(product_id).isdigit() else 0,
                "sku_id":       "",
                "phase":        1,
                "source":       "live_research"
            }]
        )
        stored += 1

    print(f"  Merged {stored} new rows into vet_products")
    return stored


# ─────────────────────────────────────────────────────────────
# DEEPSEEK — PHASE 1 REASONING
# ─────────────────────────────────────────────────────────────

def normalise_confidence(conf: str) -> str:
    """Converts any confidence value to High/Medium/Low/Very Low."""
    if not conf:
        return "Low"
    s = str(conf).lower().replace("%", "").strip()
    try:
        v = float(s)
        if v >= 0.7 or v >= 70:   return "High"
        elif v >= 0.5 or v >= 50: return "Medium"
        else:                      return "Low"
    except ValueError:
        if "very low" in s:   return "Very Low"
        elif "high" in s:     return "High"
        elif "medium" in s:   return "Medium"
        else:                  return "Low"


def ask_deepseek_phase1(query: str, candidates: list) -> dict:
    """
    DeepSeek reasons over Phase 1 candidates to identify the product family.
    Returns product_family, product_id, confidence, reason.
    """

    context = ""
    for i, c in enumerate(candidates):
        context += f"""
CANDIDATE {i+1}
  PRODUCT FAMILY : {c['product_name']}
  PRODUCT ID     : {c['product_id']}
  SIMILARITY     : {c['similarity']}
  VARIANT ROW    : {c['description']}
"""

    prompt = f"""
You are a veterinary product matching system.

Identify which product family this messy vet entry belongs to.

CRITICAL RULES:
1. If the query contains an explicit brand name, that brand MUST be the primary match.
2. Scan ALL candidates carefully before deciding — the brand may be in a lower-ranked candidate.
3. "NexGard Spectra" and "NexGard" are DIFFERENT products. Match the most specific one.
4. Use ONLY: High, Medium, Low, Very Low for confidence — no numbers or percentages.

OUTPUT FORMAT (JSON only):
{{
  "product_family": "",
  "product_id": "",
  "confidence": "",
  "reason": ""
}}

USER QUERY: {query}

CANDIDATES:
{context}
"""

    try:
        response = deepseek.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content.strip())
        result["confidence"] = normalise_confidence(result.get("confidence", ""))
        return result
    except Exception as e:
        print(f"  DeepSeek Phase 1 error: {e}")
        top = candidates[0] if candidates else {}
        return {
            "product_family": top.get("product_name", "Unknown"),
            "product_id":     top.get("product_id", ""),
            "confidence":     "Low",
            "reason":         f"DeepSeek error — fell back to top similarity match"
        }


# ─────────────────────────────────────────────────────────────
# DEEPSEEK — PHASE 2 REASONING
# ─────────────────────────────────────────────────────────────

def ask_deepseek_phase2(query: str, sku_candidates: list, product_name: str) -> dict:
    """
    DeepSeek reasons over Phase 2 SKU candidates to identify the exact SKU.
    Returns sku_id, confidence, reason.
    """

    context = ""
    for i, c in enumerate(sku_candidates):
        context += f"""
SKU CANDIDATE {i+1}
  SKU ID         : {c['sku_id']}
  SIMILARITY     : {c['similarity']}
  EXAMPLE NOTE   : {c['example_note']}
"""

    prompt = f"""
You are a veterinary product SKU matching system.

The product has already been identified as: {product_name}

Now identify the EXACT SKU variant from the candidates below.

Rules:
- Match on weight range, pack size, species, application method
- Use ONLY: High, Medium, Low, Very Low for confidence
- If unsure between two SKUs, pick the higher similarity one

OUTPUT FORMAT (JSON only):
{{
  "sku_id": "",
  "confidence": "",
  "reason": "",
  "matched_example": ""
}}

USER QUERY: {query}

SKU CANDIDATES:
{context}
"""

    try:
        response = deepseek.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content.strip())
        result["confidence"] = normalise_confidence(result.get("confidence", ""))
        return result
    except Exception as e:
        print(f"  DeepSeek Phase 2 error: {e}")
        top = sku_candidates[0] if sku_candidates else {}
        return {
            "sku_id":          top.get("sku_id", ""),
            "confidence":      "Low",
            "reason":          "DeepSeek error — fell back to top similarity SKU",
            "matched_example": top.get("example_note", "")
        }


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def identify_product(query: str) -> dict:
    """
    Full two-phase pipeline.
    Returns product_name, product_id, sku_id, confidences, reasons.
    """

    if not query.strip():
        return {"error": "Empty query"}

    print(f"\nQuery: '{query}'")

    # ── PHASE 1: Product Identification ───────────────────────

    p1_candidates  = search_vet_products(query)

    if not p1_candidates:
        return {
            "query":           query,
            "product_family":  "Unknown",
            "product_id":      None,
            "sku_id":          None,
            "p1_confidence":   "Low",
            "p2_confidence":   None,
            "reason":          "No products found in knowledge base",
            "sku_reason":      None
        }

    top_similarity = p1_candidates[0]["similarity"]
    print(f"  Phase 1 top: {top_similarity} ({p1_candidates[0]['product_name']})")

    # Trigger live Gemini research if confidence too low
    if top_similarity < CONFIDENCE_THRESHOLD:
        print(f"  Below threshold {CONFIDENCE_THRESHOLD} — researching live...")
        research = research_with_gemini(query)

        if research["table_text"]:
            merge_into_chromadb(
                research["product_name"],
                research["product_id"],
                research["table_text"]
            )
            p1_candidates  = search_vet_products(query)
            top_similarity = p1_candidates[0]["similarity"]
            print(f"  New top similarity: {top_similarity}")

    # DeepSeek Phase 1 reasoning
    p1_result = ask_deepseek_phase1(query, p1_candidates)
    print(f"  Phase 1 result: {p1_result.get('product_family')} "
          f"(ProductId: {p1_result.get('product_id')}) "
          f"— {p1_result.get('confidence')}")

    product_id = p1_result.get("product_id")

    # ── PHASE 2: SKU Identification ───────────────────────────

    sku_result = {
        "sku_id":          None,
        "confidence":      "Low",
        "reason":          "No SKU data available for this product",
        "matched_example": ""
    }

    if product_id and str(product_id).lstrip("-").isdigit():
        pid_int      = int(product_id)
        sku_candidates = search_sku(query, pid_int)

        if sku_candidates:
            print(f"  Phase 2 top SKU: {sku_candidates[0]['sku_id']} "
                  f"(similarity: {sku_candidates[0]['similarity']})")
            sku_result = ask_deepseek_phase2(
                query,
                sku_candidates,
                p1_result.get("product_family", "")
            )
            print(f"  Phase 2 result: SKU {sku_result.get('sku_id')} "
                  f"— {sku_result.get('confidence')}")
        else:
            print(f"  Phase 2: no SKU data found for ProductId {pid_int}")

    # ── COMPILE FINAL RESULT ──────────────────────────────────

    return {
        "query":          query,
        "product_family": p1_result.get("product_family", "Unknown"),
        "product_id":     product_id,
        "p1_confidence":  p1_result.get("confidence", "Low"),
        "p1_reason":      p1_result.get("reason", ""),
        "sku_id":         sku_result.get("sku_id"),
        "p2_confidence":  sku_result.get("confidence"),
        "p2_reason":      sku_result.get("reason", ""),
        "matched_example":sku_result.get("matched_example", "")
    }


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_queries = [
        "Frontline Spray 100ml for dogs",
        "FRONTLINE S/ON LGE 20-40K 6 PIP",
        "Advocate medium dog 3 pipettes",
        "fipronil spray cat",
        "Stronghold small cat 45mg 3 pack",
        "Effipro spot on large dog 6 pipettes",
        "random product xyz",
    ]

    print("=" * 65)
    print("TWO-PHASE QUERY PIPELINE TEST")
    print("=" * 65)

    for q in test_queries:
        print(f"\n{'─' * 65}")
        result = identify_product(q)
        print(f"\nRESULT:")
        print(f"  Product Family : {result.get('product_family')}")
        print(f"  Product ID     : {result.get('product_id')}")
        print(f"  P1 Confidence  : {result.get('p1_confidence')}")
        print(f"  P1 Reason      : {result.get('p1_reason')}")
        print(f"  SKU ID         : {result.get('sku_id')}")
        print(f"  P2 Confidence  : {result.get('p2_confidence')}")
        print(f"  P2 Reason      : {result.get('p2_reason')}")
        print(f"  Matched Note   : {result.get('matched_example')}")
