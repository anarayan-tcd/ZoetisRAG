# chromadb_client.py
#
# Singleton embedding model + ChromaDB client/collections, shared across
# query_agent.py and everything under knowledge/ so no script loads the
# embedding model twice or opens a second ChromaDB connection when one
# calls into another (e.g. query_agent.py's live-research fallback
# calling knowledge_agent2's research/store functions).

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = str(Path(__file__).resolve().parent / "data" / "chroma_db")
VET_PRODUCTS_COLLECTION = "vet_products"

# Base name — kept stable so knowledge/sku_examples_builder.py's own
# "<base>_v2" derivation (its rebuild target) never chases whatever the
# pipeline is currently pointed at.
SKU_EXAMPLES_COLLECTION = "sku_examples"

# What the query-time pipeline actually reads (resources.py). Separate
# from SKU_EXAMPLES_COLLECTION so flipping this to a new build is a
# one-line change here, with the previous collection left untouched as
# an instant rollback (just point this back at "sku_examples").
ACTIVE_SKU_EXAMPLES_COLLECTION = SKU_EXAMPLES_COLLECTION + "_v2"

_embedding_model = None
_chroma_client = None
_collections = {}


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model

    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model loaded!")

    return _embedding_model


def get_chroma_client():
    global _chroma_client

    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    return _chroma_client


def get_collection(name: str = VET_PRODUCTS_COLLECTION):
    if name not in _collections:
        client = get_chroma_client()
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Connected to ChromaDB collection '{name}'. {collection.count()} entr(y/ies) stored.")
        _collections[name] = collection

    return _collections[name]


def get_all_variant_documents() -> dict:
    """
    Bulk-fetches every existing vet_products document text in ONE round
    trip, keyed by id. Used to diff "did this variant's content actually
    change" in memory instead of one collection.get(ids=[vid]) call per
    variant.
    """
    collection = get_collection(VET_PRODUCTS_COLLECTION)

    if collection.count() == 0:
        return {}

    data = collection.get(include=["documents"])

    return dict(zip(data["ids"], data["documents"]))
