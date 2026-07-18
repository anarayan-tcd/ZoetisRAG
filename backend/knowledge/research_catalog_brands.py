# research_catalog_brands.py — one-shot driver: Gemini-research every
# catalog brand (Merged_Product_Variant_SKU.xlsx "Product Name") that has
# no researched pool in vet_products yet. Reuses knowledge_agent2's
# research/store machinery wholesale; only the brand list and the research
# hints differ — hints come from the brand's own catalog VariantDescriptions
# (real market names), which target the search far better than generic
# guesses. Checkpointed per brand, same as the main agent, so quota
# exhaustion partway through never loses work.
#
# Run:  python -u knowledge/research_catalog_brands.py   (from backend/)

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chromadb_client import get_all_variant_documents, get_collection
from knowledge.catalog_common import (
    catalog_brands_needing_research,
    load_catalog,
    research_hints_for_brand,
)
from knowledge.knowledge_agent2 import (
    GeminiUnavailableError,
    research_brand_with_gemini,
    store_variant_in_chromadb,
)

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "generated" / "catalog_brands_research_checkpoint.json"


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(checkpoint: dict):
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":

    catalog = load_catalog()

    vp = get_collection("vet_products")
    researched_brands = {m["brand"] for m in vp.get(include=["metadatas"])["metadatas"]}

    todo = catalog_brands_needing_research(researched_brands)
    existing_docs = get_all_variant_documents()

    checkpoint = load_checkpoint()
    remaining = [b for b in todo if b not in checkpoint]

    print(f"{len(todo)} catalog brand(s) need research; {len(checkpoint)} already checkpointed; {len(remaining)} to go.")

    for brand in remaining:
        hints = research_hints_for_brand(catalog, brand)

        try:
            data = None
            # Transient network drops (server disconnects, timeouts) killed a
            # previous run mid-loop — retry those a couple of times before
            # giving up on the brand; GeminiUnavailableError (quota on every
            # key) still stops the whole run since retrying won't help.
            for attempt in range(3):
                try:
                    data = research_brand_with_gemini(brand, hints)
                    break
                except GeminiUnavailableError:
                    raise
                except Exception as e:
                    print(f"  attempt {attempt + 1} for '{brand}' failed ({type(e).__name__}: {e}) — retrying in 15s...")
                    time.sleep(15)
            if data is None:
                print(f"'{brand}': skipped after repeated network failures — re-run to retry it.")
                continue
        except GeminiUnavailableError as e:
            print(f"\n{e}")
            print(f"Stopping — {len(checkpoint)} brand(s) checkpointed; re-run to resume.")
            break

        variants = data.get("variants", [])
        updated = sum(store_variant_in_chromadb(brand, v, existing_docs) for v in variants)
        print(f"'{brand}': {updated} variant(s) added/updated ({len(variants)} researched).")

        checkpoint[brand] = data
        save_checkpoint(checkpoint)

    print(f"\nDone. vet_products now has {get_collection('vet_products').count()} variants.")
