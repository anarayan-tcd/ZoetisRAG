# backfill_pack_color.py — one-shot driver: re-researches every brand
# whose vet_products variants are missing pack_color (or only partially
# have it), via Gemini, so the color capture added to the research prompt
# partway through this project's build gets backfilled onto brands
# researched before that point.
#
# Safe to re-run: research_brand_with_gemini + store_variant_in_chromadb
# upsert on a DETERMINISTIC id keyed to (brand, variant_name, region), so
# re-researching a brand updates its existing variants in place (adding
# color and refreshing anything else that changed) rather than duplicating
# them — the same mechanism already used for the earlier 8-brand and
# 49-brand re-research passes this session.
#
# Checkpointed per brand, same retry-hardened pattern as
# research_catalog_brands.py — a transient network drop mid-run costs at
# most the one in-flight brand, and a restart resumes from where it left
# off instead of re-spending API credits on brands already done.
#
# Run:  python -u knowledge/backfill_pack_color.py   (from backend/)

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chromadb_client import get_all_variant_documents, get_collection
from knowledge.catalog_common import load_catalog, research_hints_for_brand
from knowledge.knowledge_agent2 import (
    GeminiUnavailableError,
    research_brand_with_gemini,
    store_variant_in_chromadb,
)

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "generated" / "pack_color_backfill_checkpoint.json"


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(checkpoint: dict):
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=1), encoding="utf-8")


def brands_missing_color() -> list:
    """Brands with zero or partial pack_color coverage across their
    researched variants, sorted by variant count descending — the highest-
    value (most-prescribed) brands get backfilled first, so a run that's
    interrupted still delivers the most useful color coverage possible."""
    vp = get_collection("vet_products")
    metas = vp.get(include=["metadatas"])["metadatas"]

    counts = {}
    for m in metas:
        brand = m.get("brand")
        if not brand:
            continue
        total, colored = counts.setdefault(brand, [0, 0])
        counts[brand][0] += 1
        if (m.get("pack_color") or "").strip():
            counts[brand][1] += 1

    missing = [(b, total) for b, (total, colored) in counts.items() if colored < total]
    missing.sort(key=lambda x: -x[1])
    return [b for b, _ in missing]


if __name__ == "__main__":

    todo_all = brands_missing_color()
    print(f"{len(todo_all)} brand(s) have missing/partial pack_color coverage.")

    catalog = load_catalog()
    existing_docs = get_all_variant_documents()

    checkpoint = load_checkpoint()
    remaining = [b for b in todo_all if b not in checkpoint]
    print(f"{len(checkpoint)} already checkpointed; {len(remaining)} to go.\n")

    catalog_brands = set(catalog["Product Name"])

    for brand in remaining:
        hints = research_hints_for_brand(catalog, brand) if brand in catalog_brands else ""

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
        colored = sum(1 for v in variants if str(v.get("pack_color", "")).strip())
        updated = sum(store_variant_in_chromadb(brand, v, existing_docs) for v in variants)

        print(f"'{brand}': {updated}/{len(variants)} variant(s) updated, {colored}/{len(variants)} have color.")

        checkpoint[brand] = {"variants_researched": len(variants), "variants_colored": colored}
        save_checkpoint(checkpoint)

    print(f"\nDone. vet_products now has {get_collection('vet_products').count()} variants.")
