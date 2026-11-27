# backfill_missing_fields.py — one-shot driver: re-researches every brand
# with at least one vet_products variant missing ANY of species, region,
# active_ingredients, or pack_color, via Gemini.
#
# Generalizes backfill_pack_color.py (which targeted pack_color alone) after
# diagnosing a real gap in the structured pipeline's region field: some
# catalog rows link to a researched vet_products variant that has some
# fields populated (e.g. species, ingredients) but not others (e.g. region)
# — the SAME research pass simply didn't capture every field for every
# variant. A single completeness check across all four fields, rather than
# separate single-field backfills, fixes every such gap in one research
# pass per brand instead of re-researching the same brand repeatedly as
# each field's gap is noticed in turn.
#
# Safe to re-run: research_brand_with_gemini + store_variant_in_chromadb
# upsert on a DETERMINISTIC id keyed to (brand, variant_name, region), so
# re-researching a brand updates its existing variants in place rather than
# duplicating them.
#
# Checkpointed per brand, same retry-hardened pattern as
# research_catalog_brands.py / backfill_pack_color.py — a transient network
# drop mid-run costs at most the one in-flight brand, and a restart resumes
# instead of re-spending API credits on brands already done.
#
# Uses its OWN checkpoint file, separate from pack_color_backfill_
# checkpoint.json: a brand already checkpointed there for color alone may
# still be missing species/region/ingredients under this broader check, and
# vice versa, so the two progress states must not be conflated.
#
# Run:  python -u knowledge/backfill_missing_fields.py   (from backend/)

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

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "generated" / "missing_fields_backfill_checkpoint.json"

# The optional research fields worth completing. variant_name/brand are
# always present (they're how a researched record is identified in the
# first place), so they're not candidates for "missing" here.
TRACKED_FIELDS = ("species", "region", "active_ingredients", "pack_color")


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(checkpoint: dict):
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=1), encoding="utf-8")


def brands_missing_fields() -> list:
    """Brands with at least one variant missing at least one of
    TRACKED_FIELDS, sorted by variant count descending — the highest-value
    (most-prescribed) brands get backfilled first, so a run that's
    interrupted still delivers the most useful coverage possible."""
    vp = get_collection("vet_products")
    metas = vp.get(include=["metadatas"])["metadatas"]

    counts = {}
    for m in metas:
        brand = m.get("brand")
        if not brand:
            continue
        total, complete = counts.setdefault(brand, [0, 0])
        counts[brand][0] += 1
        if all((m.get(f) or "").strip() for f in TRACKED_FIELDS):
            counts[brand][1] += 1

    missing = [(b, total) for b, (total, complete) in counts.items() if complete < total]
    missing.sort(key=lambda x: -x[1])
    return [b for b, _ in missing]


if __name__ == "__main__":

    todo_all = brands_missing_fields()
    print(f"{len(todo_all)} brand(s) have at least one variant missing species/region/ingredients/pack_color.")

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
        field_counts = {f: sum(1 for v in variants if str(v.get(f, "")).strip()) for f in TRACKED_FIELDS}
        updated = sum(store_variant_in_chromadb(brand, v, existing_docs) for v in variants)

        summary = ", ".join(f"{f}={field_counts[f]}/{len(variants)}" for f in TRACKED_FIELDS)
        print(f"'{brand}': {updated}/{len(variants)} variant(s) updated. {summary}")

        checkpoint[brand] = {"variants_researched": len(variants), **field_counts}
        save_checkpoint(checkpoint)

    print(f"\nDone. vet_products now has {get_collection('vet_products').count()} variants.")
