# catalog_enricher.py — joins the Gemini-researched product knowledge
# (vet_products collection) onto the authoritative SKU catalog
# (Merged_Product_Variant_SKU.xlsx), producing the enriched catalog the
# structured pipeline reads:
#
#   * every catalog variant gets the best-matching researched variant's
#     name + metadata (species, pack color, region, ingredients) as extra
#     descriptor columns — market-facing text vets actually prescribe by;
#   * researched variants that match NO catalog row are appended as new
#     rows with the ID columns set to the "New Product" sentinel, so
#     market products absent from the catalog still exist in the
#     structured candidate space;
#   * admin pseudo-brands (reminders etc.) pass through flagged is_admin
#     so the pipeline can exclude them from matching.
#
# Matching is per catalog brand (exact researched brand + alias pool),
# variant-level, with the same hybrid pattern as sku_examples_builder:
#   sim >= EMBED_ACCEPT       -> accept the embedding match outright
#   sim <  EMBED_REJECT       -> unmatched (New Product candidate)
#   in between                -> DeepSeek verification against the brand's
#                                FULL catalog variant list (may say none);
#                                decisions checkpointed so re-runs are free
#
# Re-runnable and idempotent: run once now, and again after
# research_catalog_brands.py finishes to fill in the new brands' matches.
#
# Run:  python -u knowledge/catalog_enricher.py   (from backend/)

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chromadb_client import get_collection, get_embedding_model
from knowledge.catalog_common import CATALOG_BRAND_ALIASES, load_catalog

BACKEND_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_XLSX = BACKEND_ROOT / "data" / "generated" / "catalog_enriched.xlsx"
LLM_CHECKPOINT = BACKEND_ROOT / "data" / "generated" / "catalog_enricher_llm_checkpoint.json"

EMBED_ACCEPT = 0.80
EMBED_REJECT = 0.55
LLM_BATCH_SIZE = 15

NEW_PRODUCT_SENTINEL = "New Product"


# ─────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────

def load_researched_variants() -> dict:
    """researched brand -> list of variant dicts (name + prompt-relevant
    metadata), deduped by variant_name (regional duplicates share names)."""
    vp = get_collection("vet_products")
    data = vp.get(include=["metadatas"])

    by_brand = {}
    seen = set()
    for meta in data["metadatas"]:
        key = (meta.get("brand"), meta.get("variant_name"))
        if key in seen:
            continue
        seen.add(key)
        by_brand.setdefault(meta.get("brand"), []).append({
            "variant_name": meta.get("variant_name", ""),
            "species": meta.get("species", ""),
            "pack_color": meta.get("pack_color", ""),
            "region": meta.get("region", ""),
            "active_ingredients": meta.get("active_ingredients", ""),
            "pack_quantity": meta.get("pack_quantity", ""),
        })
    return by_brand


def build_brand_pools(catalog: pd.DataFrame, researched: dict) -> dict:
    """catalog Product Name -> list of researched brands to match against
    (exact case-insensitive name plus alias entries)."""
    researched_lower = {b.lower(): b for b in researched}

    pools = {}
    for name in catalog.loc[~catalog["is_admin"], "Product Name"].unique():
        pool = []
        if name.lower() in researched_lower:
            pool.append(researched_lower[name.lower()])
        alias = CATALOG_BRAND_ALIASES.get(name)
        if alias and alias in researched and alias not in pool:
            pool.append(alias)
        if pool:
            pools[name] = pool
    return pools


# ─────────────────────────────────────────────────────────────
# MATCHING
# ─────────────────────────────────────────────────────────────

def _load_llm_checkpoint() -> dict:
    if LLM_CHECKPOINT.exists():
        return json.loads(LLM_CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def _save_llm_checkpoint(cp: dict):
    LLM_CHECKPOINT.write_text(json.dumps(cp, ensure_ascii=False, indent=1), encoding="utf-8")


LLM_VERIFY_PROMPT = """
### ROLE
You match veterinary products. For each RESEARCHED product below, pick which
CATALOG VARIANT (if any) is the SAME product line: same brand family, same
species, same strength/weight band. Pack count differences do NOT matter
(catalog variants hold their pack options at a lower level). Weight-band or
species differences DO matter — "Cats 2-8kg" is NOT "Kitten under 2kg".

### CATALOG VARIANTS for brand __BRAND__
__CATALOG_LIST__

### RESEARCHED PRODUCTS TO MATCH
__RESEARCHED_LIST__

### OUTPUT (STRICT JSON ONLY)
{"matches": [{"index": <researched index>, "variant_id": <catalog VariantId or null if none matches>}]}
"""


def llm_verify_batch(brand: str, catalog_variants: pd.DataFrame, items: list) -> dict:
    """items: [(researched_index, variant_dict)]. Returns
    {researched_index: catalog VariantId or None}."""
    from pipeline.shared.resources import deepseek_json

    catalog_list = "\n".join(
        f'VariantId {int(r.VariantId)}: {r.VariantDescription}'
        for r in catalog_variants.itertuples()
    )
    researched_list = "\n".join(
        f'{i}: {v["variant_name"]} [species: {v.get("species") or "?"}]'
        for i, v in items
    )

    prompt = (
        LLM_VERIFY_PROMPT
        .replace("__BRAND__", brand)
        .replace("__CATALOG_LIST__", catalog_list)
        .replace("__RESEARCHED_LIST__", researched_list)
    )

    valid_ids = set(catalog_variants["VariantId"].astype(int))
    out = {}
    try:
        result = deepseek_json(prompt)
        for m in result.get("matches", []):
            idx = m.get("index")
            vid = m.get("variant_id")
            if idx is None:
                continue
            if isinstance(vid, (int, float)) and int(vid) in valid_ids:
                out[int(idx)] = int(vid)
            else:
                out[int(idx)] = None
    except Exception as e:
        print(f"  LLM batch failed for {brand}: {e} — leaving batch unmatched this run.")
    return out


def match_brand(brand: str, pool_brands: list, catalog: pd.DataFrame,
                researched: dict, model, llm_cp: dict):
    """Returns (variant_matches, matched_researched_keys) where
    variant_matches: {catalog VariantId: (researched variant dict, sim, method)}."""
    cat_variants = (
        catalog.loc[(catalog["Product Name"] == brand) & (~catalog["is_admin"]),
                    ["VariantId", "VariantDescription"]]
        .drop_duplicates("VariantId")
        .reset_index(drop=True)
    )
    res_variants = [v for b in pool_brands for v in researched.get(b, [])]

    if cat_variants.empty or not res_variants:
        return {}, set()

    cat_embs = model.encode(cat_variants["VariantDescription"].tolist(), normalize_embeddings=True)
    res_embs = model.encode([v["variant_name"] for v in res_variants], normalize_embeddings=True)
    sims = np.asarray(res_embs) @ np.asarray(cat_embs).T

    matches = {}          # VariantId -> (variant dict, sim, method)
    matched_keys = set()  # (pool brand irrelevant) variant_name matched somewhere
    llm_todo = []

    for i, rv in enumerate(res_variants):
        j = int(sims[i].argmax())
        best = float(sims[i][j])
        vid = int(cat_variants.iloc[j]["VariantId"])

        if best >= EMBED_ACCEPT:
            _record(matches, vid, rv, best, "embedding")
            matched_keys.add(rv["variant_name"])
        elif best >= EMBED_REJECT:
            cp_key = f"{brand}||{rv['variant_name']}"
            if cp_key in llm_cp:
                decided = llm_cp[cp_key]
                if decided is not None and int(decided) in set(cat_variants["VariantId"].astype(int)):
                    _record(matches, int(decided), rv, best, "llm")
                    matched_keys.add(rv["variant_name"])
            else:
                llm_todo.append((i, rv, best))
        # < EMBED_REJECT: unmatched

    for start in range(0, len(llm_todo), LLM_BATCH_SIZE):
        batch = llm_todo[start:start + LLM_BATCH_SIZE]
        decisions = llm_verify_batch(brand, cat_variants, [(i, rv) for i, rv, _ in batch])

        for i, rv, best in batch:
            if i not in decisions:
                continue  # LLM call failed — no checkpoint write, retried next run
            vid = decisions[i]
            llm_cp[f"{brand}||{rv['variant_name']}"] = vid
            if vid is not None:
                _record(matches, vid, rv, best, "llm")
                matched_keys.add(rv["variant_name"])
        _save_llm_checkpoint(llm_cp)

    return matches, matched_keys


def _record(matches: dict, vid: int, rv: dict, sim: float, method: str):
    """Keep the highest-similarity researched variant per catalog variant."""
    if vid not in matches or sim > matches[vid][1]:
        matches[vid] = (rv, sim, method)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def build_enriched_catalog() -> pd.DataFrame:
    catalog = load_catalog()
    researched = load_researched_variants()
    pools = build_brand_pools(catalog, researched)
    model = get_embedding_model()
    llm_cp = _load_llm_checkpoint()

    print(f"Catalog: {len(catalog)} SKU rows, {catalog['Product Name'].nunique()} brands "
          f"({catalog['is_admin'].sum()} admin rows).")
    print(f"Researched: {sum(len(v) for v in researched.values())} variants across {len(researched)} brands.")
    print(f"Brand pools resolvable: {len(pools)}.\n")

    all_matches = {}       # (Product Name, VariantId) -> (variant dict, sim, method)
    matched_names = set()  # researched variant_names matched to any catalog row

    for brand, pool in sorted(pools.items()):
        matches, keys = match_brand(brand, pool, catalog, researched, model, llm_cp)
        for vid, hit in matches.items():
            all_matches[(brand, vid)] = hit
        matched_names |= keys

    # Descriptor columns onto every SKU row of a matched variant
    def descriptor(row, field, default=""):
        hit = all_matches.get((row["Product Name"], int(row["VariantId"])))
        if not hit:
            return default
        rv, sim, method = hit
        return {
            "name": rv["variant_name"], "species": rv["species"],
            "color": rv["pack_color"], "region": rv["region"],
            "ingredients": rv["active_ingredients"],
            "sim": round(sim, 3), "method": method,
        }[field]

    enriched = catalog.copy()
    enriched["GeminiVariantName"] = enriched.apply(lambda r: descriptor(r, "name"), axis=1)
    enriched["GeminiSpecies"] = enriched.apply(lambda r: descriptor(r, "species"), axis=1)
    enriched["GeminiPackColor"] = enriched.apply(lambda r: descriptor(r, "color"), axis=1)
    enriched["GeminiRegion"] = enriched.apply(lambda r: descriptor(r, "region"), axis=1)
    enriched["GeminiIngredients"] = enriched.apply(lambda r: descriptor(r, "ingredients"), axis=1)
    enriched["GeminiMatchSim"] = enriched.apply(lambda r: descriptor(r, "sim", default=None), axis=1)
    enriched["GeminiMatchMethod"] = enriched.apply(lambda r: descriptor(r, "method", default="none"), axis=1)
    enriched["is_new_product"] = False

    # Researched variants with no catalog match anywhere -> New Product rows
    new_rows = []
    for brand, variants in sorted(researched.items()):
        for rv in variants:
            if rv["variant_name"] in matched_names:
                continue
            pq = rv.get("pack_quantity", "")
            pack_units = int(pq) if str(pq).isdigit() else 0
            new_rows.append({
                "ProductId": NEW_PRODUCT_SENTINEL,
                "Product Name": brand,
                "Supplier": "",
                "VariantId": NEW_PRODUCT_SENTINEL,
                "VariantDescription": rv["variant_name"],
                "SKU ID": NEW_PRODUCT_SENTINEL,
                "PackUnits": pack_units,
                "MlVolume": 0,
                "SKU Description": rv["variant_name"],
                "is_admin": False,
                "GeminiVariantName": rv["variant_name"],
                "GeminiSpecies": rv["species"],
                "GeminiPackColor": rv["pack_color"],
                "GeminiRegion": rv["region"],
                "GeminiIngredients": rv["active_ingredients"],
                "GeminiMatchSim": None,
                "GeminiMatchMethod": "new_product",
                "is_new_product": True,
            })

    enriched = pd.concat([enriched, pd.DataFrame(new_rows)], ignore_index=True)

    matched_variant_count = len(all_matches)
    total_variants = catalog.loc[~catalog["is_admin"]].groupby(["Product Name", "VariantId"]).ngroups
    print(f"\nMatched {matched_variant_count}/{total_variants} catalog variants to a researched descriptor.")
    print(f"Appended {len(new_rows)} New Product rows (researched, absent from catalog).")

    return enriched


if __name__ == "__main__":
    enriched = build_enriched_catalog()
    OUTPUT_XLSX.parent.mkdir(exist_ok=True)
    enriched.to_excel(OUTPUT_XLSX, index=False)
    print(f"\nSaved {len(enriched)} rows -> {OUTPUT_XLSX}")
