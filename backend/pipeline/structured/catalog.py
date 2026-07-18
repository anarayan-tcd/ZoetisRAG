# structured/catalog.py — in-memory access layer over the enriched SKU
# catalog for the structured pipeline. Loaded once per process:
#
#   * brand -> its catalog variants, each with description, Gemini
#     descriptor columns, and the variant's real SKU options
#     (PackUnits / MlVolume / SKU Description);
#   * historical occurrence priors from Labelled.csv, per variant within
#     a brand and per SKU within a variant (feeds the same
#     dominant/tied/low tiering the raw pipeline uses);
#   * a researched-brand -> catalog-brand resolution map (exact names
#     plus the alias table), since Phase 1 returns researched brand
#     names ("Vectra 3D") that may differ from catalog spelling
#     ("Vectra3D");
#   * "New Product" pseudo-variants (researched, absent from catalog) so
#     market products without catalog IDs are still offered as matches.
#
# Falls back to the raw catalog xlsx when the enriched build doesn't
# exist yet — structured mode then simply runs without Gemini descriptors.

from pathlib import Path

import pandas as pd

from knowledge.catalog_common import CATALOG_BRAND_ALIASES, load_catalog
from ..shared.query_utils import strip_pack_phrase as _strip_pack_phrase

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
ENRICHED_XLSX = BACKEND_ROOT / "data" / "generated" / "catalog_enriched.xlsx"
LABELLED_CSV = BACKEND_ROOT / "data" / "raw" / "Labelled.csv"

NEW_PRODUCT_SENTINEL = "New Product"

# A catalog VariantId spans every pack size of one product line (e.g.
# Stronghold 45mg Cat's SKUs 156/157/1319 are its 6/3/1-pipette packs) —
# but the researched Gemini market name attached to it was often written
# for ONE specific pack ("...45mg) 6 Pipette Pack"), because knowledge_
# agent2 originally researched pack sizes as separately-named variants.
# catalog_enricher.py picks whichever one embedding-matched best, so
# ~48% of GeminiVariantName values carry a pack count that's true of only
# ONE of the variant's several real SKUs. Left in place, Phase 2 copies it
# verbatim into full_product_name while Phase 3 independently picks
# whichever SKU the historical/query evidence actually supports — the two
# stages disagree whenever the picked SKU isn't the one the name happened
# to describe (e.g. name says "6 Pipette Pack", the dominant real SKU is
# the 3-pack). Stripped here at load time so full_product_name starts
# pack-agnostic; the SKU stage re-appends the CORRECT pack count from the
# actually-chosen SKU's real PackUnits (see phase3_structured.py's
# apply_resolved_pack call).

_catalog_df = None
_brand_variants = None       # catalog brand -> [variant dict]
_brand_new_products = None   # catalog/researched brand -> [pseudo-variant dict]
_brand_resolution = None     # lowercased researched/catalog name -> [catalog brands]
_variant_priors = None       # (pid) -> {vid: count}
_sku_priors = None           # (pid, vid) -> {sid: count}


def _load_enriched() -> pd.DataFrame:
    if ENRICHED_XLSX.exists():
        df = pd.read_excel(ENRICHED_XLSX)
    else:
        print(f"[structured] {ENRICHED_XLSX.name} not built yet — using raw catalog without descriptors.")
        df = load_catalog()
        for col in ("GeminiVariantName", "GeminiSpecies", "GeminiPackColor",
                    "GeminiRegion", "GeminiIngredients"):
            df[col] = ""
        df["is_new_product"] = False

    for col in ("GeminiVariantName", "GeminiSpecies", "GeminiPackColor",
                "GeminiRegion", "GeminiIngredients"):
        df[col] = df[col].fillna("")
    return df


def _load_priors():
    """Occurrence counts from Labelled.csv: how often each variant (within
    its ProductId) and each SKU (within its variant) was actually
    prescribed. Nulls collapse to 0-sentinels and are ignored here — the
    catalog is the authority on what exists; labelled data only supplies
    the commonness prior."""
    lab = pd.read_csv(LABELLED_CSV, usecols=["ProductId", "ProductVariantId", "SkuId"])
    lab = lab.dropna(subset=["ProductId"])
    lab["pid"] = lab["ProductId"].astype(int)
    lab["vid"] = lab["ProductVariantId"].fillna(0).astype(int)
    lab["sid"] = lab["SkuId"].fillna(0).astype(int)

    variant_priors = {}
    for (pid, vid), n in lab.groupby(["pid", "vid"]).size().items():
        if vid:
            variant_priors.setdefault(pid, {})[vid] = int(n)

    sku_priors = {}
    for (pid, vid, sid), n in lab.groupby(["pid", "vid", "sid"]).size().items():
        if vid and sid:
            sku_priors.setdefault((pid, vid), {})[sid] = int(n)

    return variant_priors, sku_priors


def _build():
    global _catalog_df, _brand_variants, _brand_new_products, _brand_resolution
    global _variant_priors, _sku_priors

    df = _load_enriched()
    _catalog_df = df
    _variant_priors, _sku_priors = _load_priors()

    real = df[(~df["is_admin"]) & (~df["is_new_product"])]

    _brand_variants = {}
    for (brand, pid, vid), sub in real.groupby(["Product Name", "ProductId", "VariantId"]):
        first = sub.iloc[0]
        skus = [
            {
                "sku_id": int(r["SKU ID"]),
                "description": r["SKU Description"],
                "pack_units": int(r["PackUnits"]),
                "ml_volume": float(r["MlVolume"]),
                "occurrences": _sku_priors.get((int(pid), int(vid)), {}).get(int(r["SKU ID"]), 0),
            }
            for _, r in sub.iterrows()
        ]
        _brand_variants.setdefault(brand, []).append({
            "product_id": int(pid),
            "variant_id": int(vid),
            "description": first["VariantDescription"],
            # Stripped (unlike the New Product branch below): this variant
            # has real SKUs of its own, each possibly a different pack size
            # than whatever pack the researched name happened to describe.
            "gemini_name": _strip_pack_phrase(first["GeminiVariantName"]),
            "gemini_species": first["GeminiSpecies"],
            "gemini_color": first["GeminiPackColor"],
            "gemini_region": first["GeminiRegion"],
            "occurrences": _variant_priors.get(int(pid), {}).get(int(vid), 0),
            "skus": sorted(skus, key=lambda s: -s["occurrences"]),
        })

    _brand_new_products = {}
    for _, r in df[df["is_new_product"] == True].iterrows():  # noqa: E712
        _brand_new_products.setdefault(r["Product Name"], []).append({
            "product_id": "", "variant_id": "",
            "description": r["VariantDescription"],
            # NOT stripped — a New Product has no catalog SKU list of its
            # own to reconcile against later, so whatever pack count the
            # research described is the only one that exists here.
            "gemini_name": r["GeminiVariantName"],
            "gemini_species": r["GeminiSpecies"],
            "gemini_color": r["GeminiPackColor"],
            "gemini_region": r["GeminiRegion"],
            "occurrences": 0,
            "skus": [],
            "is_new_product": True,
        })

    # researched name and catalog name both resolve to catalog brand(s)
    _brand_resolution = {}
    for brand in real["Product Name"].unique():
        _brand_resolution.setdefault(brand.lower(), []).append(brand)
    for cat_name, researched_name in CATALOG_BRAND_ALIASES.items():
        entry = _brand_resolution.setdefault(researched_name.lower(), [])
        if cat_name in (_brand_variants or {}) and cat_name not in entry:
            entry.append(cat_name)
    # New-Product-only brands (researched, zero catalog rows) still resolve
    for brand in _brand_new_products:
        _brand_resolution.setdefault(brand.lower(), [])
        if brand not in _brand_resolution[brand.lower()]:
            _brand_resolution[brand.lower()].append(brand)


def _ensure_loaded():
    if _brand_variants is None:
        _build()


def resolve_catalog_brands(brand: str) -> list:
    """Catalog brand name(s) for a Phase 1 brand answer. Usually one;
    alias families ("Veloxa" -> Veloxa Cat + Veloxa Dog) return several."""
    _ensure_loaded()
    return _brand_resolution.get((brand or "").strip().lower(), [])


def variants_for_brands(catalog_brands: list) -> list:
    """All real catalog variants plus New Product pseudo-variants for the
    given catalog brand(s) — the complete Phase 2 candidate space."""
    _ensure_loaded()
    out = []
    for b in catalog_brands:
        out.extend(_brand_variants.get(b, []))
        out.extend(_brand_new_products.get(b, []))
    return out


def variant_by_id(catalog_brands: list, variant_id) -> dict:
    _ensure_loaded()
    for v in variants_for_brands(catalog_brands):
        if str(v["variant_id"]) == str(variant_id):
            return v
    return {}


def variants_by_ids(catalog_brands: list, variant_ids: list) -> list:
    """The catalog variants for a set of ids, in the order given. Silently
    skips ids that don't resolve — callers validate membership separately."""
    wanted = [str(v) for v in variant_ids]
    by_id = {str(v["variant_id"]): v for v in variants_for_brands(catalog_brands) if v["variant_id"] != ""}
    return [by_id[v] for v in wanted if v in by_id]


def pack_options_for_variants(variants: list) -> list:
    """Distinct pack sizes across a set of equivalent variants, each with
    the historical weight of every SkuId that offers it.

    The unit of choice here is the PACK, not the SkuId, because several
    SkuIds routinely describe the very same physical pack (Milbemax
    VariantId 86 has six SkuIds at PackUnits 4, all with byte-identical
    description text — nothing in the catalog distinguishes them). Asking
    an LLM to choose between those is asking it to invent a distinction
    that doesn't exist; asking it to choose a PACK SIZE is a real question
    with a real answer. Enumeration then reports every SkuId that pack
    maps to.

    Returns [{"pack_units", "ml_volume", "occurrences", "sku_count",
    "descriptions"}], sorted by occurrences descending."""
    groups = {}

    for v in variants:
        for s in v.get("skus", []):
            key = (int(s["pack_units"] or 0), float(s["ml_volume"] or 0))
            g = groups.setdefault(key, {
                "pack_units": key[0],
                "ml_volume": key[1],
                "occurrences": 0,
                "sku_count": 0,
                "descriptions": [],
            })
            g["occurrences"] += int(s.get("occurrences") or 0)
            g["sku_count"] += 1
            desc = s.get("description") or ""
            if desc and desc not in g["descriptions"] and len(g["descriptions"]) < 3:
                g["descriptions"].append(desc)

    return sorted(groups.values(), key=lambda g: -g["occurrences"])


def enumerate_id_combinations(variants: list, pack_units, ml_volume) -> list:
    """Every real (ProductId, VariantId, SkuId) across the given equivalent
    variants that offers the chosen pack — the whole point of the
    ID-combinations output: when the catalog genuinely cannot distinguish
    several rows (duplicate/legacy codes for one product), report all of
    them rather than pick one arbitrarily.

    pack_units/ml_volume of None means "pack not resolved" -> every SkuId
    of every given variant is returned. Otherwise only SkuIds matching that
    exact pack are, so a stated "6 pack" never drags a 1-pack code in.

    Sorted by historical occurrences descending, so the most-prescribed
    combination reads first without being presented as the only answer."""
    combos = []

    for v in variants:
        for s in v.get("skus", []):
            if pack_units is not None and int(s["pack_units"] or 0) != int(pack_units):
                continue
            if ml_volume is not None and float(s["ml_volume"] or 0) != float(ml_volume):
                continue

            combos.append({
                "product_id": str(v["product_id"]),
                "product_variant_id": str(v["variant_id"]),
                "sku_id": str(s["sku_id"]),
                "variant_description": v.get("description", ""),
                "catalog_description": s.get("description", ""),
                "pack_units": int(s["pack_units"] or 0),
                "ml_volume": float(s["ml_volume"] or 0),
                "occurrences": int(s.get("occurrences") or 0),
            })

    return sorted(combos, key=lambda c: -c["occurrences"])
