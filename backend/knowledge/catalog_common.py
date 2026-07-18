# catalog_common.py — shared constants + loading for the structured-catalog
# work (Merged_Product_Variant_SKU.xlsx): the brand-alias map between
# catalog "Product Name"s and already-researched vet_products brands, the
# admin pseudo-brand exclusion list, and the canonical catalog loader.
# Used by research_catalog_brands.py (which brands still need Gemini
# research) and catalog_enricher.py (which researched pool to match each
# catalog brand's variants against).

from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parent.parent
CATALOG_XLSX = BACKEND_ROOT / "data" / "raw" / "Merged_Product_Variant_SKU.xlsx"

# Catalog "Product Name" -> researched vet_products brand whose variant pool
# covers it. Two kinds of entries: naming drift for the same product family
# ("Vectra3D" vs "Vectra 3D", Japanese script vs romanized), and sub-lines
# the researcher already covered while researching the parent brand
# ("Activyl Tick Plus" variants were researched under "Activyl"). These
# catalog brands are NOT re-researched, and their enrichment matching pools
# from the aliased researched brand.
CATALOG_BRAND_ALIASES = {
    "AMMO Paste": "Ammo",
    "Acclaim": "Vetkem Acclaim",
    "Activyl Tick Plus": "Activyl",
    "Advantage Extra": "Advantage",
    "Amflee Combo": "Amflee",
    "Big 5 Protection Pack": "Big 5",
    "Bimectin Plus": "Bimectin",
    "F10 WOUND SPRAY + INSECTICIDE": "F10",
    "Felex Plus": "Felex",
    "Felisecto plus": "Felisecto",
    "Fleascreen Combo": "Fleascreen",
    "Kiltix Collar": "Kiltix",
    "My Free Guard": "マイフリーガード",
    "Panamectin": "パナメクチン",
    "Pyratape": "Pyratape P",
    "RIP Flea": "Rip Fleas",
    "Ranvet Allwormer": "Ranvet",
    "Selapro Selamectin": "Selapro",
    "Selehold Spot On": "Selehold",
    "Vectra3D": "Vectra 3D",
    "Veloxa Cat": "Veloxa",
    "Veloxa Dog": "Veloxa",
    "Wagg & Purr All Wormer": "Wagg & Purr",
    "Wagg & Purr Imidaclop": "Wagg & Purr",
    "Wagg & Purr Selamectin": "Wagg & Purr",
    "Wagg & Purr Tapewormer": "Wagg & Purr",
}

# Catalog rows that are practice-management bookkeeping, not products a vet
# would prescribe by name: treatment reminders, an explicit invalid-entry
# sentinel, and bare generic category words ("Flea", "Wormer" — one of
# whose SKUs is literally "Kotuntersuchung", a fecal exam). Never Gemini-
# researched, and flagged in the enriched catalog so the structured
# pipeline never offers them as match candidates.
ADMIN_PSEUDO_BRANDS = {
    "Flea",
    "Wormer",
    "Invalid Parasiticide",
    "IRecall Reminder Activator",
    "Reminder - Flea",
    "Reminder - Heartworm",
    "Reminder - Tick",
    "Reminder - Worming",
    "STOP - All Groups Reminder",
    "STOP - Flea Reminder",
    "STOP - Heartworm Reminder",
    "STOP - Tick Reminder",
    "STOP - Worming Reminder",
    "STOP REMINDER",
}


def load_catalog() -> pd.DataFrame:
    """The authoritative SKU catalog, names stripped, one row per real
    (ProductId, VariantId, SKU ID) triplet."""
    df = pd.read_excel(CATALOG_XLSX)

    for col in ("Product Name", "Supplier", "VariantDescription", "SKU Description"):
        df[col] = df[col].astype(str).str.strip()

    df["is_admin"] = df["Product Name"].isin(ADMIN_PSEUDO_BRANDS)
    return df


def catalog_brands_needing_research(researched_brands: set) -> list:
    """Catalog Product Names with no researched pool: not an exact
    (case-insensitive) researched brand, not alias-covered, not admin."""
    catalog = load_catalog()
    researched_lower = {b.lower() for b in researched_brands}

    needing = []
    for name in sorted(catalog["Product Name"].unique()):
        if name in ADMIN_PSEUDO_BRANDS or name in CATALOG_BRAND_ALIASES:
            continue
        if name.lower() in researched_lower:
            continue
        needing.append(name)
    return needing


def research_hints_for_brand(catalog: pd.DataFrame, brand: str, cap: int = 10) -> str:
    """The brand's own catalog VariantDescriptions, joined as the
    known-product-lines hint for Gemini research — real market names beat
    any generic hint."""
    descriptions = (
        catalog.loc[catalog["Product Name"] == brand, "VariantDescription"]
        .drop_duplicates()
        .head(cap)
        .tolist()
    )
    return "; ".join(descriptions)
