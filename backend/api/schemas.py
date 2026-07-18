# schemas.py — request/response models for the FastAPI backend.

from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    enable_live_fallback: bool = False
    enable_llm_judge: bool = False
    # "raw" = the original researched-text pipeline (Labelled.csv only);
    # "structured" = the catalog-backed pipeline (Merged_Product_Variant_SKU
    # + Labelled.csv priors/phrasings). Structured ignores the enable_*
    # flags above.
    mode: str = "raw"


class ProductMatch(BaseModel):
    """WHAT the product is — deliberately carries no ids. One real product
    often has several id triplets (duplicate/legacy catalog listings), so
    ids live in their own list rather than being forced into a single
    arbitrary pick here."""
    brand: str = ""
    products: str = ""
    full_product_name: str = ""
    region: str = ""
    species: str = ""
    animal_size: str = ""
    pack_size: str = ""
    pack_quantity: str = ""
    animal_age: str = ""
    confidence: str = ""
    reason: str = ""


class IdCombination(BaseModel):
    """One real (ProductId, VariantId, SkuId) the matched product maps to,
    with the catalog text that justifies it being listed."""
    product_id: str = ""
    product_variant_id: str = ""
    sku_id: str = ""
    variant_description: str = ""
    catalog_description: str = ""
    pack_units: int = 0
    ml_volume: float = 0
    occurrences: int = 0


class UsageStats(BaseModel):
    elapsed_seconds: float = 0.0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Conservative (cache-miss-rate) DeepSeek pricing estimate — see
    # pipeline/shared/resources.py's usage-tracking section for caveats.
    estimated_cost_usd: float = 0.0


class QueryResponse(BaseModel):
    product_match: ProductMatch
    id_combinations: List[IdCombination] = []
    id_combinations_reason: str = ""
    usage: UsageStats = UsageStats()
