# schemas.py — request/response models for the FastAPI backend.

from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    enable_live_fallback: bool = False
    enable_llm_judge: bool = False


class ProductMatch(BaseModel):
    brand: str = ""
    products: str = ""
    full_product_name: str = ""
    product_id: str = ""
    sku_id: str = ""
    product_variant_id: str = ""
    region: str = ""
    species: str = ""
    animal_size: str = ""
    pack_size: str = ""
    pack_quantity: str = ""
    animal_age: str = ""
    confidence: str = ""
    reason: str = ""


class QueryResponse(BaseModel):
    primary_match: ProductMatch
    alternative_matches: List[ProductMatch] = []
