// types.ts — mirrors backend/schemas.py

export interface ProductMatch {
  brand: string;
  products: string;
  full_product_name: string;
  product_id: string;
  sku_id: string;
  product_variant_id: string;
  region: string;
  species: string;
  animal_size: string;
  pack_size: string;
  pack_quantity: string;
  animal_age: string;
  confidence: string;
  reason: string;
}

export interface QueryResponse {
  primary_match: ProductMatch;
  alternative_matches: ProductMatch[];
}

export interface BatchStats {
  total_rows: number;
  unique_queries: number;
  llm_calls: number;
  cache_hits: number;
  elapsed: number;
}

export interface BatchJobStatus {
  status: "pending" | "running" | "done" | "error";
  completed: number;
  total: number;
  total_rows: number;
  rows: Record<string, string>[] | null;
  stats: BatchStats | null;
  error: string | null;
}
