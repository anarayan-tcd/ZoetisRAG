// types.ts — mirrors backend/schemas.py

// WHAT the product is — carries no ids. One real product often maps to
// several id triplets (duplicate/legacy catalog listings), so ids live in
// their own list rather than being forced into one arbitrary pick here.
export interface ProductMatch {
  brand: string;
  products: string;
  full_product_name: string;
  region: string;
  species: string;
  animal_size: string;
  pack_size: string;
  pack_quantity: string;
  animal_age: string;
  confidence: string;
  reason: string;
}

export interface IdCombination {
  product_id: string;
  product_variant_id: string;
  sku_id: string;
  variant_description: string;
  catalog_description: string;
  pack_units: number;
  ml_volume: number;
  occurrences: number;
}

export interface UsageStats {
  elapsed_seconds: number;
  llm_calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
}

export interface QueryResponse {
  product_match: ProductMatch;
  id_combinations: IdCombination[];
  id_combinations_reason: string;
  usage: UsageStats;
}

export interface BatchStats {
  total_rows: number;
  unique_queries: number;
  llm_calls: number;
  cache_hits: number;
  elapsed: number;
  avg_seconds_per_resolved_row: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  billed_llm_calls: number;
  estimated_cost_usd: number;
}

export interface BatchJobStatus {
  status: "pending" | "running" | "done" | "error" | "cancelled";
  completed: number;
  total: number;
  total_rows: number;
  rows: Record<string, string>[] | null;
  stats: BatchStats | null;
  error: string | null;
}
