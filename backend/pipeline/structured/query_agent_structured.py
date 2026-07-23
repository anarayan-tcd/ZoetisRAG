# structured/query_agent_structured.py — orchestrator for the structured
# (catalog-backed) matching mode. Mirrors pipeline/query_agent.py's shape
# and result schema so batch.py/main.py can dispatch to either
# interchangeably, but the internals are much shorter: brand resolution is
# shared with the raw pipeline (same vector search + fuzzy recovery +
# DeepSeek Phase 1), then the catalog supplies the COMPLETE candidate space
# with authoritative ids — one variant call, one (often skipped) SKU call.
#
# enable_live_fallback / enable_llm_judge are accepted for signature
# compatibility but intentionally ignored: both mechanisms exist to patch
# gaps in the raw pipeline's researched-text knowledge base, which the
# authoritative catalog doesn't have.

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..shared.resources import (
    MAX_WORKERS, reset_usage_tracking, get_usage_totals, snapshot_global_usage,
    estimate_cost_usd, usage_delta,
)
from ..shared.phase1_brand import search_brands, search_brands_batch, ask_deepseek_phase1, augment_with_fuzzy_brand
from ..shared.query_utils import clean_query, unknown_result as _unknown_result
from .catalog import (
    resolve_catalog_brands, variants_for_brands, variants_by_ids,
    pack_options_for_variants, enumerate_id_combinations,
)
from .phase2_structured import ask_deepseek_phase2_structured, NEW_PRODUCT_SENTINEL
from .phase3_structured import resolve_pack_structured, apply_pack_to_match

# Separate cache from the raw pipeline's — the same query legitimately
# produces different results per mode.
_result_cache = {}


def cache_size() -> int:
    return len(_result_cache)


def clear_cache() -> None:
    _result_cache.clear()


def _resolve_structured(cleaned_query: str, brand_candidates: list) -> dict:
    brand_candidates = augment_with_fuzzy_brand(cleaned_query, brand_candidates)

    phase1_result = ask_deepseek_phase1(cleaned_query, brand_candidates) if brand_candidates else None

    if not phase1_result or not phase1_result.get("brand"):
        return _unknown_result("No matching brand found.")

    brand = phase1_result["brand"]

    catalog_brands = resolve_catalog_brands(brand)

    if not catalog_brands:
        return _unknown_result(
            f"Brand identified as {brand}, but it has no entries in the structured catalog.",
            brand=brand,
            confidence="Low",
        )

    variants = variants_for_brands(catalog_brands)

    if not variants:
        return _unknown_result(
            f"Brand identified as {brand}, but the catalog holds no variants for it.",
            brand=brand,
            confidence="Low",
        )

    phase2 = ask_deepseek_phase2_structured(cleaned_query, brand, variants)
    match = phase2["product_match"]

    if phase2["is_new_product"]:
        # Researched as real on the market but absent from the catalog —
        # there are no ids to enumerate, and saying so is the honest answer.
        return {
            "product_match": match,
            "id_combinations": [{
                "product_id": NEW_PRODUCT_SENTINEL,
                "product_variant_id": NEW_PRODUCT_SENTINEL,
                "sku_id": NEW_PRODUCT_SENTINEL,
                "variant_description": "",
                "catalog_description": "Researched as a real market product, but absent from the catalog — no ids exist for it.",
                "pack_units": 0, "ml_volume": 0, "occurrences": 0,
            }],
            "id_combinations_reason": phase2["variant_ids_reason"],
        }

    matched_variants = variants_by_ids(catalog_brands, phase2["variant_ids"])

    if not matched_variants:
        return {
            "product_match": match,
            "id_combinations": [],
            "id_combinations_reason": phase2["variant_ids_reason"] or "No catalog variant matched this query.",
        }

    # Pack size is resolved BEFORE ids are enumerated: it's what decides
    # which of each variant's SkuIds are in scope at all, so a stated "6
    # pack" can't drag a 1-pack code in, and no real 6-pack code is missed.
    packs = pack_options_for_variants(matched_variants)
    pack = resolve_pack_structured(cleaned_query, brand, match.get("full_product_name"), packs)

    apply_pack_to_match(match, pack["pack_units"], pack["ml_volume"])

    combinations = enumerate_id_combinations(matched_variants, pack["pack_units"], pack["ml_volume"])

    return {
        "product_match": match,
        "id_combinations": combinations,
        "id_combinations_reason": _combinations_reason(phase2, pack, combinations, matched_variants),
    }


def _combinations_reason(phase2: dict, pack: dict, combinations: list, matched_variants: list) -> str:
    """A short, honest account of why THESE ids are listed — which product
    they're all listings of, which pack narrowed them, and (when there's
    more than one) that the catalog genuinely can't separate them."""
    parts = []

    if phase2["variant_ids_reason"]:
        parts.append(phase2["variant_ids_reason"])

    if pack.get("reason"):
        parts.append(f"Pack size: {pack['reason']}")

    if len(combinations) > 1:
        variant_count = len({c["product_variant_id"] for c in combinations})
        if variant_count > 1:
            parts.append(
                f"{len(combinations)} catalog rows across {variant_count} VariantIds describe this same "
                f"product at this pack size — the catalog does not distinguish between them, so all are listed."
            )
        else:
            parts.append(
                f"This variant has {len(combinations)} separate SkuIds for the same pack, with identical "
                f"catalog descriptions — nothing distinguishes them, so all are listed."
            )

    return " ".join(parts)


def identify_product(query: str, use_cache: bool = True, **_ignored) -> dict:

    if not query or not query.strip():
        return {"error": "Empty query"}

    cleaned = clean_query(query)

    if not cleaned:
        return {"error": "Empty query after cleaning"}

    start = time.time()
    reset_usage_tracking()

    if use_cache and cleaned in _result_cache:
        result = _result_cache[cleaned]
    else:
        brand_candidates = search_brands(cleaned)
        result = _resolve_structured(cleaned, brand_candidates)

        if use_cache:
            _result_cache[cleaned] = result

    usage = get_usage_totals()
    return {
        **result,
        "usage": {
            "elapsed_seconds": round(time.time() - start, 3),
            "llm_calls": usage["llm_calls"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "estimated_cost_usd": round(estimate_cost_usd(usage["cache_hit_tokens"], usage["cache_miss_tokens"], usage["completion_tokens"]), 6),
        },
    }


def identify_products_batch(queries: list, max_workers: int = MAX_WORKERS,
                            progress_callback=None, cancel_check=None, **_ignored):
    """Same contract as the raw pipeline's identify_products_batch
    (results aligned with `queries`, plus a stats dict), with the same
    dedupe/cache/thread-pool/cancel behavior."""
    total = len(queries)
    cleaned = [clean_query(q) for q in queries]

    usage_start = snapshot_global_usage()

    unique_all = list(dict.fromkeys(c for c in cleaned if c))
    unique_pending = [c for c in unique_all if c not in _result_cache]

    failures = {}
    failed_count = 0

    if unique_pending:

        brand_candidates_batch = search_brands_batch(unique_pending)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:

            futures = {
                pool.submit(_resolve_structured, cq, cands): cq
                for cq, cands in zip(unique_pending, brand_candidates_batch)
            }

            cancelled = False

            for future in as_completed(futures):
                cq = futures[future]

                try:
                    _result_cache[cq] = future.result()
                except Exception as e:
                    failures[cq] = _unknown_result(f"Processing failed: {e}")
                    failed_count += 1

                completed += 1

                if progress_callback:
                    progress_callback(completed, len(unique_pending))

                if cancel_check and cancel_check():
                    cancelled = True
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

            if cancelled:
                for cq in unique_pending:
                    if cq not in _result_cache and cq not in failures:
                        failures[cq] = _unknown_result("Cancelled before processing")

    results = []
    for c in cleaned:
        if not c:
            results.append({"error": "Empty query"})
        elif c in _result_cache:
            results.append(_result_cache[c])
        else:
            results.append(failures.get(c, _unknown_result("Processing failed")))

    stats = {
        "total_rows": total,
        "unique_queries": len(unique_all),
        "llm_calls": len(unique_pending),
        "cache_hits": len(unique_all) - len(unique_pending),
        "failed": failed_count,
        **usage_delta(usage_start, snapshot_global_usage()),
    }

    return results, stats
