# query_agent.py — core execution path for the query-time matching
# pipeline. Deliberately kept thin: input cleaning, the result cache, and
# top-level orchestration only. Each pipeline stage's prompt templates and
# call logic live in their own module (phase1_brand.py, phase2_variant.py,
# phase3b_sku.py, llm_judge.py, live_fallback.py), all built on the shared
# setup in resources.py.

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from knowledge.extract_brands import guess_brand_token
from ..shared.resources import (
    MAX_WORKERS, DEFAULT_ENABLE_LIVE_FALLBACK, DEFAULT_ENABLE_LLM_JUDGE,
    reset_usage_tracking, get_usage_totals, snapshot_global_usage, estimate_cost_usd, usage_delta,
)
from ..shared.phase1_brand import search_brands, search_brands_batch, ask_deepseek_phase1, augment_with_fuzzy_brand
from ..shared.query_utils import clean_query, unknown_result as _unknown_result
from .phase2_variant import search_variants_for_brand, search_real_examples, ask_deepseek_phase2
from .llm_judge import apply_llm_judge
from .live_fallback import live_research_fallback


# ─────────────────────────────────────────────────────────────
# RESULT CACHE
# ─────────────────────────────────────────────────────────────
# Vet prescription lists repeat the same product strings constantly
# (many patients on the same product). Caching by cleaned query text
# means repeats are served instantly instead of re-hitting DeepSeek.

_result_cache = {}


def cache_size() -> int:
    return len(_result_cache)


def clear_cache() -> None:
    _result_cache.clear()


def _is_low_confidence_result(result: dict) -> bool:
    match = result.get("product_match", {})
    return (
        not match.get("brand")
        or match.get("brand") == "Unknown"
        or match.get("confidence") in ("Low", "Very Low")
    )


def _cache_usable(cleaned_query: str, enable_live_fallback: bool, enable_llm_judge: bool = False) -> bool:
    """
    A cache hit is only skipped when the caller now wants a feature that
    the cached result wasn't produced with, AND that feature would
    actually have changed something:
      - live fallback: cached result was low-confidence (fallback would
        have tried to improve it).
      - LLM judge: cached result is High/Medium confidence (judge would
        actually run on it) and wasn't already judged (see the "_judged"
        marker _resolve_with_brand_candidates sets).
    Otherwise a query resolved once would silently keep returning that
    same stale answer forever, even after the caller asks this call to
    try harder. Good, already-fully-processed cached answers are always
    reused regardless.
    """
    if cleaned_query not in _result_cache:
        return False

    cached = _result_cache[cleaned_query]

    if enable_live_fallback and _is_low_confidence_result(cached):
        return False

    if enable_llm_judge and not cached.get("_judged") and not _is_low_confidence_result(cached):
        return False

    return True


def _resolve_with_brand_candidates(
    cleaned_query: str,
    brand_candidates: list,
    enable_live_fallback: bool = DEFAULT_ENABLE_LIVE_FALLBACK,
    enable_llm_judge: bool = DEFAULT_ENABLE_LLM_JUDGE
) -> dict:

    # Recovers a brand the vector search missed entirely due to heavy
    # input noise (leetspeak, missing letters, decorative symbols) via
    # deterministic fuzzy text matching — no extra DB/LLM cost, and safe
    # even when brand_candidates is already [] (turns it into exactly the
    # one recovered candidate instead of skipping Phase 1 altogether).
    brand_candidates = augment_with_fuzzy_brand(cleaned_query, brand_candidates)

    phase1_result = ask_deepseek_phase1(cleaned_query, brand_candidates) if brand_candidates else None

    low_confidence = (
        not brand_candidates
        or phase1_result is None
        or phase1_result.get("confidence") in ("Low", "Very Low")
    )

    if low_confidence and enable_live_fallback:

        guessed_brand = (
            phase1_result.get("brand") if phase1_result and phase1_result.get("brand")
            else guess_brand_token(cleaned_query).title()
        )

        if guessed_brand and live_research_fallback(guessed_brand, cleaned_query):
            brand_candidates = search_brands(cleaned_query)
            phase1_result = ask_deepseek_phase1(cleaned_query, brand_candidates) if brand_candidates else None

    if not brand_candidates or not phase1_result or not phase1_result.get("brand"):
        return _unknown_result("No matching brand found in ChromaDB.")

    brand = phase1_result["brand"]

    variant_candidates = search_variants_for_brand(cleaned_query, brand)

    if not variant_candidates:
        # Phase 1's confidence answers "how sure are we of the BRAND" —
        # it says nothing about whether we can actually match a PRODUCT,
        # which here we can't at all (zero researched variants for this
        # brand). Reusing it directly was a real bug: a query could come
        # back "Confidence: High" next to an entirely blank product match,
        # which reads as a confident empty answer rather than the honest
        # "we know the brand but have no catalogue data for it" this
        # actually is.
        return _unknown_result(
            f"Brand identified as {brand}, but no stored variants matched.",
            brand=brand,
            confidence="Low"
        )

    real_examples = search_real_examples(cleaned_query, brand)

    result = ask_deepseek_phase2(cleaned_query, brand, variant_candidates, real_examples)

    if enable_llm_judge:
        result = apply_llm_judge(cleaned_query, brand, result)

    # Records whether THIS resolution ran with judging enabled, regardless
    # of whether apply_llm_judge found anything to flag — lets
    # _cache_usable tell "already judged" apart from "never asked to be".
    # Stripped automatically from API responses (not part of ProductMatch/
    # QueryResponse), and harmless in batch export (only product_match /
    # id_combinations keys are read there).
    result["_judged"] = enable_llm_judge

    return result


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE — SINGLE QUERY
# ─────────────────────────────────────────────────────────────

def identify_product(
    query: str,
    use_cache: bool = True,
    enable_live_fallback: bool = DEFAULT_ENABLE_LIVE_FALLBACK,
    enable_llm_judge: bool = DEFAULT_ENABLE_LLM_JUDGE
) -> dict:

    if not query or not query.strip():
        return {"error": "Empty query"}

    cleaned = clean_query(query)

    if not cleaned:
        return {"error": "Empty query after cleaning"}

    start = time.time()
    reset_usage_tracking()

    if use_cache and _cache_usable(cleaned, enable_live_fallback, enable_llm_judge):
        result = _result_cache[cleaned]
    else:
        brand_candidates = search_brands(cleaned)
        result = _resolve_with_brand_candidates(cleaned, brand_candidates, enable_live_fallback, enable_llm_judge)

        if use_cache:
            _result_cache[cleaned] = result

    # Computed fresh every call (not cached alongside the match itself) so
    # a cache hit honestly reports ~0s / $0 rather than the original
    # resolution's cost.
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


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE — LARGE FILE BATCHES
# ─────────────────────────────────────────────────────────────

def identify_products_batch(
    queries: list,
    max_workers: int = MAX_WORKERS,
    progress_callback=None,
    enable_live_fallback: bool = DEFAULT_ENABLE_LIVE_FALLBACK,
    enable_llm_judge: bool = DEFAULT_ENABLE_LLM_JUDGE,
    cancel_check=None
):
    """
    Efficiently classifies a large list of raw prescription strings:

    1. Cleans and de-duplicates the queries — identical entries are
       only reasoned over once, and reuse the shared result cache.
    2. Embeds + searches ChromaDB for Phase 1 brand candidates for every
       unique query in ONE round trip instead of one call per row.
    3. Fans the remaining two-phase DeepSeek reasoning out across a
       thread pool (DeepSeek is I/O-bound, so this overlaps network
       latency instead of processing the file one row at a time).

    progress_callback(completed, total_unique_calls) is invoked as each
    query finishes its reasoning, so a UI can render live progress.

    cancel_check, if given, is polled after every completed query; once
    it returns True, any not-yet-started futures are cancelled and the
    loop stops early (already-running futures still finish — Python
    threads can't be safely force-killed). Rows that never got
    processed are reported as failures rather than silently dropped.

    Returns (results, stats) where results is aligned index-for-index
    with `queries`, and stats reports how much work was skipped.
    """

    total = len(queries)
    cleaned = [clean_query(q) for q in queries]

    usage_start = snapshot_global_usage()

    unique_all = list(dict.fromkeys(c for c in cleaned if c))
    unique_pending = [c for c in unique_all if not _cache_usable(c, enable_live_fallback, enable_llm_judge)]

    # Rows that failed this run (e.g. a repeated malformed DeepSeek reply)
    # go here, NOT into _result_cache — so one bad row can't abort the
    # whole batch, and a later re-run retries it instead of serving a
    # cached failure. Only genuine successes are cached.
    failures = {}
    failed_count = 0

    if unique_pending:

        brand_candidates_batch = search_brands_batch(unique_pending)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:

            futures = {
                pool.submit(_resolve_with_brand_candidates, cq, cands, enable_live_fallback, enable_llm_judge): cq
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


# ─────────────────────────────────────────────────────────────
# TESTING
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_queries = [
        "Frontline Spray 100ml for dogs",
        "Frontline spot on large dog",
        "fleas treatment cat 3 pipettes",
        "frontline 250ml spray",
        "tick treatment small dog frontline",
        "frontline spot on dog xl 6",
        "frontline plus dog 3 pack"
    ]

    print("=" * 60)
    print("DEEPSEEK VETERINARY PRODUCT CLASSIFIER")
    print("=" * 60)

    for q in test_queries:

        print(f"\nQUERY: {q}")

        result = identify_product(q)

        print(json.dumps(result, indent=2))
        print("-" * 60)
