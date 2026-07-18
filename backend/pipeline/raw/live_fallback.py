# live_fallback.py — query-time live-research fallback (opt-in per call,
# see enable_live_fallback). When Phase 1 can't confidently identify a
# brand, optionally research it live via Gemini search grounding and
# merge into ChromaDB on the spot, rather than waiting for the next
# knowledge_agent2.py batch run.


def live_research_fallback(guessed_brand: str, cleaned_query: str) -> bool:
    """
    Researches a hypothesis brand live via Gemini search grounding and
    merges any results into ChromaDB, reusing knowledge_agent2.py's own
    research/store functions — imported lazily so a normal query run
    (fallback disabled) never requires a Gemini API key at all.

    The full failed query — not just the guessed brand — is passed in as
    research context via the existing "known product lines" hint
    parameter: since this Gemini call is already being paid for, it
    should be pointed at the SPECIFIC variant the query describes
    (species, pack size, region, everything in the text), not left to
    generically catalog the brand. A brand-only search can easily surface
    the brand's most common products while missing the one specific
    variant this particular query was actually asking about.
    """

    from knowledge.knowledge_agent2 import research_brand_with_gemini, store_variant_in_chromadb
    from chromadb_client import get_all_variant_documents

    print(f"Phase 1 confidence too low — live-researching '{guessed_brand}' for query \"{cleaned_query}\"...")

    hint = (
        f'The exact real-world query that triggered this research: "{cleaned_query}". '
        f"Prioritize finding the specific variant (species, pack size, region) this describes — "
        f"don't just catalog the brand's most common products."
    )
    data = research_brand_with_gemini(guessed_brand, known_product_lines=hint)
    variants = data.get("variants", [])

    if not variants:
        return False

    existing_docs = get_all_variant_documents()
    updated = sum(store_variant_in_chromadb(guessed_brand, v, existing_docs) for v in variants)

    print(f"Live research added/updated {updated} variant(s) for '{guessed_brand}'.")

    return updated > 0
