# phase2_variant.py — Phase 2 of the query-time matching pipeline:
# within the brand Phase 1 already confirmed, identify WHICH VARIANT
# (pack size, species, region-specific naming) the query refers to, and
# assign product_id/product_variant_id at the product-LINE level using
# real historical examples (Rule 5). Exact pack size (sku_id) is then
# resolved by phase3b_sku.py's resolve_sku, called from ask_deepseek_phase2
# below — kept in this file because id-assembly and final validation of
# the full triplet naturally belong with the function that owns the
# overall Phase 2 result.

from collections import defaultdict

from ..shared.resources import (
    collection, sku_examples_collection, embedding_model,
    MAX_VARIANT_CANDIDATES, MAX_EXAMPLE_HINTS, BRAND_PRODUCT_IDS,
    deepseek_json, brand_fallback_product_id, detect_language_hints,
    detect_species_for_widening, classify_probability_tier, id_str,
)
from .phase3b_sku import resolve_sku
from ..shared.query_utils import append_color_if_missing


def _pack_quantity_sort_key(candidate: dict):
    try:
        return int(str(candidate.get("pack_quantity") or "0").strip())
    except ValueError:
        return 0


def _candidate_from_meta(doc: str, meta: dict, similarity) -> dict:
    return {
        "variant_name": meta.get("variant_name", ""),
        "region": meta.get("region", ""),
        "active_ingredients": meta.get("active_ingredients", ""),
        "species": meta.get("species", ""),
        "pack_size": meta.get("pack_size", ""),
        "pack_quantity": meta.get("pack_quantity", ""),
        # Only present on variants researched after pack-color capture was
        # added to knowledge_agent2.py — older entries return "".
        "pack_color": meta.get("pack_color", ""),
        "official_brand_url": meta.get("official_brand_url", ""),
        "description": doc,
        "similarity": similarity,
    }


def _widen_with_species_match(query: str, brand: str, candidates: list) -> list:
    """
    Confirmed on a real failure: a query stating its species via a foreign
    word ("gato" = cat) returned ZERO cat candidates among the top-11 for
    Bravecto — every single one was a dog chew — because the embedding
    model doesn't weight a non-English species word strongly enough to
    outrank a brand's more numerous dog variants (plain English "cat"
    retrieves correctly, 11/11; this is specifically a foreign-word gap,
    the same class of failure already fixed for brand identification in
    phase1_brand.py's fuzzy matcher).

    detect_species_for_widening() is the same deterministic, zero-cost
    multilingual lookup already used for the prompt's language hint — if
    it names a species genuinely absent from the initial candidates, this
    pulls in every real variant of that species for the brand, so
    DeepSeek is never stuck choosing among the wrong species entirely.
    """
    target_species = detect_species_for_widening(query)

    if not target_species:
        return candidates

    if any(target_species in (c.get("species") or "") for c in candidates):
        return candidates

    try:
        data = collection.get(where={"brand": brand}, include=["documents", "metadatas"])
    except Exception:
        return candidates

    seen_names = {c["variant_name"] for c in candidates}
    added = []

    for doc, meta in zip(data["documents"], data["metadatas"]):
        if target_species not in (meta.get("species") or ""):
            continue
        name = meta.get("variant_name", "")
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        added.append(_candidate_from_meta(
            doc, meta,
            f"n/a (added — query names the species \"{target_species}\" but no candidate of that species was otherwise ranked)",
        ))

    return candidates + added if added else candidates


def _widen_with_pack_siblings(brand: str, candidates: list) -> list:
    """
    Vector search ranks candidates on OVERALL text similarity to the whole
    query — for a brand whose variants are otherwise near-identical text
    differing only in a trailing pack-count number (e.g. every
    "22.1-44.0 lbs (Teal)" pack option of Simparica Trio), similarity
    barely discriminates between them at all. Confirmed on a real failure:
    for a query explicitly asking for "pack 6", every one of the top-8
    candidates was pack_quantity 1 or 12 — NONE were 3 or 6 — because the
    embedding scores were clustered within 0.01 of each other. DeepSeek
    can't pick a pack count that was never shown to it, no matter how well
    it reasons about the query text.

    Once a product LINE surfaces among the initial candidates — identified
    by (species, pack_size), the two fields that stay constant across an
    otherwise-identical brand's pack-count siblings (pack_size is the
    per-unit dose/format, e.g. "72mg sarolaner...", not the pack count) —
    this deterministically pulls in every OTHER real pack-count variant of
    that same line, so the full set of real options is always visible
    regardless of how the embedding happened to rank them. Sibling
    candidates are grouped next to each other (sorted by species, then
    pack_size, then ascending pack_quantity) instead of appended at the
    ranked list's tail, and marked with a `synthetic_candidate` flag since
    they didn't come from the similarity search and have no similarity
    score of their own to report.
    """
    if not candidates:
        return candidates

    seen_names = {c["variant_name"] for c in candidates}
    families = {(c["species"], c["pack_size"]) for c in candidates if c["pack_size"]}

    added = []

    for species, pack_size in families:
        try:
            data = collection.get(
                where={"$and": [{"brand": brand}, {"species": species}, {"pack_size": pack_size}]},
                include=["documents", "metadatas"],
            )
        except Exception:
            continue

        for doc, meta in zip(data["documents"], data["metadatas"]):
            name = meta.get("variant_name", "")
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            candidate = _candidate_from_meta(
                doc, meta, "n/a (added as a same-line pack-size sibling of a ranked candidate)"
            )
            candidate["synthetic_candidate"] = True
            added.append(candidate)

    if not added:
        return candidates

    merged = candidates + added
    merged.sort(key=lambda c: (c["species"], c["pack_size"], _pack_quantity_sort_key(c)))
    return merged


def search_variants_for_brand(query: str, brand: str, n_results: int = MAX_VARIANT_CANDIDATES) -> list:
    """
    Searches only the chosen brand's variants, using ChromaDB's native
    `where` + `query_embeddings` filtering in a single call — not a
    manual pull-everything-then-rank-in-Python loop. Widened afterward,
    first by species (_widen_with_species_match) then by pack-count
    sibling (_widen_with_pack_siblings), so the candidate pool always
    includes every real option the query's own explicit details call for,
    not just whichever ones happened to win the similarity ranking.
    """

    if collection.count() == 0:
        return []

    embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        where={"brand": brand},
        n_results=min(n_results, collection.count())
    )

    candidates = [
        _candidate_from_meta(doc, meta, round(1 - distance, 3))
        for doc, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]

    candidates.sort(key=lambda x: x["similarity"], reverse=True)

    candidates = _widen_with_species_match(query, brand, candidates)
    return _widen_with_pack_siblings(brand, candidates)


def search_real_examples(query: str, brand: str, n: int = MAX_EXAMPLE_HINTS) -> list:
    """
    Pulls real historical vet-entered PHRASING for this brand from the
    sku_examples collection (see sku_examples_builder.py) — a Gemini-
    researched description doesn't capture how vets actually abbreviate
    things ("s/on", "3pk"), so a few authentic examples help Phase 2 read
    the query's shorthand. This is a semantic search over free text, used
    for STYLE/disambiguation context only — id ASSIGNMENT is a separate,
    deterministic job handled by triplets_for_candidate_variant() below,
    which is authoritative because it's keyed to the exact candidate
    variant rather than whatever text happens to be semantically similar.
    """

    if sku_examples_collection.count() == 0:
        return []

    embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

    try:
        results = sku_examples_collection.query(
            query_embeddings=[embedding],
            where={"brand": brand},
            n_results=min(n, sku_examples_collection.count())
        )
    except Exception:
        return []

    if not results["documents"] or not results["documents"][0]:
        return []

    examples = []

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        examples.append({
            "text": doc,
            "product_id": meta.get("product_id") or "",
            "sku_id": meta.get("sku_id") or "",
            # sku_examples_builder.py stores 0 as a "not available" sentinel
            "product_variant_id": meta.get("product_variant_id") or "",
        })

    return examples


def triplets_for_candidate_variant(brand: str, variant_name: str) -> list:
    """
    Every (product_id, product_variant_id) pair the triplet-first
    sku_examples builder linked to THIS EXACT catalogue variant as its top
    match, with each pair's combined historical-occurrence probability
    (summed across whichever SkuIds happen to fall under it — Phase 2
    doesn't care about pack size, that's Phase 3b's job). This is the
    authoritative source for Phase 2's id assignment: unlike a generic
    phrasing search, it's precomputed and keyed to the specific variant
    being considered, so there's no correlation guesswork left to the LLM.
    Sorted by probability, highest first.
    """
    if sku_examples_collection.count() == 0:
        return []

    try:
        data = sku_examples_collection.get(
            where={"$and": [{"brand": brand}, {"gemini_variant": variant_name}]},
            include=["metadatas"],
        )
    except Exception:
        return []

    # Dedupe to one entry per distinct (product_id, product_variant_id,
    # sku_id) triplet first — each triplet is stored on up to
    # EXAMPLES_PER_TRIPLET note rows that all carry the SAME
    # occurrence_probability, so summing directly over metadata rows would
    # multiply-count it (and be worse the more phrasings a triplet has).
    seen_triplets = {}
    for meta in data["metadatas"]:
        pid = id_str(meta.get("product_id"))
        vid = id_str(meta.get("product_variant_id"))
        if not pid or not vid:
            continue
        key = (pid, vid, meta.get("sku_id"))
        if key not in seen_triplets:
            seen_triplets[key] = float(meta.get("occurrence_probability") or 0.0)

    pair_probability = defaultdict(float)
    for (pid, vid, _sid), prob in seen_triplets.items():
        pair_probability[(pid, vid)] += prob

    return sorted(
        (
            {"product_id": pid, "product_variant_id": vid, "probability": p}
            for (pid, vid), p in pair_probability.items()
        ),
        key=lambda t: t["probability"],
        reverse=True,
    )


def format_variant_triplet_hint(triplets: list) -> str:
    """
    Renders the dominant/tied/low-confidence historical-id guidance for one
    candidate variant, per classify_probability_tier's three tiers:
      - "dominant": one id pair clearly leads (>=65%) -> shown alone.
      - "tied": no dominant leader but a real second contender (>=25%)
        exists (e.g. 55/45, 40/60) -> BOTH shown, so DeepSeek can use the
        query's own details to break the tie instead of the code guessing.
      - "low": fragmented, nothing close -> best available is still shown
        (never withheld), flagged as low confidence.
    """
    if not triplets:
        return "(no historical id-triplet recorded for this variant)"

    shares = [((t["product_id"], t["product_variant_id"]), t["probability"]) for t in triplets]
    tier = classify_probability_tier(shares)

    def fmt(t):
        return f'ProductId {t["product_id"]}, ProductVariantId {t["product_variant_id"]}'

    if tier == "dominant":
        t = triplets[0]
        return f'{fmt(t)} -- dominant, {round(t["probability"] * 100)}% of this variant\'s historical triplets.'

    if tier == "tied":
        a, b = triplets[0], triplets[1]
        return (
            f'Roughly tied: [{fmt(a)}] ({round(a["probability"] * 100)}%) '
            f'vs [{fmt(b)}] ({round(b["probability"] * 100)}%) -- use the '
            f'query\'s other details (species, size, region) to decide between these two.'
        )

    t = triplets[0]
    return f'Low confidence -- best available: {fmt(t)} (only {round(t["probability"] * 100)}%, no dominant historical pattern).'


def build_context_phase2(candidates: list, brand: str) -> str:

    context = ""

    for i, c in enumerate(candidates):
        color_line = f"\nPACK COLOR: {c['pack_color']}" if c.get("pack_color") else ""
        id_hint = format_variant_triplet_hint(triplets_for_candidate_variant(brand, c["variant_name"]))
        context += f"""
────────────────────────────────────────────
CANDIDATE {i + 1}
VARIANT: {c['variant_name']}
REGION: {c['region']}
SIMILARITY: {c['similarity']}{color_line}
HISTORICAL IDS: {id_hint}

FULL DETAILS:
{c['description']}
"""

    return context


def build_examples_block(examples: list) -> str:
    """Real historical PHRASINGS for Phase 2 — style/shorthand context only
    (see search_real_examples). Each candidate's own "HISTORICAL IDS" line
    (see build_context_phase2 / triplets_for_candidate_variant) is now the
    authoritative id-assignment source; these ids are shown alongside the
    phrasing purely as extra corroboration, not as the primary signal."""

    if not examples:
        return "(none available)"

    lines = []
    for e in examples:
        lines.append(
            f'- "{e["text"]}" '
            f'[ProductId: {e["product_id"]}, ProductVariantId: {e["product_variant_id"]}]'
        )
    return "\n".join(lines)


def build_known_product_id_block(brand: str) -> str:
    """Renders this brand's known ProductId(s) from BRAND_PRODUCT_IDS for
    the Phase 2 prompt — the guaranteed brand-level fallback, restored
    alongside Rule 5."""

    info = BRAND_PRODUCT_IDS.get(brand)

    if not info:
        return "(no known ProductId on file for this brand)"

    if len(info["all"]) == 1:
        return str(info["primary"])

    others = [str(i) for i in info["all"] if i != info["primary"]]
    return f'{info["primary"]} (most common) — other known ids for this brand: {", ".join(others)}'


# Plain (non f-string) template — the golden examples and output schema
# below are full of literal JSON braces, which would otherwise all need
# escaping as {{ }} in an f-string. Substitution is done with .replace()
# on the __TOKEN__ placeholders instead (same pattern as
# knowledge/knowledge_agent2.py's RESEARCH_PROMPT_TEMPLATE).
PHASE2_PROMPT_TEMPLATE = """
### ROLE
You are a veterinary product matching system. Your job is to pick the
exact product variant (pack size, species, region-specific naming) that
a messy vet prescription entry refers to, from a shortlist of candidate
variants already narrowed down to one confirmed brand, and to identify
the ProductId and ProductVariantId for that variant (product-LINE level).
You do NOT decide the exact pack size's SkuId here — that is resolved by
a separate, later stage once the variant/product line is confirmed.

### CONTEXT
The brand has already been identified by an earlier stage — your job is
to disambiguate WHICH variant of that brand's product line the query
refers to (size, pack count, species, region-specific name), and to
attach its ProductId/ProductVariantId. Each CANDIDATE VARIANT below comes
with its own "HISTORICAL IDS" line — the id-triplet(s) real historical
prescribing data actually linked to THAT exact variant, precomputed with
how often each occurred, so you don't have to correlate ids from a
separate list yourself (see Rule 5). These ids form a hierarchy: ProductId
is the BRAND-level id (one per brand, occasionally a couple across catalog
eras — always known); ProductVariantId identifies a specific product LINE
within the brand (e.g. "Spray" vs "Combo Cat") — match at THIS level,
ignoring exact pack size/count differences. Exact pack size (SkuId) is
deliberately not your concern — a later stage handles it once your variant
match is locked in. Often shorthand WILL be used in the query (e.g. "3pk"
for "3 Pipette Pack", "S/on" for "Spot On", "sml/med/lge" for size) — you
must interpret these correctly. If the query is ambiguous about pack size,
species, or other details, pick the most commonly prescribed variant as
primary with Medium confidence, and list the other plausible variants as
alternatives. Never invent a variant that isn't in the candidate list.

### INFO PROVIDED
BRAND (already confirmed): __BRAND__

KNOWN PRODUCT ID(S) FOR THIS BRAND (from historical records): __KNOWN_PRODUCT_ID__

USER QUERY:
__QUERY__

MULTILINGUAL TERM HINT (deterministic token scan — non-English words found
in the query, with their English meaning):
__SPECIES_HINT__

CANDIDATE VARIANTS:
__CONTEXT__

REAL HISTORICAL PHRASING EXAMPLES FOR THIS BRAND:
__EXAMPLES__

### RULES
1. Identify the BEST matching product variant from the candidates, and
   return exactly ONE. Do not produce alternative candidates.
2. Use ONLY the provided candidate data — never invent variants or sizes
3. "animal_age" is not available in this data — always return it as an empty string, never guess a value for it
4. "products" and "full_product_name" are DIFFERENT fields and must NEVER contain the same text:
   - "products" is the SHORT product line name only — 1-3 words, no size, no
     species, no pack count. Examples: "Spot On", "Combo", "Spray", "Plus", "Gold".
   - "full_product_name" is the COMPLETE variant name exactly as it appears in
     the matched candidate, including species/size/pack details. Example:
     "Spot On Medium Dog (10-20kg) 6 Pipette Pack".
   - If you find yourself about to write the same string into both fields,
     you have the wrong value for "products" — go back and extract just the
     product line portion of the name.
5. Assign product_id/product_variant_id using the CHOSEN candidate's own
   "HISTORICAL IDS" line (never a different candidate's):
   - If it names ONE id pair as "dominant", copy that ProductId AND
     ProductVariantId exactly — this is the well-established historical
     assignment for this variant.
   - If it shows "Roughly tied" between two id pairs, use the query's own
     details (species, weight/size band, region — anything the two
     historically-tied groups might actually differ on) to pick between
     them. Do not default to whichever number is listed first or has the
     marginally higher percentage if the query's details point the other
     way; if genuinely nothing distinguishes them, pick the higher one.
   - If it says "Low confidence — best available", still use that id pair
     (better than leaving it blank), but this weak historical signal
     should pull your overall confidence for this match down a level.
   - If it says "(no historical id-triplet recorded for this variant)",
     fall back to "KNOWN PRODUCT ID(S) FOR THIS BRAND" above for
     product_id (use the one marked "most common" if more than one is
     listed), and leave product_variant_id as an empty string.
   product_id must NEVER be left blank once a brand has been confirmed —
   every brand has at least one known real ProductId. Never invent or
   guess an id number.
   IMPORTANT — this dominant/tied signal must also drive WHICH CANDIDATE
   you pick in the first place, not just which ids you copy afterward:
   when two or more candidates plausibly describe the same real
   strength/species/weight band, don't let one candidate's fuller or more
   keyword-matching description (e.g. it spells out an exact decimal like
   "11.4mg" while another candidate's text only says "11mg") win just
   because it reads more specifically — a terser description means less
   research text was attached to that candidate, not that it's the wrong
   product. Prefer whichever of the plausible candidates has the
   strongest HISTORICAL IDS signal (dominant beats a richer-sounding
   minority candidate) unless the query states a detail that genuinely
   rules the dominant one out (an actually different species, weight
   band, or strength — not just a more literal text match).
6. "animal_size" should include tags like S, M, L, XL if known, plus the weight range if present, eg: 2-10kg, 10-20kg
7. "region" must be copied exactly from the candidate's REGION field — never invent or guess a region
8. Use ONLY: High, Medium, Low, Very Low for confidence — no numbers or percentages
9. Do NOT output a sku_id field — pack size is resolved by a later stage.
10. If the MULTILINGUAL TERM HINT is given, treat each hinted meaning
    exactly like the equivalent explicit English word in the query — vets
    often mix other languages into a note ("perro" = dog, "chat" = cat,
    "caja 3" = pack of 3, "comprimes" = tablets, "pipeta" = pipette).
    In particular the chosen variant's SPECIES MUST match a hinted species;
    never pick a dog variant for a query hinted as Cat, or vice versa.
    Hinted form words (tablet, pipette, pack count, bottle...) count as the
    query specifying that formulation/packaging. If the hint says
    "(none detected)", ignore this rule.
11. Pack/box COLOR (shown on some candidates, e.g. "(Purple)") is
    supporting evidence only: use it to CONFIRM a match when the query
    mentions a color, but colors change over time — NEVER use color alone to
    pick between (or reject) variants whose dosage, species, weight band,
    and pack count all match; those primary characteristics always outrank
    color.
12. If the query states an explicit pack count/quantity (a number next to
    "pack", "chews", "pipettes", "tablets", "x", "pk", "count", or a
    hinted form word from Rule 10), and a candidate exists with that EXACT
    pack_quantity for the same product line/species/weight-band you've
    otherwise matched, you MUST pick that exact-count candidate — do not
    settle for a different pack count of the same line just because it
    ranked first among the candidates shown. Every real pack-count option
    for a matched product line is always included in the candidate list
    (see any candidate marked "n/a (added as a same-line pack-size
    sibling...)" — these exist specifically so the count you need is
    available), so a missing exact match should be rare; only fall back to
    the closest available count, with lower confidence, when genuinely
    none matches.

### GOLDEN EXAMPLES

QUERY: "frontline spot on small dog 3pk"
CANDIDATES include: "Frontline Spot On Small Dog (2-10kg) 3 Pipette Pack" [region: UK/IE, HISTORICAL IDS: ProductId 32, ProductVariantId 197 -- dominant, 94% of this variant's historical triplets.], "...6 Pipette Pack" [region: UK/IE]
-> primary_match: products "Spot On", full_product_name "Spot On Small Dog (2-10kg) 3 Pipette Pack", ids copied exactly (32/197), region "UK/IE", confidence High
("products" holds only "Spot On", NOT the full name; pack count itself is NOT part of the id-matching decision — species + product line matched, and the dominant historical pair was copied directly)

QUERY: "frontline spot on small dog" (no pack count given)
CANDIDATES include both the 3-pack and 6-pack Small Dog variants
-> pick the more commonly-prescribed pack size with MEDIUM confidence (the query didn't specify). product_id/product_variant_id are unaffected by this ambiguity — they're decided by product line, not pack size.

QUERY: "advocate dog ex lge >25kg pk6"
CANDIDATES include: "Advocate Spot-On for Extra Large Dogs (over 25kg) 6 Pack" [region: UK/IE, HISTORICAL IDS: ProductId 1, ProductVariantId 7 -- dominant, 88% of this variant's historical triplets.]
-> primary_match: products "Spot-On", full_product_name "Spot-On for Extra Large Dogs (over 25kg) 6 Pack", ids copied exactly (1/7), region "UK/IE", confidence High
(the historical triplets backing this variant are mostly from the 3-pack's records, not the 6-pack being matched here — but ProductVariantId is a product-LINE level id, so it still applies to every pack size of this same line)

QUERY: "revolution plus cat 5.6-11 lbs" (weight band matches TWO regionally-aliased historical groups)
CANDIDATES include: "Revolution Plus for Medium Cats (2.6-5kg / 5.6-11 lbs)" [region: US, HISTORICAL IDS: Roughly tied: [ProductId 71, ProductVariantId 812] (55%) vs [ProductId 71, ProductVariantId 940] (45%) -- use the query's other details (species, size, region) to decide between these two.]
-> Look for ANY other detail in the query or candidate (region, exact wording, species tag) that favors one of the two tied ids; if the query gives no further signal at all, pick the higher one (812) but note the near-tie in "reason" and keep confidence at Medium rather than High, since the historical signal alone didn't clearly decide it.
(a genuine near-tie is NOT resolved by defaulting to the bigger number without looking — only do that as a last resort when nothing else distinguishes them)

QUERY: "drontal cat xl"
CANDIDATES include: "Drontal Cat XL Wormer Tablet for Large Cats (over 6kg) 1 Tablet" [region: UK/IE, HISTORICAL IDS: ProductId 3, ProductVariantId 22 -- dominant, 100% of this variant's historical triplets.]
-> primary_match: products "Cat XL", full_product_name "Cat XL Wormer Tablet for Large Cats (over 6kg) 1 Tablet", ids copied exactly (3/22), confidence High
(a tablet wormer, not a spot-on; "products" is the short line "Cat XL")

QUERY: "seresto collar lge dog >8kg"
CANDIDATES include: "Seresto Flea & Tick Collar for Large Dogs (over 8kg)" [region: UK/IE, HISTORICAL IDS: ProductId 6, ProductVariantId 47 -- dominant, 91% of this variant's historical triplets.], "...for Small Dogs (under 8kg)" [region: UK/IE, HISTORICAL IDS: ProductId 6, ProductVariantId 49 -- dominant, 89% of this variant's historical triplets.]
-> primary_match: products "Collar", full_product_name "Flea & Tick Collar for Large Dogs (over 8kg)", ids copied exactly (6/47), confidence High
(the Large-dog and Small-dog collars are DIFFERENT product lines/ids, each with their OWN dominant historical pair — pick the CANDIDATE matching species+size, then use THAT candidate's own ids, never borrow another candidate's)

QUERY: "milbemax chewy dog" (no size/strength given)
CANDIDATES include the Small Dog and the Large Dog chewable variants
-> pick the more commonly-prescribed size with MEDIUM confidence — the same ambiguity rule as pack size, applied to species/weight. Use whichever candidate you pick's OWN "HISTORICAL IDS" line for the id assignment.

QUERY: "bravecto chew large dog"
CANDIDATES include: "Bravecto Chews for Dogs 20-40kg (44-88 lbs) (Blue) 1 Tablet" [HISTORICAL IDS: (no historical id-triplet recorded for this variant)]
KNOWN PRODUCT ID(S) FOR THIS BRAND: 22 (most common) — other known ids for this brand: 89
-> primary_match: products "Chews", full_product_name "Chews for Dogs 20-40kg (44-88 lbs) (Blue) 1 Tablet", product_id "22" (known brand-level id — always filled), product_variant_id "" (no historical id-triplet recorded for this variant, so fall back to the brand-level id only)

QUERY: "frontline gold for dogs 45-88lbs"
CANDIDATES only include UK/IE Spot On and Combo variants (no US "Gold" line found for this brand this run)
-> primary_match: confidence "Very Low", reason: "Query references a 'Gold' product line not present in the candidate list for this brand."
(never force-fit a query onto a wrong-region candidate just because it shares a brand — if nothing plausibly matches, say so honestly)

### OUTPUT FORMAT (STRICT JSON ONLY — one match, no alternatives, no sku_id field):

{
  "primary_match": {
    "brand": "__BRAND__",
    "products": "",
    "full_product_name": "",
    "product_id": "",
    "product_variant_id": "",
    "region": "",
    "species": "",
    "animal_size": "",
    "pack_size": "",
    "pack_quantity": "",
    "animal_age": "",
    "confidence": "",
    "reason": ""
  }
}
"""


def _get_valid_variant_pairs_for_brand(brand: str) -> set:
    """
    The set of (product_id, product_variant_id) pairs that genuinely exist
    in sku_examples for this brand — used to catch DeepSeek citing a
    product-line id combination (Phase 2's job) that doesn't correspond to
    any real historical record. Product-line level, NOT pack-size level —
    SkuId is validated separately once Phase 3b resolves it.
    """
    if sku_examples_collection.count() == 0:
        return set()

    try:
        data = sku_examples_collection.get(where={"brand": brand}, include=["metadatas"])
    except Exception:
        return set()

    return {
        (str(meta.get("product_id") or ""), str(meta.get("product_variant_id") or ""))
        for meta in data["metadatas"]
    }


def _validate_and_correct_variant_pair(match: dict, valid_pairs: set, query: str) -> dict:
    """
    Deterministic (non-LLM) safety net for Phase 2's Rule 5: if DeepSeek
    claimed a product_id/product_variant_id pair that doesn't match any
    real record for this brand, clear product_variant_id (product_id is
    backstopped separately via the brand-level fallback, so it's left
    alone here). No claim (product_variant_id blank) needs no check.
    """
    pid = str(match.get("product_id") or "")
    vid = str(match.get("product_variant_id") or "")

    if vid == "":
        return match

    if (pid, vid) not in valid_pairs:
        print(
            f"  Correcting mismatched variant id for '{query}': claimed "
            f"({pid}, {vid}) doesn't match any real record — clearing."
        )
        match["product_variant_id"] = ""

    return match


def _get_valid_id_triplets_for_brand(brand: str) -> set:
    """
    The set of (product_id, sku_id, product_variant_id) triplets that
    genuinely exist in the sku_examples collection for this brand — the
    final backstop once Phase 3b has (or hasn't) resolved a specific
    sku_id. Pure metadata lookup, no embedding/similarity involved.
    """
    if sku_examples_collection.count() == 0:
        return set()

    try:
        data = sku_examples_collection.get(where={"brand": brand}, include=["metadatas"])
    except Exception:
        return set()

    return {
        (
            str(meta.get("product_id") or ""),
            str(meta.get("sku_id") or ""),
            str(meta.get("product_variant_id") or ""),
        )
        for meta in data["metadatas"]
    }


def _validate_and_correct_ids(match: dict, valid_triplets: set, query: str) -> dict:
    """
    Deterministic (non-LLM) final backstop: only runs when a SPECIFIC
    sku_id was actually assigned (by Phase 3b, or the single-SKU
    shortcut) — that's the only case making a claim valid_triplets can
    verify. A blank sku_id (variant known but pack size not resolved,
    e.g. an alternative match, or a variant with no real SKUs) makes no
    specific-record claim and needs no check.
    """
    triplet = (
        str(match.get("product_id") or ""),
        str(match.get("sku_id") or ""),
        str(match.get("product_variant_id") or ""),
    )

    if triplet[1] == "":
        return match

    if triplet not in valid_triplets:
        print(
            f"  Correcting mismatched ids for '{query}': claimed {triplet} "
            f"doesn't match any real record — clearing sku_id."
        )
        match["sku_id"] = ""

    return match


def _finalize_match_ids(query: str, brand: str, match: dict, valid_pairs: set, run_pack_llm: bool) -> dict:
    """
    Full id-resolution pipeline for one match, run after Phase 2's LLM has
    already picked product_id/product_variant_id (Rule 5):
      1. Validate the (product_id, product_variant_id) pair against real
         sku_examples records for this brand — clears product_variant_id
         if DeepSeek claimed a pair that doesn't actually exist.
      2. Guarantee product_id via the brand-level fallback if still blank.
      3. Resolve sku_id (Phase 3b or the single-SKU shortcut).
      4. Final triplet backstop — only meaningful once sku_id is set.
    """
    match = _validate_and_correct_variant_pair(match, valid_pairs, query)
    match["product_id"] = match.get("product_id") or brand_fallback_product_id(brand)
    match = resolve_sku(query, brand, match, run_pack_llm)
    return match


def _apply_pack_color(match: dict, candidates: list) -> None:
    """Guarantees pack color shows up in the OUTPUT full_product_name
    whenever the matched candidate's researched data has it — deterministic,
    not dependent on whether the LLM happened to echo it back. Most
    researched variant_names already bake color into the text (Rule 11's
    soft-hint hasn't needed this before now), so this is usually a no-op;
    append_color_if_missing skips names that already mention it."""
    name = match.get("full_product_name", "")

    color = next((c["pack_color"] for c in candidates if c.get("variant_name") == name and c.get("pack_color")), "")
    if not color:
        # Fall back to a loose match in case the LLM paraphrased slightly.
        color = next(
            (c["pack_color"] for c in candidates
             if c.get("pack_color") and name and (c["variant_name"].lower() in name.lower() or name.lower() in c["variant_name"].lower())),
            "",
        )

    match["full_product_name"] = append_color_if_missing(name, color)


def _to_output_shape(match: dict) -> dict:
    """Splits the LLM's flat match into the app's two-part output: the
    product itself (no ids), and the id combination(s) it corresponds to.

    The raw pipeline always yields exactly ONE combination — unlike the
    structured pipeline, it has no authoritative catalog to enumerate
    duplicate/legacy listings from, so it can only report the single
    triplet it inferred from historical records. The shape is shared so
    both pipelines stay directly comparable and the API/export/frontend
    only ever handle one contract."""
    product = {k: match.get(k, "") for k in (
        "brand", "products", "full_product_name", "region", "species",
        "animal_size", "pack_size", "pack_quantity", "animal_age",
        "confidence", "reason",
    )}

    pid = match.get("product_id") or ""
    vid = match.get("product_variant_id") or ""
    sid = match.get("sku_id") or ""

    if not pid and not vid and not sid:
        return {"product_match": product, "id_combinations": [], "id_combinations_reason": ""}

    return {
        "product_match": product,
        "id_combinations": [{
            "product_id": str(pid),
            "product_variant_id": str(vid),
            "sku_id": str(sid),
            "variant_description": match.get("full_product_name", ""),
            "catalog_description": "Inferred from historical prescribing records (no authoritative catalog in raw mode).",
            "pack_units": 0,
            "ml_volume": 0,
            "occurrences": 0,
        }],
        "id_combinations_reason": (
            "Raw mode infers one id triplet by correlating researched product text against "
            "historical records, so it reports a single combination rather than enumerating "
            "every catalog listing of this product."
        ),
    }


def ask_deepseek_phase2(query: str, brand: str, candidates: list, real_examples: list) -> dict:
    """
    Phase 2: the LLM picks the variant/characteristics AND product_id /
    product_variant_id (Rule 5, matched at the product-line level, real
    historical examples as the source of truth) in one call. Exact pack
    size (sku_id) is then resolved separately in _finalize_match_ids —
    directly if the variant has only one real SkuId, otherwise via
    Phase 3b (a small, focused LLM call).
    """
    context = build_context_phase2(candidates, brand)
    examples_block = build_examples_block(real_examples)
    known_product_id_block = build_known_product_id_block(brand)

    prompt = (
        PHASE2_PROMPT_TEMPLATE
        .replace("__BRAND__", brand)
        .replace("__QUERY__", query)
        .replace("__SPECIES_HINT__", detect_language_hints(query) or "(none detected)")
        .replace("__CONTEXT__", context)
        .replace("__EXAMPLES__", examples_block)
        .replace("__KNOWN_PRODUCT_ID__", known_product_id_block)
    )

    result = deepseek_json(prompt)
    match = result.get("primary_match") or {}

    valid_pairs = _get_valid_variant_pairs_for_brand(brand)
    match = _finalize_match_ids(query, brand, match, valid_pairs, run_pack_llm=True)
    _apply_pack_color(match, candidates)

    # Final deterministic backstop against a hallucinated sku_id specifically.
    match = _validate_and_correct_ids(match, _get_valid_id_triplets_for_brand(brand), query)

    return _to_output_shape(match)
