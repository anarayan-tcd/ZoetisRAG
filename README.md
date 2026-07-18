# Veterinary Product Matching System

Matches vet prescription text (e.g. `"FRONTLINE SPOT ON XL DOG 3PK"`) against
a database of veterinary products. Built as a two-phase retrieval + LLM reasoning pipeline:
a vector search in ChromaDB narrows candidates down, then DeepSeek reasons over them to pick the best match —
first the **brand**, then the exact **product variant** within that brand, then (if ambiguous) the exact **pack size**.
An optional LLM-as-judge pass then double-checks High/Medium-confidence results against real historical
prescribing records, and can promote a better-grounded alternative if the first pick looks wrong.

The product knowledge base itself is built offline by asking Gemini to research each brand live
via Google Search grounding, cross-checked against real historical prescription phrasing pulled
from a labelled dataset.

A React + FastAPI app is the active interface. An earlier Gradio prototype has been retired to
`legacy/`.

## Project layout

```
ZoetisRAG/
├── backend/          FastAPI app + the two matching pipelines
│   ├── main.py         FastAPI entrypoint (stays at backend root — uvicorn/gunicorn target)
│   ├── api/            HTTP layer: request/response schemas, batch job store, exporters
│   ├── pipeline/
│   │   ├── shared/       modules BOTH pipelines use (LLM/DB resources, Phase 1 brand ID, query cleaning)
│   │   ├── raw/          original pipeline: ids inferred from researched text + historical phrasing
│   │   └── structured/   catalog-backed pipeline: authoritative ids from the merged SKU catalog
│   ├── knowledge/      offline scripts that build/enrich the knowledge base
│   └── data/           raw/generated data files and the ChromaDB store
├── frontend/         React + TypeScript UI (Vite)
├── legacy/           retired Gradio app (reference only)
├── results/          evaluation spreadsheets + dissertation reference material
├── scripts/          standalone debug utilities
└── start.bat         launches backend + frontend together
```

---

## `backend/` — FastAPI app

| File | Description |
|---|---|
| `main.py` | FastAPI entrypoint (kept at backend root so `uvicorn main:app --app-dir backend` and the Azure `gunicorn --chdir backend main:app` startup command both keep working); exposes `/api/query`, `/api/batch/*` (upload, status, cancel, download), and `/api/cache/*`. Both endpoints take a `mode` ("raw" or "structured") selecting which pipeline answers. |
| `chromadb_client.py` | Singleton embedding model + ChromaDB collection accessors, shared by the pipelines, the offline knowledge scripts, and `scripts/viewChromadb.py` so nothing loads the model twice. |
| `APIKeys.env` | DeepSeek / Gemini API keys (not committed to source control). |
| `requirements.txt` | Backend-only Python dependencies (used by the Azure App Service deploy). |

### `backend/api/` — HTTP layer

| File | Description |
|---|---|
| `schemas.py` | Pydantic request/response models for the API (`QueryRequest`, `ProductMatch`, `QueryResponse`). |
| `batch.py` | In-memory job store and background execution for batch file uploads (CSV/Excel/TXT), including cooperative cancellation (`cancel_job`) for the frontend's stop button, and per-job `mode` dispatch to the chosen pipeline. |
| `export.py` | Writes a completed (or cancelled) batch job's rows out to CSV, Excel, or JSON for download. |

### `backend/pipeline/shared/` — modules both pipelines use

| File | Description |
|---|---|
| `resources.py` | Shared setup every pipeline module imports FROM (DeepSeek client, embedding model, ChromaDB collections, brand→ProductId lookup, config constants, multilingual term detection, the `deepseek_json` retry helper) — centralized here specifically to avoid import cycles between phase modules. |
| `phase1_brand.py` | Phase 1: identifies WHICH BRAND a messy query most likely refers to, from vector-search candidates plus deterministic fuzzy recovery of heavily-garbled brand names. Both pipelines start here. |
| `query_utils.py` | Input normalization (`clean_query`) and the empty-result shape (`unknown_result`). |

### `backend/pipeline/raw/` — the original matching pipeline

Ids are INFERRED: researched product text is matched against historical prescription records.

| File | Description |
|---|---|
| `query_agent.py` | Raw-mode orchestration: the result cache and `identify_product`/`identify_products_batch` entry points. Deliberately thin; each phase below owns its own reasoning. THIS AGENT USES AI BASED REASONING AND COSTS MONEY. |
| `phase2_variant.py` | Phase 2: within the confirmed brand, identifies WHICH VARIANT (pack size, species, region-specific naming) the query refers to, and assigns `product_id`/`product_variant_id` from real historical examples. |
| `phase3b_sku.py` | Phase 3: resolves the exact pack size (`sku_id`). A deterministic lookup if the variant has only one real SKU on file; a small focused DeepSeek call only when genuinely ambiguous, and only for the primary match. |
| `llm_judge.py` | Optional LLM-as-judge pass (opt-in, off by default): a separate DeepSeek call reviews High/Medium-confidence primary matches against real historical phrasing for that exact variant. Can approve as-is, downgrade confidence ("flagged") when nothing supports the match, or — if one of Phase 2's own alternative matches is better grounded — promote that alternative to primary ("corrected") instead of just downgrading. Never invents a match beyond what Phase 2 already proposed, and can't fix a wrong brand from Phase 1. THIS AGENT USES AI BASED REASONING AND COSTS MONEY. |
| `live_fallback.py` | Optional live-research fallback (opt-in, off by default): when Phase 1 confidence is Low/Very Low, researches the full query (not just the guessed brand) live via Gemini and merges results into ChromaDB on the spot, reusing `knowledge_agent2.py`'s research/store functions. THIS AGENT USES AI BASED REASONING AND COSTS MONEY. |

### `backend/pipeline/structured/` — the catalog-backed matching pipeline

Ids are AUTHORITATIVE: every candidate comes from the merged SKU catalog
(`Merged_Product_Variant_SKU.xlsx`), enriched with Gemini-researched market descriptions.
Historical data still supplies commonness priors and real vet phrasings — but never the ids.
The `enable_live_fallback`/`enable_llm_judge` flags are accepted and ignored in this mode
(they patch raw-pipeline knowledge gaps the catalog doesn't have).

| File | Description |
|---|---|
| `query_agent_structured.py` | Structured-mode orchestration, mirroring the raw entry points (own result cache, same batch/cancel contract). THIS AGENT USES AI BASED REASONING AND COSTS MONEY. |
| `catalog.py` | In-memory access layer over the enriched catalog: brand→variants→SKUs (with real `PackUnits`), Labelled.csv occurrence priors, researched-name→catalog-name brand resolution, and "New Product" pseudo-variants. Falls back to the raw catalog xlsx when the enriched build doesn't exist yet. |
| `phase2_structured.py` | One DeepSeek call over the brand's COMPLETE catalog variant list (max ~31/brand — no vector search shortlisting, so no candidate starvation), each candidate printed with its real ids; hallucinated ids are cleared by a membership check. |
| `phase3_structured.py` | SKU resolution: deterministic when the variant has one SKU; otherwise a small DeepSeek call over authoritative `PackUnits`/`MlVolume` options, keeping the discriminate-first and evidence-weighting rules. |

### `backend/knowledge/` — offline knowledge-base builders

Run in order to (re)build the product knowledge base; none of these are called by the live app.

| File | Description |
|---|---|
| `extract_brands.py` | Turns the `Notes` column in `Labelled.csv` into a clean, deduplicated brand list (`brand_seed_list.csv`) via a heuristic pass plus DeepSeek classification for ambiguous groups. THIS AGENT USES AI BASED REASONING AND COSTS MONEY. |
| `knowledge_agent2.py` | For each brand in the seed list, researches it live via Gemini + Google Search grounding and stores structured product variants (one ChromaDB document per variant, tagged with region) — only writing what's new or changed, and checkpointed per-brand so a crash or credit exhaustion doesn't force re-researching already-completed brands. THIS AGENT USES AI BASED REASONING AND COSTS MONEY. GEMINI IS CURRENTLY RUNNING ON THE FLASH 2.5 FREE PLAN. FULL 146 BRAND SEARCH GROUNDING MIGHT REQUIRE ADDITIONAL TOKENS. |
| `sku_examples_builder.py` | Builds a second ChromaDB collection of real historical vet-entered phrasing per brand, used as authentic style/id context in Phase 2 matching and as grounding for the LLM judge. This is critical to building confidence for what the AI would assume to be ambiguous or low-confidence guesses. This takes the most cleanly defined combinations in Labelled.csv and uses them as samples for the AI model to take into account when classifying the messy input string. It accounts for shorthand, misspellings etc. |
| `select_evaluation_notes.py` | Samples a set of real notes from `Labelled.csv` for building evaluation sets. |
| `catalog_common.py` | Shared constants + loading for the structured-catalog work: the catalog↔researched brand alias map, the admin pseudo-brand exclusion list (reminders etc.), and the canonical catalog loader. |
| `research_catalog_brands.py` | One-shot driver: Gemini-researches every catalog brand with no researched pool yet, using the brand's own catalog descriptions as search hints. Checkpointed per brand. THIS AGENT USES AI BASED REASONING AND COSTS MONEY. |
| `catalog_enricher.py` | Joins the researched knowledge onto the authoritative catalog: per-brand embedding match of researched variants onto catalog `VariantDescription`s (auto-accept ≥0.80, auto-reject <0.55, DeepSeek-verified in between, checkpointed), attaching market names/species/pack color as descriptor columns, and appending researched-but-not-in-catalog products as "New Product" rows. Output: `data/generated/catalog_enriched.xlsx`, which `pipeline/structured/catalog.py` loads. THIS AGENT USES AI BASED REASONING AND COSTS MONEY. |

### `backend/data/`

| Path | Description |
|---|---|
| `raw/Labelled.csv` | Historical vet prescription notes (~328k rows). Source data for brand discovery, real-phrasing examples, and commonness priors. |
| `raw/Merged_Product_Variant_SKU.xlsx` | The authoritative SKU catalog: every real (ProductId, VariantId, SkuId) triplet with descriptions, `PackUnits`, and `MlVolume`. Source of truth for the structured pipeline's ids. |
| `raw/FrontlineDemo.csv` | Small sample file for manual batch-upload testing. |
| `generated/brand_seed_list.csv` | Output of `extract_brands.py`. Canonical brand list fed into `knowledge_agent2.py`. |
| `generated/brand_extraction_checkpoint.json` | Resumable checkpoint so a crash during `extract_brands.py` doesn't lose already-completed LLM calls. |
| `generated/knowledge_research_checkpoint.json` | Resumable checkpoint so a crash or Gemini credit exhaustion during `knowledge_agent2.py` doesn't lose already-researched brands or force re-spending credits on a restart. |
| `generated/evaluation_*.csv` | Evaluation sample sets and scored results from pipeline accuracy runs. |
| `chroma_db/` | Persistent ChromaDB stores `vet_products` (researched variants) and `sku_examples` (real historical phrasing) collections. |

---

## `frontend/` — React + TypeScript UI

Vite-based single-page app. The backend base URL comes from `VITE_API_BASE` at build time
(set in the Azure Static Web Apps workflow for production), falling back to
`http://localhost:8000` so local dev works unconfigured.

| File | Description |
|---|---|
| `src/App.tsx` | Top-level orchestrator — owns all shared state (loading, results, toasts) and lays out the two-column UI; the form components below are input-only and report back to it. Also owns query cancellation (`AbortController`) and batch cancellation (`cancelBatch`). |
| `src/api.ts` | Typed `fetch` wrappers for every backend endpoint, including `cancelBatch` and an abortable `queryProduct`. |
| `src/types.ts` | TypeScript types mirroring `backend/schemas.py`. |
| `src/components/QueryForm.tsx` | Single-query input form (card layout), with Run/Stop buttons side by side. |
| `src/components/BatchUpload.tsx` | File upload input form (CSV/TXT, card layout), with Run/Stop buttons side by side. |
| `src/components/ResultsPanel.tsx` | Unified results display — renders either the single-query match cards or the batch summary/downloads (including a distinct "Cancelled" state), depending on which was last run. |
| `src/components/MatchCard.tsx` | Collapsible card rendering one product match (Brand, Products, Full Product Name, ids, region, confidence, etc.). |
| `src/components/ProgressBar.tsx` | One shared progress bar — a real determinate fill for batch jobs, an indeterminate cycling-label bar for single queries (which have no intermediate backend signal). |
| `src/components/Toast.tsx` | Lightweight toast notification system (success/error/info), no external dependency. |
| `src/components/CacheResetButton.tsx` | Clears the backend's result cache. |
| `src/components/HowToSection.tsx` | In-app documentation, opened via a button as a floating modal (not an inline dropdown) — explains the two input modes, output fields, confidence levels, and optional checkboxes. |

Both input forms sit on their own card panel with a small icon-only stop button (⏹) next to the
Run button, so an in-flight single query or batch job can be cancelled from the UI. A stopped
query aborts client-side (the in-flight DeepSeek call may still finish server-side — Python
threads can't be force-killed); a stopped batch job stops dispatching new rows and keeps
whatever results already completed.

---

## `legacy/` — retired Gradio app

`gemini_app.py` and `ui_helpers.py` were the original UI, superseded by `frontend/` + `backend/`.
Kept for reference only — not part of the active app, and not updated when the pipeline changes
(so its imports of the old flat `query_agent` module are stale).

---

## `results/`

Evaluation spreadsheets, F1-score comparisons across pipeline versions, and a pipeline-logic slide
deck — dissertation/analysis reference material, not used by the running app.

---

## `scripts/`

| File | Description |
|---|---|
| `viewChromadb.py` | Dumps the full contents of a ChromaDB collection to the console for debugging. Takes an optional arg (`vet_products` (default) or `sku_examples`) to pick which collection. |

---

## Running locally

```
start.bat
```

Launches the backend (FastAPI, `http://localhost:8000`) and frontend (Vite,
`http://localhost:5173`) in separate windows. Close a window (or Ctrl+C inside it) to stop that
server.

To rebuild the knowledge base from scratch, run the three scripts in `backend/knowledge/` in
order: `extract_brands.py` → `knowledge_agent2.py` → `sku_examples_builder.py`.
