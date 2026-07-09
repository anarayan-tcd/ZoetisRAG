# Veterinary Product Matching System

Matches vet prescription text (e.g. `"FRONTLINE SPOT ON XL DOG 3PK"`) against
a database of veterinary products. Built as a two-phase retrieval + LLM reasoning pipeline:
a vector search in ChromaDB narrows candidates down, then DeepSeek reasons over them to pick the best match —
first the **brand**, then the exact **product variant** within that brand.

The product knowledge base itself is built offline by asking Gemini to research each brand live
via Google Search grounding, cross-checked against real historical prescription phrasing pulled
from a labelled dataset.

A React + FastAPI app is the active interface. An earlier Gradio prototype has been retired to
`legacy/`.

## Project layout

```
CodeProjects/
├── backend/          FastAPI app + the core matching pipeline
│   └── knowledge/     offline scripts that build the ChromaDB knowledge base
│   └── data/           raw/generated CSVs and the ChromaDB store
├── frontend/         React + TypeScript UI (Vite)
├── legacy/           retired Gradio app (reference only)
├── scripts/          standalone debug utilities
└── start.bat         launches backend + frontend together
```

---

## `backend/` — FastAPI app + matching pipeline

| File | Description |
|---|---|
| `main.py` | FastAPI app; exposes `/api/query`, `/api/batch/*`, and `/api/cache/*` endpoints. |
| `query_agent.py` | The core two-phase matching pipeline built on ChromaDB search, DeepSeek reasoning (Phase 1: brand, Phase 2: variant), result caching, the opt-in live-research fallback, and a deterministic post-check that clears any product/SKU id DeepSeek claims that doesn't match a real historical record. THIS AGENT USES AI BASED REASONING AND COSTS MONEY. |
| `schemas.py` | Pydantic request/response models for the API (`QueryRequest`, `ProductMatch`, `QueryResponse`). |
| `batch.py` | In-memory job store and background execution for batch file uploads. |
| `export.py` | Writes a completed batch job's rows out to CSV, Excel, or JSON for download. |
| `chromadb_client.py` | Singleton embedding model + ChromaDB collection accessors, shared by every backend script so nothing loads the model twice. |
| `APIKeys.env` | DeepSeek / Gemini API keys (not committed to source control). |

### `backend/knowledge/` — offline knowledge-base builders

Run in order to (re)build the product knowledge base; none of these are called by the live app.

| File | Description |
|---|---|
| `extract_brands.py` | Turns the `Notes` column in `Labelled.csv` into a clean, deduplicated brand list (`brand_seed_list.csv`) via a heuristic pass plus DeepSeek classification for ambiguous groups. THIS AGENT USES AI BASED REASONING AND COSTS MONEY. |
| `knowledge_agent2.py` | For each brand in the seed list, researches it live via Gemini + Google Search grounding and stores structured product variants (one ChromaDB document per variant, tagged with region) — only writing what's new or changed, and checkpointed per-brand so a crash or credit exhaustion doesn't force re-researching already-completed brands. THIS AGENT USES AI BASED REASONING AND COSTS MONEY. GEMINI IS CURRENTLY RUNNING ON THE FLASH 2.5 FREE PLAN. FULL 146 BRAND SEARCH GROUNDING MIGHT REQUIRE ADDITIONAL TOKENS. |
| `sku_examples_builder.py` | Builds a second ChromaDB collection of real historical vet-entered phrasing per brand, used as authentic style/id context in Phase 2 matching. This is critical to building confidence for what the AI would assume to be ambiguous or low-confidence guesses. This takes the most cleanly defined combinations in Labelled.csv and uses them as samples for the AI model to take into account when classifying the messy input string. It accounts for shorthand, misspellings etc. |

### `backend/data/`

| Path | Description |
|---|---|
| `raw/Labelled.csv` | Historical vet prescription notes (~328k rows). Source data for brand discovery and real-phrasing examples. |
| `raw/FrontlineDemo.csv` | Small sample file for manual batch-upload testing. |
| `generated/brand_seed_list.csv` | Output of `extract_brands.py`. Canonical brand list fed into `knowledge_agent2.py`. |
| `generated/brand_extraction_checkpoint.json` | Resumable checkpoint so a crash during `extract_brands.py` doesn't lose already-completed LLM calls. |
| `generated/knowledge_research_checkpoint.json` | Resumable checkpoint so a crash or Gemini credit exhaustion during `knowledge_agent2.py` doesn't lose already-researched brands or force re-spending credits on a restart. |
| `chroma_db/` | Persistent ChromaDB stores `vet_products` (researched variants) and `sku_examples` (real historical phrasing) collections. |

---

## `frontend/` — React + TypeScript UI

Vite-based single-page app. Runs entirely locally against the FastAPI backend at
`http://localhost:8000` (hardcoded — this app isn't deployed anywhere else).

| File | Description |
|---|---|
| `src/App.tsx` | Top-level orchestrator — owns all shared state (loading, results, toasts) and lays out the two-column UI; the form components below are input-only and report back to it. |
| `src/api.ts` | Typed `fetch` wrappers for every backend endpoint. |
| `src/types.ts` | TypeScript types mirroring `backend/schemas.py`. |
| `src/components/QueryForm.tsx` | Single-query input form. |
| `src/components/BatchUpload.tsx` | File upload input form (CSV/TXT). |
| `src/components/ResultsPanel.tsx` | Unified results display — renders either the single-query match cards or the batch summary/downloads, depending on which was last run. |
| `src/components/MatchCard.tsx` | Collapsible card rendering one product match (Brand, Products, Full Product Name, ids, region, confidence, etc.). |
| `src/components/ProgressBar.tsx` | One shared progress bar — a real determinate fill for batch jobs, an indeterminate cycling-label bar for single queries (which have no intermediate backend signal). |
| `src/components/Toast.tsx` | Lightweight toast notification system (success/error/info), no external dependency. |
| `src/components/CacheResetButton.tsx` | Clears the backend's result cache. |
| `src/components/HowToSection.tsx` | Collapsible in-app documentation — explains the two input modes, output fields, and confidence levels. |

---

## `legacy/` — retired Gradio app

`gemini_app.py` and `ui_helpers.py` were the original UI, superseded by `frontend/` + `backend/`.
Kept for reference only — not part of the active app, but still runnable from this location if needed.

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
