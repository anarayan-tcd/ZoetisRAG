# main.py — FastAPI backend, replacing the Gradio app one step at a time.

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import batch
import export as exporters
from pipeline import query_agent
from schemas import QueryRequest, QueryResponse

app = FastAPI(title="Veterinary Product Matching API")

# Vite's default dev server port, plus the common alternate — this app
# runs locally only, so no other origins are ever needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────
# SINGLE QUERY
# ─────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
def query_product(request: QueryRequest):

    result = query_agent.identify_product(
        request.query,
        enable_live_fallback=request.enable_live_fallback,
        enable_llm_judge=request.enable_llm_judge
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


# ─────────────────────────────────────────────────────────────
# BATCH FILE PROCESSING
# ─────────────────────────────────────────────────────────────

@app.post("/api/batch/upload")
async def upload_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    enable_live_fallback: bool = False,
    enable_llm_judge: bool = False,
):
    content = await file.read()
    queries = batch.read_uploaded_content(file.filename, content)

    if not queries:
        raise HTTPException(status_code=400, detail="No readable rows found in the uploaded file.")

    job_id = batch.create_job(total_rows=len(queries))
    background_tasks.add_task(batch.run_batch_job, job_id, queries, enable_live_fallback, enable_llm_judge)

    return {"job_id": job_id, "total_rows": len(queries)}


@app.get("/api/batch/{job_id}/status")
def batch_status(job_id: str):
    job = batch.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@app.post("/api/batch/{job_id}/cancel")
def cancel_batch(job_id: str):
    if not batch.cancel_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found or already finished")

    return {"cancelled": True}


@app.get("/api/batch/{job_id}/download")
def batch_download(job_id: str, format: str = "csv"):
    job = batch.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] not in ("done", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Job is not finished (status: {job['status']})")

    exporters_by_format = {
        "csv": (exporters.export_csv, "text/csv"),
        "xlsx": (exporters.export_excel, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "json": (exporters.export_json, "application/json"),
    }

    if format not in exporters_by_format:
        raise HTTPException(status_code=400, detail="format must be csv, xlsx, or json")

    export_fn, media_type = exporters_by_format[format]
    path = export_fn(job["rows"])

    return FileResponse(path, media_type=media_type, filename=path.name)


# ─────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────

@app.post("/api/cache/reset")
def reset_cache():
    size = query_agent.cache_size()
    query_agent.clear_cache()
    return {"cleared": size}


@app.get("/api/cache/size")
def get_cache_size():
    return {"size": query_agent.cache_size()}
