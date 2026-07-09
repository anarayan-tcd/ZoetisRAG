// api.ts — plain fetch calls against the local FastAPI backend.
// This app runs locally only, so the base URL is just hardcoded.

import type { BatchJobStatus, QueryResponse } from "./types";

const API_BASE = "http://localhost:8000";

async function unwrap<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

export async function queryProduct(
  query: string,
  enableLiveFallback: boolean
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, enable_live_fallback: enableLiveFallback }),
  });

  return unwrap<QueryResponse>(response);
}

export async function uploadBatch(
  file: File,
  enableLiveFallback: boolean
): Promise<{ job_id: string; total_rows: number }> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(
    `${API_BASE}/api/batch/upload?enable_live_fallback=${enableLiveFallback}`,
    { method: "POST", body: formData }
  );

  return unwrap(response);
}

export async function getBatchStatus(jobId: string): Promise<BatchJobStatus> {
  const response = await fetch(`${API_BASE}/api/batch/${jobId}/status`);
  return unwrap<BatchJobStatus>(response);
}

export function batchDownloadUrl(jobId: string, format: "csv" | "xlsx" | "json"): string {
  return `${API_BASE}/api/batch/${jobId}/download?format=${format}`;
}

export async function resetCache(): Promise<{ cleared: number }> {
  const response = await fetch(`${API_BASE}/api/cache/reset`, { method: "POST" });
  return unwrap(response);
}
