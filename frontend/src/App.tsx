import { useRef, useState } from "react";
import QueryForm from "./components/QueryForm";
import BatchUpload from "./components/BatchUpload";
import CacheResetButton from "./components/CacheResetButton";
import ResultsPanel from "./components/ResultsPanel";
import type { ResultMode } from "./components/ResultsPanel";
import ProgressBar from "./components/ProgressBar";
import HowToSection from "./components/HowToSection";
import { ToastContainer, useToasts } from "./components/Toast";
import { getBatchStatus, queryProduct, uploadBatch } from "./api";
import type { BatchJobStatus, QueryResponse } from "./types";
import "./App.css";

const QUERY_STEP_LABELS = [
  "Cleaning input...",
  "Searching product database...",
  "Identifying brand...",
  "Matching exact variant...",
];
const QUERY_STEP_INTERVAL_MS = 1200;
const BATCH_POLL_INTERVAL_MS = 1500;

function App() {
  const [mode, setMode] = useState<ResultMode>("idle");
  const [loading, setLoading] = useState(false);

  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [queryStepLabel, setQueryStepLabel] = useState(QUERY_STEP_LABELS[0]);
  const queryStepTimer = useRef<number | null>(null);

  const [batchStatus, setBatchStatus] = useState<BatchJobStatus | null>(null);
  const [batchJobId, setBatchJobId] = useState<string | null>(null);
  const batchPollTimer = useRef<number | null>(null);

  const { toasts, pushToast, dismissToast } = useToasts();

  async function handleQuerySubmit(query: string, enableLiveFallback: boolean) {
    if (!query.trim()) {
      pushToast("error", "Enter a product query before running.");
      return;
    }

    setMode("query");
    setLoading(true);
    setQueryResult(null);

    let stepIndex = 0;
    setQueryStepLabel(QUERY_STEP_LABELS[0]);
    queryStepTimer.current = window.setInterval(() => {
      stepIndex = Math.min(stepIndex + 1, QUERY_STEP_LABELS.length - 1);
      setQueryStepLabel(QUERY_STEP_LABELS[stepIndex]);
    }, QUERY_STEP_INTERVAL_MS);

    try {
      const result = await queryProduct(query, enableLiveFallback);
      setQueryResult(result);
      pushToast(
        "success",
        `Matched "${query.trim()}" → ${result.primary_match.brand || "Unknown"} (${result.primary_match.confidence || "—"})`
      );
    } catch (err) {
      pushToast("error", err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      if (queryStepTimer.current !== null) {
        window.clearInterval(queryStepTimer.current);
        queryStepTimer.current = null;
      }
      setLoading(false);
    }
  }

  function stopBatchPolling() {
    if (batchPollTimer.current !== null) {
      window.clearInterval(batchPollTimer.current);
      batchPollTimer.current = null;
    }
  }

  async function handleBatchSubmit(file: File, enableLiveFallback: boolean) {
    setMode("batch");
    setLoading(true);
    setBatchStatus(null);
    setBatchJobId(null);

    try {
      const { job_id, total_rows } = await uploadBatch(file, enableLiveFallback);
      setBatchJobId(job_id);
      pushToast("info", `Processing ${total_rows} row(s)...`);

      batchPollTimer.current = window.setInterval(async () => {
        try {
          const s = await getBatchStatus(job_id);
          setBatchStatus(s);

          if (s.status === "done") {
            stopBatchPolling();
            setLoading(false);
            pushToast("success", `Batch complete — ${s.stats?.total_rows ?? 0} row(s) processed.`);
          } else if (s.status === "error") {
            stopBatchPolling();
            setLoading(false);
            pushToast("error", s.error || "Batch processing failed.");
          }
        } catch (err) {
          stopBatchPolling();
          setLoading(false);
          pushToast("error", err instanceof Error ? err.message : "Polling failed.");
        }
      }, BATCH_POLL_INTERVAL_MS);
    } catch (err) {
      setLoading(false);
      pushToast("error", err instanceof Error ? err.message : "Upload failed.");
    }
  }

  const batchProgressPct =
    batchStatus && batchStatus.total > 0
      ? Math.round((batchStatus.completed / batchStatus.total) * 100)
      : 0;

  return (
    <main className="app">
      <h1>Veterinary Product Matching System</h1>

      <HowToSection />

      <ProgressBar
        active={loading}
        determinate={mode === "batch"}
        percent={batchProgressPct}
        label={
          mode === "batch"
            ? batchStatus
              ? `${batchStatus.completed}/${batchStatus.total} reasoned`
              : "Uploading..."
            : queryStepLabel
        }
      />

      <div className="layout">
        <div className="layout-col">
          <QueryForm onSubmit={handleQuerySubmit} disabled={loading} />
          <BatchUpload onSubmit={handleBatchSubmit} disabled={loading} />
          <CacheResetButton onToast={pushToast} />
        </div>

        <div className="layout-col">
          <ResultsPanel
            mode={mode}
            queryResult={queryResult}
            batchStatus={batchStatus}
            batchJobId={batchJobId}
          />
        </div>
      </div>

      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </main>
  );
}

export default App;
