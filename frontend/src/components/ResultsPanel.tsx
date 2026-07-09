import MatchCard from "./MatchCard";
import { batchDownloadUrl } from "../api";
import type { BatchJobStatus, QueryResponse } from "../types";

export type ResultMode = "idle" | "query" | "batch";

interface ResultsPanelProps {
  mode: ResultMode;
  queryResult: QueryResponse | null;
  batchStatus: BatchJobStatus | null;
  batchJobId: string | null;
}

export default function ResultsPanel({ mode, queryResult, batchStatus, batchJobId }: ResultsPanelProps) {
  if (mode === "idle") {
    return (
      <div className="results-placeholder">
        Run a single query or upload a file on the left to see results here.
      </div>
    );
  }

  if (mode === "query") {
    if (!queryResult) return null;

    return (
      <div className="results">
        <MatchCard title="Primary Match" match={queryResult.primary_match} />
        {queryResult.alternative_matches.map((alt, i) => (
          <MatchCard key={i} title={`Alternative ${i + 1}`} match={alt} />
        ))}
      </div>
    );
  }

  // mode === "batch"
  if (!batchStatus) return null;

  if (batchStatus.status === "error") {
    return <p className="error-banner">⚠️ {batchStatus.error}</p>;
  }

  if (batchStatus.status !== "done" || !batchStatus.stats || !batchJobId) {
    return null;
  }

  return (
    <div className="batch-summary">
      <p>
        <strong>Done</strong> — {batchStatus.stats.total_rows} row(s),{" "}
        {batchStatus.stats.unique_queries} unique, {batchStatus.stats.cache_hits} served from
        cache, {batchStatus.stats.llm_calls} reasoned by DeepSeek — {batchStatus.stats.elapsed}s
      </p>
      <div className="download-buttons">
        <a href={batchDownloadUrl(batchJobId, "csv")}>Download CSV</a>
        <a href={batchDownloadUrl(batchJobId, "xlsx")}>Download Excel</a>
        <a href={batchDownloadUrl(batchJobId, "json")}>Download JSON</a>
      </div>
    </div>
  );
}
