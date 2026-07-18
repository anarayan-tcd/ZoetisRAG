import MatchCard from "./MatchCard";
import IdCombinationsCard from "./IdCombinationsCard";
import { BatchUsageSummary, QueryUsageSummary } from "./UsageSummary";
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
        <QueryUsageSummary usage={queryResult.usage} />
        <MatchCard match={queryResult.product_match} />
        <IdCombinationsCard
          combinations={queryResult.id_combinations}
          reason={queryResult.id_combinations_reason}
        />
      </div>
    );
  }

  // mode === "batch"
  if (!batchStatus) return null;

  if (batchStatus.status === "error") {
    return <p className="error-banner">⚠️ {batchStatus.error}</p>;
  }

  const isFinished = batchStatus.status === "done" || batchStatus.status === "cancelled";

  if (!isFinished || !batchStatus.stats || !batchJobId) {
    return null;
  }

  return (
    <div className="batch-summary">
      <p>
        <strong>{batchStatus.status === "cancelled" ? "Cancelled" : "Done"}</strong> —{" "}
        {batchStatus.stats.total_rows} row(s), {batchStatus.stats.unique_queries} unique,{" "}
        {batchStatus.stats.cache_hits} served from cache, {batchStatus.stats.llm_calls} reasoned
        by DeepSeek — {batchStatus.stats.elapsed}s
      </p>
      <BatchUsageSummary
        elapsed={batchStatus.stats.elapsed}
        avgSecondsPerRow={batchStatus.stats.avg_seconds_per_resolved_row}
        totalTokens={batchStatus.stats.total_tokens}
        estimatedCostUsd={batchStatus.stats.estimated_cost_usd}
      />
      <div className="download-buttons">
        <a href={batchDownloadUrl(batchJobId, "csv")}>Download CSV</a>
        <a href={batchDownloadUrl(batchJobId, "xlsx")}>Download Excel</a>
        <a href={batchDownloadUrl(batchJobId, "json")}>Download JSON</a>
      </div>
    </div>
  );
}
