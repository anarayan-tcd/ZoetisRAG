import type { UsageStats } from "../types";

function formatSeconds(s: number): string {
  if (s < 1) return `${Math.round(s * 1000)}ms`;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s % 60)}s`;
}

function formatCost(usd: number): string {
  if (usd === 0) return "$0.00";
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}

interface QueryUsageSummaryProps {
  usage: UsageStats;
}

export function QueryUsageSummary({ usage }: QueryUsageSummaryProps) {
  return (
    <div className="usage-summary" title="Estimated DeepSeek cost — priced per-call from DeepSeek's actual cache-hit/cache-miss token split, so it should track the real invoice closely">
      <span>⏱ {formatSeconds(usage.elapsed_seconds)}</span>
      <span>·</span>
      <span>{usage.llm_calls} LLM call{usage.llm_calls === 1 ? "" : "s"}</span>
      <span>·</span>
      <span>{usage.total_tokens.toLocaleString()} tokens</span>
      <span>·</span>
      <span>~{formatCost(usage.estimated_cost_usd)}</span>
    </div>
  );
}

interface BatchUsageSummaryProps {
  elapsed: number;
  avgSecondsPerRow: number;
  totalTokens: number;
  estimatedCostUsd: number;
}

export function BatchUsageSummary({ elapsed, avgSecondsPerRow, totalTokens, estimatedCostUsd }: BatchUsageSummaryProps) {
  return (
    <div className="usage-summary" title="Estimated DeepSeek cost — priced per-call from DeepSeek's actual cache-hit/cache-miss token split, so it should track the real invoice closely">
      <span>⏱ {formatSeconds(elapsed)} total</span>
      <span>·</span>
      <span>~{formatSeconds(avgSecondsPerRow)}/row</span>
      <span>·</span>
      <span>{totalTokens.toLocaleString()} tokens</span>
      <span>·</span>
      <span>~{formatCost(estimatedCostUsd)}</span>
    </div>
  );
}
