import { useState } from "react";
import type { FormEvent } from "react";

interface QueryFormProps {
  onSubmit: (query: string, enableLiveFallback: boolean, enableLlmJudge: boolean) => void;
  onStop: () => void;
  disabled: boolean;
}

export default function QueryForm({ onSubmit, onStop, disabled }: QueryFormProps) {
  const [query, setQuery] = useState("");
  const [enableLiveFallback, setEnableLiveFallback] = useState(false);
  const [enableLlmJudge, setEnableLlmJudge] = useState(false);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    onSubmit(query, enableLiveFallback, enableLlmJudge);
  }

  return (
    <section className="query-form form-card">
      <h2>Single Query</h2>
      <form onSubmit={handleSubmit}>
        <div className="field">
          <label className="field-label" htmlFor="query-input">
            Product Query
          </label>
          <textarea
            id="query-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. Frontline Spray 100ml dog"
            rows={2}
            disabled={disabled}
          />
        </div>

        <div className="option-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={enableLiveFallback}
              onChange={(e) => setEnableLiveFallback(e.target.checked)}
              disabled={disabled}
            />
            Enable live web research fallback
          </label>
          <p className="checkbox-disclaimer">
            ⚠️ Triggers a live Gemini web-search call per unmatched product. Costs tokens/money
            and adds latency.
          </p>
        </div>

        <div className="option-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={enableLlmJudge}
              onChange={(e) => setEnableLlmJudge(e.target.checked)}
              disabled={disabled}
            />
            Enable LLM-as-judge review
          </label>
          <p className="checkbox-disclaimer">
            ⚠️ Adds a second DeepSeek check on High/Medium-confidence matches — roughly +50% more
            LLM calls. Costs more and takes longer.
          </p>
        </div>

        <div className="run-row">
          <button type="submit" className="run-button" disabled={disabled}>
            {disabled ? "⏳ Working..." : "▶ Run"}
          </button>
          <button
            type="button"
            className="stop-button"
            onClick={onStop}
            disabled={!disabled}
            title="Stop processing"
            aria-label="Stop processing"
          >
            ⏹
          </button>
        </div>
      </form>
    </section>
  );
}
