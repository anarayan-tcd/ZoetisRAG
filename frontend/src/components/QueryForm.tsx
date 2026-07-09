import { useState } from "react";
import type { FormEvent } from "react";

interface QueryFormProps {
  onSubmit: (query: string, enableLiveFallback: boolean) => void;
  disabled: boolean;
}

export default function QueryForm({ onSubmit, disabled }: QueryFormProps) {
  const [query, setQuery] = useState("");
  const [enableLiveFallback, setEnableLiveFallback] = useState(false);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    onSubmit(query, enableLiveFallback);
  }

  return (
    <section className="query-form">
      <h2>Single Query</h2>
      <form onSubmit={handleSubmit}>
        <label htmlFor="query-input">Product Query</label>
        <textarea
          id="query-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. Frontline Spray 100ml dog"
          rows={2}
          disabled={disabled}
        />

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
          ⚠️ When nothing matches confidently, this triggers a live Gemini web-search call to
          research the product on the spot. That's a real API call — it costs tokens/money and
          adds latency. Leave off unless you specifically need to discover a new product.
        </p>

        <button type="submit" disabled={disabled}>
          {disabled ? "⏳ Working..." : "▶ Run"}
        </button>
      </form>
    </section>
  );
}
