import { useState } from "react";

interface BatchUploadProps {
  onSubmit: (file: File, enableLiveFallback: boolean, enableLlmJudge: boolean) => void;
  onStop: () => void;
  disabled: boolean;
}

export default function BatchUpload({ onSubmit, onStop, disabled }: BatchUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [enableLiveFallback, setEnableLiveFallback] = useState(false);
  const [enableLlmJudge, setEnableLlmJudge] = useState(false);

  function handleRun() {
    if (!file) return;
    onSubmit(file, enableLiveFallback, enableLlmJudge);
  }

  return (
    <section className="batch-upload form-card">
      <h2>Batch File Processing</h2>

      <div className="field">
        <label className="field-label" htmlFor="batch-file-input">
          CSV or TXT File
        </label>
        <input
          id="batch-file-input"
          type="file"
          accept=".csv,.txt"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
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
          ⚠️ Triggers a live Gemini web-search call per unmatched product. Costs tokens/money and
          adds latency.
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
          ⚠️ Adds a second DeepSeek check on High/Medium-confidence rows — roughly +50% more LLM
          calls across the file. Costs more and takes longer.
        </p>
      </div>

      <div className="run-row">
        <button type="button" className="run-button" onClick={handleRun} disabled={disabled || !file}>
          {disabled ? "⏳ Processing..." : "▶ Run Batch"}
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
    </section>
  );
}
