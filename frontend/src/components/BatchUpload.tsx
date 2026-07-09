import { useState } from "react";

interface BatchUploadProps {
  onSubmit: (file: File, enableLiveFallback: boolean) => void;
  disabled: boolean;
}

export default function BatchUpload({ onSubmit, disabled }: BatchUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [enableLiveFallback, setEnableLiveFallback] = useState(false);

  function handleRun() {
    if (!file) return;
    onSubmit(file, enableLiveFallback);
  }

  return (
    <section className="batch-upload">
      <h2>Batch File Processing</h2>

      <input
        type="file"
        accept=".csv,.txt"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
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
        ⚠️ When a row can't be confidently matched, this triggers a live Gemini web-search call
        per unmatched product — that's real API calls, so it costs tokens/money and adds latency.
        Leave off unless you specifically need to discover new products.
      </p>

      <button type="button" onClick={handleRun} disabled={disabled || !file}>
        {disabled ? "⏳ Processing..." : "▶ Run Batch"}
      </button>
    </section>
  );
}
