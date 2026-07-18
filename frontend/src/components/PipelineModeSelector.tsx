import type { PipelineMode } from "../api";

interface PipelineModeSelectorProps {
  mode: PipelineMode;
  onChange: (mode: PipelineMode) => void;
  disabled: boolean;
}

const OPTIONS: { value: PipelineMode; title: string; description: string }[] = [
  {
    value: "structured",
    title: "Structured data",
    description:
      "Verified product catalog with authoritative Product/Variant/SKU IDs, enriched with live-researched market descriptions. Recommended.",
  },
  {
    value: "raw",
    title: "Raw data",
    description:
      "Original pipeline: IDs inferred by matching live-researched product descriptions against historical prescription records only.",
  },
];

export default function PipelineModeSelector({ mode, onChange, disabled }: PipelineModeSelectorProps) {
  return (
    <section className="pipeline-mode-selector" role="radiogroup" aria-label="Matching method">
      <span className="pipeline-mode-label">Matching method</span>
      <div className="pipeline-mode-options">
        {OPTIONS.map((opt) => (
          <label
            key={opt.value}
            className={`pipeline-mode-card${mode === opt.value ? " selected" : ""}${disabled ? " disabled" : ""}`}
          >
            <input
              type="radio"
              name="pipeline-mode"
              value={opt.value}
              checked={mode === opt.value}
              onChange={() => onChange(opt.value)}
              disabled={disabled}
            />
            <span className="pipeline-mode-title">{opt.title}</span>
            <span className="pipeline-mode-description">{opt.description}</span>
          </label>
        ))}
      </div>
    </section>
  );
}
