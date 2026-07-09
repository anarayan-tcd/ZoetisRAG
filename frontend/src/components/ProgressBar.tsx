// ProgressBar — one shared bar for both flows. Batch processing reports
// real completed/total counts from the backend, so that renders as a
// determinate fill. A single query is one synchronous request with no
// intermediate signal from the backend, so it renders as an indeterminate
// sliding bar with a cycling step label instead of a fabricated percentage.

interface ProgressBarProps {
  active: boolean;
  determinate: boolean;
  percent?: number;
  label: string;
}

export default function ProgressBar({ active, determinate, percent = 0, label }: ProgressBarProps) {
  if (!active) return null;

  return (
    <div
      className="progress-bar"
      role="progressbar"
      aria-valuenow={determinate ? percent : undefined}
    >
      <div
        className={determinate ? "progress-bar-fill" : "progress-bar-fill indeterminate"}
        style={determinate ? { width: `${percent}%` } : undefined}
      />
      <span className="progress-bar-label">{label}</span>
    </div>
  );
}
