import { resetCache } from "../api";

interface CacheResetButtonProps {
  onToast: (type: "success" | "error" | "info", message: string) => void;
}

export default function CacheResetButton({ onToast }: CacheResetButtonProps) {
  async function handleClick() {
    try {
      const { cleared } = await resetCache();
      onToast("success", `🗑️ Cache cleared — ${cleared} entr${cleared === 1 ? "y" : "ies"} removed.`);
    } catch (err) {
      onToast("error", err instanceof Error ? err.message : "Failed to reset cache.");
    }
  }

  return (
    <button type="button" className="secondary" onClick={handleClick}>
      🗑️ Reset Cache
    </button>
  );
}
