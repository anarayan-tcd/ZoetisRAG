import { useCallback, useState } from "react";

export interface ToastMessage {
  id: number;
  type: "success" | "error" | "info";
  message: string;
}

let nextToastId = 1;

export function useToasts() {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const pushToast = useCallback(
    (type: ToastMessage["type"], message: string) => {
      const id = nextToastId++;
      setToasts((prev) => [...prev, { id, type, message }]);
      window.setTimeout(() => dismissToast(id), 4500);
    },
    [dismissToast]
  );

  return { toasts, pushToast, dismissToast };
}

const ICONS: Record<ToastMessage["type"], string> = {
  success: "✅",
  error: "⚠️",
  info: "ℹ️",
};

export function ToastContainer({
  toasts,
  onDismiss,
}: {
  toasts: ToastMessage[];
  onDismiss: (id: number) => void;
}) {
  if (toasts.length === 0) return null;

  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`toast toast-${t.type}`}
          onClick={() => onDismiss(t.id)}
          role="status"
        >
          <span>{ICONS[t.type]}</span>
          <span>{t.message}</span>
        </div>
      ))}
    </div>
  );
}
