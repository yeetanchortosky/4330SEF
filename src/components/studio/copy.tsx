import { useState } from "react";
import { Check, Copy } from "lucide-react";

export function CopyButton({ text, label }: { text: string; label: string }) {
  const [on, setOn] = useState(false);
  return (
    <button
      type="button"
      className="inline-flex h-11 items-center gap-2 rounded-md border border-line bg-surface px-3 text-sm font-semibold text-fg"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setOn(true);
          window.setTimeout(() => setOn(false), 1400);
        } catch {
          setOn(false);
        }
      }}
    >
      {on ? <Check className="size-4 text-copper" /> : <Copy className="size-4" />}
      {on ? "Copied" : label}
    </button>
  );
}
