"use client";

import { strategyDescription } from "@/lib/strategies";

export function StrategyHint({
  id,
  label,
}: {
  id?: string | null;
  label?: string | null;
}) {
  const text = label || id?.replaceAll("_", " ") || null;
  const blurb = strategyDescription(id);

  if (!text) return <span className="text-mute">—</span>;
  if (!blurb) return <>{text}</>;

  return (
    <span className="relative inline-flex">
      <span className="peer cursor-help border-b border-dotted border-mute/50">{text}</span>
      <span
        role="tooltip"
        className="pointer-events-none invisible absolute left-1/2 top-[calc(100%+6px)] z-50 w-60 -translate-x-1/2 rounded border border-line bg-panel2 px-2.5 py-2 text-left text-[11px] font-sans font-normal normal-case leading-snug tracking-normal text-ink shadow-lg peer-hover:visible"
      >
        {blurb}
      </span>
    </span>
  );
}
