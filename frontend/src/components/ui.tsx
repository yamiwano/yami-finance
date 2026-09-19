import { dirBg } from "@/lib/format";

export function Badge({ children, tone }: { children: React.ReactNode; tone?: string }) {
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded border text-[10px] font-mono uppercase tracking-wide ${tone || "border-line text-mute"}`}>
      {children}
    </span>
  );
}

export function DirBadge({ d }: { d: string }) {
  return <Badge tone={dirBg(d)}>{d}</Badge>;
}

export function ScoreBar({ score }: { score: number }) {
  const w = Math.max(0, Math.min(100, score));
  const color = score >= 75 ? "#3dd68c" : score >= 60 ? "#e7c547" : "#f0616d";
  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <div className="h-1.5 flex-1 bg-line rounded overflow-hidden">
        <div className="h-full" style={{ width: `${w}%`, background: color }} />
      </div>
      <span className="font-mono text-xs w-8 text-right">{Math.round(score)}</span>
    </div>
  );
}

export function Panel({ title, right, children, className = "" }: { title?: string; right?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <section className={`bg-panel border border-line rounded ${className}`}>
      {(title || right) && (
        <header className="flex items-center justify-between px-3 py-2 border-b border-line">
          <h2 className="text-[11px] uppercase tracking-[0.16em] text-mute">{title}</h2>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Select({
  value,
  onChange,
  options,
  label,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  label?: string;
}) {
  return (
    <label className="text-[11px] text-mute flex items-center gap-2">
      {label}
      <select
        className="bg-panel2 border border-line rounded px-2 py-1 text-ink text-xs"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
