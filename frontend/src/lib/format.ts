export function fmtPx(n: number | null | undefined, d = 4): string {
  if (n == null || Number.isNaN(n)) return "—";
  const abs = Math.abs(n);
  const digits = abs >= 1000 ? 2 : abs >= 10 ? 3 : d;
  return n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtNum(n: number | null | undefined, d = 2): string {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: 0 });
}

export function fmtVol(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  const abs = Math.abs(n);
  if (abs >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `$${(n / 1e3).toFixed(0)}K`;
  return `$${n.toFixed(0)}`;
}

export function fmtPct(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

export function fmtTime(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

export function fmtAgo(iso?: string | null): string {
  if (!iso) return "never";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "—";
  const sec = Math.max(0, (Date.now() - t) / 1000);
  if (sec < 45) return "just now";
  if (sec < 3600) return `${Math.max(1, Math.round(sec / 60))}m ago`;
  if (sec < 86400) return `${Math.max(1, Math.round(sec / 3600))}h ago`;
  if (sec < 86400 * 7) return `${Math.max(1, Math.round(sec / 86400))}d ago`;
  return fmtTime(iso);
}

export function dirClass(d: string): string {
  if (d === "LONG") return "text-long";
  if (d === "SHORT") return "text-short";
  if (d === "WATCH") return "text-watch";
  return "text-avoid";
}

export function dirBg(d: string): string {
  if (d === "LONG") return "bg-long/15 text-long border-long/30";
  if (d === "SHORT") return "bg-short/15 text-short border-short/30";
  if (d === "WATCH") return "bg-watch/15 text-watch border-watch/30";
  return "bg-avoid/15 text-avoid border-avoid/30";
}

export function statusLabel(s: string): string {
  return s.replaceAll("_", " ");
}
