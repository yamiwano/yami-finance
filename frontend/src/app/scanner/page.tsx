"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { ScannerRow } from "@/lib/types";
import { DirBadge, Panel, ScoreBar, Select } from "@/components/ui";
import { StrategyHint } from "@/components/StrategyHint";
import { fmtPct, fmtPx, fmtVol } from "@/lib/format";

export default function ScannerPage() {
  const [rows, setRows] = useState<ScannerRow[]>([]);
  const [q, setQ] = useState("");
  const [direction, setDirection] = useState("");
  const [strategy, setStrategy] = useState("");
  const [timeframe, setTimeframe] = useState("");
  const [minScore, setMinScore] = useState("0");
  const [sort, setSort] = useState<"score" | "symbol" | "change" | "volume">("score");
  const [strategies, setStrategies] = useState<{ id: string; label: string }[]>([]);

  useEffect(() => {
    api.meta().then((m) => setStrategies(m.strategies)).catch(() => undefined);
  }, []);

  useEffect(() => {
    const params = {
      q: q || undefined,
      direction: direction || undefined,
      strategy: strategy || undefined,
      timeframe: timeframe || undefined,
      min_score: Number(minScore) || 0,
    };
    api.scanner(params).then(setRows).catch(() => undefined);
    const t = setInterval(() => {
      api.scanner(params).then(setRows).catch(() => undefined);
    }, 5000);
    return () => clearInterval(t);
  }, [q, direction, strategy, timeframe, minScore]);

  const sorted = [...rows].sort((a, b) => {
    if (sort === "symbol") return a.symbol.localeCompare(b.symbol);
    if (sort === "change") return (b.change_pct || 0) - (a.change_pct || 0);
    if (sort === "volume") return (b.volume || 0) - (a.volume || 0);
    return (b.signal?.score || -1) - (a.signal?.score || -1);
  });

  return (
    <div className="p-4 space-y-4">
      <header>
        <div className="text-[11px] uppercase tracking-[0.18em] text-mute">Universe</div>
        <h1 className="text-xl font-semibold">Binance USDT scanner</h1>
        <p className="text-[11px] text-mute mt-1">Last price, 24h quote volume, and bid/ask spread from Binance spot.</p>
      </header>
      <div className="flex flex-wrap gap-3 items-center">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search pair"
          className="bg-panel border border-line rounded px-2 py-1 text-xs w-40"
        />
        <Select label="Dir" value={direction} onChange={setDirection} options={[{ value: "", label: "All" }, { value: "LONG", label: "Long" }, { value: "SHORT", label: "Short" }, { value: "WATCH", label: "Watch" }]} />
        <Select
          label="Strategy"
          value={strategy}
          onChange={setStrategy}
          options={[{ value: "", label: "All" }, ...strategies.map((s) => ({ value: s.id, label: s.label }))]}
        />
        <Select label="TF" value={timeframe} onChange={setTimeframe} options={[{ value: "", label: "All" }, { value: "5m", label: "5m" }, { value: "15m", label: "15m" }, { value: "1h", label: "1h" }]} />
        <label className="text-[11px] text-mute flex items-center gap-2">
          Min score
          <input
            type="number"
            min={0}
            max={100}
            value={minScore}
            onChange={(e) => setMinScore(e.target.value)}
            className="bg-panel border border-line rounded px-2 py-1 text-xs w-16"
          />
        </label>
        <Select
          label="Sort"
          value={sort}
          onChange={(v) => setSort(v as "score" | "symbol" | "change" | "volume")}
          options={[
            { value: "score", label: "Score" },
            { value: "symbol", label: "Symbol" },
            { value: "change", label: "Change" },
            { value: "volume", label: "24h vol" },
          ]}
        />
      </div>
      <Panel>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-mute text-[10px] uppercase tracking-wider">
              <tr className="border-b border-line">
                <th className="text-left px-3 py-2">Pair</th>
                <th className="text-left px-2 py-2">Name</th>
                <th className="text-right px-2 py-2">Last</th>
                <th className="text-right px-2 py-2">Chg 24h</th>
                <th className="text-right px-2 py-2">24h USDT</th>
                <th className="text-right px-2 py-2">Spread</th>
                <th className="text-left px-2 py-2">Signal</th>
                <th className="text-left px-2 py-2">Strategy</th>
                <th className="text-left px-2 py-2">TF</th>
                <th className="text-left px-2 py-2">Score</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr key={r.symbol} className="border-b border-line/70 hover:bg-line/40">
                  <td className="px-3 py-2 font-mono">
                    {r.signal ? (
                      <Link className="text-accent hover:underline" href={`/signals/${r.signal.id}`}>
                        {r.symbol}
                      </Link>
                    ) : (
                      r.symbol
                    )}
                  </td>
                  <td className="px-2 py-2 text-mute">{r.name}</td>
                  <td className="px-2 py-2 text-right font-mono">{fmtPx(r.price)}</td>
                  <td className={`px-2 py-2 text-right font-mono ${(r.change_pct || 0) >= 0 ? "text-long" : "text-short"}`}>
                    {fmtPct(r.change_pct)}
                  </td>
                  <td className="px-2 py-2 text-right font-mono text-mute">{fmtVol(r.volume)}</td>
                  <td className="px-2 py-2 text-right font-mono text-mute">{r.spread_bps == null ? "—" : `${r.spread_bps.toFixed(1)} bps`}</td>
                  <td className="px-2 py-2">{r.signal ? <DirBadge d={r.signal.direction} /> : <span className="text-mute">—</span>}</td>
                  <td className="px-2 py-2">
                    <StrategyHint id={r.signal?.strategy} label={r.signal?.strategy_label} />
                  </td>
                  <td className="px-2 py-2 font-mono">{r.signal?.timeframe || "—"}</td>
                  <td className="px-2 py-2">{r.signal ? <ScoreBar score={r.signal.score} /> : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
