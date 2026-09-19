"use client";

import Link from "next/link";
import type { Signal } from "@/lib/types";
import { DirBadge, ScoreBar } from "./ui";
import { StrategyHint } from "./StrategyHint";
import { fmtPx, fmtTime } from "@/lib/format";

export function OpportunityTable({ rows }: { rows: Signal[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead className="text-mute text-[10px] uppercase tracking-wider">
          <tr className="border-b border-line">
            <th className="text-left px-3 py-2 font-medium">Pair</th>
            <th className="text-left px-2 py-2 font-medium">Dir</th>
            <th className="text-right px-2 py-2 font-medium">Entry Point</th>
            <th className="text-left px-2 py-2 font-medium">Strategy</th>
            <th className="text-left px-2 py-2 font-medium">TF</th>
            <th className="text-left px-2 py-2 font-medium">Score</th>
            <th className="text-right px-2 py-2 font-medium">Px</th>
            <th className="text-right px-2 py-2 font-medium">R:R</th>
            <th className="text-right px-2 py-2 font-medium">Detected</th>
            <th className="text-left px-2 py-2 font-medium">Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr>
              <td colSpan={10} className="px-3 py-8 text-center text-mute">
                No qualifying setups. Most pairs carry no signal by design.
              </td>
            </tr>
          )}
          {rows.map((s) => (
            <tr key={s.id} className="border-b border-line/70 hover:bg-line/40">
              <td className="px-3 py-2 font-mono">
                <Link href={`/signals/${s.id}`} className="text-accent hover:underline">
                  {s.symbol}
                </Link>
              </td>
              <td className="px-2 py-2">
                <DirBadge d={s.direction} />
              </td>
              <td className="px-2 py-2 text-right font-mono">
                {s.entry_low || s.entry_high ? fmtPx((s.entry_low + s.entry_high) / 2) : "—"}
              </td>
              <td className="px-2 py-2">
                <StrategyHint id={s.strategy} label={s.strategy_label} />
              </td>
              <td className="px-2 py-2 font-mono">{s.timeframe}</td>
              <td className="px-2 py-2">
                <ScoreBar score={s.score} />
              </td>
              <td className="px-2 py-2 text-right font-mono">{fmtPx(s.current_price)}</td>
              <td className="px-2 py-2 text-right font-mono">{s.risk_reward?.toFixed(2)}R</td>
              <td className="px-2 py-2 text-right text-mute">{fmtTime(s.detected_at)}</td>
              <td className="px-2 py-2 text-mute">{s.status.replaceAll("_", " ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
