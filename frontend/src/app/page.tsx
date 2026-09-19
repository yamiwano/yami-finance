"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useLiveSignals } from "@/hooks/useLive";
import { OpportunityTable } from "@/components/OpportunityTable";
import { LearnerPulse } from "@/components/LearnerPulse";
import { DirBadge, Panel, Select } from "@/components/ui";
import { fmtTime } from "@/lib/format";

export default function DashboardPage() {
  const [direction, setDirection] = useState("");
  const params = useMemo(
    () => ({ status: "open", direction: direction || undefined, limit: 40 }),
    [direction]
  );
  const { signals, activity, connected, scans, health } = useLiveSignals(params);
  const tape = health?.tape;
  const tapeLive = Boolean(connected && tape?.ws_connected !== false && (tape?.ready ?? health?.ready ?? true));

  return (
    <div className="p-4 space-y-4">
      <header className="flex items-end justify-between gap-4">
        <div>
          <div className="text-[11px] uppercase tracking-[0.18em] text-mute">Live book</div>
          <h1 className="text-xl font-semibold">Top crypto opportunities</h1>
          <p className="text-[11px] text-mute mt-1">
            Binance USDT spot · closed-candle scanner · last trade + bid/ask book. Not a broker.
          </p>
        </div>
        <div className="flex items-center gap-3 text-[11px]">
          <span className={`font-mono ${tapeLive ? "text-long" : "text-short"}`}>
            {tapeLive ? "BINANCE LIVE" : "TAPE OFFLINE"}
          </span>
          <span className="text-mute">{tape?.universe ?? "—"} pairs</span>
          <span className="text-mute">scans {scans}</span>
          <Select
            label="Side"
            value={direction}
            onChange={setDirection}
            options={[
              { value: "", label: "All" },
              { value: "LONG", label: "Long" },
              { value: "SHORT", label: "Short" },
              { value: "WATCH", label: "Watch" },
            ]}
          />
        </div>
      </header>

      <div className="grid grid-cols-12 gap-4">
        <Panel title="Ranked setups" className="col-span-12 xl:col-span-8" right={<span className="text-[10px] text-mute">{signals.length} open</span>}>
          <OpportunityTable rows={signals} />
        </Panel>
        <div className="col-span-12 xl:col-span-4 space-y-4">
          {health?.learner && <LearnerPulse learner={health.learner} compact />}
          <Panel title="Scanner activity">
            <ul className="max-h-[420px] overflow-auto divide-y divide-line">
              {activity.length === 0 && <li className="px-3 py-6 text-xs text-mute">Waiting for first closed-candle scan…</li>}
              {activity.map((a, i) => (
                <li key={`${a.ts}-${i}`} className="px-3 py-2 text-xs">
                  <div className="flex justify-between text-mute">
                    <span className={`uppercase ${a.kind === "learner" ? "text-accent" : ""}`}>{a.kind}</span>
                    <span>{fmtTime(a.ts)}</span>
                  </div>
                  <div className="mt-0.5">
                    {a.id ? (
                      <Link href={`/signals/${a.id}`} className="hover:text-accent">
                        {a.message}
                      </Link>
                    ) : (
                      a.message
                    )}
                  </div>
                  {a.kind === "learner" && (a.changes || []).length > 0 && (
                    <ul className="mt-1 space-y-0.5 text-[11px] text-mute">
                      {a.changes!.slice(0, 3).map((c) => (
                        <li key={`${c.kind}-${c.strategy || ""}-${c.key}`}>
                          {c.strategy_label ? `${c.strategy_label}: ` : ""}
                          {c.label} {c.prior_text} → {c.live_text}
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
              ))}
            </ul>
          </Panel>
          <Panel title="Tape snapshot">
            <div className="p-3 grid grid-cols-2 gap-2">
              {["LONG", "SHORT", "WATCH", "AVOID"].map((d) => (
                <div key={d} className="border border-line rounded p-2">
                  <DirBadge d={d} />
                  <div className="font-mono text-lg mt-1">{signals.filter((s) => s.direction === d).length}</div>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
