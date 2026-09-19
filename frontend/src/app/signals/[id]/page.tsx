"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { api } from "@/lib/api";
import type { ChartPayload, Signal } from "@/lib/types";
import { RadarChart } from "@/components/RadarChart";
import { DirBadge, Panel, ScoreBar } from "@/components/ui";
import { fmtPx, fmtTime } from "@/lib/format";

const WEIGHTS = [
  { key: "technical_structure", label: "Technical structure", prior: 0.3 },
  { key: "volume_activity", label: "Volume / activity", prior: 0.2 },
  { key: "market_context", label: "Market context", prior: 0.15 },
  { key: "setup_quality", label: "Setup quality / R:R", prior: 0.2 },
  { key: "catalyst_context", label: "Catalyst / news", prior: 0.15 },
] as const;

export default function SignalDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [signal, setSignal] = useState<Signal | null>(null);
  const [chart, setChart] = useState<ChartPayload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [watching, setWatching] = useState(false);

  useEffect(() => {
    if (!id) return;
    Promise.all([api.signal(id), api.signalChart(id)])
      .then(([s, c]) => {
        setSignal(s);
        setChart(c);
      })
      .catch((e) => setErr(String(e)));
    const t = setInterval(() => {
      api.signal(id).then(setSignal).catch(() => undefined);
      api.signalChart(id).then(setChart).catch(() => undefined);
    }, 8000);
    return () => clearInterval(t);
  }, [id]);

  if (err) return <div className="p-6 text-short">{err}</div>;
  if (!signal || !chart) return <div className="p-6 text-mute">Loading setup…</div>;
  const c = signal.score_components;

  return (
    <div className="p-4 space-y-4">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-[11px] uppercase tracking-[0.18em] text-mute">Signal detail</div>
          <h1 className="text-2xl font-semibold font-mono flex items-center gap-3">
            {signal.symbol}
            <DirBadge d={signal.direction} />
            <span className="text-sm font-sans text-mute">{signal.strategy_label}</span>
          </h1>
          <div className="text-xs text-mute mt-1">
            {signal.timeframe} · Binance spot · detected {fmtTime(signal.detected_at)} · {signal.status.replaceAll("_", " ")}
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div>
            <div className="text-[10px] text-mute uppercase">Opportunity score</div>
            <div className="w-40">
              <ScoreBar score={signal.score} />
            </div>
            {c?.raw_total != null && Math.abs(c.raw_total - signal.score) >= 1 && (
              <div className="text-[10px] text-mute mt-1 font-mono">
                base {Math.round(c.raw_total)} → calibrated {Math.round(signal.score)}
              </div>
            )}
          </div>
          <button
            className="text-xs border border-line rounded px-3 py-1.5 hover:border-accent"
            onClick={async () => {
              await api.addWatch(signal.symbol);
              setWatching(true);
            }}
          >
            {watching ? "On watchlist" : "Add to watchlist"}
          </button>
        </div>
      </header>

      <div className="grid grid-cols-12 gap-4">
        <Panel title={`${signal.symbol} ${signal.timeframe} · EMA / VWAP / volume / RSI`} className="col-span-12 xl:col-span-8">
          <RadarChart chart={chart} signal={signal} />
        </Panel>
        <div className="col-span-12 xl:col-span-4 space-y-4">
          <Panel title="Hypothetical levels (not guaranteed)">
            <dl className="grid grid-cols-2 gap-x-3 gap-y-2 p-3 text-xs">
              {[
                ["Last", fmtPx(signal.current_price)],
                ["Bid / ask", `${fmtPx((signal.snapshot as { bid?: number }).bid)} / ${fmtPx((signal.snapshot as { ask?: number }).ask)}`],
                ["Entry zone", `${fmtPx(signal.entry_low)} – ${fmtPx(signal.entry_high)}`],
                ["Stop", fmtPx(signal.stop)],
                ["Invalidation", fmtPx(signal.invalidation)],
                ["Target 1", fmtPx(signal.target_1)],
                ["Target 2", fmtPx(signal.target_2)],
                ["R:R to T1", `${signal.risk_reward?.toFixed(2)}R`],
                ["R multiple", signal.r_multiple == null ? "—" : `${signal.r_multiple.toFixed(2)}R`],
              ].map(([k, v]) => (
                <div key={k}>
                  <dt className="text-mute">{k}</dt>
                  <dd className="font-mono">{v}</dd>
                </div>
              ))}
            </dl>
          </Panel>
          <Panel title="Score breakdown">
            <div className="p-3 space-y-2">
              {WEIGHTS.map((w) => {
                const live = c?.learner_weights?.[w.key];
                const pct = `${Math.round((live ?? w.prior) * 100)}%`;
                const shifted = live != null && Math.abs(live - w.prior) >= 0.02;
                return (
                  <div key={w.key}>
                    <div className="flex justify-between text-[11px] text-mute">
                      <span>
                        {w.label}{" "}
                        <span className="opacity-60">
                          ({pct}
                          {shifted ? ` · was ${Math.round(w.prior * 100)}%` : ""})
                        </span>
                      </span>
                      <span className="font-mono text-ink">{Number(c?.[w.key] ?? 0).toFixed(1)}</span>
                    </div>
                    <div className="h-1 bg-line rounded mt-1">
                      <div className="h-full bg-accent/80 rounded" style={{ width: `${Number(c?.[w.key] ?? 0)}%` }} />
                    </div>
                  </div>
                );
              })}
              <div className="flex justify-between text-[11px] pt-1 border-t border-line">
                <span className="text-mute">Penalties</span>
                <span className="font-mono text-short">{Number(c?.penalties ?? 0).toFixed(1)}</span>
              </div>
              {(c?.penalty_reasons || []).map((p) => (
                <div key={p} className="text-[11px] text-short/90">
                  {p}
                </div>
              ))}
              {c?.learner_note && (
                <div className="text-[11px] text-mute pt-2 border-t border-line leading-relaxed">{c.learner_note}</div>
              )}
              {(c?.learner_tweaks || []).length > 0 && (
                <ul className="text-[11px] text-mute space-y-1 pt-1">
                  {(c.learner_tweaks || []).map((t) => (
                    <li key={t.key}>
                      {t.label}: {t.prior_text ?? String(t.prior)} → {t.live_text ?? String(t.live)}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </Panel>
        </div>

        <Panel title="Why this fired" className="col-span-12 xl:col-span-6">
          <ul className="p-3 space-y-2 text-xs list-disc pl-6">
            {signal.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
          {signal.evidence && (
            <pre className="text-[10px] text-mute bg-bg m-3 p-2 rounded overflow-auto max-h-40">
              {JSON.stringify(signal.evidence, null, 2)}
            </pre>
          )}
        </Panel>
        <Panel title="Risk flags" className="col-span-12 xl:col-span-6">
          <div className="p-3 space-y-2">
            {(signal.risk_flags || []).length === 0 && <div className="text-xs text-mute">No hard risk flags on this snapshot.</div>}
            {(signal.risk_flags || []).map((f, i) => (
              <div key={i} className="border border-line rounded px-2 py-1.5 text-xs">
                <span className="font-mono text-watch mr-2">{f.severity}</span>
                <span className="font-mono mr-2">{f.code}</span>
                {f.detail}
              </div>
            ))}
          </div>
        </Panel>
        <Panel
          title="AI explanation (facts-only, does not change the score)"
          className="col-span-12"
          right={<span className="text-[10px] text-mute">{signal.ai_explanation?.provider || "pending"}</span>}
        >
          {signal.ai_explanation ? (
            <div className="p-3 grid md:grid-cols-2 gap-4 text-xs">
              <div>
                <div className="text-mute uppercase text-[10px] mb-1">Thesis</div>
                <p>{signal.ai_explanation.thesis}</p>
                <div className="text-mute uppercase text-[10px] mt-3 mb-1">Evidence</div>
                <ul className="list-disc pl-4 space-y-1">
                  {(signal.ai_explanation.evidence || []).map((e, i) => (
                    <li key={i}>{e}</li>
                  ))}
                </ul>
              </div>
              <div>
                <div className="text-mute uppercase text-[10px] mb-1">Bear case</div>
                <p>{signal.ai_explanation.bear_case}</p>
                <div className="text-mute uppercase text-[10px] mt-3 mb-1">What to watch</div>
                <p>{signal.ai_explanation.what_to_watch}</p>
                <div className="text-mute uppercase text-[10px] mt-3 mb-1">Summary</div>
                <p className="text-ink/90">{signal.ai_explanation.summary}</p>
              </div>
            </div>
          ) : (
            <div className="p-3 text-xs text-mute">
              No AI note yet. Explanations run only on new or materially changed high-score setups, and never invent numbers.
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
