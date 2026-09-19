"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Research } from "@/lib/types";
import { Panel } from "@/components/ui";
import { fmtPct } from "@/lib/format";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-line rounded p-3">
      <div className="text-[10px] uppercase tracking-wider text-mute">{label}</div>
      <div className="font-mono text-2xl mt-1">{value}</div>
    </div>
  );
}

function fmtNum(value: number | null | undefined, digits = 3) {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export default function ResearchPage() {
  const [r, setR] = useState<Research | null>(null);
  const [training, setTraining] = useState(false);
  useEffect(() => {
    api.research().then(setR).catch(() => undefined);
    const t = setInterval(() => api.research().then(setR).catch(() => undefined), 8000);
    return () => clearInterval(t);
  }, []);

  async function train() {
    setTraining(true);
    try {
      await api.trainResearch();
      setR(await api.research());
    } catch {
      /* keep polling */
    } finally {
      setTraining(false);
    }
  }

  if (!r) return <div className="p-6 text-mute">Loading research loop…</div>;
  const model = r.active_model || r.last_run;
  const metrics = model?.metrics;
  const pos = r.samples.positive_rate == null ? "—" : fmtPct(r.samples.positive_rate * 100);
  const livePos = r.live.positive_rate == null ? "—" : fmtPct(r.live.positive_rate * 100);

  return (
    <div className="p-4 space-y-4">
      <header className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[11px] uppercase tracking-[0.18em] text-mute">Historical testing</div>
          <h1 className="text-xl font-semibold">Research</h1>
          <p className="text-xs text-mute mt-1 max-w-2xl">
            Market state at time T (closed 1h bars only) → what happened over the next {r.samples.horizon_hours}h.
            The model is kept only if walk-forward metrics beat a constant base-rate baseline. This does not predict prices.
          </p>
          <p className="text-[11px] text-mute mt-1.5 font-mono">
            {r.status.phase} · {r.status.message}
          </p>
          {r.status.error && <p className="text-[11px] text-short mt-1 font-mono">{r.status.error}</p>}
        </div>
        <button
          type="button"
          className="text-xs border border-line rounded px-3 py-1.5 hover:border-accent shrink-0"
          onClick={train}
          disabled={training || r.status.busy}
        >
          {training || r.status.busy ? "Working…" : "Train now"}
        </button>
      </header>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="1h candles" value={String(r.candles.count)} />
        <Stat label="Labeled samples" value={`${r.samples.labeled} / ${r.samples.count}`} />
        <Stat label={`Move ≥ ${fmtPct(r.samples.move_pct * 100)} rate`} value={pos} />
        <Stat label="Live resolved" value={String(r.live.resolved)} />
      </div>

      <Panel title="What this measures">
        <p className="px-3 py-3 text-xs text-mute leading-relaxed">{r.disclaimer}</p>
      </Panel>

      <div className="grid md:grid-cols-2 gap-4">
        <Panel
          title="Walk-forward"
          right={
            <span className="text-[10px] text-mute">
              {model ? `${model.status} · ${metrics?.fold_count ?? 0} folds` : "no run yet"}
            </span>
          }
        >
          <div className="grid grid-cols-2 gap-3 p-3">
            <Stat label="Mean AUC" value={fmtNum(metrics?.mean_auc)} />
            <Stat label="Mean Brier" value={fmtNum(metrics?.mean_brier)} />
            <Stat label="Baseline Brier" value={fmtNum(metrics?.mean_baseline_brier)} />
            <Stat label="Train samples" value={metrics?.train_samples != null ? String(metrics.train_samples) : "—"} />
          </div>
          <table className="w-full text-xs">
            <thead>
              <tr className="text-mute text-[10px] uppercase">
                {["Fold", "N", "AUC", "Brier", "Baseline", "Lift"].map((h) => (
                  <th key={h} className="text-left px-3 py-2 font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(metrics?.folds || []).length === 0 && (
                <tr>
                  <td className="px-3 py-6 text-mute" colSpan={6}>
                    Need enough dated 1h history for expanding train / embargo / test windows.
                  </td>
                </tr>
              )}
              {(metrics?.folds || []).map((fold, i) => (
                <tr key={i} className="border-t border-line">
                  <td className="px-3 py-2">{i + 1}</td>
                  <td className="px-3 py-2 font-mono">{fold.n}</td>
                  <td className="px-3 py-2 font-mono">{fmtNum(fold.auc)}</td>
                  <td className="px-3 py-2 font-mono">{fmtNum(fold.brier)}</td>
                  <td className="px-3 py-2 font-mono">{fmtNum(fold.baseline_brier)}</td>
                  <td className="px-3 py-2 font-mono">{fmtNum(fold.top_quintile_lift, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {model?.notes && <p className="px-3 py-3 text-[11px] text-mute border-t border-line leading-relaxed">{model.notes}</p>}
        </Panel>

        <Panel title="Live book" right={<span className="text-[10px] text-mute">{r.live.pending} pending</span>}>
          <div className="grid grid-cols-2 gap-3 p-3">
            <Stat label="Live AUC" value={fmtNum(r.live.auc)} />
            <Stat label="Live Brier" value={fmtNum(r.live.brier)} />
            <Stat label="Realized move rate" value={livePos} />
            <Stat label="Top-quintile lift" value={fmtNum(r.live.top_quintile_lift, 2)} />
          </div>
          <p className="px-3 py-3 text-[11px] text-mute border-t border-line leading-relaxed">{r.live.note}</p>
        </Panel>
      </div>

      <Panel title="Dataset">
        <div className="px-3 py-3 text-xs space-y-1 font-mono text-mute">
          <div>
            timeframe {r.candles.timeframe} · symbols {r.candles.symbols} · stride {r.samples.stride} bars
          </div>
          <div>
            oldest {r.candles.oldest || "—"} · newest {r.candles.newest || "—"}
          </div>
          <div>
            significant move: |return| ≥ {fmtPct(r.samples.move_pct * 100)} or max upside / drawdown over{" "}
            {r.samples.horizon_hours}h
          </div>
        </div>
      </Panel>
    </div>
  );
}
