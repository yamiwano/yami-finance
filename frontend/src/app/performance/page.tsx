"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { LearnerEvent, LearnerProfile, Performance } from "@/lib/types";
import { Panel } from "@/components/ui";
import { fmtAgo, fmtPct, fmtTime } from "@/lib/format";
import { LearnerPulse, changeLine, stanceTone } from "@/components/LearnerPulse";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-line rounded p-3">
      <div className="text-[10px] uppercase tracking-wider text-mute">{label}</div>
      <div className="font-mono text-2xl mt-1">{value}</div>
    </div>
  );
}

export default function PerformancePage() {
  const [p, setP] = useState<Performance | null>(null);
  const [confirmReset, setConfirmReset] = useState(false);
  const [resetting, setResetting] = useState(false);
  useEffect(() => {
    api.performance().then(setP).catch(() => undefined);
    const t = setInterval(() => api.performance().then(setP).catch(() => undefined), 10000);
    return () => clearInterval(t);
  }, []);
  if (!p) return <div className="p-6 text-mute">Loading performance…</div>;

  async function resetPeriod() {
    if (!confirmReset) {
      setConfirmReset(true);
      return;
    }
    setResetting(true);
    try {
      setP(await api.resetPerformance());
      setConfirmReset(false);
    } catch {
      /* keep confirm so they can retry */
    } finally {
      setResetting(false);
    }
  }

  return (
    <div className="p-4 space-y-4">
      <header className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[11px] uppercase tracking-[0.18em] text-mute">Track record</div>
          <h1 className="text-xl font-semibold">Performance</h1>
          <p className="text-xs text-mute mt-1">
            Outcomes vs scanner levels on the live tape. Calibration re-ranks future setups from closed results; it does not predict prices.
          </p>
          <p className="text-[11px] text-mute mt-1.5 font-mono">
            {confirmReset
              ? "Zeros win/loss counters from now. History and calibration stay."
              : p.period_started_at
                ? `This period · started ${fmtTime(p.period_started_at)} (${fmtAgo(p.period_started_at)})`
                : "All-time"}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0 pt-1">
          {confirmReset && (
            <button
              type="button"
              className="text-xs border border-line rounded px-3 py-1.5 hover:border-mute"
              onClick={() => setConfirmReset(false)}
              disabled={resetting}
            >
              Cancel
            </button>
          )}
          <button
            type="button"
            className={`text-xs border rounded px-3 py-1.5 ${
              confirmReset
                ? "border-short/50 bg-short/15 text-short hover:border-short"
                : "border-line hover:border-accent"
            }`}
            onClick={resetPeriod}
            disabled={resetting}
          >
            {resetting ? "Resetting…" : confirmReset ? "Confirm reset" : "Reset counters"}
          </button>
        </div>
      </header>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Stat label="Total signals" value={String(p.total_signals)} />
        <Stat label="Active" value={String(p.active)} />
        <Stat label="Wins / losses" value={`${p.wins} / ${p.losses}`} />
        <Stat label="Win rate" value={p.win_rate == null ? "—" : fmtPct(p.win_rate * 100)} />
        <Stat label="Avg R" value={p.average_r.toFixed(2)} />
      </div>
      <div className="grid md:grid-cols-3 gap-4">
        <Panel title="By strategy">
          <Table
            rows={Object.entries(p.by_strategy).map(([k, v]) => [k.replaceAll("_", " "), String(v.count), v.win_rate == null ? "—" : fmtPct(v.win_rate * 100), v.avg_r == null ? "—" : v.avg_r.toFixed(2)])}
            head={["Strategy", "N", "Win", "Avg R"]}
          />
        </Panel>
        <Panel title="By score range">
          <Table
            rows={Object.entries(p.by_score_range).map(([k, v]) => [k, String(v.count), v.win_rate == null ? "—" : fmtPct(v.win_rate * 100)])}
            head={["Score", "N", "Win"]}
          />
        </Panel>
        <Panel title="Long vs short">
          <Table
            rows={Object.entries(p.by_direction).map(([k, v]) => [k, String(v.count), v.win_rate == null ? "—" : fmtPct(v.win_rate * 100), v.avg_r == null ? "—" : v.avg_r.toFixed(2)])}
            head={["Side", "N", "Win", "Avg R"]}
          />
        </Panel>
      </div>
      {p.learner && <LearnerPanel learner={p.learner} />}
    </div>
  );
}

function LearnerPanel({ learner }: { learner: LearnerProfile }) {
  const stratRows = Object.entries(learner.by_strategy).map(([k, v]) => [
    learner.strategy_labels[k] || k.replaceAll("_", " "),
    v.stance_label || "—",
    `${v.wins} / ${v.losses}`,
    v.avg_r == null ? "—" : v.avg_r.toFixed(2),
    v.confidence > 0 ? `×${v.multiplier.toFixed(2)}` : "—",
    v.min_score_bump >= 1 ? `+${v.min_score_bump.toFixed(0)}` : "—",
  ]);
  const weightRows = Object.entries(learner.prior_weights || {}).map(([k, prior]) => {
    const live = learner.weights[k] ?? prior;
    const delta = learner.weight_deltas?.[k] ?? 0;
    return [
      learner.weight_labels[k] || k.replaceAll("_", " "),
      `${Math.round(prior * 100)}%`,
      `${Math.round(live * 100)}%`,
      Math.abs(delta) < 1 ? "—" : `${delta > 0 ? "+" : ""}${delta.toFixed(0)}`,
    ];
  });
  return (
    <div className="space-y-4">
      <LearnerPulse learner={learner} />
      <JournalPanel events={learner.journal || []} calibratedAt={learner.updated_at} />
      <Panel
        title="Outcome calibration"
        right={
          <span className="text-[10px] text-mute">
            {learner.enabled ? "ON" : "PAUSED"} · {learner.decided} closed · {learner.half_life_days}d half-life
          </span>
        }
      >
        <div className="p-3 space-y-2 text-xs">
          <p className="text-mute leading-relaxed">
            After a setup hits target 2, the stop, or invalidation, the scanner updates score weights, raises the listing
            bar on weak patterns, and nudges each strategy's existing gates (volume, RSI, wick, stop) when the tape says
            it is salvageable or firing too loosely. Small samples barely move. Toggle this under Settings.
          </p>
          <ul className="list-disc pl-5 space-y-1">
            {learner.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </div>
      </Panel>
      <div className="grid md:grid-cols-2 gap-4">
        <Panel title="Learned score weights">
          <Table rows={weightRows} head={["Component", "Prior", "Now", "Win−loss gap"]} />
        </Panel>
        <Panel title="Strategy bar">
          <Table rows={stratRows} head={["Strategy", "Stance", "W / L", "Avg R", "Score ×", "Bar"]} />
        </Panel>
      </div>
      <StrategyCards learner={learner} />
    </div>
  );
}

function JournalPanel({ events, calibratedAt }: { events: LearnerEvent[]; calibratedAt?: string | null }) {
  return (
    <Panel
      title="Change log"
      right={
        <span className="text-[10px] text-mute">
          Recalculated {fmtAgo(calibratedAt ?? null)}
        </span>
      }
    >
      {events.length === 0 ? (
        <div className="p-3 text-xs text-mute leading-relaxed">
          No recorded tweaks yet. When a closed trade actually moves a weight or a strategy gate, it lands here with the
          timestamp and the before → after values.
        </div>
      ) : (
        <ol className="divide-y divide-line">
          {events.map((ev) => (
            <li key={ev.ts} className="p-3 space-y-1.5">
              <div className="flex justify-between gap-3 text-[11px] text-mute">
                <span className="uppercase tracking-wide">{ev.origin === "snapshot" ? "Snapshot" : "Tweak"}</span>
                <span className="font-mono" title={fmtTime(ev.ts)}>
                  {fmtAgo(ev.ts)} · {fmtTime(ev.ts)}
                </span>
              </div>
              <p className="text-xs">{ev.summary}</p>
              <ul className="text-[11px] space-y-1">
                {ev.changes.map((c) => (
                  <li key={`${c.kind}-${c.strategy || ""}-${c.key}`}>
                    <span className="font-mono">{changeLine(c)}</span>
                    <span className="text-mute"> — {c.why}</span>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  );
}

function StrategyCards({ learner }: { learner: LearnerProfile }) {
  const ids = Object.keys(learner.strategy_labels || {});
  if (ids.length === 0) return null;
  return (
    <Panel title="Live strategy knobs">
      <div className="grid md:grid-cols-2 xl:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-line">
        {ids.map((sid) => {
          const label = learner.strategy_labels[sid] || sid.replaceAll("_", " ");
          const t = learner.strategy_tweaks?.[sid];
          const b = learner.by_strategy[sid];
          const changes = t?.changes || [];
          const knobs = learner.knobs?.[sid] || {};
          const labels = learner.knob_labels || {};
          const moved = new Set(changes.map((c) => c.key));
          return (
            <div key={sid} className="p-3 space-y-2 min-w-0">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="text-xs">{label}</div>
                  <div className="text-[11px] text-mute font-mono mt-0.5">
                    {b ? `${b.wins}W / ${b.losses}L` : "no closed tape"}
                    {b?.avg_r != null ? ` · ${b.avg_r >= 0 ? "+" : ""}${b.avg_r.toFixed(2)}R` : ""}
                  </div>
                </div>
                <span className={`shrink-0 px-1.5 py-0.5 rounded border text-[10px] ${stanceTone(t?.stance)}`}>
                  {t?.stance_label || "Watching"}
                </span>
              </div>
              {changes.length > 0 ? (
                <ul className="text-[11px] space-y-1">
                  {changes.map((c) => (
                    <li key={c.key}>
                      <span className="text-mute">{c.label}:</span>{" "}
                      <span className="font-mono">
                        {c.prior_text ?? String(c.prior)} → {c.live_text ?? String(c.live)}
                      </span>
                      <span className="text-mute"> — {c.why}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-[11px] text-mute leading-relaxed">
                  Still on defaults. Needs enough closed T2 / stop prints on this pattern before gates move.
                </p>
              )}
              {Object.keys(knobs).length > 0 && (
                <dl className="grid grid-cols-2 gap-x-2 gap-y-1 text-[10px] text-mute pt-1 border-t border-line">
                  {Object.entries(knobs).map(([key, val]) => (
                    <div key={key} className={moved.has(key) ? "text-ink" : ""}>
                      <dt className="truncate">{labels[key] || key.replaceAll("_", " ")}</dt>
                      <dd className="font-mono">
                        {typeof val === "boolean" ? (val ? "on" : "off") : String(val)}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

function Table({ head, rows }: { head: string[]; rows: string[][] }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-mute text-[10px] uppercase">
          {head.map((h) => (
            <th key={h} className="text-left px-3 py-2 font-medium">
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.length === 0 && (
          <tr>
            <td className="px-3 py-6 text-mute" colSpan={head.length}>
              No closed outcomes yet.
            </td>
          </tr>
        )}
        {rows.map((r, i) => (
          <tr key={i} className="border-t border-line">
            {r.map((c, j) => (
              <td key={j} className={`px-3 py-2 ${j ? "font-mono" : ""}`}>
                {c}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
