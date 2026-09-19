"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Health, Settings } from "@/lib/types";
import { Panel } from "@/components/ui";
import { fmtAgo } from "@/lib/format";
import Link from "next/link";

export default function SettingsPage() {
  const [s, setS] = useState<Settings | null>(null);
  const [saved, setSaved] = useState(false);
  const [strats, setStrats] = useState<{ id: string; label: string }[]>([]);
  const [source, setSource] = useState<string>("");
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.settings().then(setS).catch(() => undefined);
    api.health().then(setHealth).catch(() => undefined);
    api.meta().then((m) => {
      setStrats(m.strategies);
      setSource(m.data_source || "");
    }).catch(() => undefined);
  }, []);

  if (!s) return <div className="p-6 text-mute">Loading settings…</div>;

  function toggle(list: string[], id: string): string[] {
    return list.includes(id) ? list.filter((x) => x !== id) : [...list, id];
  }

  return (
    <div className="p-4 space-y-4 max-w-3xl">
      <header>
        <div className="text-[11px] uppercase tracking-[0.18em] text-mute">Preferences</div>
        <h1 className="text-xl font-semibold">Scanner settings</h1>
      </header>
      <Panel title="Universe & threshold">
        <div className="p-4 space-y-4 text-sm">
          <p className="text-mute text-xs leading-relaxed">
            Universe is the most liquid Binance USDT spot pairs by 24h quote volume. Equities are not scanned.
          </p>
          <label className="flex items-center justify-between gap-4">
            Minimum opportunity score
            <input
              type="number"
              min={0}
              max={100}
              className="bg-bg border border-line rounded px-2 py-1 w-20 font-mono"
              value={s.min_score}
              onChange={(e) => setS({ ...s, min_score: Number(e.target.value) })}
            />
          </label>
          <label className="flex items-start justify-between gap-4">
            <span>
              Learn from closed outcomes
              <span className="block text-[11px] text-mute mt-1 leading-relaxed font-normal">
                Reweight scores, raise the listing bar on weak patterns, and tweak each strategy's existing gates
                (volume, RSI, wick, stop) when closed results say it is salvageable or firing too loosely. Does not
                invent new strategies or prices.
              </span>
            </span>
            <input
              type="checkbox"
              className="mt-1"
              checked={s.learn_from_outcomes !== false}
              onChange={(e) => setS({ ...s, learn_from_outcomes: e.target.checked })}
            />
          </label>
          {health?.learner && (
            <div className="border border-line rounded p-3 text-xs space-y-1">
              <div className="flex justify-between gap-3 text-mute">
                <span>{health.learner.status_label}</span>
                <span className="font-mono">last change {fmtAgo(health.learner.last_change_at)}</span>
              </div>
              {health.learner.last_change_summary ? (
                <p>{health.learner.last_change_summary}</p>
              ) : (
                <p className="text-mute">No knob or weight change recorded yet.</p>
              )}
              {(health.learner.tweaked_strategies || []).length > 0 && (
                <p className="text-mute">
                  Moving: {health.learner.tweaked_strategies.map((t) => t.label).join(", ")}
                </p>
              )}
              <Link href="/performance" className="inline-block text-accent hover:underline pt-1">
                Open the changelog on Performance →
              </Link>
            </div>
          )}
        </div>
      </Panel>
      <Panel title="Timeframes">
        <div className="p-4 flex gap-4 text-sm">
          {["5m", "15m", "1h"].map((tf) => (
            <label key={tf} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={s.enabled_timeframes.includes(tf)}
                onChange={() => setS({ ...s, enabled_timeframes: toggle(s.enabled_timeframes, tf) })}
              />
              {tf}
            </label>
          ))}
        </div>
      </Panel>
      <Panel title="Strategies">
        <div className="p-4 grid sm:grid-cols-2 gap-2 text-sm">
          {strats.map((st) => (
            <label key={st.id} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={s.enabled_strategies.includes(st.id)}
                onChange={() => setS({ ...s, enabled_strategies: toggle(s.enabled_strategies, st.id) })}
              />
              {st.label}
            </label>
          ))}
        </div>
      </Panel>
      <button
        className="text-sm bg-accent/20 border border-accent/50 rounded px-4 py-2"
        onClick={async () => {
          await api.saveSettings({
            min_score: s.min_score,
            enabled_strategies: s.enabled_strategies,
            enabled_timeframes: s.enabled_timeframes,
            learn_from_outcomes: s.learn_from_outcomes !== false,
          });
          api.health().then(setHealth).catch(() => undefined);
          setSaved(true);
          setTimeout(() => setSaved(false), 1500);
        }}
      >
        {saved ? "Saved" : "Save settings"}
      </button>
      <p className="text-[11px] text-mute leading-relaxed">
        {source || "Tape: Binance spot WebSocket."} Set <code>MARKET_DATA_PROVIDER=mock</code> only for offline UI work — it is not suitable for live research.
      </p>
    </div>
  );
}
