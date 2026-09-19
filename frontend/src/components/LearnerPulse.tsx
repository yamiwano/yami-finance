"use client";

import Link from "next/link";
import type { LearnerChange, LearnerStatus, LearnerStatusCode } from "@/lib/types";
import { fmtAgo } from "@/lib/format";
import { Panel } from "@/components/ui";

const TONE: Record<LearnerStatusCode, string> = {
  paused: "text-mute",
  waiting: "text-watch",
  watching: "text-accent",
  adjusting: "text-long",
};

const DOT: Record<LearnerStatusCode, string> = {
  paused: "bg-mute",
  waiting: "bg-watch",
  watching: "bg-accent",
  adjusting: "bg-long",
};

export function stanceTone(stance?: string | null): string {
  if (stance === "working") return "text-long border-long/40";
  if (stance === "salvage") return "text-watch border-watch/40";
  if (stance === "strict") return "text-short border-short/40";
  return "text-mute border-line";
}

export function changeLine(c: LearnerChange): string {
  const who = c.strategy_label ? `${c.strategy_label} · ` : "";
  return `${who}${c.label} ${c.prior_text} → ${c.live_text}`;
}

export function LearnerPulse({
  learner,
  compact = false,
}: {
  learner: LearnerStatus;
  compact?: boolean;
}) {
  const status = learner.status || (learner.enabled ? "waiting" : "paused");
  const tweaked = learner.tweaked_strategies || [];
  const lastChanges = learner.last_changes || [];

  return (
    <Panel
      title="Strategy learner"
      right={
        <span className={`text-[10px] font-mono uppercase tracking-wide ${TONE[status]}`}>
          <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1.5 align-middle ${DOT[status]} ${status === "adjusting" ? "animate-pulse" : ""}`} />
          {status === "paused" ? "PAUSED" : status === "adjusting" ? "LIVE" : status === "watching" ? "WATCHING" : "WAITING"}
        </span>
      }
    >
      <div className="p-3 space-y-3 text-xs">
        <p className="text-mute leading-relaxed">{learner.status_label}</p>
        <div className="grid grid-cols-2 gap-2">
          <div className="border border-line rounded p-2">
            <div className="text-[10px] uppercase tracking-wider text-mute">Last change</div>
            <div className="font-mono mt-0.5">{fmtAgo(learner.last_change_at)}</div>
          </div>
          <div className="border border-line rounded p-2">
            <div className="text-[10px] uppercase tracking-wider text-mute">Closed tape</div>
            <div className="font-mono mt-0.5">{learner.decided} T2 / stop</div>
          </div>
        </div>
        {learner.last_change_summary ? (
          <div>
            <div className="text-[10px] uppercase tracking-wider text-mute mb-1">What moved</div>
            <p className="leading-relaxed">{learner.last_change_summary}</p>
            {!compact && lastChanges.length > 0 && (
              <ul className="mt-2 space-y-1 text-[11px]">
                {lastChanges.map((c) => (
                  <li key={`${c.kind}-${c.strategy || ""}-${c.key}`}>
                    <span className="font-mono">{changeLine(c)}</span>
                    {c.why && <span className="text-mute"> — {c.why}</span>}
                  </li>
                ))}
              </ul>
            )}
          </div>
        ) : (
          <p className="text-mute leading-relaxed">
            No gate or weight change yet. The learner needs about 6 closed target-2 / stop outcomes on a
            pattern before it will nudge volume, RSI, wick, or stop knobs.
          </p>
        )}
        {tweaked.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {tweaked.map((s) => (
              <span key={s.id} className={`px-1.5 py-0.5 rounded border text-[10px] ${stanceTone(s.stance)}`}>
                {s.label}
                {s.n_changes > 0 ? ` · ${s.n_changes}` : ""}
              </span>
            ))}
          </div>
        )}
        {compact && (
          <Link href="/performance" className="inline-block text-accent hover:underline">
            Full changelog on Performance →
          </Link>
        )}
      </div>
    </Panel>
  );
}
