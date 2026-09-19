"""Calibrate ranking and strategy knobs from closed outcomes.

This is not a price predictor. It reweights the 0–100 score, raises the
listing bar on weak patterns, and nudges each strategy's existing gates
(volume, RSI, wick, stop distance) when the closed tape says a rule is
salvageable or firing too loosely. It does not invent new setups.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Sequence

from app.domain import STRATEGY_LABELS, Direction, SignalStatus, StrategyId, utcnow
from app.models.signal import Signal
from app.strategies.knobs import (
    DEFAULT_KNOBS,
    KNOB_LABELS,
    STANCE_LABELS,
    clamp_knobs,
    format_knob,
)

PRIOR_WEIGHTS: dict[str, float] = {
    "technical_structure": 0.30,
    "volume_activity": 0.20,
    "market_context": 0.15,
    "setup_quality": 0.20,
    "catalyst_context": 0.15,
}
WEIGHT_KEYS = tuple(PRIOR_WEIGHTS)
WEIGHT_LABELS = {
    "technical_structure": "Technical structure",
    "volume_activity": "Volume / activity",
    "market_context": "Market context",
    "setup_quality": "Setup quality / R:R",
    "catalyst_context": "Catalyst / news",
}

WIN_STATUSES = {SignalStatus.TARGET_2_HIT.value}
LOSS_STATUSES = {SignalStatus.STOPPED.value, SignalStatus.INVALIDATED.value}
DECIDED_STATUSES = WIN_STATUSES | LOSS_STATUSES

WEIGHT_SHRINK = 40.0
BUCKET_SHRINK = 12.0
HALF_LIFE_DAYS = 21.0
MIN_RAW_TO_MOVE = 6
COMBO_MIN_RAW = 8
MULT_LO, MULT_HI = 0.70, 1.22
BUMP_CAP = 18.0
JOURNAL_MAX = 40
STATUS_LABELS = {
    "paused": "Paused in Settings",
    "waiting": "Waiting for closed trades",
    "watching": "Watching — knobs still at defaults",
    "adjusting": "Live — gates are being nudged",
}


def empty_profile() -> dict[str, Any]:
    return {
        "version": 1,
        "decided": 0,
        "weighted_n": 0.0,
        "updated_at": None,
        "weights": dict(PRIOR_WEIGHTS),
        "weight_deltas": {k: 0.0 for k in WEIGHT_KEYS},
        "by_strategy": {},
        "by_timeframe": {},
        "by_direction": {},
        "by_combo": {},
        "knobs": {sid: dict(vals) for sid, vals in DEFAULT_KNOBS.items()},
        "strategy_tweaks": {},
        "journal": [],
        "last_change_at": None,
        "last_change_summary": None,
        "notes": ["No closed wins/losses yet. Ranking uses the original 0–100 weights."],
    }


def build_profile(rows: Sequence[Signal], *, now: datetime | None = None) -> dict[str, Any]:
    now = now or utcnow()
    decided = [s for s in rows if s.status in DECIDED_STATUSES]
    profile = empty_profile()
    if not decided:
        return profile

    w_sum = 0.0
    wins_w = 0.0
    losses_w = 0.0
    win_comp = defaultdict(float)
    loss_comp = defaultdict(float)
    buckets: dict[str, dict[str, dict[str, float | int]]] = {
        "strategy": defaultdict(_empty_bucket),
        "timeframe": defaultdict(_empty_bucket),
        "direction": defaultdict(_empty_bucket),
        "combo": defaultdict(_empty_bucket),
    }

    for sig in decided:
        w = _recency_weight(sig.closed_at or sig.detected_at, now)
        if w <= 0:
            continue
        win = sig.status in WIN_STATUSES
        r = float(sig.r_multiple) if sig.r_multiple is not None else (1.0 if win else -1.0)
        r = max(-2.5, min(3.0, r))
        comps = sig.score_components or {}
        w_sum += w
        if win:
            wins_w += w
            for k in WEIGHT_KEYS:
                win_comp[k] += w * float(comps.get(k) or 0)
        else:
            losses_w += w
            for k in WEIGHT_KEYS:
                loss_comp[k] += w * float(comps.get(k) or 0)

        keys = {
            "strategy": sig.strategy or "unknown",
            "timeframe": sig.timeframe or "unknown",
            "direction": sig.direction or "unknown",
            "combo": f"{sig.strategy or 'unknown'}|{sig.timeframe or 'unknown'}",
        }
        for family, key in keys.items():
            b = buckets[family][key]
            b["n"] += w
            b["r_sum"] += w * r
            b["raw_n"] += 1
            if win:
                b["wins_w"] += w
                b["raw_wins"] += 1
            else:
                b["losses_w"] += w
                b["raw_losses"] += 1

    profile["decided"] = len(decided)
    profile["weighted_n"] = round(w_sum, 2)
    profile["updated_at"] = now.isoformat()
    profile["weights"], profile["weight_deltas"] = _mix_weights(win_comp, loss_comp, wins_w, losses_w, w_sum)

    profile["by_strategy"] = {k: _finalize_bucket(v) for k, v in buckets["strategy"].items()}
    profile["by_timeframe"] = {k: _finalize_bucket(v) for k, v in buckets["timeframe"].items()}
    profile["by_direction"] = {k: _finalize_bucket(v) for k, v in buckets["direction"].items()}
    profile["by_combo"] = {
        k: _finalize_bucket(v) for k, v in buckets["combo"].items() if int(v["raw_n"]) >= COMBO_MIN_RAW
    }
    knobs, tweaks = _tune_all(decided)
    profile["knobs"] = knobs
    profile["strategy_tweaks"] = tweaks
    for sid, t in tweaks.items():
        if sid in profile["by_strategy"]:
            profile["by_strategy"][sid]["stance"] = t["stance"]
            profile["by_strategy"][sid]["stance_label"] = t["stance_label"]
    profile["notes"] = _notes(profile)
    return profile


def apply_profile(
    profile: dict[str, Any] | None,
    components: dict[str, Any],
    *,
    strategy: str,
    timeframe: str,
    direction: str,
    enabled: bool,
) -> dict[str, Any]:
    out = dict(components)
    raw = float(out.get("total") or 0)
    out["raw_total"] = round(raw, 2)
    if not enabled or not profile or not profile.get("decided"):
        out["total"] = round(raw, 2)
        out["learner_multiplier"] = 1.0
        out["learner_min_score_bump"] = 0.0
        out["learner_weights"] = dict(PRIOR_WEIGHTS)
        out["learner_note"] = "Outcome calibration is off or has no closed trades yet."
        return out

    weights = profile.get("weights") or PRIOR_WEIGHTS
    weighted = sum(float(weights.get(k, PRIOR_WEIGHTS[k])) * float(out.get(k) or 0) for k in WEIGHT_KEYS)
    penalty = float(out.get("penalties") or 0)
    reweighted = _clip(weighted + penalty)

    bucket = _lookup_bucket(profile, strategy, timeframe)
    mult = float(bucket.get("multiplier") or 1.0)
    tf_bucket = (profile.get("by_timeframe") or {}).get(timeframe) or {}
    if int(tf_bucket.get("raw_n") or 0) >= MIN_RAW_TO_MOVE:
        tf_mult = float(tf_bucket.get("multiplier") or 1.0)
        mult *= 0.80 + 0.20 * tf_mult
    if direction in {Direction.WATCH.value, Direction.AVOID.value}:
        mult = 1.0
        bump = 0.0
        bucket_note = bucket.get("note") or "Not enough closed outcomes to calibrate this setup."
        note = (
            f"Watch/avoid listings keep the raw rank. Closed record for this strategy: {bucket_note}"
            if int(bucket.get("raw_n") or 0)
            else bucket_note
        )
    else:
        mult = max(MULT_LO, min(MULT_HI, mult))
        bump = float(bucket.get("min_score_bump") or 0)
        note = bucket.get("note") or "Not enough closed outcomes to calibrate this setup."

    learned = _clip(reweighted * mult)
    tweaks = (profile.get("strategy_tweaks") or {}).get(strategy) or {}
    changes = tweaks.get("changes") or []
    if changes and direction not in {Direction.WATCH.value, Direction.AVOID.value}:
        extra = "; ".join(c["why"] for c in changes[:2])
        note = f"{note} Tweaks: {extra}"
    out["total"] = round(learned, 2)
    out["learner_multiplier"] = round(mult, 3)
    out["learner_min_score_bump"] = round(bump, 1)
    out["learner_weights"] = {k: round(float(weights.get(k, PRIOR_WEIGHTS[k])), 3) for k in WEIGHT_KEYS}
    out["learner_note"] = note
    out["learner_stance"] = tweaks.get("stance")
    out["learner_tweaks"] = changes
    return out


def effective_min_score(base: float, components: dict[str, Any], direction: str) -> float:
    if direction in {Direction.WATCH.value, Direction.AVOID.value, Direction.NONE.value}:
        return base
    return base + float(components.get("learner_min_score_bump") or 0)


def public_profile(profile: dict[str, Any], *, enabled: bool, journal: list[dict] | None = None) -> dict[str, Any]:
    out = dict(profile)
    journal = list(journal if journal is not None else profile.get("journal") or [])
    out["enabled"] = enabled
    out["prior_weights"] = dict(PRIOR_WEIGHTS)
    out["weight_labels"] = dict(WEIGHT_LABELS)
    out["strategy_labels"] = {k.value: v for k, v in STRATEGY_LABELS.items()}
    out["knob_labels"] = dict(KNOB_LABELS)
    out["stance_labels"] = dict(STANCE_LABELS)
    out["status_labels"] = dict(STATUS_LABELS)
    out["half_life_days"] = HALF_LIFE_DAYS
    out["journal"] = journal[:24]
    if journal:
        out["last_change_at"] = journal[0].get("ts")
        out["last_change_summary"] = journal[0].get("summary")
        out["last_changes"] = _featured_changes(list(journal[0].get("changes") or []))
    else:
        out["last_change_at"] = profile.get("last_change_at")
        out["last_change_summary"] = profile.get("last_change_summary")
        out["last_changes"] = _featured_changes(list(profile.get("last_changes") or []))
    status = _learning_status(enabled, out)
    out["status"] = status
    out["status_label"] = STATUS_LABELS[status]
    out["tweaked_strategies"] = _tweaked_strategies(out)
    out["calibrated_at"] = out.get("updated_at")
    return out


def public_status(profile: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    """Compact pulse for health / dashboard — not the full calibration tables."""
    journal = list(profile.get("journal") or [])
    status = _learning_status(enabled, profile)
    last = journal[0] if journal else None
    return {
        "enabled": enabled,
        "decided": int(profile.get("decided") or 0),
        "status": status,
        "status_label": STATUS_LABELS[status],
        "calibrated_at": profile.get("updated_at"),
        "last_change_at": (last or {}).get("ts") or profile.get("last_change_at"),
        "last_change_summary": (last or {}).get("summary") or profile.get("last_change_summary"),
        "tweaked_strategies": _tweaked_strategies(profile),
        "last_changes": _featured_changes(list((last or {}).get("changes") or [])),
    }


def compact_snapshot(profile: dict[str, Any]) -> dict[str, Any]:
    """Persist enough state to detect the next real tweak after a restart."""
    return {
        "decided": int(profile.get("decided") or 0),
        "weights": dict(profile.get("weights") or PRIOR_WEIGHTS),
        "knobs": {sid: dict(vals) for sid, vals in (profile.get("knobs") or {}).items()},
        "strategy_tweaks": dict(profile.get("strategy_tweaks") or {}),
        "by_strategy": {
            sid: {
                "multiplier": b.get("multiplier"),
                "min_score_bump": b.get("min_score_bump"),
                "stance": b.get("stance"),
                "wins": b.get("wins"),
                "losses": b.get("losses"),
                "avg_r": b.get("avg_r"),
                "confidence": b.get("confidence"),
            }
            for sid, b in (profile.get("by_strategy") or {}).items()
        },
    }


def attach_journal(profile: dict[str, Any], journal: list[dict] | None) -> dict[str, Any]:
    journal = list(journal or [])
    profile["journal"] = journal[:JOURNAL_MAX]
    if journal:
        profile["last_change_at"] = journal[0].get("ts")
        profile["last_change_summary"] = journal[0].get("summary")
    return profile


def diff_profiles(before: dict[str, Any] | None, after: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any] | None:
    """Return a journal event only when knobs, weights, stance, or rank actually moved."""
    before = before or empty_profile()
    after = after or empty_profile()
    changes: list[dict[str, Any]] = []

    before_w = before.get("weights") or PRIOR_WEIGHTS
    after_w = after.get("weights") or PRIOR_WEIGHTS
    deltas = after.get("weight_deltas") or {}
    for k in WEIGHT_KEYS:
        prior = float(before_w.get(k, PRIOR_WEIGHTS[k]))
        live = float(after_w.get(k, PRIOR_WEIGHTS[k]))
        if abs(live - prior) < 0.015:
            continue
        delta = float(deltas.get(k) or 0)
        side = "winners" if delta > 0 else "losers"
        why = (
            f"Ran {abs(delta):.0f} pts higher on {side}."
            if abs(delta) >= 1
            else "Closed tape shifted this score weight."
        )
        changes.append(
            _change(
                "weight",
                key=k,
                label=WEIGHT_LABELS[k],
                prior=prior,
                live=live,
                prior_text=f"{prior * 100:.0f}%",
                live_text=f"{live * 100:.0f}%",
                why=why,
            )
        )

    before_tweaks = before.get("strategy_tweaks") or {}
    after_tweaks = after.get("strategy_tweaks") or {}
    before_knobs = before.get("knobs") or {}
    after_knobs = after.get("knobs") or {}
    after_buckets = after.get("by_strategy") or {}
    before_buckets = before.get("by_strategy") or {}

    for sid, priors in DEFAULT_KNOBS.items():
        label = _strategy_label(sid)
        bt = before_tweaks.get(sid) or {}
        at = after_tweaks.get(sid) or {}
        old_stance = bt.get("stance") or "observe"
        new_stance = at.get("stance") or "observe"
        if old_stance != new_stance:
            bucket = after_buckets.get(sid) or {}
            changes.append(
                _change(
                    "stance",
                    key="stance",
                    label="Stance",
                    prior=old_stance,
                    live=new_stance,
                    prior_text=STANCE_LABELS.get(old_stance, old_stance),
                    live_text=STANCE_LABELS.get(new_stance, new_stance),
                    why=bucket.get("note") or "Closed tape changed how this pattern is treated.",
                    strategy=sid,
                    strategy_label=label,
                )
            )
        old_k = before_knobs.get(sid) or dict(priors)
        new_k = after_knobs.get(sid) or dict(priors)
        why_by_key = {c.get("key"): c.get("why") for c in (at.get("changes") or [])}
        for key, default in priors.items():
            prior_v = old_k.get(key, default)
            live_v = new_k.get(key, default)
            if not _value_moved(key, prior_v, live_v):
                continue
            changes.append(
                _change(
                    "knob",
                    key=key,
                    label=KNOB_LABELS.get(key, key.replace("_", " ")),
                    prior=prior_v,
                    live=live_v,
                    prior_text=format_knob(key, prior_v),
                    live_text=format_knob(key, live_v),
                    why=why_by_key.get(key) or "Closed tape nudged this gate.",
                    strategy=sid,
                    strategy_label=label,
                )
            )
        bb = before_buckets.get(sid) or {}
        ab = after_buckets.get(sid) or {}
        old_m = float(bb.get("multiplier") or 1.0)
        new_m = float(ab.get("multiplier") or 1.0)
        if abs(new_m - old_m) >= 0.02 and float(ab.get("confidence") or 0) > 0:
            changes.append(
                _change(
                    "rank",
                    key="multiplier",
                    label="Score multiplier",
                    prior=old_m,
                    live=new_m,
                    prior_text=f"×{old_m:.2f}",
                    live_text=f"×{new_m:.2f}",
                    why=ab.get("note") or "Ranking weight for this pattern shifted.",
                    strategy=sid,
                    strategy_label=label,
                )
            )
        old_b = float(bb.get("min_score_bump") or 0)
        new_b = float(ab.get("min_score_bump") or 0)
        if abs(new_b - old_b) >= 1 and float(ab.get("confidence") or 0) > 0:
            changes.append(
                _change(
                    "rank",
                    key="min_score_bump",
                    label="Listing bar",
                    prior=old_b,
                    live=new_b,
                    prior_text=f"+{old_b:.0f}" if old_b else "0",
                    live_text=f"+{new_b:.0f}" if new_b else "0",
                    why=ab.get("note") or "Listing bar moved for this pattern.",
                    strategy=sid,
                    strategy_label=label,
                )
            )

    if not changes:
        return None
    origin = "snapshot" if int(before.get("decided") or 0) == 0 else "tweak"
    summary = _event_summary(changes)
    if origin == "snapshot":
        summary = f"Loaded calibration from {int(after.get('decided') or 0)} closed trades. {summary}"
    return {
        "ts": (now or utcnow()).isoformat(),
        "decided": int(after.get("decided") or 0),
        "origin": origin,
        "summary": summary,
        "changes": changes,
    }


def _empty_bucket() -> dict[str, float | int]:
    return {
        "n": 0.0,
        "r_sum": 0.0,
        "wins_w": 0.0,
        "losses_w": 0.0,
        "raw_n": 0,
        "raw_wins": 0,
        "raw_losses": 0,
    }


def _finalize_bucket(raw: dict[str, float | int]) -> dict[str, Any]:
    raw_n = int(raw["raw_n"])
    raw_wins = int(raw["raw_wins"])
    raw_losses = int(raw["raw_losses"])
    n = float(raw["n"])
    r_sum = float(raw["r_sum"])
    conf = min(1.0, raw_n / 15.0) if raw_n >= MIN_RAW_TO_MOVE else 0.0
    shrunk_r = r_sum / (n + BUCKET_SHRINK) if n else 0.0
    avg_r = (r_sum / n) if n else None
    decided = raw_wins + raw_losses
    win_rate = (raw_wins / decided) if decided else None
    mult = 1.0 + conf * 0.38 * math.tanh(shrunk_r)
    mult = max(MULT_LO, min(MULT_HI, mult))
    bump = 0.0
    if conf > 0 and shrunk_r < 0:
        bump = min(BUMP_CAP, conf * (4.0 + 14.0 * min(1.0, -shrunk_r)))
    note = _bucket_note(raw_wins, raw_losses, avg_r, mult, bump, conf)
    return {
        "raw_n": raw_n,
        "wins": raw_wins,
        "losses": raw_losses,
        "avg_r": None if avg_r is None else round(avg_r, 3),
        "win_rate": None if win_rate is None else round(win_rate, 4),
        "multiplier": round(mult, 3),
        "min_score_bump": round(bump, 1),
        "confidence": round(conf, 2),
        "note": note,
    }


def _bucket_note(wins: int, losses: int, avg_r: float | None, mult: float, bump: float, conf: float) -> str:
    if conf <= 0:
        return f"{wins}W / {losses}L so far — too few closed trades to change ranking."
    rtxt = "n/a" if avg_r is None else f"{avg_r:+.2f}R"
    parts = [f"{wins}W / {losses}L, avg {rtxt}"]
    if abs(mult - 1.0) >= 0.03:
        parts.append(f"score ×{mult:.2f}")
    if bump >= 1:
        parts.append(f"listing bar +{bump:.0f}")
    return "; ".join(parts)


def _mix_weights(
    win_comp: dict[str, float],
    loss_comp: dict[str, float],
    wins_w: float,
    losses_w: float,
    n_eff: float,
) -> tuple[dict[str, float], dict[str, float]]:
    if wins_w < 1e-6 or losses_w < 1e-6 or n_eff < MIN_RAW_TO_MOVE:
        return dict(PRIOR_WEIGHTS), {k: 0.0 for k in WEIGHT_KEYS}

    usefulness: dict[str, float] = {}
    deltas: dict[str, float] = {}
    for k in WEIGHT_KEYS:
        mw = win_comp[k] / wins_w
        ml = loss_comp[k] / losses_w
        delta = mw - ml
        deltas[k] = delta
        usefulness[k] = math.exp(max(-2.5, min(2.5, delta / 25.0)))

    u_sum = sum(usefulness.values()) or 1.0
    mix = n_eff / (n_eff + WEIGHT_SHRINK)
    weights: dict[str, float] = {}
    for k in WEIGHT_KEYS:
        emp = usefulness[k] / u_sum
        weights[k] = mix * emp + (1.0 - mix) * PRIOR_WEIGHTS[k]
    w_sum = sum(weights.values()) or 1.0
    weights = {k: round(v / w_sum, 4) for k, v in weights.items()}
    return weights, {k: round(deltas[k], 2) for k in WEIGHT_KEYS}


def _lookup_bucket(profile: dict[str, Any], strategy: str, timeframe: str) -> dict[str, Any]:
    combo = (profile.get("by_combo") or {}).get(f"{strategy}|{timeframe}")
    if combo and int(combo.get("raw_n") or 0) >= COMBO_MIN_RAW:
        return combo
    return (profile.get("by_strategy") or {}).get(strategy) or {
        "multiplier": 1.0,
        "min_score_bump": 0.0,
        "note": "Not enough closed outcomes to calibrate this setup.",
    }


def _recency_weight(when: datetime | None, now: datetime) -> float:
    if when is None:
        return 1.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    days = max(0.0, (now - when).total_seconds() / 86400.0)
    return 0.5 ** (days / HALF_LIFE_DAYS)


def _notes(profile: dict[str, Any]) -> list[str]:
    notes: list[str] = [
        f"{profile['decided']} closed wins/losses (T2 vs stop/invalidation). "
        f"Half-life {HALF_LIFE_DAYS:.0f}d so older tape fades. Expired setups are ignored."
    ]
    deltas: dict[str, float] = profile.get("weight_deltas") or {}
    for k, delta in sorted(deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]:
        prior_pct = PRIOR_WEIGHTS[k] * 100
        live_pct = float(profile["weights"][k]) * 100
        if abs(live_pct - prior_pct) < 2:
            continue
        side = "winners" if delta > 0 else "losers"
        notes.append(
            f"{WEIGHT_LABELS[k]} ran {abs(delta):.0f} pts higher on {side} "
            f"(weight {prior_pct:.0f}% → {live_pct:.0f}%)."
        )
    strategies: dict[str, dict] = profile.get("by_strategy") or {}
    ranked = sorted(
        strategies.items(),
        key=lambda kv: (kv[1].get("avg_r") is None, kv[1].get("avg_r") or 0, -kv[1].get("raw_n", 0)),
    )
    for sid, b in ranked[:3]:
        if b.get("confidence", 0) <= 0:
            continue
        label = _strategy_label(sid)
        avg = b.get("avg_r")
        rtxt = "n/a" if avg is None else f"{avg:+.2f}R"
        extra = []
        if b.get("min_score_bump", 0) >= 1:
            extra.append(f"bar +{b['min_score_bump']:.0f}")
        if abs(float(b.get("multiplier") or 1) - 1) >= 0.03:
            extra.append(f"×{b['multiplier']:.2f}")
        adj = f" ({', '.join(extra)})" if extra else ""
        notes.append(f"{label}: {b['wins']}W / {b['losses']}L, avg {rtxt}{adj}.")
    for sid, t in (profile.get("strategy_tweaks") or {}).items():
        changes = t.get("changes") or []
        if not changes:
            continue
        label = _strategy_label(sid)
        bits = [
            f"{c['label']} {format_knob(c['key'], c['prior'])} → {format_knob(c['key'], c['live'])}"
            for c in changes[:2]
        ]
        notes.append(f"{label} ({t.get('stance_label', 'tweak')}): {'; '.join(bits)}.")
    return notes[:8]


def _clip(v: float) -> float:
    return max(0.0, min(100.0, v))


def summarize_shift(after: dict[str, Any]) -> str | None:
    if not after.get("decided"):
        return None
    worst = None
    worst_r = 1.0
    for sid, b in (after.get("by_strategy") or {}).items():
        avg = b.get("avg_r")
        if avg is None or b.get("confidence", 0) <= 0:
            continue
        if avg < worst_r:
            worst_r = avg
            worst = (sid, b)
    if worst is None:
        return f"Calibrated from {after['decided']} closed outcomes."
    sid, b = worst
    return f"Learner: {_strategy_label(sid)} {b['wins']}W/{b['losses']}L avg {b['avg_r']:+.2f}R"


def _strategy_label(sid: str) -> str:
    if sid in StrategyId._value2member_map_:
        return STRATEGY_LABELS[StrategyId(sid)]
    return sid.replace("_", " ")


def _tune_all(rows: Sequence[Signal]) -> tuple[dict[str, dict], dict[str, dict]]:
    by_sid: dict[str, list[Signal]] = defaultdict(list)
    for sig in rows:
        if sig.strategy:
            by_sid[sig.strategy].append(sig)
    knobs = {sid: dict(vals) for sid, vals in DEFAULT_KNOBS.items()}
    tweaks: dict[str, dict] = {}
    for sid, priors in DEFAULT_KNOBS.items():
        live, meta = _tune_strategy(sid, dict(priors), by_sid.get(sid) or [])
        knobs[sid] = live
        tweaks[sid] = meta
    return knobs, tweaks


def _tune_strategy(sid: str, priors: dict[str, Any], rows: list[Signal]) -> tuple[dict[str, Any], dict[str, Any]]:
    knobs = dict(priors)
    wins = [s for s in rows if s.status in WIN_STATUSES]
    losses = [s for s in rows if s.status in LOSS_STATUSES]
    n = len(wins) + len(losses)
    r_vals = []
    for sig in rows:
        r = float(sig.r_multiple) if sig.r_multiple is not None else (1.0 if sig.status in WIN_STATUSES else -1.0)
        r_vals.append(max(-2.5, min(3.0, r)))
    avg_r = (sum(r_vals) / len(r_vals)) if r_vals else None
    separable = _separable(wins, losses)
    stance = _stance(len(wins), n, avg_r, separable)
    conf = min(1.0, n / 15.0) if n >= MIN_RAW_TO_MOVE else 0.0
    empty = {"stance": stance, "stance_label": STANCE_LABELS[stance], "changes": []}
    if stance == "observe" or conf <= 0:
        return clamp_knobs(sid, knobs), empty

    stopped_frac = (
        sum(1 for s in losses if s.status == SignalStatus.STOPPED.value) / len(losses) if losses else 0.0
    )
    w_vol, l_vol = _mean_feat(wins, "relative_volume"), _mean_feat(losses, "relative_volume")
    w_rsi, l_rsi = _mean_feat(wins, "rsi"), _mean_feat(losses, "rsi")
    whys: dict[str, str] = {}

    if "min_rel_vol" in knobs:
        bump = 0.0
        if w_vol is not None and l_vol is not None and l_vol + 0.12 < w_vol:
            bump = conf * min(0.35, (w_vol - l_vol) * 0.65)
            why = f"Winners printed {w_vol:.2f}x volume vs {l_vol:.2f}x on losers."
        elif stance == "strict":
            bump = conf * 0.16
            why = "Pattern is leaking; requiring more volume before it fires."
        elif stance == "salvage" and l_vol is not None:
            bump = conf * 0.10
            why = "Needs work: losers often came in on thinner volume."
        if bump > 0:
            knobs["min_rel_vol"] = float(priors["min_rel_vol"]) + bump
            whys["min_rel_vol"] = why

    if "rsi_hi" in knobs:
        cut = 0.0
        if w_rsi is not None and l_rsi is not None and l_rsi > w_rsi + 4:
            cut = conf * min(6.0, l_rsi - w_rsi) * 0.4
            why = f"Losers were more stretched (RSI {l_rsi:.0f} vs {w_rsi:.0f} on winners)."
        elif stance == "strict":
            cut = conf * 3.0
            why = "Tightening the RSI ceiling so late/stretched copies fire less."
        if cut > 0:
            knobs["rsi_hi"] = float(priors["rsi_hi"]) - cut
            whys["rsi_hi"] = why

    if "rsi_lo" in knobs and sid != "parabolic_exhaustion":
        lift = 0.0
        if w_rsi is not None and l_rsi is not None and l_rsi + 4 < w_rsi:
            lift = conf * min(6.0, w_rsi - l_rsi) * 0.35
            why = f"Losers were weaker (RSI {l_rsi:.0f} vs {w_rsi:.0f} on winners)."
        elif stance == "strict":
            lift = conf * 2.5
            why = "Raising the RSI floor so unfinished moves fire less."
        if lift > 0:
            knobs["rsi_lo"] = float(priors["rsi_lo"]) + lift
            whys["rsi_lo"] = why

    if sid == "parabolic_exhaustion":
        knobs["require_roll_and_fade"] = True
        whys["require_roll_and_fade"] = "Exhaustion shorts now need both a rollover and a volume fade."
        ext_l = _mean_feat(losses, "extension_atr")
        if stance == "strict" or (ext_l is not None and ext_l < float(priors["min_extension_atr"]) + 0.3):
            knobs["min_extension_atr"] = float(priors["min_extension_atr"]) + conf * 0.35
            whys["min_extension_atr"] = "Requiring a larger extension before fading a move."
        knobs["rsi_ob"] = float(priors["rsi_ob"]) + conf * 3.0
        whys["rsi_ob"] = "RSI must be more overbought before this short fires."

    if sid == "trend_pullback":
        if stance in {"salvage", "strict"} or (l_vol is not None and l_vol > float(priors["max_rel_vol_dry"])):
            knobs["require_dry"] = True
            whys["require_dry"] = "Pullbacks now have to dry up; heavy volume on the dip has been failing."

    if sid == "failed_breakout":
        w_wick, l_wick = _mean_feat(wins, "upper_wick"), _mean_feat(losses, "upper_wick")
        bump = 0.0
        if l_wick is not None and (w_wick is None or l_wick + 0.04 < w_wick):
            bump = conf * 0.06
            why = "Winners had a clearer rejection wick than losers."
        elif stance == "salvage":
            bump = conf * 0.04
            why = "Needs work: asking for a slightly cleaner rejection wick."
        if bump:
            knobs["min_wick"] = float(priors["min_wick"]) + bump
            whys["min_wick"] = why

    if sid == "volume_confirmed_breakout":
        w_loc = _mean_feat(wins, "close_in_range")
        l_loc = _mean_feat(losses, "close_in_range")
        if w_loc is not None and l_loc is not None and l_loc + 0.06 < w_loc:
            knobs["min_close_in_range"] = float(priors["min_close_in_range"]) + conf * 0.06
            whys["min_close_in_range"] = "Winners closed stronger in the bar than losers."
        elif stance == "strict":
            knobs["min_body"] = float(priors["min_body"]) + conf * 0.05
            whys["min_body"] = "Requiring a more convincing breakout body."

    if stance == "salvage" and stopped_frac >= 0.55 and "stop_atr" in knobs:
        knobs["stop_atr"] = float(priors["stop_atr"]) + conf * 0.08
        whys["stop_atr"] = "Most losses were stop-outs; the stop is a little farther from entry."

    knobs = clamp_knobs(sid, knobs)
    changes = []
    for key, prior in priors.items():
        live = knobs.get(key, prior)
        moved = (bool(live) != bool(prior)) if isinstance(prior, bool) else abs(float(live) - float(prior)) >= (
            0.6 if "rsi" in key else 0.025
        )
        if not moved:
            knobs[key] = prior
            continue
        changes.append(
            {
                "key": key,
                "label": KNOB_LABELS.get(key, key.replace("_", " ")),
                "prior": prior,
                "live": live,
                "prior_text": format_knob(key, prior),
                "live_text": format_knob(key, live),
                "why": whys.get(key, "Closed tape nudged this gate."),
            }
        )
    return knobs, {"stance": stance, "stance_label": STANCE_LABELS[stance], "changes": changes}


def _stance(wins: int, n: int, avg_r: float | None, separable: bool) -> str:
    if n < MIN_RAW_TO_MOVE:
        return "observe"
    if avg_r is not None and avg_r >= 0.12 and wins >= 2:
        return "working"
    if separable or wins >= 2:
        return "salvage"
    if wins == 0 and n >= MIN_RAW_TO_MOVE:
        return "strict"
    if n >= 8:
        return "strict"
    return "observe"


def _separable(wins: list[Signal], losses: list[Signal]) -> bool:
    if not wins or not losses:
        return False
    checks = (
        ("relative_volume", 0.18),
        ("rsi", 5.0),
        ("upper_wick", 0.06),
        ("extension_atr", 0.4),
        ("close_in_range", 0.08),
    )
    for key, gap in checks:
        mw, ml = _mean_feat(wins, key), _mean_feat(losses, key)
        if mw is not None and ml is not None and abs(mw - ml) >= gap:
            return True
    return False


def _mean_feat(rows: list[Signal], key: str) -> float | None:
    vals = [v for v in (_feat(s, key) for s in rows) if v is not None]
    return sum(vals) / len(vals) if vals else None


def _feat(sig: Signal, key: str) -> float | None:
    ev = sig.evidence or {}
    if ev.get(key) is not None:
        try:
            return float(ev[key])
        except (TypeError, ValueError):
            pass
    ind = (sig.snapshot or {}).get("indicators") or {}
    if ind.get(key) is not None:
        try:
            return float(ind[key])
        except (TypeError, ValueError):
            pass
    return None


def _learning_status(enabled: bool, profile: dict[str, Any]) -> str:
    if not enabled:
        return "paused"
    if int(profile.get("decided") or 0) == 0:
        return "waiting"
    if _tweaked_strategies(profile) or _weights_moved(profile) or _rank_moved(profile):
        return "adjusting"
    return "watching"


def _weights_moved(profile: dict[str, Any]) -> bool:
    weights = profile.get("weights") or PRIOR_WEIGHTS
    return any(abs(float(weights.get(k, PRIOR_WEIGHTS[k])) - PRIOR_WEIGHTS[k]) >= 0.015 for k in WEIGHT_KEYS)


def _rank_moved(profile: dict[str, Any]) -> bool:
    for b in (profile.get("by_strategy") or {}).values():
        if float(b.get("confidence") or 0) <= 0:
            continue
        if abs(float(b.get("multiplier") or 1) - 1) >= 0.03:
            return True
        if float(b.get("min_score_bump") or 0) >= 1:
            return True
    return False


def _tweaked_strategies(profile: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    tweaks = profile.get("strategy_tweaks") or {}
    buckets = profile.get("by_strategy") or {}
    for sid in DEFAULT_KNOBS:
        t = tweaks.get(sid) or {}
        b = buckets.get(sid) or {}
        n = len(t.get("changes") or [])
        rank = abs(float(b.get("multiplier") or 1) - 1) >= 0.03 or float(b.get("min_score_bump") or 0) >= 1
        if n == 0 and not rank:
            continue
        out.append(
            {
                "id": sid,
                "label": _strategy_label(sid),
                "stance": t.get("stance") or b.get("stance"),
                "stance_label": t.get("stance_label") or b.get("stance_label"),
                "n_changes": n,
            }
        )
    return out


def _change(
    kind: str,
    *,
    key: str,
    label: str,
    prior: Any,
    live: Any,
    prior_text: str,
    live_text: str,
    why: str,
    strategy: str | None = None,
    strategy_label: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "kind": kind,
        "key": key,
        "label": label,
        "prior": prior,
        "live": live,
        "prior_text": prior_text,
        "live_text": live_text,
        "why": why,
    }
    if strategy:
        item["strategy"] = strategy
        item["strategy_label"] = strategy_label
    return item


def _value_moved(key: str, prior: Any, live: Any) -> bool:
    if isinstance(prior, bool) or isinstance(live, bool):
        return bool(live) != bool(prior)
    try:
        return abs(float(live) - float(prior)) >= (0.6 if "rsi" in key else 0.025)
    except (TypeError, ValueError):
        return prior != live


def _event_summary(changes: list[dict[str, Any]]) -> str:
    for kind in ("knob", "stance", "weight", "rank"):
        hit = next((c for c in changes if c.get("kind") == kind), None)
        if not hit:
            continue
        who = f"{hit['strategy_label']}: " if hit.get("strategy_label") else ""
        extra = len(changes) - 1
        more = f" (+{extra} more)" if extra > 0 else ""
        return f"{who}{hit['label']} {hit['prior_text']} → {hit['live_text']}{more}"
    return "Calibration updated."


def _featured_changes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"knob": 0, "stance": 1, "rank": 2, "weight": 3}
    ranked = sorted(changes, key=lambda c: order.get(str(c.get("kind")), 9))
    return ranked[:4]

