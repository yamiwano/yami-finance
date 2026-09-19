"""Point-in-time features. Only bars up to and including T. Default knobs only."""

from __future__ import annotations

import math

from app import indicators
from app.domain import Bar, Direction, Timeframe
from app.strategies import STRATEGY_FNS, evaluate_all
from app.strategies.common import body_pct, close_in_range_pct, lower_wick_pct, upper_wick_pct
from app.strategies.knobs import DEFAULT_KNOBS
from app.research.spec import LOOKBACK_BARS

STRATEGY_IDS = [sid for sid, _ in STRATEGY_FNS]


def _nan(value: float | None) -> float:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return math.nan
    return float(value)


def _ret(bars: list[Bar], n: int) -> float:
    if len(bars) <= n or bars[-1 - n].close <= 0:
        return math.nan
    return bars[-1].close / bars[-1 - n].close - 1.0


def _dist_pct(level: float | None, close: float) -> float:
    if level is None or close <= 0:
        return math.nan
    return level / close - 1.0


def extract_features(bars: list[Bar], timeframe: str) -> dict[str, float] | None:
    """Describe the closed bar at T using only information available then."""
    if len(bars) < LOOKBACK_BARS:
        return None
    window = bars[-LOOKBACK_BARS:]
    last = window[-1]
    if last.close <= 0:
        return None
    ind, _ = indicators.compute(window, timeframe)
    tf = Timeframe(timeframe)
    hits = evaluate_all(window, ind, tf, DEFAULT_KNOBS)
    by_id = {h.strategy.value: h for h in hits if h.strategy}

    ema_stack = 0.0
    if ind.ema9 and ind.ema21 and ind.ema50:
        if ind.ema9 >= ind.ema21 >= ind.ema50:
            ema_stack = 1.0
        elif ind.ema9 <= ind.ema21 <= ind.ema50:
            ema_stack = -1.0

    any_long = 0.0
    any_short = 0.0
    feats: dict[str, float] = {
        "ret_1": _ret(window, 1),
        "ret_3": _ret(window, 3),
        "ret_6": _ret(window, 6),
        "ret_12": _ret(window, 12),
        "ret_24": _ret(window, 24),
        "rsi": _nan(ind.rsi),
        "atr_pct": _nan(ind.atr_pct),
        "relative_volume": _nan(ind.relative_volume),
        "ema9_dist": _dist_pct(ind.ema9, last.close),
        "ema21_dist": _dist_pct(ind.ema21, last.close),
        "ema50_dist": _dist_pct(ind.ema50, last.close),
        "ema_stack": ema_stack,
        "close_vs_vwap": _dist_pct(ind.vwap, last.close),
        "dist_resist_pct": _dist_pct(ind.resistance, last.close),
        "dist_support_pct": _dist_pct(ind.support, last.close),
        "body_pct": body_pct(last),
        "close_in_range": close_in_range_pct(last),
        "upper_wick": upper_wick_pct(last),
        "lower_wick": lower_wick_pct(last),
        "hour_utc": float(last.ts.hour),
        "dow_utc": float(last.ts.weekday()),
    }
    for sid in STRATEGY_IDS:
        hit = by_id.get(sid)
        fired = 1.0 if hit and hit.fired else 0.0
        quality = 0.0
        if hit and hit.fired:
            quality = (hit.structure_quality + hit.setup_quality) / 200.0
            if hit.direction == Direction.LONG:
                any_long = 1.0
            elif hit.direction == Direction.SHORT:
                any_short = 1.0
        feats[f"strat_{sid}"] = fired
        feats[f"strat_quality_{sid}"] = quality
    feats["any_long"] = any_long
    feats["any_short"] = any_short
    feats["conflict"] = 1.0 if any_long and any_short else 0.0
    return feats


def jsonable_features(feats: dict[str, float]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, value in feats.items():
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            out[key] = None
        else:
            out[key] = float(value)
    return out


def feature_names() -> list[str]:
    return [
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "rsi",
        "atr_pct",
        "relative_volume",
        "ema9_dist",
        "ema21_dist",
        "ema50_dist",
        "ema_stack",
        "close_vs_vwap",
        "dist_resist_pct",
        "dist_support_pct",
        "body_pct",
        "close_in_range",
        "upper_wick",
        "lower_wick",
        "hour_utc",
        "dow_utc",
        *[f"strat_{sid}" for sid in STRATEGY_IDS],
        *[f"strat_quality_{sid}" for sid in STRATEGY_IDS],
        "any_long",
        "any_short",
        "conflict",
    ]
