"""Shared helpers for deterministic strategy rules."""

from __future__ import annotations

from app.domain import Bar, IndicatorSnapshot, StrategyHit, Timeframe


def rr(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    return abs(target - entry) / risk


def body_pct(bar: Bar) -> float:
    rng = bar.high - bar.low
    if rng <= 0:
        return 0.0
    return abs(bar.close - bar.open) / rng


def close_in_range_pct(bar: Bar) -> float:
    rng = bar.high - bar.low
    if rng <= 0:
        return 0.5
    return (bar.close - bar.low) / rng


def upper_wick_pct(bar: Bar) -> float:
    rng = bar.high - bar.low
    if rng <= 0:
        return 0.0
    return (bar.high - max(bar.open, bar.close)) / rng


def lower_wick_pct(bar: Bar) -> float:
    rng = bar.high - bar.low
    if rng <= 0:
        return 0.0
    return (min(bar.open, bar.close) - bar.low) / rng


def mid(a: float, b: float) -> float:
    return (a + b) / 2


def empty_hit(tf: Timeframe) -> StrategyHit:
    return StrategyHit(timeframe=tf)


def need(ind: IndicatorSnapshot, *fields: str) -> bool:
    return all(getattr(ind, f) is not None for f in fields)
