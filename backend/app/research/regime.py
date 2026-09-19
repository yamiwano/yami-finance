"""Regime analysis: characterize market context at signal time T.

All regime variables use ONLY information available at or before T.
Thresholds are fit on development data only and frozen before evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np

from app.domain import Bar
from app.research.history import as_utc
from app.research.ranking import MIN_CROSS_SECTION, group_by_timestamp

MIN_REGIME_SAMPLE = 30
STATUS_NO_RELATION = "no_clear_regime_relationship"
STATUS_CANDIDATE = "candidate_regime_relationships"
STATUS_STRONG = "strong_regime_dependence"
STATUS_INSUFFICIENT = "insufficient_data"


def _feat(row: Any, name: str) -> float:
    feats = getattr(row, "features", None) or {}
    value = feats.get(name)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def trailing_return(bars: Sequence[Bar], ts: datetime, hours: int) -> float | None:
    """Return from close at ts-hours to close at ts (past data only)."""
    end = as_utc(ts)
    start = end - timedelta(hours=hours)
    past = [b for b in bars if as_utc(b.ts) <= end]
    if len(past) < 2:
        return None
    ref = None
    for b in reversed(past):
        if as_utc(b.ts) <= start:
            ref = b
            break
    if ref is None or ref.close <= 0:
        return None
    return float(past[-1].close / ref.close - 1.0)


def realized_volatility(bars: Sequence[Bar], ts: datetime, hours: int) -> float | None:
    """Std of hourly log returns over trailing window ending at ts."""
    end = as_utc(ts)
    start = end - timedelta(hours=hours)
    window = [b for b in bars if start < as_utc(b.ts) <= end]
    if len(window) < 3:
        return None
    rets = np.array([np.log(window[i].close / window[i - 1].close) for i in range(1, len(window))], dtype=float)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return None
    return float(np.std(rets, ddof=1))


def btc_trend_regime(btc_bars: Sequence[Bar], ts: datetime) -> str:
    """Fixed thresholds: 24h return > +1% bullish, < -1% bearish, else neutral."""
    r24 = trailing_return(btc_bars, ts, 24)
    if r24 is None:
        return "unknown"
    if r24 > 0.01:
        return "bullish"
    if r24 < -0.01:
        return "bearish"
    return "neutral"


def btc_volatility_regime(btc_bars: Sequence[Bar], ts: datetime, q33: float, q66: float) -> str:
    vol = realized_volatility(btc_bars, ts, 24)
    if vol is None:
        return "unknown"
    if vol <= q33:
        return "low"
    if vol <= q66:
        return "medium"
    return "high"


def market_breadth_regime(rows: Sequence[Any]) -> str:
    """Fixed: >60% positive 24h = strong, <40% = weak, else neutral."""
    rets = [_feat(r, "ret_24") for r in rows]
    rets = [r for r in rets if np.isfinite(r)]
    if not rets:
        return "unknown"
    frac_pos = float(np.mean([r > 0 for r in rets]))
    if frac_pos > 0.60:
        return "strong"
    if frac_pos < 0.40:
        return "weak"
    return "neutral"


def market_volatility_regime(rows: Sequence[Any], q33: float, q66: float) -> str:
    atrs = [_feat(r, "atr_pct") for r in rows]
    atrs = [a for a in atrs if np.isfinite(a)]
    if not atrs:
        return "unknown"
    med = float(np.median(atrs))
    if med <= q33:
        return "low"
    if med <= q66:
        return "medium"
    return "high"


def momentum_dispersion_regime(rows: Sequence[Any], q33: float, q66: float) -> str:
    rets = [_feat(r, "ret_24") for r in rows]
    rets = [r for r in rets if np.isfinite(r)]
    if len(rets) < 3:
        return "unknown"
    disp = float(np.std(rets, ddof=1))
    if disp <= q33:
        return "low"
    if disp <= q66:
        return "medium"
    return "high"


def volume_regime(rows: Sequence[Any], q33: float, q66: float) -> str:
    vols = [_feat(r, "relative_volume") for r in rows]
    vols = [v for v in vols if np.isfinite(v)]
    if not vols:
        return "unknown"
    med = float(np.median(vols))
    if med <= q33:
        return "low"
    if med <= q66:
        return "normal"
    return "high"


def fit_regime_thresholds(
    rows: Sequence[Any],
    btc_bars: Sequence[Bar],
    *,
    train_end: datetime,
) -> dict[str, Any]:
    """Fit quantile thresholds on development data only (<= train_end)."""
    train_end = as_utc(train_end)
    dev_rows = [r for r in rows if as_utc(r.ts) <= train_end]
    btc_vols = []
    for r in dev_rows:
        v = realized_volatility(btc_bars, as_utc(r.ts), 24)
        if v is not None:
            btc_vols.append(v)
    market_atrs = [_feat(r, "atr_pct") for r in dev_rows]
    market_atrs = [a for a in market_atrs if np.isfinite(a)]
    dispersions = []
    volumes = []
    by_ts = group_by_timestamp(dev_rows)
    for _ts, bucket in by_ts.items():
        rets = [_feat(r, "ret_24") for r in bucket]
        rets = [r for r in rets if np.isfinite(r)]
        if len(rets) >= 3:
            dispersions.append(float(np.std(rets, ddof=1)))
        vols = [_feat(r, "relative_volume") for r in bucket]
        vols = [v for v in vols if np.isfinite(v)]
        if vols:
            volumes.append(float(np.median(vols)))

    def _q(vals: list[float]) -> tuple[float, float]:
        if len(vals) < 10:
            return float("nan"), float("nan")
        q33, q66 = np.quantile(vals, [1 / 3, 2 / 3])
        return float(q33), float(q66)

    return {
        "btc_volatility": _q(btc_vols),
        "market_volatility": _q(market_atrs),
        "momentum_dispersion": _q(dispersions),
        "volume": _q(volumes),
        "train_end": train_end.isoformat(),
        "n_dev_rows": len(dev_rows),
    }


def assign_regimes(
    rows: Sequence[Any],
    btc_bars: Sequence[Bar],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    """Assign regime labels per timestamp using only past data."""
    by_ts = group_by_timestamp(rows)
    out = []
    for ts, bucket in sorted(by_ts.items()):
        btc_trend = btc_trend_regime(btc_bars, ts)
        btc_vol = btc_volatility_regime(btc_bars, ts, *thresholds["btc_volatility"])
        breadth = market_breadth_regime(bucket)
        market_vol = market_volatility_regime(bucket, *thresholds["market_volatility"])
        dispersion = momentum_dispersion_regime(bucket, *thresholds["momentum_dispersion"])
        volume = volume_regime(bucket, *thresholds["volume"])
        out.append(
            {
                "ts": ts,
                "btc_trend": btc_trend,
                "btc_volatility": btc_vol,
                "market_breadth": breadth,
                "market_volatility": market_vol,
                "momentum_dispersion": dispersion,
                "volume": volume,
                "n_assets": len(bucket),
            }
        )
    return out


def regime_transition(prev: str | None, curr: str) -> str:
    if prev is None or prev == "unknown" or curr == "unknown":
        return "unknown"
    if prev == curr:
        return f"stable_{curr}"
    return f"{prev}_to_{curr}"


def interaction_label(a: str, b: str) -> str:
    if a == "unknown" or b == "unknown":
        return "unknown"
    return f"{a}×{b}"


def group_rows_by_regime(
    rows: Sequence[Any],
    regime_by_ts: dict[datetime, dict[str, str]],
    dimension: str,
) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        ts = as_utc(row.ts)
        regime = regime_by_ts.get(ts, {}).get(dimension, "unknown")
        groups[regime].append(row)
    return dict(groups)


def period_composition(
    periods: list[dict[str, Any]],
    regime_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Share of timestamps in each regime per period."""
    out = []
    for p in periods:
        start = as_utc(p["start_ts"])
        end = as_utc(p["end_ts"])
        rows = [r for r in regime_rows if start <= as_utc(r["ts"]) <= end]
        comp: dict[str, Any] = {"period": p["period"], "start": p["start"], "end": p["end"], "timestamps": len(rows)}
        for dim in ("btc_trend", "btc_volatility", "market_breadth", "market_volatility", "momentum_dispersion", "volume"):
            counts: dict[str, int] = defaultdict(int)
            for r in rows:
                counts[r[dim]] += 1
            total = max(1, len(rows))
            comp[dim] = {k: v / total for k, v in counts.items()}
        out.append(comp)
    return out


def classify_regime_relationship(regime_results: dict[str, Any]) -> str:
    """Objective classification based on variation across regimes."""
    if not regime_results:
        return STATUS_INSUFFICIENT
    # Measure spread of full-model AUC across regimes per dimension/barrier
    spreads = []
    for dim, by_regime in regime_results.items():
        for regime, by_barrier in by_regime.items():
            if regime in ("unknown", "insufficient_sample"):
                continue
            for barrier, metrics in by_barrier.items():
                full = metrics.get("full") or {}
                auc = full.get("auc")
                if auc is not None:
                    spreads.append((dim, barrier, auc))
    if not spreads:
        return STATUS_INSUFFICIENT
    by_dim: dict[str, list[float]] = defaultdict(list)
    for dim, _barrier, auc in spreads:
        by_dim[dim].append(auc)
    dim_ranges = [max(v) - min(v) for v in by_dim.values() if len(v) >= 2]
    if not dim_ranges:
        return STATUS_INSUFFICIENT
    max_range = max(dim_ranges)
    if max_range >= 0.15:
        return STATUS_STRONG
    if max_range >= 0.05:
        return STATUS_CANDIDATE
    return STATUS_NO_RELATION
