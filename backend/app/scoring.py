"""Transparent 0–100 opportunity score. Components are persisted for the UI."""

from __future__ import annotations

from app.domain import AssetType, Direction, IndicatorSnapshot, ScoreComponents, StrategyHit


def score_setup(
    hit: StrategyHit,
    ind: IndicatorSnapshot,
    *,
    asset_type: AssetType,
    spread_bps: float,
    catalyst: float,
    higher_tf_aligned: bool,
    price: float,
) -> ScoreComponents:
    technical = _clip(hit.structure_quality)
    volume = _volume_score(ind)
    context = _context_score(ind, hit, higher_tf_aligned)
    quality = _clip(hit.setup_quality if hit.setup_quality else _rr_quality(hit))
    catalyst_s = _clip(catalyst)

    weighted = (
        technical * 0.30
        + volume * 0.20
        + context * 0.15
        + quality * 0.20
        + catalyst_s * 0.15
    )

    penalty, reasons = _penalties(ind, asset_type, spread_bps, price, hit)
    total = _clip(weighted + penalty)
    if hit.direction == Direction.WATCH:
        total = min(total, 64)
    if hit.direction == Direction.AVOID:
        total = min(total, 35)

    return ScoreComponents(
        technical_structure=round(technical, 2),
        volume_activity=round(volume, 2),
        market_context=round(context, 2),
        setup_quality=round(quality, 2),
        catalyst_context=round(catalyst_s, 2),
        penalties=round(penalty, 2),
        penalty_reasons=reasons,
        total=round(total, 2),
    )


def _volume_score(ind: IndicatorSnapshot) -> float:
    rv = ind.relative_volume or 1.0
    if rv >= 2.2:
        return 88
    if rv >= 1.6:
        return 76
    if rv >= 1.2:
        return 62
    if rv >= 0.9:
        return 48
    return 32


def _context_score(ind: IndicatorSnapshot, hit: StrategyHit, aligned: bool) -> float:
    s = 50.0
    if aligned:
        s += 18
    if ind.vwap and hit.direction == Direction.LONG:
        s += 8 if (ind.ema9 or 0) >= ind.vwap else -6
    if ind.vwap and hit.direction == Direction.SHORT:
        s += 8 if (ind.ema9 or 0) <= ind.vwap else -6
    if ind.rsi is not None:
        if hit.direction == Direction.LONG and 48 <= ind.rsi <= 68:
            s += 8
        if hit.direction == Direction.SHORT and 32 <= ind.rsi <= 52:
            s += 8
    return _clip(s)


def _rr_quality(hit: StrategyHit) -> float:
    if hit.risk_reward >= 2.4:
        return 88
    if hit.risk_reward >= 1.8:
        return 74
    if hit.risk_reward >= 1.3:
        return 58
    if hit.risk_reward >= 1.0:
        return 42
    return 28


def _penalties(
    ind: IndicatorSnapshot,
    asset_type: AssetType,
    spread_bps: float,
    price: float,
    hit: StrategyHit,
) -> tuple[float, list[str]]:
    p = 0.0
    reasons: list[str] = []
    if (ind.relative_volume or 1) < 0.7:
        p -= 12
        reasons.append("Low relative volume (<0.7x) — liquidity/activity penalty.")
    if (ind.atr_pct or 0) > 8.0:
        p -= 8
        reasons.append(f"Elevated ATR% ({ind.atr_pct:.1f}%) — chop/whipsaw risk.")
    if spread_bps >= 12:
        p -= 12
        reasons.append(f"Wide spot spread ({spread_bps:.1f} bps) — execution penalty.")
    elif spread_bps >= 6:
        p -= 6
        reasons.append(f"Elevated spot spread ({spread_bps:.1f} bps).")
    if hit.risk_reward and hit.risk_reward < 1.1:
        p -= 8
        reasons.append(f"Sub-1.1R first target (R={hit.risk_reward:.2f}).")
    if ind.rsi is not None and hit.direction.value == "LONG" and ind.rsi > 78:
        p -= 8
        reasons.append(f"RSI {ind.rsi:.0f} stretched for a long.")
    if ind.rsi is not None and hit.direction.value == "SHORT" and ind.rsi < 22:
        p -= 8
        reasons.append(f"RSI {ind.rsi:.0f} stretched for a short.")
    return p, reasons


def _clip(v: float) -> float:
    return max(0.0, min(100.0, v))
