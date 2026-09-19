"""Hard risk gates. Returning flags does not invent signals — it annotates / can veto."""

from __future__ import annotations

from app.domain import AssetType, Direction, IndicatorSnapshot, StrategyHit


def evaluate(
    hit: StrategyHit,
    ind: IndicatorSnapshot,
    *,
    asset_type: AssetType,
    spread_bps: float,
    price: float,
) -> tuple[bool, list[dict]]:
    flags: list[dict] = []
    veto = False

    if (ind.atr_pct or 0) > 12:
        flags.append({"code": "EXTREME_VOL", "severity": "high", "detail": f"ATR% {ind.atr_pct:.1f} is extreme."})
        veto = True
    elif (ind.atr_pct or 0) > 7:
        flags.append({"code": "HIGH_VOL", "severity": "med", "detail": f"ATR% {ind.atr_pct:.1f} elevated."})
    if spread_bps >= 15:
        flags.append({"code": "WIDE_SPREAD", "severity": "high", "detail": f"Spot spread {spread_bps:.1f} bps."})
        veto = True
    elif spread_bps >= 8:
        flags.append({"code": "SPREAD", "severity": "med", "detail": f"Spot spread {spread_bps:.1f} bps."})
    if (ind.relative_volume or 1) < 0.45:
        flags.append({"code": "THIN", "severity": "high", "detail": "Relative volume under 0.45x."})
        veto = True
    if hit.stop <= 0 or hit.target_1 <= 0:
        flags.append({"code": "BAD_LEVELS", "severity": "high", "detail": "Non-positive stop or target."})
        veto = True
    if hit.direction == Direction.LONG and not (hit.stop < hit.entry_low <= hit.entry_high < hit.target_1):
        flags.append({"code": "LEVEL_ORDER", "severity": "med", "detail": "Long levels are not strictly ordered."})
    if hit.direction == Direction.SHORT and not (hit.target_1 < hit.entry_low <= hit.entry_high < hit.stop):
        flags.append({"code": "LEVEL_ORDER", "severity": "med", "detail": "Short levels are not strictly ordered."})

    allow = (not veto) or hit.direction in {Direction.WATCH, Direction.AVOID}
    if veto and hit.direction in {Direction.LONG, Direction.SHORT}:
        allow = False
    return allow, flags
