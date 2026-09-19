from typing import Any

from app.domain import Bar, Direction, IndicatorSnapshot, StrategyHit, Timeframe
from app.strategies.longs import momentum_continuation, trend_pullback, volume_confirmed_breakout
from app.strategies.shorts import breakdown_continuation, failed_breakout, parabolic_exhaustion

STRATEGY_FNS = [
    ("volume_confirmed_breakout", volume_confirmed_breakout),
    ("trend_pullback", trend_pullback),
    ("momentum_continuation", momentum_continuation),
    ("failed_breakout", failed_breakout),
    ("parabolic_exhaustion", parabolic_exhaustion),
    ("breakdown_continuation", breakdown_continuation),
]


def evaluate_all(
    bars: list[Bar],
    ind: IndicatorSnapshot,
    tf: Timeframe,
    knobs_by_strategy: dict[str, dict[str, Any]] | None = None,
) -> list[StrategyHit]:
    """Run every strategy and return all hits (fired or not). Used as model features."""
    extra = knobs_by_strategy or {}
    return [fn(bars, ind, tf, extra.get(sid)) for sid, fn in STRATEGY_FNS]


def detect(
    bars: list[Bar],
    ind: IndicatorSnapshot,
    tf: Timeframe,
    knobs_by_strategy: dict[str, dict[str, Any]] | None = None,
) -> list[StrategyHit]:
    """Run all six strategies. Most return no signal. Prefer the strongest directional hit."""
    hits = evaluate_all(bars, ind, tf, knobs_by_strategy)
    fired = [h for h in hits if h.fired]
    if not fired:
        return []
    directional = [h for h in fired if h.direction in {Direction.LONG, Direction.SHORT}]
    if directional:
        directional.sort(key=lambda h: (h.structure_quality + h.setup_quality), reverse=True)
        best = directional[0]
        # Conflicting long+short at once → AVOID
        dirs = {h.direction for h in directional}
        if Direction.LONG in dirs and Direction.SHORT in dirs:
            best.direction = Direction.AVOID
            best.reasons = ["AVOID: long and short strategies fired on the same bar — conflicting structure."] + best.reasons
        return [best]
    fired.sort(key=lambda h: h.structure_quality, reverse=True)
    return [fired[0]]
