from typing import Any

from app.domain import Bar, Direction, IndicatorSnapshot, StrategyHit, StrategyId, Timeframe
from app.strategies.common import empty_hit, mid, need, rr, upper_wick_pct
from app.strategies.knobs import resolve


def failed_breakout(
    bars: list[Bar], ind: IndicatorSnapshot, tf: Timeframe, knobs: dict[str, Any] | None = None
) -> StrategyHit:
    """SHORT: poke through resistance then close back below, with rejection wick and volume."""
    hit = empty_hit(tf)
    k = resolve(StrategyId.FAILED_BREAKOUT.value, knobs)
    if len(bars) < 30 or not need(ind, "resistance", "atr", "relative_volume", "rsi"):
        return hit
    resistance = ind.resistance or 0
    atr = ind.atr or 0
    rsi = ind.rsi or 0
    rel_vol = ind.relative_volume or 0
    bar = bars[-1]
    window = bars[-8:-1]
    poked = any(b.high > resistance for b in window) or bar.high > resistance
    closed_back = bar.close < resistance
    wick = upper_wick_pct(bar) >= k["min_wick"] or any(upper_wick_pct(b) >= k["min_wick"] + 0.04 for b in window[-3:])
    vol_ok = rel_vol >= k["min_rel_vol"]
    rsi_ok = rsi <= k["rsi_hi"]

    facts = [
        f"20-bar resistance {resistance:.6g}. Last high {bar.high:.6g}, close {bar.close:.6g}.",
        f"Poke-through observed: {poked}. Close back below: {closed_back}.",
        f"Rejection wick present: {wick}. Rel volume {rel_vol:.2f}x. RSI {rsi:.1f}.",
    ]

    if poked and closed_back and not (wick and vol_ok):
        hit.fired = True
        hit.direction = Direction.WATCH
        hit.strategy = StrategyId.FAILED_BREAKOUT
        hit.reasons = ["WATCH: failed to hold above resistance; awaiting volume/wick confirmation."] + facts
        hit.evidence = {"resistance": resistance, "rsi": rsi, "relative_volume": rel_vol}
        hit.entry_low = min(bar.close, resistance)
        hit.entry_high = resistance
        failed_high = max(b.high for b in bars[-8:])
        hit.stop = failed_high + 0.2 * atr
        hit.invalidation = hit.stop
        hit.target_1 = bar.close - 1.4 * atr
        hit.target_2 = bar.close - 2.3 * atr
        hit.structure_quality = 48
        hit.setup_quality = 44
        return hit

    if not (poked and closed_back and wick and vol_ok and rsi_ok):
        return hit

    failed_high = max(b.high for b in bars[-8:])
    entry = mid(bar.close, resistance)
    stop = failed_high + k["stop_atr"] * atr
    t1 = entry - k["t1_r"] * (stop - entry)
    t2 = entry - (k["t1_r"] + 0.9) * (stop - entry)
    hit.fired = True
    hit.direction = Direction.SHORT
    hit.strategy = StrategyId.FAILED_BREAKOUT
    hit.reasons = ["SHORT: failed breakout — price traded through resistance and closed back below."] + facts
    hit.evidence = {
        "resistance": resistance,
        "failed_high": failed_high,
        "relative_volume": rel_vol,
        "rsi": rsi,
        "atr": atr,
        "upper_wick": upper_wick_pct(bar),
    }
    hit.entry_low = min(bar.close, resistance)
    hit.entry_high = max(bar.close, resistance)
    hit.stop = stop
    hit.invalidation = failed_high
    hit.target_1 = t1
    hit.target_2 = t2
    hit.risk_reward = rr(entry, stop, t1)
    hit.structure_quality = 80
    hit.setup_quality = min(93, 50 + hit.risk_reward * 17)
    return hit


def parabolic_exhaustion(
    bars: list[Bar], ind: IndicatorSnapshot, tf: Timeframe, knobs: dict[str, Any] | None = None
) -> StrategyHit:
    """SHORT: extension from EMA21, RSI rolling over from overbought, volume climax fading."""
    hit = empty_hit(tf)
    k = resolve(StrategyId.PARABOLIC_EXHAUSTION.value, knobs)
    if len(bars) < 40 or not need(ind, "ema9", "ema21", "rsi", "atr", "relative_volume"):
        return hit
    ema21 = ind.ema21 or 0
    ema9 = ind.ema9 or 0
    rsi = ind.rsi or 0
    atr = ind.atr or 0
    bar = bars[-1]
    extension = (bar.close - ema21) / atr if atr else 0
    extended = extension >= k["min_extension_atr"] or (ema21 > 0 and (bar.close - ema21) / ema21 >= 0.055)
    rsi_ob = rsi >= k["rsi_ob"]
    vols = [b.volume for b in bars[-8:]]
    climax = max(vols) >= k["climax_mult"] * (sum(vols) / len(vols))
    fade = bar.volume < max(vols) * k["fade_frac"]
    rolling = bar.close < bars[-2].close or bar.close < ema9

    facts = [
        f"Close {bar.close:.6g} is {extension:.2f} ATR above EMA21 {ema21:.6g}.",
        f"RSI(14) {rsi:.1f}. Volume climax {climax}, fade {fade}.",
        f"Price rolling vs prior close / EMA9: {rolling}.",
    ]

    if extended and rsi_ob and not (rolling or fade):
        hit.fired = True
        hit.direction = Direction.WATCH
        hit.strategy = StrategyId.PARABOLIC_EXHAUSTION
        hit.reasons = ["WATCH: parabolic extension / overbought RSI without rollover confirmation."] + facts
        hit.evidence = {"extension_atr": extension, "rsi": rsi, "ema21": ema21}
        hit.entry_low = min(bar.close, ema9)
        hit.entry_high = bar.close
        recent_high = max(b.high for b in bars[-6:])
        hit.stop = recent_high + 0.3 * atr
        hit.invalidation = hit.stop
        hit.target_1 = ema21
        hit.target_2 = (ind.ema50 or ema21)
        hit.structure_quality = 50
        hit.setup_quality = 42
        return hit

    confirmed = (rolling and fade) if k["require_roll_and_fade"] else (rolling or fade)
    if not (extended and rsi_ob and confirmed and (climax or fade or rolling)):
        return hit

    recent_high = max(b.high for b in bars[-6:])
    entry = bar.close
    stop = recent_high + k["stop_atr"] * atr
    t1 = ema21
    t2 = ind.ema50 or (ema21 - 1.2 * atr)
    if t1 >= entry:
        t1 = entry - 1.5 * (stop - entry)
    if t2 >= t1:
        t2 = entry - 2.4 * (stop - entry)
    hit.fired = True
    hit.direction = Direction.SHORT
    hit.strategy = StrategyId.PARABOLIC_EXHAUSTION
    hit.reasons = ["SHORT: parabolic exhaustion — extended from mean with RSI rollover and volume fade."] + facts
    hit.evidence = {
        "extension_atr": extension,
        "rsi": rsi,
        "ema9": ema9,
        "ema21": ema21,
        "relative_volume": ind.relative_volume,
        "atr": atr,
        "volume_climax": climax,
    }
    hit.entry_low = min(entry, ema9)
    hit.entry_high = entry
    hit.stop = stop
    hit.invalidation = recent_high
    hit.target_1 = t1
    hit.target_2 = t2
    hit.risk_reward = rr(entry, stop, t1)
    hit.structure_quality = 76
    hit.setup_quality = min(90, 46 + hit.risk_reward * 18)
    return hit


def breakdown_continuation(
    bars: list[Bar], ind: IndicatorSnapshot, tf: Timeframe, knobs: dict[str, Any] | None = None
) -> StrategyHit:
    """SHORT: stacked down EMAs, close through 20-bar support, volume and RSI < 45."""
    hit = empty_hit(tf)
    k = resolve(StrategyId.BREAKDOWN_CONTINUATION.value, knobs)
    if len(bars) < 55 or not need(ind, "ema9", "ema21", "ema50", "support", "atr", "rsi", "relative_volume"):
        return hit
    ema9, ema21, ema50 = ind.ema9 or 0, ind.ema21 or 0, ind.ema50 or 0
    support = ind.support or 0
    atr, rsi = ind.atr or 0, ind.rsi or 0
    rel_vol = ind.relative_volume or 0
    bar = bars[-1]
    stacked_down = ema9 < ema21 < ema50 and bar.close < ema9
    broke = bar.close < support and any(b.close >= support * 0.999 for b in bars[-5:-1])
    vol_ok = rel_vol >= k["min_rel_vol"]
    rsi_ok = rsi <= k["rsi_hi"]

    facts = [
        f"Stacked down EMAs: {ema9:.6g} < {ema21:.6g} < {ema50:.6g}.",
        f"20-bar support {support:.6g}. Close {bar.close:.6g}. Broke: {broke}.",
        f"Rel volume {rel_vol:.2f}x. RSI {rsi:.1f}.",
    ]

    if stacked_down and (broke or bar.close <= support * 1.008) and not (broke and vol_ok and rsi_ok):
        hit.fired = True
        hit.direction = Direction.WATCH
        hit.strategy = StrategyId.BREAKDOWN_CONTINUATION
        hit.reasons = ["WATCH: bearish stack into support without full breakdown confirmation."] + facts
        hit.evidence = {"support": support, "rsi": rsi, "relative_volume": rel_vol}
        hit.entry_low = min(bar.close, support)
        hit.entry_high = max(bar.close, support)
        hit.stop = ema21 + 0.25 * atr
        hit.invalidation = hit.stop
        hit.target_1 = bar.close - 1.5 * atr
        hit.target_2 = bar.close - 2.4 * atr
        hit.structure_quality = 50
        hit.setup_quality = 44
        return hit

    if not (stacked_down and broke and vol_ok and rsi_ok):
        return hit

    entry = bar.close
    stop = max(support, ema21) + k["stop_atr"] * atr
    t1 = entry - k["t1_r"] * (stop - entry)
    t2 = entry - (k["t1_r"] + 0.9) * (stop - entry)
    hit.fired = True
    hit.direction = Direction.SHORT
    hit.strategy = StrategyId.BREAKDOWN_CONTINUATION
    hit.reasons = ["SHORT: breakdown continuation through 20-bar support with volume."] + facts
    hit.evidence = {
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "support": support,
        "relative_volume": rel_vol,
        "rsi": rsi,
        "atr": atr,
    }
    hit.entry_low = min(entry, support)
    hit.entry_high = max(entry, support) if support > 0 else entry
    hit.stop = stop
    hit.invalidation = support
    hit.target_1 = t1
    hit.target_2 = t2
    hit.risk_reward = rr(entry, stop, t1)
    hit.structure_quality = 81
    hit.setup_quality = min(93, 50 + hit.risk_reward * 17)
    return hit
