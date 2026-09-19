from typing import Any

from app.domain import Bar, Direction, IndicatorSnapshot, StrategyHit, StrategyId, Timeframe
from app.strategies.common import body_pct, close_in_range_pct, empty_hit, mid, need, rr
from app.strategies.knobs import resolve


def volume_confirmed_breakout(
    bars: list[Bar], ind: IndicatorSnapshot, tf: Timeframe, knobs: dict[str, Any] | None = None
) -> StrategyHit:
    """LONG: close through 20-bar resistance with volume expansion and bullish close location."""
    hit = empty_hit(tf)
    k = resolve(StrategyId.VOLUME_BREAKOUT.value, knobs)
    if len(bars) < 30 or not need(ind, "resistance", "atr", "relative_volume", "rsi"):
        return hit
    bar = bars[-1]
    resistance = ind.resistance or 0
    atr = ind.atr or 0
    if resistance <= 0 or atr <= 0:
        return hit
    rel_vol = ind.relative_volume or 0
    rsi = ind.rsi or 0

    broke = bar.close > resistance and any(b.close <= resistance * 1.002 for b in bars[-5:-1])
    vol_ok = rel_vol >= k["min_rel_vol"]
    rsi_ok = k["rsi_lo"] <= rsi <= k["rsi_hi"]
    conviction = close_in_range_pct(bar) >= k["min_close_in_range"] and body_pct(bar) >= k["min_body"]
    not_gap_wild = (bar.high - bar.low) <= k["max_range_atr"] * atr

    facts = [
        f"Last close {bar.close:.6g} vs 20-bar resistance {resistance:.6g} (excluding last 2 bars).",
        f"Relative volume {rel_vol:.2f}x vs 20-bar average.",
        f"RSI(14) {rsi:.1f}. Close location in bar {close_in_range_pct(bar) * 100:.0f}%.",
    ]

    if not broke and vol_ok and bar.close >= resistance * 0.992 and rsi_ok:
        hit.fired = True
        hit.direction = Direction.WATCH
        hit.strategy = StrategyId.VOLUME_BREAKOUT
        hit.reasons = ["WATCH: volume expansion into resistance without a confirmed close-through."] + facts
        hit.evidence = {"resistance": resistance, "relative_volume": rel_vol, "rsi": rsi}
        hit.entry_low = min(resistance, bar.close)
        hit.entry_high = max(resistance, bar.close)
        hit.stop = resistance - atr
        hit.invalidation = hit.stop
        hit.target_1 = bar.close + 1.5 * atr
        hit.target_2 = bar.close + 2.5 * atr
        hit.structure_quality = 46
        hit.setup_quality = 42
        return hit

    if not (broke and vol_ok and rsi_ok and conviction and not_gap_wild):
        return hit

    entry_low = resistance
    entry_high = bar.close
    entry = mid(entry_low, entry_high)
    stop = min(bar.low, resistance) - k["stop_atr"] * atr
    if ind.swing_low:
        stop = min(stop, ind.swing_low)
    t1 = entry + 1.6 * (entry - stop)
    t2 = entry + 2.6 * (entry - stop)
    hit.fired = True
    hit.direction = Direction.LONG
    hit.strategy = StrategyId.VOLUME_BREAKOUT
    hit.reasons = ["LONG: volume-confirmed close through 20-bar resistance.", *facts]
    hit.evidence = {
        "resistance": resistance,
        "relative_volume": rel_vol,
        "rsi": rsi,
        "close_in_range": close_in_range_pct(bar),
        "atr": atr,
    }
    hit.entry_low = min(entry_low, entry_high)
    hit.entry_high = max(entry_low, entry_high)
    hit.stop = stop
    hit.invalidation = resistance * 0.997
    hit.target_1 = t1
    hit.target_2 = t2
    hit.risk_reward = rr(entry, stop, t1)
    hit.structure_quality = 82
    hit.setup_quality = min(95, 50 + hit.risk_reward * 18)
    return hit


def trend_pullback(
    bars: list[Bar], ind: IndicatorSnapshot, tf: Timeframe, knobs: dict[str, Any] | None = None
) -> StrategyHit:
    """LONG: uptrend (EMA21 > EMA50), pullback into EMA21, reclaim with RSI recovering."""
    hit = empty_hit(tf)
    k = resolve(StrategyId.TREND_PULLBACK.value, knobs)
    if len(bars) < 55 or not need(ind, "ema9", "ema21", "ema50", "rsi", "atr", "relative_volume"):
        return hit
    ema9, ema21, ema50 = ind.ema9 or 0, ind.ema21 or 0, ind.ema50 or 0
    rsi, atr = ind.rsi or 0, ind.atr or 0
    bar = bars[-1]
    recent = bars[-4:]
    uptrend = ema21 > ema50 and ema9 >= ema50 * 0.998
    touched = any(b.low <= ema21 * 1.004 for b in recent)
    reclaim = bar.close > ema21
    rsi_ok = k["rsi_lo"] <= rsi <= k["rsi_hi"]
    dry = (ind.relative_volume or 1) <= k["max_rel_vol_dry"]
    rsi_rising = True
    if len(bars) >= 16:
        # compare last close location vs prior bar — RSI series not passed; use close slope proxy
        rsi_rising = bar.close >= bars[-2].close

    facts = [
        f"EMA9 {ema9:.6g} / EMA21 {ema21:.6g} / EMA50 {ema50:.6g}.",
        f"Pullback touched EMA21: {touched}. Last close {bar.close:.6g} reclaim={reclaim}.",
        f"RSI(14) {rsi:.1f}. Relative volume {ind.relative_volume or 0:.2f}x.",
    ]

    if uptrend and touched and not reclaim and rsi_ok:
        hit.fired = True
        hit.direction = Direction.WATCH
        hit.strategy = StrategyId.TREND_PULLBACK
        hit.reasons = ["WATCH: uptrend pullback into EMA21 without reclaim close."] + facts
        hit.evidence = {"ema21": ema21, "ema50": ema50, "rsi": rsi}
        hit.entry_low = min(ema21, bar.close)
        hit.entry_high = max(ema21, bar.close)
        hit.stop = (ind.ema50 or bar.low) - 0.2 * atr
        hit.invalidation = hit.stop
        hit.target_1 = bar.close + 1.5 * atr
        hit.target_2 = bar.close + 2.5 * atr
        hit.structure_quality = 50
        hit.setup_quality = 45
        return hit

    if not (uptrend and touched and reclaim and rsi_ok and rsi_rising):
        return hit
    if k["require_dry"] and not dry:
        return hit

    entry_low = min(ema21, bar.close)
    entry_high = max(bar.close, ema21)
    entry = mid(entry_low, entry_high)
    stop = min(min(b.low for b in recent), ema50) - k["stop_atr"] * atr
    t1 = entry + k["t1_r"] * (entry - stop)
    t2 = entry + (k["t1_r"] + 1.0) * (entry - stop)
    hit.fired = True
    hit.direction = Direction.LONG
    hit.strategy = StrategyId.TREND_PULLBACK
    hit.reasons = [
        "LONG: trend pullback — price tagged EMA21 in an EMA21>EMA50 uptrend and reclaimed.",
        *facts,
        f"Volume dry-up during pullback: {dry}.",
    ]
    hit.evidence = {
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "rsi": rsi,
        "relative_volume": ind.relative_volume,
        "atr": atr,
        "vwap": ind.vwap,
    }
    hit.entry_low = entry_low
    hit.entry_high = entry_high
    hit.stop = stop
    hit.invalidation = ema50
    hit.target_1 = t1
    hit.target_2 = t2
    hit.risk_reward = rr(entry, stop, t1)
    hit.structure_quality = 78 + (4 if dry else 0)
    hit.setup_quality = min(92, 52 + hit.risk_reward * 16)
    return hit


def momentum_continuation(
    bars: list[Bar], ind: IndicatorSnapshot, tf: Timeframe, knobs: dict[str, Any] | None = None
) -> StrategyHit:
    """LONG: stacked EMAs, higher closes, RSI in continuation band, volume and VWAP support."""
    hit = empty_hit(tf)
    k = resolve(StrategyId.MOMENTUM_CONTINUATION.value, knobs)
    if len(bars) < 55 or not need(ind, "ema9", "ema21", "ema50", "rsi", "atr", "relative_volume"):
        return hit
    ema9, ema21, ema50 = ind.ema9 or 0, ind.ema21 or 0, ind.ema50 or 0
    rsi, atr = ind.rsi or 0, ind.atr or 0
    bar = bars[-1]
    stacked = ema9 > ema21 > ema50 and bar.close > ema9
    higher = all(bars[-i].close > bars[-i - 1].close for i in range(1, 4))
    rsi_ok = k["rsi_lo"] <= rsi <= k["rsi_hi"]
    vol_ok = (ind.relative_volume or 0) >= k["min_rel_vol"]
    above_vwap = ind.vwap is None or bar.close >= ind.vwap

    facts = [
        f"Stacked EMAs: {ema9:.6g} > {ema21:.6g} > {ema50:.6g}. Close {bar.close:.6g}.",
        f"Last 3 closes higher: {higher}. RSI(14) {rsi:.1f}.",
        f"Relative volume {ind.relative_volume or 0:.2f}x. Above VWAP: {above_vwap}.",
    ]

    if stacked and rsi_ok and not (higher and vol_ok):
        hit.fired = True
        hit.direction = Direction.WATCH
        hit.strategy = StrategyId.MOMENTUM_CONTINUATION
        hit.reasons = ["WATCH: stacked-EMA trend without full continuation confirmation."] + facts
        hit.evidence = {"ema9": ema9, "rsi": rsi, "relative_volume": ind.relative_volume}
        hit.entry_low = min(ema9, bar.close)
        hit.entry_high = bar.close
        hit.stop = ema21 - 0.2 * atr
        hit.invalidation = hit.stop
        hit.target_1 = bar.close + 1.4 * atr
        hit.target_2 = bar.close + 2.3 * atr
        hit.structure_quality = 52
        hit.setup_quality = 48
        return hit

    if not (stacked and higher and rsi_ok and vol_ok and above_vwap):
        return hit

    entry = bar.close
    stop = ema21 - k["stop_atr"] * atr
    t1 = entry + k["t1_r"] * (entry - stop)
    t2 = entry + (k["t1_r"] + 0.9) * (entry - stop)
    hit.fired = True
    hit.direction = Direction.LONG
    hit.strategy = StrategyId.MOMENTUM_CONTINUATION
    hit.reasons = ["LONG: momentum continuation with stacked EMAs, higher closes, and volume."] + facts
    hit.evidence = {
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "rsi": rsi,
        "relative_volume": ind.relative_volume,
        "vwap": ind.vwap,
        "atr": atr,
    }
    hit.entry_low = min(ema9, entry)
    hit.entry_high = entry
    hit.stop = stop
    hit.invalidation = ema21
    hit.target_1 = t1
    hit.target_2 = t2
    hit.risk_reward = rr(entry, stop, t1)
    hit.structure_quality = 80
    hit.setup_quality = min(90, 48 + hit.risk_reward * 18)
    return hit
