"""Deterministic technical indicators. No ML, no interpolation of missing rules."""

from __future__ import annotations

from app.domain import Bar, IndicatorSnapshot


def ema(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    k = 2 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    running = sum(values[:period])
    out[period - 1] = running / period
    for i in range(period, len(values)):
        running += values[i] - values[i - period]
        out[i] = running / period
    return out


def rsi(closes: list[float], period: int = 14) -> list[float | None]:
    """Wilder RSI."""
    out: list[float | None] = [None] * len(closes)
    if len(closes) <= period:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100 - (100 / (1 + avg_gain / avg_loss))
    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            out[i] = 100 - (100 / (1 + avg_gain / avg_loss))
    return out


def true_range(bars: list[Bar]) -> list[float]:
    trs: list[float] = []
    for i, bar in enumerate(bars):
        if i == 0:
            trs.append(bar.high - bar.low)
        else:
            prev_close = bars[i - 1].close
            trs.append(max(bar.high - bar.low, abs(bar.high - prev_close), abs(bar.low - prev_close)))
    return trs


def atr(bars: list[Bar], period: int = 14) -> list[float | None]:
    trs = true_range(bars)
    return sma(trs, period)


def session_vwap(bars: list[Bar], lookback: int) -> list[float | None]:
    """Rolling VWAP over `lookback` bars (session approximation)."""
    out: list[float | None] = [None] * len(bars)
    cum_pv = 0.0
    cum_v = 0.0
    typical: list[float] = []
    for i, bar in enumerate(bars):
        tp = (bar.high + bar.low + bar.close) / 3
        typical.append(tp)
        cum_pv += tp * bar.volume
        cum_v += bar.volume
        if i >= lookback:
            old = bars[i - lookback]
            old_tp = typical[i - lookback]
            cum_pv -= old_tp * old.volume
            cum_v -= old.volume
        if cum_v > 0 and i + 1 >= min(lookback, 5):
            out[i] = cum_pv / cum_v
    return out


def last(values: list[float | None], offset: int = 0) -> float | None:
    idx = len(values) - 1 - offset
    if idx < 0:
        return None
    return values[idx]


def swing_points(bars: list[Bar], left: int = 3, right: int = 3) -> tuple[float | None, float | None]:
    """Most recent confirmed swing high / swing low."""
    swing_high = None
    swing_low = None
    end = len(bars) - right
    for i in range(left, end):
        high = bars[i].high
        low = bars[i].low
        if all(high >= bars[i - j].high for j in range(1, left + 1)) and all(
            high >= bars[i + j].high for j in range(1, right + 1)
        ):
            swing_high = high
        if all(low <= bars[i - j].low for j in range(1, left + 1)) and all(
            low <= bars[i + j].low for j in range(1, right + 1)
        ):
            swing_low = low
    return swing_high, swing_low


def donchian(bars: list[Bar], lookback: int, exclude_last: int = 1) -> tuple[float | None, float | None]:
    window = bars[-(lookback + exclude_last) : -exclude_last] if exclude_last else bars[-lookback:]
    if len(window) < 5:
        return None, None
    return max(b.high for b in window), min(b.low for b in window)


VWAP_LOOKBACK = {"5m": 78, "15m": 26, "1h": 24}


def compute(bars: list[Bar], timeframe: str) -> tuple[IndicatorSnapshot, dict[str, list[float | None]]]:
    if not bars:
        return IndicatorSnapshot(), {}
    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)
    rsi14 = rsi(closes, 14)
    atr14 = atr(bars, 14)
    vol_sma = sma(volumes, 20)
    vwap = session_vwap(bars, VWAP_LOOKBACK.get(timeframe, 24))
    swing_high, swing_low = swing_points(bars)
    resistance, support = donchian(bars, 20, exclude_last=2)

    last_close = closes[-1]
    last_atr = last(atr14)
    last_vol = volumes[-1]
    last_vol_sma = last(vol_sma)
    rel_vol = (last_vol / last_vol_sma) if last_vol_sma else None
    atr_pct = (last_atr / last_close * 100) if last_atr and last_close else None

    snapshot = IndicatorSnapshot(
        ema9=last(ema9),
        ema21=last(ema21),
        ema50=last(ema50),
        vwap=last(vwap),
        rsi=last(rsi14),
        atr=last_atr,
        atr_pct=atr_pct,
        volume_sma20=last_vol_sma,
        relative_volume=rel_vol,
        swing_high=swing_high,
        swing_low=swing_low,
        resistance=resistance,
        support=support,
    )
    series = {
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "rsi": rsi14,
        "vwap": vwap,
        "atr": atr14,
        "volume_sma": vol_sma,
    }
    return snapshot, series
