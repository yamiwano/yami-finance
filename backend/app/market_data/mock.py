"""Deterministic mock tape with injected regimes so the scanner has real setups."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from app.domain import AssetType, Bar, Quote, Timeframe, TF_SECONDS
from app.market_data.base import MarketDataProvider
from app.market_data.universe import REGIMES, UNIVERSE


def _seed(symbol: str) -> int:
    return sum(ord(c) for c in symbol) * 97 + 13


class MockProvider(MarketDataProvider):
    def __init__(self, acceleration: float = 40.0) -> None:
        self.acceleration = acceleration
        self._bars: dict[tuple[str, str], list[Bar]] = {}
        self._quotes: dict[str, Quote] = {}
        self._sim_now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self._news = self._build_news()
        for spec in UNIVERSE:
            self._bootstrap(spec)

    def universe(self) -> list[dict]:
        return [
            {
                "symbol": s["symbol"],
                "name": s["name"],
                "asset_type": s["asset_type"].value if isinstance(s["asset_type"], AssetType) else s["asset_type"],
                "exchange": s["exchange"],
            }
            for s in UNIVERSE
        ]

    def _build_news(self) -> dict[str, tuple[float, list[str]]]:
        # Fact-only mock catalysts. Scores are explicit so the UI can disclose the source.
        return {
            "BTCUSDT": (62.0, ["Mock catalyst: ETF flow print positive vs 20-day average (simulated)."]),
            "ETHUSDT": (55.0, ["Mock catalyst: staking yield commentary; no hard event (simulated)."]),
            "SOLUSDT": (70.0, ["Mock catalyst: ecosystem unlock calendar in 5 sessions (simulated)."]),
            "DOGEUSDT": (48.0, ["Mock catalyst: social volume elevated; no fundamental print (simulated)."]),
        }

    def _bootstrap(self, spec: dict) -> None:
        rng = random.Random(_seed(spec["symbol"]))
        regime = REGIMES.get(spec["symbol"], "range")
        base = float(spec["price"])
        for tf in Timeframe:
            n = 320
            step = timedelta(seconds=TF_SECONDS[tf])
            start = self._sim_now - step * n
            bars = self._generate_series(rng, spec["symbol"], spec["asset_type"], regime, base, start, step, n)
            self._bars[(spec["symbol"], tf.value)] = bars
            base = bars[-1].close
        last = self._bars[(spec["symbol"], Timeframe.M5.value)][-1]
        self._quotes[spec["symbol"]] = Quote(
            symbol=spec["symbol"],
            price=last.close,
            change_pct=self._change_pct(spec["symbol"]),
            volume=last.volume,
            spread_bps=self._spread(spec),
            source="mock",
            ts=last.ts,
        )

    def _spread(self, spec: dict) -> float:
        if spec["symbol"] in {"BTCUSDT", "ETHUSDT"}:
            return 2.0
        return 6.0

    def _change_pct(self, symbol: str) -> float:
        bars = self._bars[(symbol, Timeframe.M5.value)]
        if len(bars) < 80:
            return 0.0
        prev = bars[-80].close
        return (bars[-1].close - prev) / prev * 100 if prev else 0.0

    def _generate_series(
        self,
        rng: random.Random,
        symbol: str,
        asset_type: AssetType | str,
        regime: str,
        start_price: float,
        start: datetime,
        step: timedelta,
        n: int,
    ) -> list[Bar]:
        price = start_price
        bars: list[Bar] = []
        vol_base = 8.5e5
        if start_price > 1000:
            vol_base = 420.0
        elif start_price > 50:
            vol_base = 18000.0

        # Precompute a path that ends in the assigned regime.
        drift = {
            "momentum": 0.0014,
            "pullback": 0.0009,
            "breakout": 0.0004,
            "failed_breakout": 0.0006,
            "parabolic": 0.0022,
            "breakdown": -0.0015,
            "range": 0.00005,
        }[regime]
        vol = {
            "momentum": 0.006,
            "pullback": 0.007,
            "breakout": 0.008,
            "failed_breakout": 0.009,
            "parabolic": 0.014,
            "breakdown": 0.009,
            "range": 0.0045,
        }[regime]

        for i in range(n):
            t = start + step * i
            progress = i / n
            local_drift = drift
            local_vol = vol
            volume_mult = 1.0

            # Inject textbook structure in the last ~40 bars.
            if regime == "breakout" and progress > 0.78:
                if 0.78 < progress < 0.90:
                    local_drift = 0.0001
                    local_vol = 0.003
                    volume_mult = 0.7
                else:
                    local_drift = 0.0045
                    local_vol = 0.006
                    volume_mult = 2.4
            elif regime == "pullback" and progress > 0.72:
                if progress < 0.88:
                    local_drift = 0.0022
                    volume_mult = 1.4
                else:
                    local_drift = -0.0018
                    volume_mult = 0.65
                if progress > 0.96:
                    local_drift = 0.0035
                    volume_mult = 1.5
            elif regime == "momentum" and progress > 0.80:
                local_drift = 0.003
                volume_mult = 1.6
            elif regime == "failed_breakout" and progress > 0.80:
                if progress < 0.92:
                    local_drift = 0.0038
                    volume_mult = 1.8
                else:
                    local_drift = -0.0055
                    volume_mult = 2.1
            elif regime == "parabolic" and progress > 0.82:
                local_drift = 0.007
                local_vol = 0.012
                volume_mult = 2.8 if progress < 0.96 else 1.1
            elif regime == "breakdown" and progress > 0.78:
                local_drift = -0.004
                volume_mult = 1.9

            shock = rng.gauss(local_drift, local_vol)
            nxt = price * (1 + shock)
            nxt = max(nxt, 0.01)
            spread = abs(rng.gauss(0, local_vol)) * price
            high = max(price, nxt) + spread * 0.6
            low = min(price, nxt) - spread * 0.6
            low = max(low, 0.01)
            o = price
            c = nxt
            # Force a reclaim close above the pullback EMA zone on the last few bars.
            if regime == "pullback" and i >= n - 2:
                c = max(c, o * 1.004)
                high = max(high, c)
            if regime == "breakout" and i >= n - 1:
                c = max(c, o * 1.012)
                high = max(high, c)
            if regime == "failed_breakout" and i >= n - 1:
                c = min(c, o * 0.985)
                low = min(low, c)
            if regime == "momentum" and i >= n - 3:
                c = max(c, o * 1.002)
                high = max(high, c)
            vol_noise = abs(rng.gauss(1.0, 0.25)) * volume_mult
            volume = vol_base * vol_noise * (1.15 + 0.25 * math.sin(i / 9))
            bars.append(Bar(ts=t, open=round(o, 6), high=round(high, 6), low=round(low, 6), close=round(c, 6), volume=round(volume, 2)))
            price = c
        return _stamp_regime(bars, regime, vol_base)

    async def historical(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 300,
        *,
        closed_only: bool = False,
    ) -> list[Bar]:
        bars = self._bars.get((symbol, timeframe.value), [])
        if closed_only:
            bars = [b for b in bars if b.closed]
        return bars[-limit:]

    async def quote(self, symbol: str) -> Quote:
        return self._quotes[symbol]

    async def quotes(self) -> dict[str, Quote]:
        return dict(self._quotes)

    async def tick(self) -> None:
        """Advance forming 5m bar; roll a new bar when simulated time crosses the boundary."""
        self._sim_now += timedelta(seconds=max(5.0, 300 / max(self.acceleration, 1)))
        for spec in UNIVERSE:
            symbol = spec["symbol"]
            rng = random.Random(_seed(symbol) + int(self._sim_now.timestamp()))
            q = self._quotes[symbol]
            regime = REGIMES.get(symbol, "range")
            vol = 0.0004 if regime == "range" else 0.0007
            if regime in {"parabolic", "failed_breakout"}:
                vol = 0.0009
            shock = rng.gauss(0.00015 if "break" in regime or regime == "momentum" else 0.0, vol)
            nxt = q.price * (1 + shock)
            nxt = max(nxt, 0.01)
            self._apply_tick(symbol, nxt, rng)
            bars5 = self._bars[(symbol, Timeframe.M5.value)]
            self._quotes[symbol] = Quote(
                symbol=symbol,
                price=bars5[-1].close,
                change_pct=self._change_pct(symbol),
                volume=bars5[-1].volume,
                spread_bps=q.spread_bps,
                source="mock",
                ts=self._sim_now,
            )
            self._rebuild_higher_tfs(symbol)

    def _apply_tick(self, symbol: str, price: float, rng: random.Random) -> None:
        bars = self._bars[(symbol, Timeframe.M5.value)]
        last = bars[-1]
        bucket = self._sim_now.replace(second=0, microsecond=0)
        bucket = bucket.replace(minute=(bucket.minute // 5) * 5)
        vol_add = abs(rng.gauss(last.volume / 40, last.volume / 80))
        if last.ts.replace(tzinfo=timezone.utc) if last.ts.tzinfo is None else last.ts < bucket:
            bars.append(
                Bar(
                    ts=bucket,
                    open=last.close,
                    high=max(last.close, price),
                    low=min(last.close, price),
                    close=price,
                    volume=vol_add,
                )
            )
            if len(bars) > 400:
                del bars[: len(bars) - 400]
        else:
            bars[-1] = Bar(
                ts=last.ts,
                open=last.open,
                high=max(last.high, price),
                low=min(last.low, price),
                close=price,
                volume=last.volume + vol_add,
            )

    def _rebuild_higher_tfs(self, symbol: str) -> None:
        m5 = self._bars[(symbol, Timeframe.M5.value)]
        self._bars[(symbol, Timeframe.M15.value)] = _aggregate(m5, 3)
        self._bars[(symbol, Timeframe.H1.value)] = _aggregate(m5, 12)

    def status(self) -> dict:
        return {"provider": "mock", "source": "mock", "ready": True, "universe": len(UNIVERSE)}

    async def news_score(self, symbol: str, asset_type: AssetType) -> tuple[float, list[str]]:
        if symbol in self._news:
            return self._news[symbol]
        return 50.0, ["No catalyst feed configured; catalyst component scored as unknown (50)."]


def _stamp_regime(bars: list[Bar], regime: str, vol_base: float) -> list[Bar]:
    """Overwrite the last few bars with textbook structure so rules can fire on a mock tape."""
    if len(bars) < 30:
        return bars
    out = list(bars)
    avg = sum(b.close for b in out[-30:-8]) / 22
    ts = [b.ts for b in out]

    def bar(i: int, o: float, h: float, l: float, c: float, vol: float) -> Bar:
        return Bar(ts=ts[i], open=o, high=max(o, h, c), low=min(o, l, c), close=c, volume=vol)

    n = len(out)
    if regime == "breakout":
        cap = avg * 1.01
        for i in range(n - 22, n - 3):
            out[i] = bar(i, cap * 0.992, cap, cap * 0.985, cap * 0.996, vol_base * 0.8)
        out[n - 3] = bar(n - 3, cap * 0.998, cap * 1.004, cap * 0.996, cap * 1.002, vol_base * 1.2)
        out[n - 2] = bar(n - 2, cap * 1.002, cap * 1.018, cap * 1.001, cap * 1.016, vol_base * 2.6)
        out[n - 1] = bar(n - 1, cap * 1.016, cap * 1.028, cap * 1.014, cap * 1.026, vol_base * 2.3)
    elif regime == "pullback":
        for i in range(n - 40, n - 8):
            px = avg * (1 + 0.0012 * (i - (n - 40)))
            out[i] = bar(i, px * 0.998, px * 1.008, px * 0.994, px * 1.004, vol_base * 1.3)
        ema_proxy = out[n - 9].close
        out[n - 6] = bar(n - 6, ema_proxy * 1.01, ema_proxy * 1.012, ema_proxy * 0.997, ema_proxy * 1.001, vol_base * 0.6)
        out[n - 4] = bar(n - 4, ema_proxy * 1.0, ema_proxy * 1.004, ema_proxy * 0.992, ema_proxy * 0.996, vol_base * 0.55)
        out[n - 2] = bar(n - 2, ema_proxy * 0.997, ema_proxy * 1.01, ema_proxy * 0.995, ema_proxy * 1.008, vol_base * 1.4)
        out[n - 1] = bar(n - 1, ema_proxy * 1.008, ema_proxy * 1.018, ema_proxy * 1.006, ema_proxy * 1.016, vol_base * 1.5)
    elif regime == "momentum":
        px = avg
        for i in range(n - 18, n):
            o = px
            px = px * 1.0048
            out[i] = bar(i, o, px * 1.003, o * 0.998, px, vol_base * 1.7)
    elif regime == "failed_breakout":
        cap = avg * 1.02
        for i in range(n - 20, n - 6):
            out[i] = bar(i, cap * 0.99, cap, cap * 0.984, cap * 0.995, vol_base)
        out[n - 5] = bar(n - 5, cap * 0.998, cap * 1.022, cap * 0.997, cap * 1.016, vol_base * 2.0)
        out[n - 2] = bar(n - 2, cap * 1.01, cap * 1.02, cap * 0.992, cap * 0.994, vol_base * 2.2)
        out[n - 1] = bar(n - 1, cap * 0.994, cap * 0.998, cap * 0.978, cap * 0.982, vol_base * 1.9)
    elif regime == "parabolic":
        px = avg
        for i in range(n - 16, n - 2):
            o = px
            px = px * 1.012
            out[i] = bar(i, o, px * 1.006, o * 0.997, px, vol_base * (2.8 if i < n - 4 else 1.1))
        out[n - 1] = bar(n - 1, px, px * 1.004, px * 0.985, px * 0.988, vol_base * 0.9)
    elif regime == "breakdown":
        px = avg
        for i in range(n - 24, n - 3):
            o = px
            px = px * 0.996
            out[i] = bar(i, o, o * 1.004, px * 0.997, px, vol_base * 1.2)
        floor = min(b.low for b in out[-22:-3])
        out[n - 2] = bar(n - 2, floor * 1.002, floor * 1.004, floor * 0.978, floor * 0.982, vol_base * 2.1)
        out[n - 1] = bar(n - 1, floor * 0.982, floor * 0.986, floor * 0.968, floor * 0.972, vol_base * 1.8)
    return out


def _aggregate(m5: list[Bar], factor: int) -> list[Bar]:
    if not m5:
        return []
    out: list[Bar] = []
    for i in range(0, len(m5), factor):
        chunk = m5[i : i + factor]
        if not chunk:
            continue
        out.append(
            Bar(
                ts=chunk[0].ts,
                open=chunk[0].open,
                high=max(b.high for b in chunk),
                low=min(b.low for b in chunk),
                close=chunk[-1].close,
                volume=sum(b.volume for b in chunk),
            )
        )
    return out[-320:]
