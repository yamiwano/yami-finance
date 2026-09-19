"""Binance USDT spot: REST seed + WebSocket klines / bookTicker.

No mock fallback. If Binance is unreachable the scanner waits and retries.
Signals are evaluated on closed candles only; the forming bar is for charts/quotes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime, timezone

import httpx
import websockets

from app.config import get_settings
from app.domain import AssetType, Bar, Quote, Timeframe
from app.market_data.base import MarketDataProvider
from app.market_data.universe import asset_name, is_tradable_usdt_spot

log = logging.getLogger("radar.binance")

_TF_MAP = {Timeframe.M5: "5m", Timeframe.M15: "15m", Timeframe.H1: "1h"}
_INTERVAL_TF = {v: k for k, v in _TF_MAP.items()}

REST_HOSTS = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
)
WS_HOSTS = (
    "wss://data-stream.binance.vision/stream",
    "wss://stream.binance.com:9443/stream",
    "wss://stream.binance.com:443/stream",
)

_MAX_BARS = 400
_SUBSCRIBE_CHUNK = 40


def _parse_kline_row(row: list, *, now: datetime) -> Bar:
    close_ms = int(row[6])
    close_at = datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc)
    return Bar(
        ts=datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        closed=close_at <= now,
    )


def _spread_bps(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    if mid <= 0 or ask < bid:
        return 0.0
    return (ask - bid) / mid * 10_000


class BinanceProvider(MarketDataProvider):
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: httpx.AsyncClient | None = None
        self._rest_base = self._settings.binance_rest_base.rstrip("/")
        self._universe: list[dict] = []
        self._wanted: set[str] = set()
        self._bars: dict[tuple[str, str], list[Bar]] = {}
        self._quotes: dict[str, Quote] = {}
        self._closed: set[tuple[str, Timeframe]] = set()
        self._lock = asyncio.Lock()
        self._ws_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._ws_connected = False
        self._last_message_at: datetime | None = None
        self._ready = asyncio.Event()
        self._streams = 0
        self._ws_url = ""

    def universe(self) -> list[dict]:
        return list(self._universe)

    def is_ready(self) -> bool:
        return self._ready.is_set() and bool(self._quotes)

    def status(self) -> dict:
        stale_ms = None
        if self._last_message_at:
            stale_ms = int((datetime.now(timezone.utc) - self._last_message_at).total_seconds() * 1000)
        return {
            "provider": "binance",
            "source": "binance_spot",
            "ready": self.is_ready(),
            "ws_connected": self._ws_connected,
            "ws_url": self._ws_url,
            "universe": len(self._universe),
            "quotes": len(self._quotes),
            "streams": self._streams,
            "last_message_at": self._last_message_at.isoformat() if self._last_message_at else None,
            "stale_ms": stale_ms,
        }

    def drain_closed(self) -> list[tuple[str, Timeframe]]:
        out = list(self._closed)
        self._closed.clear()
        return out

    async def start(self) -> None:
        self._stop.clear()
        self._client = httpx.AsyncClient(timeout=20.0, headers={"Accept": "application/json"})
        await self._bootstrap_with_retry()
        self._ws_task = asyncio.create_task(self._ws_loop(), name="binance-ws")
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=45)
        except TimeoutError:
            if self._quotes:
                log.warning("Binance WebSocket slow to connect; scanning REST-seeded closed candles until WS is up")
                self._ready.set()
            else:
                raise RuntimeError("Binance returned no market data — refusing to scan")

    async def close(self) -> None:
        self._stop.set()
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass
            self._ws_task = None
        if self._client:
            await self._client.aclose()
            self._client = None
        self._ws_connected = False

    async def tick(self) -> None:
        return

    async def historical(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 300,
        *,
        closed_only: bool = False,
    ) -> list[Bar]:
        bars = list(self._bars.get((symbol, timeframe.value), []))
        if closed_only:
            bars = [b for b in bars if b.closed]
        return bars[-limit:]

    async def quote(self, symbol: str) -> Quote:
        q = self._quotes.get(symbol)
        if q is None:
            raise KeyError(symbol)
        return q

    async def quotes(self) -> dict[str, Quote]:
        return dict(self._quotes)

    async def news_score(self, symbol: str, asset_type: AssetType) -> tuple[float, list[str]]:
        return 50.0, [
            "No news/catalyst feed is wired. Catalyst component is scored as unknown (50), not as confirmation."
        ]

    async def _bootstrap_with_retry(self) -> None:
        delay = 2.0
        while not self._stop.is_set():
            try:
                await self._bootstrap()
                return
            except Exception:
                log.exception("Binance REST bootstrap failed; retrying in %.0fs", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 1.8, 30.0)

    async def _bootstrap(self) -> None:
        info = await self._get_json("/api/v3/exchangeInfo")
        tickers = await self._get_json("/api/v3/ticker/24hr")
        if not isinstance(tickers, list):
            raise RuntimeError("Unexpected Binance 24h ticker payload")

        tradable = {
            s["symbol"]: s
            for s in info.get("symbols", [])
            if isinstance(s, dict) and is_tradable_usdt_spot(s)
        }
        ranked: list[tuple[float, dict]] = []
        for t in tickers:
            sym = t.get("symbol")
            meta = tradable.get(sym)
            if not meta:
                continue
            try:
                qv = float(t.get("quoteVolume") or 0)
            except (TypeError, ValueError):
                continue
            if qv < self._settings.min_quote_volume_usdt:
                continue
            ranked.append((qv, {**meta, "quoteVolume": qv, "lastPrice": t.get("lastPrice"), "priceChangePercent": t.get("priceChangePercent")}))

        ranked.sort(key=lambda x: x[0], reverse=True)
        size = max(8, self._settings.universe_size)
        picked = [row for _, row in ranked[:size]]
        have = {p["symbol"] for p in picked}
        for must in ("BTCUSDT", "ETHUSDT"):
            if must not in have:
                extra = next((row for _, row in ranked if row["symbol"] == must), None)
                if extra:
                    picked.append(extra)

        self._universe = [
            {
                "symbol": row["symbol"],
                "name": asset_name(row["baseAsset"]),
                "asset_type": AssetType.CRYPTO.value,
                "exchange": "BINANCE",
                "base": row["baseAsset"],
                "quote": row["quoteAsset"],
            }
            for row in picked
        ]
        self._wanted = {s["symbol"] for s in self._universe}
        if not self._universe:
            raise RuntimeError("Binance universe empty after liquidity filters")

        ticker_by_sym = {row["symbol"]: row for row in picked}
        await self._seed_klines()
        await self._seed_book()

        async with self._lock:
            for spec in self._universe:
                sym = spec["symbol"]
                t = ticker_by_sym.get(sym, {})
                last = self._last_close(sym) or float(t.get("lastPrice") or 0)
                chg = float(t.get("priceChangePercent") or 0)
                vol = float(t.get("quoteVolume") or 0)
                existing = self._quotes.get(sym)
                self._quotes[sym] = Quote(
                    symbol=sym,
                    price=last,
                    change_pct=chg if chg else self._change_pct(sym),
                    volume=vol,
                    spread_bps=existing.spread_bps if existing else 3.0,
                    bid=existing.bid if existing else None,
                    ask=existing.ask if existing else None,
                    source="binance_spot",
                    ts=datetime.now(timezone.utc),
                )

        log.info(
            "Binance universe ready n=%s top=%s",
            len(self._universe),
            ",".join(s["symbol"] for s in self._universe[:8]),
        )
        if self._quotes:
            self._ready.set()

    async def _seed_klines(self) -> None:
        sem = asyncio.Semaphore(6)
        tasks = [
            self._load_klines(sem, spec["symbol"], tf)
            for spec in self._universe
            for tf in Timeframe
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failed = [r for r in results if isinstance(r, Exception)]
        if failed:
            log.warning("Kline seed errors: %s/%s", len(failed), len(results))
        missing = [spec["symbol"] for spec in self._universe if (spec["symbol"], Timeframe.M5.value) not in self._bars]
        if missing:
            self._universe = [s for s in self._universe if s["symbol"] not in missing]
            self._wanted = {s["symbol"] for s in self._universe}
            log.warning("Dropped symbols with no 5m klines: %s", ",".join(missing))

    async def _load_klines(self, sem: asyncio.Semaphore, symbol: str, tf: Timeframe) -> None:
        interval = _TF_MAP[tf]
        async with sem:
            data = await self._get_json(
                "/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": 300},
            )
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"no klines for {symbol} {interval}")
        now = datetime.now(timezone.utc)
        bars = [_parse_kline_row(row, now=now) for row in data]
        self._bars[(symbol, tf.value)] = bars[-_MAX_BARS:]

    async def _seed_book(self) -> None:
        try:
            data = await self._get_json("/api/v3/ticker/bookTicker")
        except Exception:
            log.exception("bookTicker snapshot failed")
            return
        if not isinstance(data, list):
            return
        wanted = self._wanted
        now = datetime.now(timezone.utc)
        for row in data:
            sym = row.get("symbol")
            if sym not in wanted:
                continue
            try:
                bid = float(row["bidPrice"])
                ask = float(row["askPrice"])
            except (KeyError, TypeError, ValueError):
                continue
            prev = self._quotes.get(sym)
            price = prev.price if prev else (bid + ask) / 2
            self._quotes[sym] = Quote(
                symbol=sym,
                price=price,
                change_pct=prev.change_pct if prev else 0.0,
                volume=prev.volume if prev else 0.0,
                spread_bps=_spread_bps(bid, ask),
                bid=bid,
                ask=ask,
                source="binance_spot",
                ts=now,
            )

    def _last_close(self, symbol: str) -> float:
        bars = self._bars.get((symbol, Timeframe.M5.value)) or []
        return bars[-1].close if bars else 0.0

    def _change_pct(self, symbol: str) -> float:
        bars = self._bars.get((symbol, Timeframe.M5.value)) or []
        if len(bars) < 2:
            return 0.0
        ref = bars[-288] if len(bars) >= 288 else bars[0]
        if not ref.close:
            return 0.0
        return (bars[-1].close - ref.close) / ref.close * 100.0

    async def _get_json(self, path: str, params: dict | None = None):
        assert self._client is not None
        hosts = [self._rest_base, *[h for h in REST_HOSTS if h != self._rest_base]]
        last_exc: Exception | None = None
        for host in hosts:
            try:
                resp = await self._client.get(host + path, params=params)
                if resp.status_code in {418, 429, 451}:
                    log.warning("REST %s%s -> %s, trying next host", host, path, resp.status_code)
                    if resp.status_code == 429:
                        await asyncio.sleep(1.5 + random.random())
                    continue
                resp.raise_for_status()
                self._rest_base = host
                return resp.json()
            except Exception as exc:
                last_exc = exc
                log.warning("REST %s%s failed: %s", host, path, exc)
                continue
        raise last_exc or RuntimeError(f"REST failed {path}")

    def _stream_names(self) -> list[str]:
        names: list[str] = []
        for spec in self._universe:
            s = spec["symbol"].lower()
            names.append(f"{s}@miniTicker")
            names.append(f"{s}@bookTicker")
            for interval in _TF_MAP.values():
                names.append(f"{s}@kline_{interval}")
        return names

    async def _ws_loop(self) -> None:
        delay = 1.0
        while not self._stop.is_set():
            hosts = list(WS_HOSTS)
            if self._settings.binance_ws_base:
                custom = self._settings.binance_ws_base.rstrip("/")
                if custom not in hosts:
                    hosts.insert(0, custom)
            connected = False
            for url in hosts:
                if self._stop.is_set():
                    return
                try:
                    await self._run_socket(url)
                    connected = True
                    delay = 1.0
                    break
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("Binance WS failed url=%s", url)
            if not connected:
                self._ws_connected = False
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    async def _run_socket(self, url: str) -> None:
        streams = self._stream_names()
        self._streams = len(streams)
        connect_url = f"{url}?streams={streams[0]}" if streams else url
        self._ws_url = connect_url
        log.info("Connecting Binance WS %s streams=%s", connect_url, len(streams))
        async with websockets.connect(
            connect_url,
            ping_interval=15,
            ping_timeout=20,
            close_timeout=5,
            max_size=8 * 1024 * 1024,
            compression=None,
        ) as ws:
            self._ws_connected = True
            rest = streams[1:]
            for i in range(0, len(rest), _SUBSCRIBE_CHUNK):
                chunk = rest[i : i + _SUBSCRIBE_CHUNK]
                await ws.send(json.dumps({"method": "SUBSCRIBE", "params": chunk, "id": i + 1}))
                await asyncio.sleep(0.25)
            log.info("Subscribed to %s Binance streams", len(streams))
            while not self._stop.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                except TimeoutError:
                    log.warning("Binance WS silent 30s — reconnecting")
                    break
                self._last_message_at = datetime.now(timezone.utc)
                self._ready.set()
                await self._on_message(raw)
            self._ws_connected = False
            # Gap-fill closed candles after a drop so we do not scan a hole.
            try:
                await self._resync_recent()
            except Exception:
                log.exception("kline resync after WS drop failed")

    async def _resync_recent(self) -> None:
        sem = asyncio.Semaphore(4)
        async def one(symbol: str, tf: Timeframe) -> None:
            interval = _TF_MAP[tf]
            async with sem:
                data = await self._get_json(
                    "/api/v3/klines",
                    params={"symbol": symbol, "interval": interval, "limit": 12},
                )
            if not isinstance(data, list):
                return
            now = datetime.now(timezone.utc)
            incoming = [_parse_kline_row(row, now=now) for row in data]
            async with self._lock:
                key = (symbol, tf.value)
                existing = list(self._bars.get(key, []))
                by_ts = {b.ts: b for b in existing}
                for bar in incoming:
                    prev = by_ts.get(bar.ts)
                    by_ts[bar.ts] = bar
                    if bar.closed and (prev is None or not prev.closed):
                        self._closed.add((symbol, tf))
                merged = sorted(by_ts.values(), key=lambda b: b.ts)[-_MAX_BARS:]
                self._bars[key] = merged

        await asyncio.gather(
            *[one(spec["symbol"], tf) for spec in self._universe for tf in Timeframe],
            return_exceptions=True,
        )

    async def _on_message(self, raw: str | bytes) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(msg, dict):
            return
        if "result" in msg and "id" in msg:
            return
        data = msg.get("data", msg)
        if isinstance(data, list):
            await self._on_mini_tickers(data)
            return
        if not isinstance(data, dict):
            return
        event = data.get("e")
        if event == "kline" or "k" in data:
            await self._on_kline(data)
        elif event == "bookTicker" or ("b" in data and "a" in data and "s" in data and "E" not in data):
            await self._on_book(data)
        elif event == "24hrMiniTicker":
            await self._on_mini_tickers([data])

    async def _on_kline(self, data: dict) -> None:
        k = data.get("k") or {}
        symbol = k.get("s") or data.get("s")
        interval = k.get("i")
        tf = _INTERVAL_TF.get(interval)
        if not symbol or tf is None:
            return
        try:
            bar = Bar(
                ts=datetime.fromtimestamp(int(k["t"]) / 1000, tz=timezone.utc),
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
                closed=bool(k.get("x")),
            )
        except (KeyError, TypeError, ValueError):
            return
        key = (symbol, tf.value)
        async with self._lock:
            bars = self._bars.get(key)
            if not bars:
                self._bars[key] = [bar]
            elif bars[-1].ts == bar.ts:
                was_open = not bars[-1].closed
                bars[-1] = bar
                if bar.closed and was_open:
                    self._closed.add((symbol, tf))
            elif bar.ts > bars[-1].ts:
                bars.append(bar)
                if bar.closed:
                    self._closed.add((symbol, tf))
                if len(bars) > _MAX_BARS:
                    del bars[: len(bars) - _MAX_BARS]
            q = self._quotes.get(symbol)
            if q and tf == Timeframe.M5:
                self._quotes[symbol] = q.model_copy(
                    update={
                        "price": bar.close,
                        "change_pct": self._change_pct(symbol),
                        "ts": datetime.now(timezone.utc),
                        "source": "binance_spot",
                    }
                )

    async def _on_book(self, data: dict) -> None:
        symbol = data.get("s")
        if not symbol or symbol not in self._wanted:
            return
        try:
            bid = float(data["b"])
            ask = float(data["a"])
        except (KeyError, TypeError, ValueError):
            return
        async with self._lock:
            prev = self._quotes.get(symbol)
            if not prev:
                return
            self._quotes[symbol] = prev.model_copy(
                update={
                    "bid": bid,
                    "ask": ask,
                    "spread_bps": _spread_bps(bid, ask),
                    "ts": datetime.now(timezone.utc),
                    "source": "binance_spot",
                }
            )

    async def _on_mini_tickers(self, rows: list) -> None:
        wanted = self._wanted
        now = datetime.now(timezone.utc)
        async with self._lock:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                symbol = row.get("s")
                if symbol not in wanted:
                    continue
                prev = self._quotes.get(symbol)
                if not prev:
                    continue
                try:
                    last = float(row.get("c") or prev.price)
                    open_ = float(row.get("o") or 0)
                    qvol = float(row.get("q") or prev.volume)
                except (TypeError, ValueError):
                    continue
                chg = ((last - open_) / open_ * 100.0) if open_ else prev.change_pct
                self._quotes[symbol] = prev.model_copy(
                    update={
                        "price": last,
                        "change_pct": chg,
                        "volume": qvol,
                        "ts": now,
                        "source": "binance_spot",
                    }
                )
