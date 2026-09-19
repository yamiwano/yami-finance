"""Market Data → Indicators → Strategies → Risk → Scoring → Persist → AI → WS."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app import indicators
from app.ai import Explainer, create_explainer
from app.cache import cache_set
from app.config import get_settings
from app.database import SessionLocal
from app import learn
from app.domain import (
    STRATEGY_LABELS,
    AssetType,
    Bar,
    Direction,
    IndicatorSnapshot,
    SignalStatus,
    StrategyHit,
    Timeframe,
    utcnow,
)
from app.market_data import create_provider
from app.market_data.base import MarketDataProvider
from app.models.asset import Asset
from app.models.settings import AppSettings
from app.models.signal import Signal, SignalEvent
from app.outcomes import update_outcomes
from app.research.history import persist_closed_pairs, persist_provider_window
from app import risk
from app.scoring import score_setup
from app.strategies import detect
from app.ws import hub

log = logging.getLogger("radar.scanner")

ACTIVITY: list[dict[str, Any]] = []
MAX_ACTIVITY = 80


def _push_activity(kind: str, message: str, extra: dict | None = None) -> None:
    item = {"ts": utcnow().isoformat(), "kind": kind, "message": message, **(extra or {})}
    ACTIVITY.insert(0, item)
    del ACTIVITY[MAX_ACTIVITY:]


class Scanner:
    def __init__(self) -> None:
        self.provider: MarketDataProvider = create_provider()
        self.explainer: Explainer = create_explainer()
        self._task: asyncio.Task | None = None
        self._running = False
        self.asset_ids: dict[str, UUID] = {}
        self.last_scan_at = None
        self.scans = 0
        self._last_full = 0.0
        self._last_tick_broadcast = 0.0
        self.learner_profile: dict[str, Any] = learn.empty_profile()
        self.learn_enabled = True

    async def start(self) -> None:
        await self.provider.start()
        async with SessionLocal() as db:
            await self._ensure_assets(db)
            await self._ensure_settings(db)
            await self._expire_non_crypto(db)
            await self._reload_learner(db)
            try:
                n = await persist_provider_window(db, self.provider, self.asset_ids, timeframe=Timeframe.H1, limit=400)
                if n:
                    log.info("Persisted %s closed 1h bars for research", n)
            except Exception:
                log.exception("initial candle persist failed")
        self._running = True
        self._task = asyncio.create_task(self._loop())
        log.info("Scanner started provider=%s", type(self.provider).__name__)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        await self.provider.close()

    async def _loop(self) -> None:
        settings = get_settings()
        while self._running:
            try:
                if not self.provider.is_ready():
                    await asyncio.sleep(1)
                    continue
                await self.provider.tick()
                quotes = await self._refresh_quotes_and_outcomes()
                closed = self.provider.drain_closed()
                if closed:
                    async with SessionLocal() as db:
                        await persist_closed_pairs(db, self.provider, self.asset_ids, closed)
                now = time.monotonic()
                if self.scans == 0:
                    await self.scan_universe(quotes)
                    self._last_full = now
                elif closed:
                    await self.scan_pairs(quotes, closed)
                elif now - self._last_full >= settings.full_rescan_seconds:
                    await self.scan_universe(quotes)
                    self._last_full = now
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("scan loop error")
            await asyncio.sleep(settings.scan_interval_seconds)

    async def _ensure_assets(self, db: AsyncSession) -> None:
        for spec in self.provider.universe():
            row = (await db.execute(select(Asset).where(Asset.symbol == spec["symbol"]))).scalar_one_or_none()
            if row is None:
                row = Asset(
                    symbol=spec["symbol"],
                    name=spec["name"],
                    asset_type=spec["asset_type"] if isinstance(spec["asset_type"], str) else spec["asset_type"].value,
                    exchange=spec["exchange"],
                )
                db.add(row)
                await db.flush()
            else:
                row.name = spec["name"]
                row.exchange = spec["exchange"]
                row.asset_type = spec["asset_type"] if isinstance(spec["asset_type"], str) else spec["asset_type"].value
                row.is_active = True
            self.asset_ids[spec["symbol"]] = row.id
        live = {s["symbol"] for s in self.provider.universe()}
        others = (await db.execute(select(Asset))).scalars().all()
        for row in others:
            if row.symbol not in live:
                row.is_active = False
        await db.commit()

    async def _expire_non_crypto(self, db: AsyncSession) -> None:
        live = {s["symbol"] for s in self.provider.universe()}
        rows = (
            await db.execute(
                select(Signal).where(Signal.status.in_([SignalStatus.ACTIVE.value, SignalStatus.TARGET_1_HIT.value]))
            )
        ).scalars().all()
        n = 0
        for sig in rows:
            if sig.asset_type == AssetType.STOCK.value or sig.symbol not in live:
                sig.status = SignalStatus.EXPIRED.value
                sig.closed_at = utcnow()
                n += 1
        if n:
            await db.commit()
            log.info("Expired %s leftover non-universe / stock signals", n)

    async def _ensure_settings(self, db: AsyncSession) -> None:
        row = await db.get(AppSettings, "default")
        if row is None:
            s = get_settings()
            db.add(
                AppSettings(
                    id="default",
                    payload={
                        "min_score": s.min_opportunity_score,
                        "enable_crypto": True,
                        "enable_stocks": False,
                        "learn_from_outcomes": True,
                        "enabled_strategies": [x.value for x in STRATEGY_LABELS],
                        "enabled_timeframes": [t.value for t in Timeframe],
                    },
                )
            )
            await db.commit()
        else:
            payload = dict(row.payload or {})
            payload["enable_stocks"] = False
            payload["enable_crypto"] = True
            payload.setdefault("learn_from_outcomes", True)
            row.payload = payload
            await db.commit()

    async def get_prefs(self, db: AsyncSession) -> dict:
        row = await db.get(AppSettings, "default")
        s = get_settings()
        if not row:
            return {
                "min_score": s.min_opportunity_score,
                "enable_crypto": True,
                "enable_stocks": False,
                "learn_from_outcomes": True,
                "enabled_strategies": [x.value for x in STRATEGY_LABELS],
                "enabled_timeframes": [t.value for t in Timeframe],
            }
        payload = dict(row.payload)
        payload["enable_stocks"] = False
        payload["enable_crypto"] = True
        payload.setdefault("learn_from_outcomes", True)
        return payload

    async def _reload_learner(self, db: AsyncSession) -> None:
        rows = (await db.execute(select(Signal))).scalars().all()
        profile = learn.build_profile(rows)
        prefs = await self.get_prefs(db)
        self.learn_enabled = bool(prefs.get("learn_from_outcomes", True))
        journal, snapshot = await self._load_learner_journal(db)
        previous = snapshot or learn.empty_profile()
        event = learn.diff_profiles(previous, profile)
        if event:
            journal = [event, *journal][: learn.JOURNAL_MAX]
            await self._save_learner_journal(db, journal, learn.compact_snapshot(profile))
            _push_activity("learner", event["summary"], {"changes": event["changes"][:4]})
        else:
            await self._save_learner_journal(db, journal, learn.compact_snapshot(profile))
        self.learner_profile = learn.attach_journal(profile, journal)

    async def _load_learner_journal(self, db: AsyncSession) -> tuple[list[dict], dict]:
        row = await db.get(AppSettings, "learner")
        payload = dict(row.payload or {}) if row else {}
        return list(payload.get("journal") or []), dict(payload.get("snapshot") or {})

    async def _save_learner_journal(self, db: AsyncSession, journal: list[dict], snapshot: dict) -> None:
        payload = {"journal": journal, "snapshot": snapshot}
        row = await db.get(AppSettings, "learner")
        if row is None:
            db.add(AppSettings(id="learner", payload=payload))
        else:
            row.payload = payload
            flag_modified(row, "payload")
        await db.commit()

    async def _refresh_quotes_and_outcomes(self) -> dict:
        quotes = await self.provider.quotes()
        await cache_set(
            "radar:quotes",
            json.dumps({k: q.model_dump() for k, q in quotes.items()}, default=str),
            30,
        )
        prices = {sym: q.price for sym, q in quotes.items()}
        async with SessionLocal() as db:
            changed = await update_outcomes(db, prices)
            closed = False
            for sig in changed:
                await hub.broadcast("signal.updated", _signal_payload(sig))
                _push_activity("outcome", f"{sig.symbol} → {sig.status}", {"symbol": sig.symbol})
                if sig.status in learn.DECIDED_STATUSES:
                    closed = True
            if closed:
                await self._reload_learner(db)
        now = time.monotonic()
        if now - self._last_tick_broadcast >= 2.0:
            self._last_tick_broadcast = now
            await hub.broadcast(
                "scanner.tick",
                {
                    "scans": self.scans,
                    "ts": utcnow().isoformat(),
                    "quotes": {k: q.model_dump() for k, q in quotes.items()},
                    "tape": self.provider.status(),
                },
            )
        return quotes

    async def scan_once(self) -> None:
        quotes = await self._refresh_quotes_and_outcomes()
        await self.scan_universe(quotes)

    async def scan_universe(self, quotes: dict) -> None:
        pairs = [(spec["symbol"], tf) for spec in self.provider.universe() for tf in Timeframe]
        await self.scan_pairs(quotes, pairs)

    async def scan_pairs(self, quotes: dict, pairs: list[tuple[str, Timeframe]]) -> None:
        async with SessionLocal() as db:
            prefs = await self.get_prefs(db)
            enabled_tf = set(prefs.get("enabled_timeframes", [t.value for t in Timeframe]))
            by_symbol = {s["symbol"]: s for s in self.provider.universe()}
            for symbol, tf in pairs:
                spec = by_symbol.get(symbol)
                if not spec:
                    continue
                if tf.value not in enabled_tf:
                    continue
                quote = quotes.get(symbol)
                if not quote:
                    continue
                bars = await self.provider.historical(symbol, tf, 280, closed_only=True)
                if len(bars) < 60:
                    continue
                ind, _series = indicators.compute(bars, tf.value)
                learn_on = bool(prefs.get("learn_from_outcomes", True))
                self.learn_enabled = learn_on
                knobs = self.learner_profile.get("knobs") if learn_on else None
                hits = detect(bars, ind, tf, knobs)
                for hit in hits:
                    if hit.strategy and hit.strategy.value not in prefs.get(
                        "enabled_strategies", [x.value for x in STRATEGY_LABELS]
                    ):
                        continue
                    await self._handle_hit(db, spec, quote, bars, ind, hit, prefs)
            self.scans += 1
            self.last_scan_at = utcnow()

    async def _handle_hit(
        self,
        db: AsyncSession,
        spec: dict,
        quote,
        bars: list[Bar],
        ind: IndicatorSnapshot,
        hit: StrategyHit,
        prefs: dict,
    ) -> None:
        asset_type = AssetType(spec["asset_type"]) if not isinstance(spec["asset_type"], AssetType) else spec["asset_type"]
        if asset_type != AssetType.CRYPTO:
            return
        allow, flags = risk.evaluate(
            hit, ind, asset_type=asset_type, spread_bps=quote.spread_bps, price=quote.price
        )
        if not allow:
            return
        higher_aligned = await self._higher_tf_aligned(spec["symbol"], hit)
        catalyst, catalyst_facts = await self.provider.news_score(spec["symbol"], asset_type)
        components = score_setup(
            hit,
            ind,
            asset_type=asset_type,
            spread_bps=quote.spread_bps,
            catalyst=catalyst,
            higher_tf_aligned=higher_aligned,
            price=quote.price,
        )
        learn_on = bool(prefs.get("learn_from_outcomes", True))
        scored = learn.apply_profile(
            self.learner_profile,
            components.model_dump(),
            strategy=hit.strategy.value if hit.strategy else "",
            timeframe=hit.timeframe.value,
            direction=hit.direction.value,
            enabled=learn_on,
        )
        min_score = float(prefs.get("min_score", get_settings().min_opportunity_score))
        floor = learn.effective_min_score(min_score, scored, hit.direction.value)

        existing = (
            await db.execute(
                select(Signal).where(
                    Signal.symbol == spec["symbol"],
                    Signal.strategy == (hit.strategy.value if hit.strategy else ""),
                    Signal.timeframe == hit.timeframe.value,
                    Signal.status.in_([SignalStatus.ACTIVE.value, SignalStatus.TARGET_1_HIT.value]),
                )
            )
        ).scalar_one_or_none()

        if existing is None:
            if hit.direction in {Direction.LONG, Direction.SHORT} and scored["total"] < floor:
                return
            if hit.direction == Direction.WATCH and scored["total"] < min(min_score, 48):
                return

        snapshot = {
            "price": quote.price,
            "bid": quote.bid,
            "ask": quote.ask,
            "spread_bps": quote.spread_bps,
            "change_pct": quote.change_pct,
            "source": getattr(quote, "source", "unknown"),
            "quote_ts": quote.ts.isoformat() if quote.ts else None,
            "indicators": ind.model_dump(),
            "last_bar": bars[-1].model_dump(mode="json"),
            "catalyst_facts": catalyst_facts,
            "higher_tf_aligned": higher_aligned,
        }
        reasons = list(hit.reasons) + catalyst_facts
        note = scored.get("learner_note")
        raw_total = scored.get("raw_total")
        if learn_on and note and raw_total is not None and abs(float(scored["total"]) - float(raw_total)) >= 2:
            reasons.append(f"Outcome calibration: {float(raw_total):.0f} → {float(scored['total']):.0f}. {note}")
        tweaks = scored.get("learner_tweaks") or []
        if learn_on and tweaks:
            reasons.append("Strategy knobs: " + "; ".join(str(c.get("why") or "") for c in tweaks[:2] if c.get("why")))

        material = existing is None or abs(existing.score - scored["total"]) >= 8
        if existing:
            existing.current_price = quote.price
            existing.score = scored["total"]
            existing.score_components = scored
            existing.risk_flags = flags
            existing.reasons = reasons
            existing.evidence = hit.evidence
            existing.snapshot = snapshot
            existing.entry_low = hit.entry_low
            existing.entry_high = hit.entry_high
            existing.stop = hit.stop
            existing.target_1 = hit.target_1
            existing.target_2 = hit.target_2
            existing.invalidation = hit.invalidation
            existing.risk_reward = hit.risk_reward
            existing.direction = hit.direction.value
            await db.commit()
            await db.refresh(existing)
            sig = existing
            event = "signal.updated"
        else:
            asset_id = self.asset_ids.get(spec["symbol"])
            if asset_id is None:
                return
            sig = Signal(
                asset_id=asset_id,
                symbol=spec["symbol"],
                asset_type=AssetType.CRYPTO.value,
                direction=hit.direction.value,
                strategy=hit.strategy.value if hit.strategy else "unknown",
                timeframe=hit.timeframe.value,
                status=SignalStatus.ACTIVE.value,
                score=scored["total"],
                current_price=quote.price,
                entry_low=hit.entry_low,
                entry_high=hit.entry_high,
                invalidation=hit.invalidation,
                stop=hit.stop,
                target_1=hit.target_1,
                target_2=hit.target_2,
                risk_reward=hit.risk_reward,
                score_components=scored,
                risk_flags=flags,
                reasons=reasons,
                evidence=hit.evidence,
                snapshot=snapshot,
            )
            db.add(sig)
            await db.flush()
            db.add(
                SignalEvent(
                    signal_id=sig.id,
                    status=SignalStatus.ACTIVE.value,
                    price=quote.price,
                    note="Detected by deterministic scanner on a closed Binance candle.",
                    ts=utcnow(),
                )
            )
            await db.commit()
            await db.refresh(sig)
            event = "signal.created"
            _push_activity(
                "signal",
                f"{sig.direction} {sig.symbol} {STRATEGY_LABELS.get(hit.strategy, sig.strategy)} ({sig.timeframe}) score {sig.score:.0f}",
                {"symbol": sig.symbol, "id": str(sig.id)},
            )

        if material and scored["total"] >= get_settings().ai_min_score and hit.direction != Direction.NONE:
            try:
                sig.ai_explanation = await self.explainer.explain(_facts(sig))
                await db.commit()
            except Exception:
                log.exception("AI explain failed for %s", sig.symbol)

        await hub.broadcast(event, _signal_payload(sig))

    async def _higher_tf_aligned(self, symbol: str, hit: StrategyHit) -> bool:
        higher = {Timeframe.M5: Timeframe.M15, Timeframe.M15: Timeframe.H1}.get(hit.timeframe)
        if not higher:
            return True
        bars = await self.provider.historical(symbol, higher, 80, closed_only=True)
        if len(bars) < 55:
            return False
        ind, _ = indicators.compute(bars, higher.value)
        if hit.direction == Direction.LONG:
            return bool(ind.ema21 and ind.ema50 and ind.ema21 >= ind.ema50)
        if hit.direction == Direction.SHORT:
            return bool(ind.ema21 and ind.ema50 and ind.ema21 <= ind.ema50)
        return False


def _facts(sig: Signal) -> dict[str, Any]:
    from app.domain import StrategyId

    label = STRATEGY_LABELS.get(StrategyId(sig.strategy), sig.strategy) if sig.strategy in StrategyId._value2member_map_ else sig.strategy
    return {
        "symbol": sig.symbol,
        "asset_type": sig.asset_type,
        "direction": sig.direction,
        "strategy": sig.strategy,
        "strategy_label": label,
        "timeframe": sig.timeframe,
        "score": sig.score,
        "score_components": sig.score_components,
        "current_price": sig.current_price,
        "entry_low": sig.entry_low,
        "entry_high": sig.entry_high,
        "stop": sig.stop,
        "target_1": sig.target_1,
        "target_2": sig.target_2,
        "invalidation": sig.invalidation,
        "risk_reward": sig.risk_reward,
        "reasons": sig.reasons,
        "evidence": sig.evidence,
        "risk_flags": sig.risk_flags,
        "detected_at": sig.detected_at.isoformat() if sig.detected_at else None,
        "data_source": (sig.snapshot or {}).get("source", "binance_spot"),
        "disclaimer": "Levels are scanner-computed zones, not guarantees. Do not invent additional numbers.",
    }


def _signal_payload(sig: Signal) -> dict[str, Any]:
    from app.domain import StrategyId

    label = None
    if sig.strategy in StrategyId._value2member_map_:
        label = STRATEGY_LABELS[StrategyId(sig.strategy)]
    return {
        "id": str(sig.id),
        "asset_id": str(sig.asset_id),
        "symbol": sig.symbol,
        "asset_type": sig.asset_type,
        "direction": sig.direction,
        "strategy": sig.strategy,
        "strategy_label": label,
        "timeframe": sig.timeframe,
        "status": sig.status,
        "detected_at": sig.detected_at.isoformat() if sig.detected_at else None,
        "closed_at": sig.closed_at.isoformat() if sig.closed_at else None,
        "score": sig.score,
        "current_price": sig.current_price,
        "entry_low": sig.entry_low,
        "entry_high": sig.entry_high,
        "invalidation": sig.invalidation,
        "stop": sig.stop,
        "target_1": sig.target_1,
        "target_2": sig.target_2,
        "risk_reward": sig.risk_reward,
        "r_multiple": sig.r_multiple,
        "score_components": sig.score_components,
        "risk_flags": sig.risk_flags,
        "reasons": sig.reasons,
        "ai_explanation": sig.ai_explanation,
    }


scanner = Scanner()
