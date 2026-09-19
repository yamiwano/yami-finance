"""Live paper trading system.

Frozen model + simulated execution only. No real orders. No retraining.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import SessionLocal
from app.domain import Bar, Timeframe, utcnow
from app.market_data.base import MarketDataProvider
from app.models.paper import PaperAccount, PaperPosition, PaperSignal, PaperTrade
from app.models.research import ModelVersion
from app.research.features import extract_features, jsonable_features
from app.research.history import as_utc, load_bars, load_recent_bars
from app.research.model import active_model, load_payload, predict_proba
from app.research.opportunity import BARRIER_SPECS
from app.research.portfolio import COST_PER_EVENT
from app.research.spec import horizon_bars, horizon_hours, timeframe
from app.research.universe import select_universe_at

log = logging.getLogger("radar.paper")

PAPER_TRADING_ENABLED = True
FROZEN_MODEL = True
STARTING_CAPITAL = 10_000.0
TOP_K = 5
MAX_GROSS_EXPOSURE = 1.0


class PaperTradingError(Exception):
    pass


def _ensure_paper_mode() -> None:
    if not get_settings().paper_trading_enabled:
        raise PaperTradingError("Paper trading disabled")


def _ensure_no_real_orders() -> None:
    """Hard guard: this module must never call exchange order endpoints."""
    # No exchange client is created here; any attempt to import/order is a bug.
    pass


class PaperTradingService:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._running = False
        self._payload: dict[str, Any] | None = None
        self._model_version: ModelVersion | None = None
        self.status: dict[str, Any] = {
            "enabled": get_settings().paper_trading_enabled,
            "active": False,
            "last_cycle_at": None,
            "error": None,
        }

    async def start(self, provider: MarketDataProvider) -> None:
        _ensure_paper_mode()
        _ensure_no_real_orders()
        if not get_settings().paper_trading_enabled:
            self.status["active"] = False
            return
        self._running = True
        await self._load_frozen_model()
        self._task = asyncio.create_task(self._loop(provider), name="paper-trading")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _load_frozen_model(self) -> None:
        async with SessionLocal() as db:
            model = await active_model(db)
            if model is None or not model.blob:
                raise PaperTradingError("No active frozen model available for paper trading")
            self._model_version = model
            self._payload = load_payload(model.blob)
            log.info("Paper trading using model %s", model.id)

    async def _loop(self, provider: MarketDataProvider) -> None:
        while self._running:
            try:
                await self._cycle(provider)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("paper trading cycle failed")
                self.status["error"] = str(exc)
            await asyncio.sleep(60)

    async def _cycle(self, provider: MarketDataProvider) -> None:
        """Evaluate at each completed 1h candle close."""
        tf = timeframe()
        now = utcnow()
        # Align to last completed hour
        last_candle = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        async with SessionLocal() as db:
            account = await self._get_account(db)
            # Recover open positions
            open_positions = await self._open_positions(db)
            # Update exits for positions whose barrier/horizon has passed
            await self._update_exits(db, open_positions, provider)
            # Generate new signals if we haven't processed this candle
            exists = await self._signal_exists(db, last_candle)
            if exists:
                return
            await self._generate_signals(db, provider, last_candle, account)
            self.status["last_cycle_at"] = utcnow().isoformat()
            self.status["error"] = None

    async def _get_account(self, db: AsyncSession) -> PaperAccount:
        row = (await db.execute(select(PaperAccount))).scalar_one_or_none()
        if row is None:
            row = PaperAccount(
                id=uuid.uuid4(),
                cash=STARTING_CAPITAL,
                equity=STARTING_CAPITAL,
                starting_capital=STARTING_CAPITAL,
                realized_pnl=0.0,
                total_fees=0.0,
                total_slippage=0.0,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)
        return row

    async def _open_positions(self, db: AsyncSession) -> list[PaperPosition]:
        return list(
            (await db.execute(select(PaperPosition).where(PaperPosition.status == "open"))).scalars().all()
        )

    async def _signal_exists(self, db: AsyncSession, ts: datetime) -> bool:
        row = (
            await db.execute(select(PaperSignal.id).where(PaperSignal.ts == as_utc(ts)).limit(1))
        ).scalar_one_or_none()
        return row is not None

    async def _update_exits(
        self, db: AsyncSession, positions: list[PaperPosition], provider: MarketDataProvider
    ) -> None:
        for pos in positions:
            bars = await load_bars(db, pos.asset_id, timeframe().value)
            idx = {as_utc(b.ts): i for i, b in enumerate(bars)}
            entry_i = idx.get(as_utc(pos.entry_ts))
            if entry_i is None:
                continue
            up_pct, down_pct = pos.up_pct, pos.down_pct
            up_level = pos.entry_price * (1.0 + up_pct)
            down_level = pos.entry_price * (1.0 - down_pct)
            exit_price = None
            exit_ts = None
            exit_reason = None
            for j in range(entry_i + 1, min(entry_i + pos.horizon_bars + 1, len(bars))):
                bar = bars[j]
                hit_up = bar.high >= up_level
                hit_down = bar.low <= down_level
                if hit_up and hit_down:
                    exit_price = bar.open
                    exit_ts = as_utc(bar.ts)
                    exit_reason = "AMBIGUOUS"
                    break
                if hit_up:
                    exit_price = up_level
                    exit_ts = as_utc(bar.ts)
                    exit_reason = "UP_BARRIER"
                    break
                if hit_down:
                    exit_price = down_level
                    exit_ts = as_utc(bar.ts)
                    exit_reason = "DOWN_BARRIER"
                    break
            if exit_price is None and entry_i + pos.horizon_bars < len(bars):
                last = bars[entry_i + pos.horizon_bars]
                exit_price = last.close
                exit_ts = as_utc(last.ts)
                exit_reason = "TIMEOUT"
            if exit_price is not None:
                gross = exit_price / pos.entry_price - 1.0
                net = gross - pos.cost_rate
                notional = pos.weight * pos.account_starting_capital
                pnl = notional * net
                pos.status = "closed"
                pos.exit_ts = exit_ts
                pos.exit_price = exit_price
                pos.exit_reason = exit_reason
                pos.gross_return = gross
                pos.net_return = net
                pos.holding_hours = (exit_ts - pos.entry_ts).total_seconds() / 3600.0
                account = await self._get_account(db)
                account.cash += notional * (1.0 + net)
                account.realized_pnl += pnl
                account.total_fees += notional * pos.cost_rate
                account.updated_at = utcnow()
                db.add(
                    PaperTrade(
                        id=uuid.uuid4(),
                        signal_id=pos.signal_id,
                        position_id=pos.id,
                        symbol=pos.symbol,
                        entry_ts=pos.entry_ts,
                        entry_price=pos.entry_price,
                        exit_ts=exit_ts,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        gross_return=gross,
                        net_return=net,
                        cost=pos.cost_rate,
                        holding_hours=pos.holding_hours,
                        created_at=utcnow(),
                    )
                )
        await db.commit()

    async def _generate_signals(
        self,
        db: AsyncSession,
        provider: MarketDataProvider,
        ts: datetime,
        account: PaperAccount,
    ) -> None:
        tf = timeframe()
        # Build point-in-time universe from candles <= ts
        assets = (await db.execute(select(Asset).where(Asset.is_active.is_(True)))).scalars().all()
        bars_by = {}
        for asset in assets:
            bars_by[asset.symbol] = await load_bars(db, asset.id, tf.value)
        universe_rows = select_universe_at(bars_by, ts, top_n=get_settings().universe_size)
        selected = [r["symbol"] for r in universe_rows if r["selected"]]
        if not selected:
            log.info("No eligible universe at %s", ts)
            return

        # Feature generation and prediction
        candidates = []
        for symbol in selected:
            asset = next((a for a in assets if a.symbol == symbol), None)
            if asset is None:
                continue
            bars = bars_by.get(symbol) or []
            idx = {as_utc(b.ts): i for i, b in enumerate(bars)}
            i = idx.get(as_utc(ts))
            if i is None or i < 80:
                continue
            feats = extract_features(bars[: i + 1], tf.value)
            if not feats:
                continue
            p = predict_proba(self._payload, feats)
            candidates.append(
                {
                    "symbol": symbol,
                    "asset_id": asset.id,
                    "features": jsonable_features(feats),
                    "probability": p,
                    "bars": bars,
                    "bar_index": i,
                }
            )
        if not candidates:
            return
        candidates.sort(key=lambda c: c["probability"], reverse=True)

        # Record all signals
        for rank, cand in enumerate(candidates, start=1):
            db.add(
                PaperSignal(
                    id=uuid.uuid4(),
                    ts=as_utc(ts),
                    symbol=cand["symbol"],
                    asset_id=cand["asset_id"],
                    direction="LONG",
                    model_id=self._model_version.id,
                    model_version=str(self._model_version.id),
                    feature_version="36-feature-v1",
                    experiment_version="paper_trading_v1",
                    probability=cand["probability"],
                    rank=rank,
                    barrier="up3_before_down2",
                    status="selected" if rank <= TOP_K else "rejected",
                    features=cand["features"],
                    created_at=utcnow(),
                )
            )
        await db.commit()

        # Enter top-K at next candle open with available capital
        available = account.cash
        open_weight = sum(p.weight for p in await self._open_positions(db))
        for cand in candidates[:TOP_K]:
            if available <= 0 or open_weight >= MAX_GROSS_EXPOSURE:
                break
            weight = min(1.0 / TOP_K, MAX_GROSS_EXPOSURE - open_weight, available / STARTING_CAPITAL)
            if weight <= 0:
                break
            bars = cand["bars"]
            i = cand["bar_index"]
            if i + 1 >= len(bars):
                continue
            entry_bar = bars[i + 1]
            entry_price = float(entry_bar.open)
            if entry_price <= 0:
                continue
            notional = weight * STARTING_CAPITAL
            account.cash -= notional
            open_weight += weight
            signal = (
                await db.execute(
                    select(PaperSignal)
                    .where(PaperSignal.ts == as_utc(ts), PaperSignal.symbol == cand["symbol"])
                    .order_by(PaperSignal.created_at.desc())
                )
            ).scalar_one()
            db.add(
                PaperPosition(
                    id=uuid.uuid4(),
                    signal_id=signal.id,
                    account_id=account.id,
                    symbol=cand["symbol"],
                    asset_id=cand["asset_id"],
                    direction="LONG",
                    entry_ts=as_utc(entry_bar.ts),
                    entry_price=entry_price,
                    weight=weight,
                    up_pct=0.03,
                    down_pct=0.02,
                    horizon_bars=horizon_bars(tf, horizon_hours()),
                    cost_rate=COST_PER_EVENT,
                    status="open",
                    account_starting_capital=STARTING_CAPITAL,
                    created_at=utcnow(),
                )
            )
        account.updated_at = utcnow()
        await db.commit()

    async def snapshot(self) -> dict[str, Any]:
        async with SessionLocal() as db:
            account = await self._get_account(db)
            open_positions = await self._open_positions(db)
            trades = (await db.execute(select(PaperTrade))).scalars().all()
            signals = (await db.execute(select(PaperSignal).order_by(PaperSignal.ts.desc()).limit(50))).scalars().all()
            returns = np.array([t.net_return for t in trades], dtype=float) if trades else np.array([])
            equity = account.cash + sum(
                p.weight * p.account_starting_capital for p in open_positions
            )
            return {
                "status": dict(self.status),
                "account": {
                    "starting_capital": account.starting_capital,
                    "cash": account.cash,
                    "equity": equity,
                    "realized_pnl": account.realized_pnl,
                    "total_fees": account.total_fees,
                    "total_slippage": account.total_slippage,
                    "open_positions": len(open_positions),
                },
                "positions": [
                    {
                        "symbol": p.symbol,
                        "direction": p.direction,
                        "entry_ts": p.entry_ts.isoformat(),
                        "entry_price": p.entry_price,
                        "weight": p.weight,
                        "status": p.status,
                    }
                    for p in open_positions
                ],
                "signals": [
                    {
                        "ts": s.ts.isoformat(),
                        "symbol": s.symbol,
                        "probability": s.probability,
                        "rank": s.rank,
                        "status": s.status,
                        "model_version": s.model_version,
                    }
                    for s in signals
                ],
                "performance": {
                    "n_trades": len(trades),
                    "win_rate": float(np.mean(returns > 0)) if len(returns) else None,
                    "avg_trade": float(np.mean(returns)) if len(returns) else None,
                    "median_trade": float(np.median(returns)) if len(returns) else None,
                    "cumulative_return": equity / account.starting_capital - 1.0,
                },
            }


paper_trading_service = PaperTradingService()
