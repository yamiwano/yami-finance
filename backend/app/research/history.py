"""Persist closed OHLCV into the candles table."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain import Bar, Timeframe, utcnow
from app.market_data.base import MarketDataProvider
from app.models.asset import Asset
from app.models.candle import Candle
from app.research.spec import backfill_days, timeframe as research_tf

log = logging.getLogger("radar.research.history")


def as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


async def upsert_bars(
    db: AsyncSession,
    asset_id: UUID,
    timeframe: str,
    bars: list[Bar],
) -> int:
    closed = [b for b in bars if b.closed]
    if not closed:
        return 0
    tss = [as_utc(b.ts) for b in closed]
    existing = (
        await db.execute(
            select(Candle.ts).where(
                Candle.asset_id == asset_id,
                Candle.timeframe == timeframe,
                Candle.ts.in_(tss),
            )
        )
    ).scalars().all()
    have = {as_utc(t) for t in existing}
    added = 0
    for bar, ts in zip(closed, tss):
        if ts in have:
            continue
        db.add(
            Candle(
                asset_id=asset_id,
                timeframe=timeframe,
                ts=ts,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
        )
        have.add(ts)
        added += 1
    return added


async def _commit_candles(db: AsyncSession) -> None:
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()


async def persist_closed_pairs(
    db: AsyncSession,
    provider: MarketDataProvider,
    asset_ids: dict[str, UUID],
    pairs: list[tuple[str, Timeframe]],
) -> int:
    added = 0
    for symbol, tf in pairs:
        asset_id = asset_ids.get(symbol)
        if asset_id is None:
            continue
        bars = await provider.historical(symbol, tf, 2, closed_only=True)
        if not bars:
            continue
        added += await upsert_bars(db, asset_id, tf.value, bars[-1:])
    if added:
        await _commit_candles(db)
    return added


async def persist_provider_window(
    db: AsyncSession,
    provider: MarketDataProvider,
    asset_ids: dict[str, UUID],
    *,
    timeframe: Timeframe | None = None,
    limit: int = 400,
) -> int:
    tf = timeframe or research_tf()
    added = 0
    for symbol, asset_id in asset_ids.items():
        bars = await provider.historical(symbol, tf, limit, closed_only=True)
        added += await upsert_bars(db, asset_id, tf.value, bars)
    if added:
        await _commit_candles(db)
    return added


async def load_recent_bars(
    db: AsyncSession,
    asset_id: UUID,
    timeframe: str,
    limit: int = 80,
) -> list[Bar]:
    rows = (
        await db.execute(
            select(Candle)
            .where(Candle.asset_id == asset_id, Candle.timeframe == timeframe)
            .order_by(Candle.ts.desc())
            .limit(limit)
        )
    ).scalars().all()
    rows = list(reversed(list(rows)))
    return [
        Bar(ts=as_utc(r.ts), open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume, closed=True)
        for r in rows
    ]


async def load_bars_since(
    db: AsyncSession,
    asset_id: UUID,
    timeframe: str,
    start: datetime,
) -> list[Bar]:
    rows = (
        await db.execute(
            select(Candle)
            .where(Candle.asset_id == asset_id, Candle.timeframe == timeframe, Candle.ts >= start)
            .order_by(Candle.ts.asc())
        )
    ).scalars().all()
    return [
        Bar(ts=as_utc(r.ts), open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume, closed=True)
        for r in rows
    ]


async def load_bars(
    db: AsyncSession,
    asset_id: UUID,
    timeframe: str,
) -> list[Bar]:
    rows = (
        await db.execute(
            select(Candle)
            .where(Candle.asset_id == asset_id, Candle.timeframe == timeframe)
            .order_by(Candle.ts.asc())
        )
    ).scalars().all()
    return [
        Bar(ts=as_utc(r.ts), open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume, closed=True)
        for r in rows
    ]


async def candle_stats(db: AsyncSession, timeframe: str) -> dict:
    count = (await db.execute(select(func.count()).select_from(Candle).where(Candle.timeframe == timeframe))).scalar_one()
    oldest = (await db.execute(select(func.min(Candle.ts)).where(Candle.timeframe == timeframe))).scalar_one()
    newest = (await db.execute(select(func.max(Candle.ts)).where(Candle.timeframe == timeframe))).scalar_one()
    symbols = (
        await db.execute(
            select(func.count(func.distinct(Candle.asset_id))).where(Candle.timeframe == timeframe)
        )
    ).scalar_one()
    return {
        "count": int(count or 0),
        "symbols": int(symbols or 0),
        "oldest": as_utc(oldest).isoformat() if oldest else None,
        "newest": as_utc(newest).isoformat() if newest else None,
        "timeframe": timeframe,
    }


async def backfill(
    db: AsyncSession,
    provider: MarketDataProvider,
    asset_ids: dict[str, UUID],
    *,
    days: int | None = None,
) -> int:
    tf = research_tf()
    end = utcnow()
    start = end - timedelta(days=days or backfill_days())
    added = 0
    symbols = list(asset_ids.items())
    log.info("Research backfill start symbols=%s tf=%s days=%s", len(symbols), tf.value, days or backfill_days())
    for i, (symbol, asset_id) in enumerate(symbols, start=1):
        try:
            bars = await provider.historical_range(symbol, tf, start, end)
        except Exception:
            log.exception("backfill failed %s %s", symbol, tf.value)
            continue
        n = await upsert_bars(db, asset_id, tf.value, bars)
        added += n
        if i % 8 == 0 or i == len(symbols):
            await _commit_candles(db)
            log.info("Research backfill %s/%s +%s bars (total +%s)", i, len(symbols), n, added)
    await _commit_candles(db)
    return added


async def asset_map(db: AsyncSession) -> dict[str, UUID]:
    rows = (await db.execute(select(Asset).where(Asset.is_active.is_(True)))).scalars().all()
    return {row.symbol: row.id for row in rows}
