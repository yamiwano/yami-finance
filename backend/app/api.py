from datetime import datetime, timezone
from uuid import UUID
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import indicators, learn, performance as perf
from app.database import get_db
from app.domain import STRATEGY_LABELS, StrategyId, Timeframe
from app.models.asset import Asset
from app.models.settings import AppSettings
from app.models.signal import Signal
from app.models.watchlist import WatchlistItem
from app.market_data.universe import resolve_symbol
from app.research.service import research_service
from app.scanner import ACTIVITY, scanner
from app.schemas import ChartOut, CandleOut, SettingsIn, SignalOut, WatchlistIn

router = APIRouter()


def _label(strategy: str) -> str | None:
    if strategy in StrategyId._value2member_map_:
        return STRATEGY_LABELS[StrategyId(strategy)]
    return strategy


def to_out(s: Signal) -> SignalOut:
    return SignalOut(
        id=s.id,
        asset_id=s.asset_id,
        symbol=s.symbol,
        asset_type=s.asset_type,
        direction=s.direction,
        strategy=s.strategy,
        strategy_label=_label(s.strategy),
        timeframe=s.timeframe,
        status=s.status,
        detected_at=s.detected_at,
        closed_at=s.closed_at,
        score=s.score,
        current_price=s.current_price,
        entry_low=s.entry_low,
        entry_high=s.entry_high,
        invalidation=s.invalidation,
        stop=s.stop,
        target_1=s.target_1,
        target_2=s.target_2,
        risk_reward=s.risk_reward,
        r_multiple=s.r_multiple,
        exit_price=s.exit_price,
        score_components=s.score_components or {},
        risk_flags=s.risk_flags or [],
        reasons=s.reasons or [],
        evidence=s.evidence or {},
        snapshot=s.snapshot or {},
        ai_explanation=s.ai_explanation,
    )


@router.get("/health")
async def health():
    tape = scanner.provider.status()
    return {
        "ok": True,
        "scans": scanner.scans,
        "last_scan_at": scanner.last_scan_at,
        "provider": type(scanner.provider).__name__,
        "ready": scanner.provider.is_ready(),
        "tape": tape,
        "learner": learn.public_status(scanner.learner_profile, enabled=scanner.learn_enabled),
        "research": {
            "phase": research_service.status.get("phase"),
            "message": research_service.status.get("message"),
            "busy": research_service.status.get("busy"),
        },
    }


@router.get("/activity")
async def activity(limit: int = 40):
    return ACTIVITY[:limit]


@router.get("/quotes")
async def quotes():
    q = await scanner.provider.quotes()
    return {k: v.model_dump() for k, v in q.items()}


def _live_symbols() -> set[str]:
    return {s["symbol"] for s in scanner.provider.universe()}


@router.get("/assets")
async def assets(db: AsyncSession = Depends(get_db)):
    live = _live_symbols()
    if not live:
        return []
    rows = (await db.execute(select(Asset).where(Asset.symbol.in_(live)).order_by(Asset.symbol))).scalars().all()
    quotes = await scanner.provider.quotes()
    out = []
    for a in rows:
        quote = quotes.get(a.symbol)
        out.append(
            {
                "id": str(a.id),
                "symbol": a.symbol,
                "name": a.name,
                "asset_type": a.asset_type,
                "exchange": a.exchange,
                "price": quote.price if quote else None,
                "change_pct": quote.change_pct if quote else None,
                "volume": quote.volume if quote else None,
                "spread_bps": quote.spread_bps if quote else None,
                "bid": quote.bid if quote else None,
                "ask": quote.ask if quote else None,
            }
        )
    return out


def _signal_query(
    market: str | None,
    direction: str | None,
    strategy: str | None,
    timeframe: str | None,
    min_score: float | None,
    status: str | None,
    q: str | None,
) -> Select:
    stmt = select(Signal)
    if market:
        stmt = stmt.where(Signal.asset_type == market)
    else:
        stmt = stmt.where(Signal.asset_type == "crypto")
    if direction:
        stmt = stmt.where(Signal.direction == direction.upper())
    if strategy:
        stmt = stmt.where(Signal.strategy == strategy)
    if timeframe:
        stmt = stmt.where(Signal.timeframe == timeframe)
    if min_score is not None:
        stmt = stmt.where(Signal.score >= min_score)
    if status:
        if status == "open":
            stmt = stmt.where(Signal.status.in_(["active", "target_1_hit"]))
        else:
            stmt = stmt.where(Signal.status == status)
    if q:
        stmt = stmt.where(Signal.symbol.ilike(f"%{q}%"))
    return stmt.order_by(Signal.score.desc(), Signal.detected_at.desc())


@router.get("/signals", response_model=list[SignalOut])
async def list_signals(
    market: str | None = None,
    direction: str | None = None,
    strategy: str | None = None,
    timeframe: str | None = None,
    min_score: float | None = None,
    status: str | None = "open",
    q: str | None = None,
    limit: int = Query(100, le=300),
    db: AsyncSession = Depends(get_db),
):
    stmt = _signal_query(market, direction, strategy, timeframe, min_score, status, q).limit(limit)
    rows = (await db.execute(stmt)).scalars().all()
    return [to_out(s) for s in rows]


@router.get("/signals/{signal_id}", response_model=SignalOut)
async def get_signal(signal_id: UUID, db: AsyncSession = Depends(get_db)):
    row = await db.get(Signal, signal_id)
    if not row:
        raise HTTPException(404, "Signal not found")
    return to_out(row)


@router.get("/signals/{signal_id}/chart", response_model=ChartOut)
async def signal_chart(signal_id: UUID, db: AsyncSession = Depends(get_db)):
    row = await db.get(Signal, signal_id)
    if not row:
        raise HTTPException(404, "Signal not found")
    tf = Timeframe(row.timeframe)
    bars = await scanner.provider.historical(row.symbol, tf, 240)
    _, series = indicators.compute(bars, tf.value)
    return ChartOut(
        symbol=row.symbol,
        timeframe=row.timeframe,
        candles=[CandleOut(**b.model_dump()) for b in bars],
        ema9=series.get("ema9", []),
        ema21=series.get("ema21", []),
        ema50=series.get("ema50", []),
        vwap=series.get("vwap", []),
        rsi=series.get("rsi", []),
    )


@router.get("/charts/{symbol}")
async def symbol_chart(symbol: str, timeframe: str = "15m"):
    tf = Timeframe(timeframe)
    live = _live_symbols()
    resolved = resolve_symbol(symbol, live) or symbol.upper()
    bars = await scanner.provider.historical(resolved, tf, 240)
    if not bars:
        raise HTTPException(404, "Unknown symbol")
    _, series = indicators.compute(bars, tf.value)
    return ChartOut(
        symbol=resolved,
        timeframe=tf.value,
        candles=[CandleOut(**b.model_dump()) for b in bars],
        ema9=series.get("ema9", []),
        ema21=series.get("ema21", []),
        ema50=series.get("ema50", []),
        vwap=series.get("vwap", []),
        rsi=series.get("rsi", []),
    )


@router.get("/scanner")
async def scanner_table(
    market: str | None = None,
    direction: str | None = None,
    strategy: str | None = None,
    timeframe: str | None = None,
    min_score: float = 0,
    q: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    live = _live_symbols()
    if not live:
        return []
    assets = (
        await db.execute(select(Asset).where(Asset.symbol.in_(live)).order_by(Asset.symbol))
    ).scalars().all()
    quotes = await scanner.provider.quotes()
    open_signals = (
        await db.execute(select(Signal).where(Signal.status.in_(["active", "target_1_hit"])))
    ).scalars().all()
    best: dict[str, Signal] = {}
    for s in open_signals:
        prev = best.get(s.symbol)
        if prev is None or s.score > prev.score:
            best[s.symbol] = s

    rows = []
    for a in assets:
        if market and a.asset_type != market:
            continue
        if q and q.upper() not in a.symbol:
            continue
        quote = quotes.get(a.symbol)
        sig = best.get(a.symbol)
        if direction and (not sig or sig.direction != direction.upper()):
            continue
        if strategy and (not sig or sig.strategy != strategy):
            continue
        if timeframe and (not sig or sig.timeframe != timeframe):
            continue
        score = sig.score if sig else 0
        if score < min_score:
            continue
        rows.append(
            {
                "asset_id": str(a.id),
                "symbol": a.symbol,
                "name": a.name,
                "asset_type": a.asset_type,
                "exchange": a.exchange,
                "price": quote.price if quote else None,
                "change_pct": quote.change_pct if quote else None,
                "volume": quote.volume if quote else None,
                "spread_bps": quote.spread_bps if quote else None,
                "bid": quote.bid if quote else None,
                "ask": quote.ask if quote else None,
                "signal": to_out(sig).model_dump() if sig else None,
            }
        )
    rows.sort(key=lambda r: (r["signal"]["score"] if r["signal"] else -1), reverse=True)
    return rows


@router.get("/history", response_model=list[SignalOut])
async def history(
    status: str | None = None,
    limit: int = 200,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Signal).where(Signal.asset_type == "crypto").order_by(Signal.detected_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(Signal.status == status)
    rows = (await db.execute(stmt)).scalars().all()
    return [to_out(s) for s in rows]


@router.get("/performance")
async def performance(db: AsyncSession = Depends(get_db)):
    return await perf.summarize(db)


@router.post("/performance/reset")
async def reset_performance(db: AsyncSession = Depends(get_db)):
    return await perf.reset_period(db)


@router.get("/research")
async def research_status():
    return await research_service.snapshot()


@router.post("/research/train")
async def research_train():
    if research_service.status.get("busy"):
        return {"ok": True, "status": "already_running", "message": research_service.status.get("message")}
    asyncio.create_task(research_service.train_now())
    return {"ok": True, "status": "started"}


@router.get("/settings")
async def get_settings_route(db: AsyncSession = Depends(get_db)):
    return await scanner.get_prefs(db)


@router.put("/settings")
async def put_settings(body: SettingsIn, db: AsyncSession = Depends(get_db)):
    row = await db.get(AppSettings, "default")
    payload = dict(row.payload) if row else {}
    data = body.model_dump(exclude_none=True)
    payload.update(data)
    if row is None:
        row = AppSettings(id="default", payload=payload)
        db.add(row)
    else:
        row.payload = payload
    await db.commit()
    scanner.learn_enabled = bool(payload.get("learn_from_outcomes", True))
    return payload


@router.get("/watchlist")
async def get_watchlist(db: AsyncSession = Depends(get_db)):
    items = (await db.execute(select(WatchlistItem))).scalars().all()
    quotes = await scanner.provider.quotes()
    out = []
    for it in items:
        asset = await db.get(Asset, it.asset_id)
        if not asset:
            continue
        q = quotes.get(asset.symbol)
        out.append(
            {
                "id": str(it.id),
                "asset_id": str(asset.id),
                "symbol": asset.symbol,
                "name": asset.name,
                "asset_type": asset.asset_type,
                "notes": it.notes,
                "created_at": it.created_at,
                "price": q.price if q else None,
                "change_pct": q.change_pct if q else None,
            }
        )
    return out


@router.post("/watchlist")
async def add_watchlist(body: WatchlistIn, db: AsyncSession = Depends(get_db)):
    live = {s["symbol"] for s in scanner.provider.universe()}
    symbol = resolve_symbol(body.symbol, live)
    if not symbol:
        raise HTTPException(404, "Unknown Binance USDT pair")
    asset = (await db.execute(select(Asset).where(Asset.symbol == symbol))).scalar_one_or_none()
    if not asset:
        raise HTTPException(404, "Unknown symbol")
    existing = (await db.execute(select(WatchlistItem).where(WatchlistItem.asset_id == asset.id))).scalar_one_or_none()
    if existing:
        existing.notes = body.notes
        await db.commit()
        return {"ok": True, "id": str(existing.id)}
    item = WatchlistItem(asset_id=asset.id, notes=body.notes)
    db.add(item)
    await db.commit()
    return {"ok": True, "id": str(item.id)}


@router.delete("/watchlist/{item_id}")
async def delete_watchlist(item_id: UUID, db: AsyncSession = Depends(get_db)):
    row = await db.get(WatchlistItem, item_id)
    if not row:
        raise HTTPException(404, "Not on watchlist")
    await db.delete(row)
    await db.commit()
    return {"ok": True}


@router.get("/meta")
async def meta():
    return {
        "strategies": [{"id": k.value, "label": v} for k, v in STRATEGY_LABELS.items()],
        "timeframes": [t.value for t in Timeframe],
        "data_source": "Binance spot (REST seed + WebSocket klines/bookTicker). No simulated prices.",
        "disclaimer": "Yami Financier is an analytical decision-support tool for Binance USDT spot. Signals are not trade recommendations and levels are not guaranteed.",
        "now": datetime.now(timezone.utc).isoformat(),
        "tape": scanner.provider.status(),
    }
