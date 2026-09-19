from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain import CLOSED_STATUSES, Direction, SignalStatus, utcnow
from app.models.signal import Signal, SignalEvent


def _status_for_price(sig: Signal, price: float) -> SignalStatus | None:
    if sig.direction == Direction.LONG.value:
        if price <= sig.stop:
            return SignalStatus.STOPPED
        if price <= sig.invalidation and price < sig.entry_low:
            return SignalStatus.INVALIDATED
        if price >= sig.target_2:
            return SignalStatus.TARGET_2_HIT
        if price >= sig.target_1:
            return SignalStatus.TARGET_1_HIT
    elif sig.direction == Direction.SHORT.value:
        if price >= sig.stop:
            return SignalStatus.STOPPED
        if price >= sig.invalidation and price > sig.entry_high:
            return SignalStatus.INVALIDATED
        if price <= sig.target_2:
            return SignalStatus.TARGET_2_HIT
        if price <= sig.target_1:
            return SignalStatus.TARGET_1_HIT
    return None


def _r_multiple(sig: Signal, exit_price: float) -> float:
    entry = (sig.entry_low + sig.entry_high) / 2
    risk = abs(entry - sig.stop)
    if risk <= 0:
        return 0.0
    if sig.direction == Direction.LONG.value:
        return (exit_price - entry) / risk
    return (entry - exit_price) / risk


def _aware(dt: datetime | None) -> datetime:
    if dt is None:
        return utcnow()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _ttl(timeframe: str) -> timedelta:
    return {"5m": timedelta(hours=18), "15m": timedelta(days=2), "1h": timedelta(days=5)}.get(
        timeframe, timedelta(days=2)
    )


async def update_outcomes(db: AsyncSession, prices: dict[str, float]) -> list[Signal]:
    result = await db.execute(
        select(Signal).where(Signal.status.in_([SignalStatus.ACTIVE.value, SignalStatus.TARGET_1_HIT.value]))
    )
    changed: list[Signal] = []
    now = utcnow()
    for sig in result.scalars():
        price = prices.get(sig.symbol)
        if price is None:
            continue
        sig.current_price = price
        nxt = _status_for_price(sig, price)
        expired = now - _aware(sig.detected_at) > _ttl(sig.timeframe)
        new_status = None
        note = ""
        if nxt == SignalStatus.STOPPED:
            new_status = SignalStatus.STOPPED.value
            note = "Stop traded."
        elif nxt == SignalStatus.INVALIDATED and sig.status == SignalStatus.ACTIVE.value:
            new_status = SignalStatus.INVALIDATED.value
            note = "Invalidation traded before targets."
        elif nxt == SignalStatus.TARGET_2_HIT:
            new_status = SignalStatus.TARGET_2_HIT.value
            note = "Target 2 traded."
        elif nxt == SignalStatus.TARGET_1_HIT and sig.status == SignalStatus.ACTIVE.value:
            new_status = SignalStatus.TARGET_1_HIT.value
            note = "Target 1 traded; trade still tracked."
        elif expired and sig.status == SignalStatus.ACTIVE.value:
            new_status = SignalStatus.EXPIRED.value
            note = "Signal expired without a decisive level being traded."

        if new_status and new_status != sig.status:
            sig.status = new_status
            if new_status in {s.value for s in CLOSED_STATUSES}:
                sig.closed_at = now
                sig.exit_price = price
                sig.r_multiple = _r_multiple(sig, price)
            db.add(SignalEvent(signal_id=sig.id, status=new_status, price=price, note=note, ts=now))
            changed.append(sig)
    if changed:
        await db.commit()
        for sig in changed:
            await db.refresh(sig)
    return changed
