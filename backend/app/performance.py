from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.domain import CLOSED_STATUSES, SignalStatus
from app.learn import build_profile, public_profile
from app.models.settings import AppSettings
from app.models.signal import Signal

RESET_KEY = "performance_reset_at"


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_reset_at(raw) -> datetime | None:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return _as_utc(raw)
    try:
        return _as_utc(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
    except ValueError:
        return None


def _period_start(prefs: AppSettings | None) -> datetime | None:
    if not prefs or not isinstance(prefs.payload, dict):
        return None
    return _parse_reset_at(prefs.payload.get(RESET_KEY))


async def reset_period(db: AsyncSession) -> dict:
    now = datetime.now(timezone.utc)
    row = await db.get(AppSettings, "default")
    payload = dict(row.payload) if row and isinstance(row.payload, dict) else {}
    payload[RESET_KEY] = now.isoformat()
    if row is None:
        row = AppSettings(id="default", payload=payload)
        db.add(row)
    else:
        row.payload = payload
        flag_modified(row, "payload")
    await db.commit()
    return await summarize(db)


async def summarize(db: AsyncSession) -> dict:
    prefs = await db.get(AppSettings, "default")
    since = _period_start(prefs)
    all_rows = (await db.execute(select(Signal))).scalars().all()
    rows = all_rows
    if since:
        rows = [s for s in all_rows if (detected := _as_utc(s.detected_at)) and detected >= since]
    total = len(rows)
    closed = [s for s in rows if s.status in {x.value for x in CLOSED_STATUSES} or s.status == SignalStatus.TARGET_1_HIT.value]
    wins = [s for s in rows if s.status in {SignalStatus.TARGET_1_HIT.value, SignalStatus.TARGET_2_HIT.value}]
    losses = [s for s in rows if s.status in {SignalStatus.STOPPED.value, SignalStatus.INVALIDATED.value}]
    r_vals = [s.r_multiple for s in rows if s.r_multiple is not None]
    avg_r = sum(r_vals) / len(r_vals) if r_vals else 0.0

    by_strategy: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0, "losses": 0, "r_sum": 0.0, "r_n": 0})
    by_range: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0, "losses": 0})
    by_side: dict[str, dict] = defaultdict(lambda: {"count": 0, "wins": 0, "losses": 0, "r_sum": 0.0, "r_n": 0})

    def bucket(score: float) -> str:
        if score < 60:
            return "50-59"
        if score < 70:
            return "60-69"
        if score < 80:
            return "70-79"
        if score < 90:
            return "80-89"
        return "90-100"

    for s in rows:
        st = by_strategy[s.strategy]
        st["count"] += 1
        if s.status in {SignalStatus.TARGET_1_HIT.value, SignalStatus.TARGET_2_HIT.value}:
            st["wins"] += 1
        if s.status in {SignalStatus.STOPPED.value, SignalStatus.INVALIDATED.value}:
            st["losses"] += 1
        if s.r_multiple is not None:
            st["r_sum"] += s.r_multiple
            st["r_n"] += 1

        b = by_range[bucket(s.score)]
        b["count"] += 1
        if s.status in {SignalStatus.TARGET_1_HIT.value, SignalStatus.TARGET_2_HIT.value}:
            b["wins"] += 1
        if s.status in {SignalStatus.STOPPED.value, SignalStatus.INVALIDATED.value}:
            b["losses"] += 1

        side = by_side[s.direction]
        side["count"] += 1
        if s.status in {SignalStatus.TARGET_1_HIT.value, SignalStatus.TARGET_2_HIT.value}:
            side["wins"] += 1
        if s.status in {SignalStatus.STOPPED.value, SignalStatus.INVALIDATED.value}:
            side["losses"] += 1
        if s.r_multiple is not None:
            side["r_sum"] += s.r_multiple
            side["r_n"] += 1

    def finish(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            item = dict(v)
            n = item.get("r_n") or 0
            item["avg_r"] = (item.get("r_sum") or 0) / n if n else None
            decided = item["wins"] + item["losses"]
            item["win_rate"] = item["wins"] / decided if decided else None
            out[k] = item
        return out

    decided = len(wins) + len(losses)
    learn_on = True
    if prefs and isinstance(prefs.payload, dict):
        learn_on = bool(prefs.payload.get("learn_from_outcomes", True))
    state = await db.get(AppSettings, "learner")
    journal = []
    if state and isinstance(state.payload, dict):
        journal = list(state.payload.get("journal") or [])

    return {
        "total_signals": total,
        "active": sum(1 for s in rows if s.status in {SignalStatus.ACTIVE.value, SignalStatus.TARGET_1_HIT.value}),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / decided) if decided else None,
        "average_r": avg_r,
        "by_strategy": finish(by_strategy),
        "by_score_range": finish(by_range),
        "by_direction": finish(by_side),
        "open_count": sum(1 for s in rows if s.status == SignalStatus.ACTIVE.value),
        "tracked_partial": len(closed),
        "period_started_at": since.isoformat() if since else None,
        "learner": public_profile(build_profile(all_rows), enabled=learn_on, journal=journal),
    }
