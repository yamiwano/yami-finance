"""Assemble (features at T) → (what happened after T)."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain import utcnow
from app.models.research import ResearchSample
from app.research.features import extract_features, jsonable_features
from app.research.history import as_utc, load_bars
from app.research.labels import label_forward
from app.research.spec import LOOKBACK_BARS, horizon_bars, horizon_hours, move_pct, sample_stride, timeframe

log = logging.getLogger("radar.research.dataset")


async def build_samples(
    db: AsyncSession,
    asset_ids: dict[str, UUID],
) -> dict[str, int]:
    tf = timeframe()
    hours = horizon_hours()
    h_bars = horizon_bars(tf, hours)
    stride = sample_stride()
    created = 0
    labeled = 0
    for symbol, asset_id in asset_ids.items():
        bars = await load_bars(db, asset_id, tf.value)
        if len(bars) < LOOKBACK_BARS + h_bars:
            continue
        existing = (
            await db.execute(
                select(ResearchSample.ts, ResearchSample.significant_move).where(
                    ResearchSample.asset_id == asset_id,
                    ResearchSample.timeframe == tf.value,
                    ResearchSample.horizon_hours == hours,
                )
            )
        ).all()
        have = {as_utc(ts): move for ts, move in existing}
        last_i = len(bars) - 1
        for i in range(LOOKBACK_BARS - 1, last_i, stride):
            ts = as_utc(bars[i].ts)
            row_move = have.get(ts, "missing")
            if row_move == "missing":
                feats = extract_features(bars[: i + 1], tf.value)
                if not feats:
                    continue
                sample = ResearchSample(
                    asset_id=asset_id,
                    symbol=symbol,
                    timeframe=tf.value,
                    ts=ts,
                    horizon_hours=hours,
                    features=jsonable_features(feats),
                )
                labels = label_forward(bars[i + 1 :], bars[i].close, h_bars, move_pct())
                if labels:
                    sample.fwd_return = float(labels["fwd_return"])
                    sample.max_upside = float(labels["max_upside"])
                    sample.max_drawdown = float(labels["max_drawdown"])
                    sample.significant_move = bool(labels["significant_move"])
                    sample.labeled_at = utcnow()
                    labeled += 1
                db.add(sample)
                have[ts] = sample.significant_move
                created += 1
            elif row_move is None:
                sample = (
                    await db.execute(
                        select(ResearchSample).where(
                            ResearchSample.asset_id == asset_id,
                            ResearchSample.timeframe == tf.value,
                            ResearchSample.horizon_hours == hours,
                            ResearchSample.ts == ts,
                        )
                    )
                ).scalar_one()
                labels = label_forward(bars[i + 1 :], bars[i].close, h_bars, move_pct())
                if labels:
                    sample.fwd_return = float(labels["fwd_return"])
                    sample.max_upside = float(labels["max_upside"])
                    sample.max_drawdown = float(labels["max_drawdown"])
                    sample.significant_move = bool(labels["significant_move"])
                    sample.labeled_at = utcnow()
                    labeled += 1
                    have[ts] = sample.significant_move
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
    log.info("Research dataset created=%s newly_labeled=%s", created, labeled)
    return {"created": created, "labeled": labeled}


async def sample_stats(db: AsyncSession) -> dict:
    tf = timeframe().value
    hours = horizon_hours()
    filters = (ResearchSample.timeframe == tf, ResearchSample.horizon_hours == hours)
    total = (await db.execute(select(func.count()).select_from(ResearchSample).where(*filters))).scalar_one()
    labeled_n = (
        await db.execute(
            select(func.count())
            .select_from(ResearchSample)
            .where(*filters, ResearchSample.significant_move.is_not(None))
        )
    ).scalar_one()
    pos = (
        await db.execute(
            select(func.count()).select_from(ResearchSample).where(*filters, ResearchSample.significant_move.is_(True))
        )
    ).scalar_one()
    rate = (float(pos) / float(labeled_n)) if labeled_n else None
    return {
        "count": int(total or 0),
        "labeled": int(labeled_n or 0),
        "positives": int(pos or 0),
        "positive_rate": rate,
        "timeframe": tf,
        "horizon_hours": hours,
        "move_pct": move_pct(),
        "stride": sample_stride(),
    }
