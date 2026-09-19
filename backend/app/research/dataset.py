"""Assemble (features at T) → (what happened after T)."""

from __future__ import annotations

import logging
from collections import defaultdict
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain import utcnow
from app.models.research import ResearchSample
from app.research.features import extract_features, jsonable_features
from app.research.history import as_utc, load_bars
from app.research.labels import label_forward, labels_from_outcomes
from app.research.spec import LOOKBACK_BARS, horizon_bars, horizon_hours, move_pct, sample_stride, timeframe

log = logging.getLogger("radar.research.dataset")


def _apply_labels(sample: ResearchSample, labels: dict) -> None:
    sample.fwd_return = float(labels["fwd_return"])
    sample.max_upside = float(labels["max_upside"])
    sample.max_drawdown = float(labels["max_drawdown"])
    sample.significant_move = bool(labels["significant_move"])
    sample.large_up_move = bool(labels["large_up_move"])
    sample.clean_up_move = bool(labels["clean_up_move"])
    sample.labeled_at = utcnow()


async def build_samples(
    db: AsyncSession,
    asset_ids: dict[str, UUID],
    *,
    stride: int | None = None,
) -> dict[str, int]:
    tf = timeframe()
    hours = horizon_hours()
    h_bars = horizon_bars(tf, hours)
    stride_n = max(1, int(stride if stride is not None else sample_stride()))
    thr = move_pct()
    created = 0
    labeled = 0
    backfilled = 0
    for symbol, asset_id in asset_ids.items():
        bars = await load_bars(db, asset_id, tf.value)
        if len(bars) < LOOKBACK_BARS + h_bars:
            continue
        existing = (
            await db.execute(
                select(ResearchSample).where(
                    ResearchSample.asset_id == asset_id,
                    ResearchSample.timeframe == tf.value,
                    ResearchSample.horizon_hours == hours,
                )
            )
        ).scalars().all()
        have = {as_utc(s.ts): s for s in existing}
        last_i = len(bars) - 1
        for i in range(LOOKBACK_BARS - 1, last_i, stride_n):
            ts = as_utc(bars[i].ts)
            sample = have.get(ts)
            if sample is None:
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
                    sample_stride=stride_n,
                )
                labels = label_forward(bars[i + 1 :], bars[i].close, h_bars, thr)
                if labels:
                    _apply_labels(sample, labels)
                    labeled += 1
                db.add(sample)
                have[ts] = sample
                created += 1
            else:
                if sample.sample_stride is None:
                    sample.sample_stride = stride_n
                needs_label = sample.significant_move is None or sample.large_up_move is None
                if needs_label:
                    labels = label_forward(bars[i + 1 :], bars[i].close, h_bars, thr)
                    if labels:
                        _apply_labels(sample, labels)
                        labeled += 1
                        if sample.significant_move is not None and sample.large_up_move is not None:
                            backfilled += 1
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
    log.info(
        "Research dataset created=%s newly_labeled=%s directional_backfill_ops=%s stride=%s",
        created,
        labeled,
        backfilled,
        stride_n,
    )
    return {"created": created, "labeled": labeled, "backfilled": backfilled, "stride": stride_n}


async def backfill_directional_labels(db: AsyncSession) -> int:
    """Fill large_up_move / clean_up_move from stored continuous outcomes without re-fetching bars."""
    thr = move_pct()
    rows = (
        await db.execute(
            select(ResearchSample).where(
                ResearchSample.max_upside.is_not(None),
                ResearchSample.max_drawdown.is_not(None),
                ResearchSample.fwd_return.is_not(None),
            )
        )
    ).scalars().all()
    updated = 0
    for sample in rows:
        derived = labels_from_outcomes(
            sample.fwd_return, sample.max_upside, sample.max_drawdown, thr
        )
        if derived is None:
            continue
        changed = False
        if sample.significant_move is None:
            sample.significant_move = derived["significant_move"]
            changed = True
        if sample.large_up_move != derived["large_up_move"]:
            sample.large_up_move = derived["large_up_move"]
            changed = True
        if sample.clean_up_move != derived["clean_up_move"]:
            sample.clean_up_move = derived["clean_up_move"]
            changed = True
        if sample.sample_stride is None:
            sample.sample_stride = sample_stride()
            changed = True
        if changed:
            updated += 1
    if updated:
        await db.commit()
    log.info("Directional label backfill updated=%s", updated)
    return updated


def select_by_stride(
    rows: list[ResearchSample],
    stride_bars: int,
    *,
    base_stride: int = 6,
) -> list[ResearchSample]:
    """Deterministically thin samples to a coarser stride.

    Assumes ``rows`` were built on ``base_stride`` (legacy default=6). Selecting
    every (stride_bars // base_stride)-th sample per symbol matches building with
    ``stride_bars`` from the same lookback origin, yielding non-overlapping
    label windows when stride_bars == horizon_hours on 1h bars.
    """
    stride_bars = max(1, int(stride_bars))
    base_stride = max(1, int(base_stride))
    by_sym: dict[str, list[ResearchSample]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r.symbol, as_utc(r.ts))):
        by_sym[row.symbol].append(row)
    if stride_bars % base_stride == 0:
        step = stride_bars // base_stride
        out: list[ResearchSample] = []
        for items in by_sym.values():
            out.extend(items[::step])
        return sorted(out, key=lambda r: as_utc(r.ts))
    # Fallback: greedy non-overlapping windows (>= stride_bars hours apart).
    gap = stride_bars
    out = []
    for items in by_sym.values():
        last_ts = None
        for row in items:
            ts = as_utc(row.ts)
            if last_ts is None or (ts - last_ts).total_seconds() >= gap * 3600:
                out.append(row)
                last_ts = ts
    return sorted(out, key=lambda r: as_utc(r.ts))


async def sample_stats(db: AsyncSession, *, target: str = "significant_move") -> dict:
    tf = timeframe().value
    hours = horizon_hours()
    filters = (ResearchSample.timeframe == tf, ResearchSample.horizon_hours == hours)
    total = (await db.execute(select(func.count()).select_from(ResearchSample).where(*filters))).scalar_one()
    target_col = getattr(ResearchSample, target, ResearchSample.significant_move)
    labeled_n = (
        await db.execute(
            select(func.count())
            .select_from(ResearchSample)
            .where(*filters, target_col.is_not(None))
        )
    ).scalar_one()
    pos = (
        await db.execute(
            select(func.count()).select_from(ResearchSample).where(*filters, target_col.is_(True))
        )
    ).scalar_one()
    rate = (float(pos) / float(labeled_n)) if labeled_n else None
    return {
        "count": int(total or 0),
        "labeled": int(labeled_n or 0),
        "positives": int(pos or 0),
        "positive_rate": rate,
        "target": target,
        "timeframe": tf,
        "horizon_hours": hours,
        "move_pct": move_pct(),
        "stride": sample_stride(),
    }
