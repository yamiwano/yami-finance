"""Record live predictions at T; resolve them from bars that closed after T."""

from __future__ import annotations

import logging
from uuid import UUID

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain import utcnow
from app.models.research import ModelVersion, Prediction
from app.research.features import extract_features, jsonable_features
from app.research.history import as_utc, load_bars_since, load_recent_bars
from app.research.labels import label_forward
from app.research.model import predict_proba
from app.research.spec import horizon_bars, horizon_hours, move_pct, timeframe

log = logging.getLogger("radar.research.live")


async def record_closed(
    db: AsyncSession,
    model: ModelVersion,
    payload: dict,
    asset_ids: dict[str, UUID],
) -> int:
    if not model.blob:
        return 0
    tf = timeframe()
    recorded = 0
    for symbol, asset_id in asset_ids.items():
        bars = await load_recent_bars(db, asset_id, tf.value, 90)
        if len(bars) < 80:
            continue
        last = bars[-1]
        ts = as_utc(last.ts)
        exists = (
            await db.execute(
                select(Prediction.id).where(
                    Prediction.model_id == model.id,
                    Prediction.symbol == symbol,
                    Prediction.timeframe == tf.value,
                    Prediction.ts == ts,
                )
            )
        ).scalar_one_or_none()
        if exists:
            continue
        feats = extract_features(bars, tf.value)
        if not feats:
            continue
        p = predict_proba(payload, feats)
        db.add(
            Prediction(
                model_id=model.id,
                asset_id=asset_id,
                symbol=symbol,
                timeframe=tf.value,
                ts=ts,
                p_significant=p,
                features=jsonable_features(feats),
            )
        )
        recorded += 1
    if recorded:
        await db.commit()
        log.info("Research live predictions +%s", recorded)
    return recorded


async def resolve_pending(db: AsyncSession, asset_ids: dict[str, UUID]) -> int:
    tf = timeframe()
    hours = horizon_hours()
    h_bars = horizon_bars(tf, hours)
    pending = (
        await db.execute(select(Prediction).where(Prediction.resolved_at.is_(None), Prediction.timeframe == tf.value))
    ).scalars().all()
    if not pending:
        return 0
    by_asset: dict[UUID, list] = {}
    for row in pending:
        by_asset.setdefault(row.asset_id, []).append(row)
    resolved = 0
    for asset_id, rows in by_asset.items():
        bars = await load_bars_since(db, asset_id, tf.value, min(as_utc(r.ts) for r in rows))
        index = {as_utc(b.ts): i for i, b in enumerate(bars)}
        for row in rows:
            i = index.get(as_utc(row.ts))
            if i is None or i + h_bars >= len(bars):
                continue
            labels = label_forward(bars[i + 1 :], bars[i].close, h_bars, move_pct())
            if not labels:
                continue
            row.fwd_return = float(labels["fwd_return"])
            row.max_upside = float(labels["max_upside"])
            row.max_drawdown = float(labels["max_drawdown"])
            row.significant_move = bool(labels["significant_move"])
            row.large_up_move = bool(labels["large_up_move"])
            row.clean_up_move = bool(labels["clean_up_move"])
            row.resolved_at = utcnow()
            resolved += 1
    if resolved:
        await db.commit()
        log.info("Research resolved predictions +%s", resolved)
    return resolved


async def live_stats(db: AsyncSession) -> dict:
    hours = horizon_hours()
    pending = (await db.execute(select(func.count()).select_from(Prediction).where(Prediction.resolved_at.is_(None)))).scalar_one()
    resolved_rows = (
        await db.execute(select(Prediction).where(Prediction.significant_move.is_not(None)))
    ).scalars().all()
    n = len(resolved_rows)
    auc = None
    brier = None
    base = None
    lift = None
    if n:
        y = np.array([1 if r.significant_move else 0 for r in resolved_rows], dtype=int)
        p = np.array([r.p_significant for r in resolved_rows], dtype=float)
        base = float(np.mean(y))
        brier = float(brier_score_loss(y, p))
        if len(set(y.tolist())) > 1:
            auc = float(roc_auc_score(y, p))
        if n >= 20 and base > 0:
            k = max(1, n // 5)
            order = np.argsort(p)[::-1][:k]
            lift = float(np.mean(y[order]) / base)
    return {
        "pending": int(pending or 0),
        "resolved": n,
        "positive_rate": base,
        "auc": auc,
        "brier": brier,
        "top_quintile_lift": lift,
        "horizon_hours": hours,
        "note": (
            "Live book compares predicted P(significant move) with realized 12h labels. "
            "Too few resolved rows means there is not yet evidence of predictive value."
            if n < 40
            else DISCLAIMER_LIVE
        ),
    }


DISCLAIMER_LIVE = (
    "Live metrics are the honest test: predicted probability vs what actually happened "
    "after the prediction was stored. They do not imply a tradable edge."
)
