"""Gradient-boosted trees on tabular market features. Keep the model only if unseen folds improve."""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research import ModelVersion, ResearchSample
from app.research.backtest import enough_folds, walk_forward_folds
from app.research.features import feature_names
from app.research.history import as_utc
from app.research.spec import PROMOTE_MIN_AUC, horizon_hours, timeframe

log = logging.getLogger("radar.research.model")

DISCLAIMER = (
    "This model estimates how often historically similar 1h states were followed by a "
    "significant move. It is not a price forecast and is kept only when walk-forward "
    "metrics beat a constant base-rate baseline."
)


def _to_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def design_matrix(rows: list[ResearchSample], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[_to_nan((row.features or {}).get(name)) for name in names] for row in rows], dtype=float)
    y = np.array([1 if row.significant_move else 0 for row in rows], dtype=int)
    return x, y


def _positive_proba(clf: HistGradientBoostingClassifier, x: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(x)
    classes = list(clf.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    if 0 in classes and len(classes) == 1:
        return np.zeros(len(x), dtype=float)
    return proba[:, -1]


def _metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    base = float(np.mean(y_true)) if len(y_true) else 0.0
    brier = float(brier_score_loss(y_true, p)) if len(y_true) else None
    baseline_brier = float(brier_score_loss(y_true, np.full(len(y_true), base))) if len(y_true) else None
    auc = None
    if len(y_true) and len(set(y_true.tolist())) > 1:
        auc = float(roc_auc_score(y_true, p))
    lift = None
    if len(y_true) >= 20 and base > 0:
        k = max(1, len(y_true) // 5)
        order = np.argsort(p)[::-1][:k]
        lift = float(np.mean(y_true[order]) / base)
    return {
        "n": int(len(y_true)),
        "positive_rate": base,
        "auc": auc,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "top_quintile_lift": lift,
    }


def _new_classifier() -> HistGradientBoostingClassifier:
    kwargs: dict[str, Any] = dict(
        max_depth=4,
        learning_rate=0.06,
        max_iter=140,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=42,
    )
    try:
        return HistGradientBoostingClassifier(class_weight="balanced", **kwargs)
    except TypeError:
        return HistGradientBoostingClassifier(**kwargs)


def fit_walk_forward(rows: list[ResearchSample]) -> dict[str, Any]:
    names = feature_names()
    ordered = sorted(rows, key=lambda r: as_utc(r.ts))
    x, y = design_matrix(ordered, names)
    timestamps = [as_utc(r.ts) for r in ordered]
    folds = walk_forward_folds(timestamps)
    fold_metrics: list[dict[str, Any]] = []
    for train_idx, test_idx in folds:
        y_train = y[train_idx]
        if len(set(y_train.tolist())) < 2:
            continue
        clf = _new_classifier()
        clf.fit(x[train_idx], y_train)
        p = _positive_proba(clf, x[test_idx])
        stats = _metrics(y[test_idx], p)
        stats["train_n"] = len(train_idx)
        stats["train_start"] = timestamps[train_idx[0]].isoformat()
        stats["train_end"] = timestamps[train_idx[-1]].isoformat()
        stats["test_start"] = timestamps[test_idx[0]].isoformat()
        stats["test_end"] = timestamps[test_idx[-1]].isoformat()
        fold_metrics.append(stats)

    aucs = [m["auc"] for m in fold_metrics if m.get("auc") is not None]
    briers = [m["brier"] for m in fold_metrics if m.get("brier") is not None]
    baselines = [m["baseline_brier"] for m in fold_metrics if m.get("baseline_brier") is not None]
    lifts = [m["top_quintile_lift"] for m in fold_metrics if m.get("top_quintile_lift") is not None]
    mean_auc = float(np.mean(aucs)) if aucs else None
    mean_brier = float(np.mean(briers)) if briers else None
    mean_baseline = float(np.mean(baselines)) if baselines else None
    mean_lift = float(np.mean(lifts)) if lifts else None
    std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else (0.0 if aucs else None)
    std_brier = float(np.std(briers, ddof=1)) if len(briers) > 1 else (0.0 if briers else None)
    promote = bool(
        enough_folds(len(fold_metrics))
        and mean_auc is not None
        and mean_brier is not None
        and mean_baseline is not None
        and mean_auc >= PROMOTE_MIN_AUC
        and mean_brier < mean_baseline
    )
    note = DISCLAIMER
    if not fold_metrics:
        note = "Not enough dated samples for walk-forward folds yet. Model was not promoted."
        promote = False
    elif not promote:
        note = (
            f"Walk-forward did not beat the base-rate baseline (mean AUC={mean_auc}, "
            f"Brier={mean_brier} vs {mean_baseline}). Candidate kept for inspection, not live scoring."
        )

    final = _new_classifier()
    if len(set(y.tolist())) < 2:
        note = "Labeled set has only one class; model was not promoted."
        promote = False
        blob = b""
    else:
        final.fit(x, y)
        blob_buf = BytesIO()
        joblib.dump({"model": final, "feature_names": names}, blob_buf)
        blob = blob_buf.getvalue()
    return {
        "promote": promote,
        "feature_names": names,
        "blob": blob,
        "metrics": {
            "folds": fold_metrics,
            "fold_count": len(fold_metrics),
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "mean_brier": mean_brier,
            "std_brier": std_brier,
            "mean_baseline_brier": mean_baseline,
            "mean_top_quintile_lift": mean_lift,
            "train_samples": int(len(ordered)),
            "positive_rate": float(np.mean(y)) if len(y) else None,
        },
        "notes": note,
    }


def predict_proba(payload: dict[str, Any], features: dict) -> float:
    names: list[str] = payload["feature_names"]
    x = np.array([[_to_nan(features.get(name)) for name in names]], dtype=float)
    return float(_positive_proba(payload["model"], x)[0])


def load_payload(blob: bytes) -> dict[str, Any]:
    return joblib.load(BytesIO(blob))


async def labeled_rows(db: AsyncSession) -> list[ResearchSample]:
    tf = timeframe().value
    hours = horizon_hours()
    return list(
        (
            await db.execute(
                select(ResearchSample).where(
                    ResearchSample.timeframe == tf,
                    ResearchSample.horizon_hours == hours,
                    ResearchSample.significant_move.is_not(None),
                )
            )
        ).scalars().all()
    )


async def active_model(db: AsyncSession) -> ModelVersion | None:
    return (
        await db.execute(
            select(ModelVersion)
            .where(ModelVersion.status == "active")
            .order_by(ModelVersion.created_at.desc())
        )
    ).scalars().first()


async def save_training_run(db: AsyncSession, result: dict[str, Any], previous: ModelVersion | None) -> ModelVersion:
    status = "active" if result["promote"] else "rejected"
    if result["promote"] and previous is not None:
        prev_auc = (previous.metrics or {}).get("mean_auc")
        new_auc = (result["metrics"] or {}).get("mean_auc")
        if prev_auc is not None and new_auc is not None and float(new_auc) + 0.005 < float(prev_auc):
            status = "rejected"
            result["notes"] = "Walk-forward passed the baseline but lost AUC vs the live model, so it was not promoted."
            result["promote"] = False
    if status == "active":
        live = (
            await db.execute(select(ModelVersion).where(ModelVersion.status == "active"))
        ).scalars().all()
        for row in live:
            row.status = "retired"
    row = ModelVersion(
        status=status,
        timeframe=timeframe().value,
        horizon_hours=horizon_hours(),
        feature_names=result["feature_names"],
        metrics=result["metrics"],
        notes=result["notes"],
        blob=result["blob"] if status == "active" else None,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    log.info("Research model %s status=%s folds=%s", row.id, status, result["metrics"].get("fold_count"))
    return row
