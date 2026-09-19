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
from app.research.experiments import ExperimentSpec
from app.research.features import feature_names
from app.research.history import as_utc
from app.research.spec import PROMOTE_MIN_AUC, horizon_hours, move_pct, sample_stride, timeframe

log = logging.getLogger("radar.research.model")

DISCLAIMER = (
    "This model estimates how often historically similar 1h states were followed by a "
    "significant move. It is not a price forecast and is kept only when walk-forward "
    "metrics beat a constant base-rate baseline."
)

STATUS_INSUFFICIENT_ATR = "insufficient_evidence_beyond_volatility"
STATUS_EXPERIMENT_PASS = "experiment_pass"
STATUS_REJECTED = "rejected"
STATUS_ACTIVE = "active"


def _to_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def _target_value(row: ResearchSample, target_name: str) -> bool | None:
    return getattr(row, target_name, None)


def design_matrix(
    rows: list[ResearchSample],
    names: list[str],
    *,
    target_name: str = "significant_move",
) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[_to_nan((row.features or {}).get(name)) for name in names] for row in rows], dtype=float)
    y = np.array([1 if _target_value(row, target_name) else 0 for row in rows], dtype=int)
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
    top_precision = None
    top_pos = None
    top_n = None
    if len(y_true) >= 20 and base > 0:
        k = max(1, len(y_true) // 5)
        order = np.argsort(p)[::-1][:k]
        top_slice = y_true[order]
        top_precision = float(np.mean(top_slice))
        top_pos = int(np.sum(top_slice))
        top_n = int(k)
        lift = float(top_precision / base)
    return {
        "n": int(len(y_true)),
        "positive_rate": base,
        "auc": auc,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "top_quintile_lift": lift,
        "top_quintile_precision": top_precision,
        "top_quintile_positive_count": top_pos,
        "top_quintile_n": top_n,
    }


def _aggregate(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    aucs = [m["auc"] for m in fold_metrics if m.get("auc") is not None]
    briers = [m["brier"] for m in fold_metrics if m.get("brier") is not None]
    baselines = [m["baseline_brier"] for m in fold_metrics if m.get("baseline_brier") is not None]
    lifts = [m["top_quintile_lift"] for m in fold_metrics if m.get("top_quintile_lift") is not None]
    precisions = [m["top_quintile_precision"] for m in fold_metrics if m.get("top_quintile_precision") is not None]
    mean_auc = float(np.mean(aucs)) if aucs else None
    mean_brier = float(np.mean(briers)) if briers else None
    mean_baseline = float(np.mean(baselines)) if baselines else None
    mean_lift = float(np.mean(lifts)) if lifts else None
    mean_precision = float(np.mean(precisions)) if precisions else None
    std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else (0.0 if aucs else None)
    std_brier = float(np.std(briers, ddof=1)) if len(briers) > 1 else (0.0 if briers else None)
    return {
        "folds": fold_metrics,
        "fold_count": len(fold_metrics),
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_brier": mean_brier,
        "std_brier": std_brier,
        "mean_baseline_brier": mean_baseline,
        "mean_top_quintile_lift": mean_lift,
        "mean_top_quintile_precision": mean_precision,
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


def _walk_forward_once(
    rows: list[ResearchSample],
    names: list[str],
    *,
    target_name: str,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda r: as_utc(r.ts))
    x, y = design_matrix(ordered, names, target_name=target_name)
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
    agg = _aggregate(fold_metrics)
    agg["train_samples"] = int(len(ordered))
    agg["positive_rate"] = float(np.mean(y)) if len(y) else None
    agg["feature_names"] = list(names)
    agg["target_name"] = target_name
    return {"ordered": ordered, "x": x, "y": y, "names": names, "agg": agg}


def fit_walk_forward(rows: list[ResearchSample]) -> dict[str, Any]:
    """Legacy bidirectional path: significant_move + full features."""
    names = feature_names()
    run = _walk_forward_once(rows, names, target_name="significant_move")
    agg = run["agg"]
    mean_auc = agg["mean_auc"]
    mean_brier = agg["mean_brier"]
    mean_baseline = agg["mean_baseline_brier"]
    promote = bool(
        enough_folds(agg["fold_count"])
        and mean_auc is not None
        and mean_brier is not None
        and mean_baseline is not None
        and mean_auc >= PROMOTE_MIN_AUC
        and mean_brier < mean_baseline
    )
    note = DISCLAIMER
    if not agg["folds"]:
        note = "Not enough dated samples for walk-forward folds yet. Model was not promoted."
        promote = False
    elif not promote:
        note = (
            f"Walk-forward did not beat the base-rate baseline (mean AUC={mean_auc}, "
            f"Brier={mean_brier} vs {mean_baseline}). Candidate kept for inspection, not live scoring."
        )

    y = run["y"]
    final = _new_classifier()
    if len(set(y.tolist())) < 2:
        note = "Labeled set has only one class; model was not promoted."
        promote = False
        blob = b""
    else:
        final.fit(run["x"], y)
        blob_buf = BytesIO()
        joblib.dump({"model": final, "feature_names": names, "target_name": "significant_move"}, blob_buf)
        blob = blob_buf.getvalue()
    return {
        "promote": promote,
        "feature_names": names,
        "blob": blob,
        "metrics": agg,
        "notes": note,
    }


def fit_experiment_walk_forward(
    rows: list[ResearchSample],
    experiment: ExperimentSpec,
) -> dict[str, Any]:
    """Directional (or other) experiment: full features vs ATR-only on the same folds."""
    target = experiment.target_name
    labeled = [r for r in rows if _target_value(r, target) is not None]
    full_names = feature_names()
    atr_names = [n for n in experiment.atr_feature_names if n in full_names]
    if not atr_names:
        raise ValueError(f"ATR baseline features missing from feature contract: {experiment.atr_feature_names}")

    full_run = _walk_forward_once(labeled, full_names, target_name=target)
    atr_run = _walk_forward_once(labeled, atr_names, target_name=target)
    full = full_run["agg"]
    atr = atr_run["agg"]

    auc_delta = None
    brier_delta = None
    if full["mean_auc"] is not None and atr["mean_auc"] is not None:
        auc_delta = float(full["mean_auc"] - atr["mean_auc"])
    if full["mean_brier"] is not None and atr["mean_brier"] is not None:
        brier_delta = float(full["mean_brier"] - atr["mean_brier"])

    beats_baseline = bool(
        full["mean_brier"] is not None
        and full["mean_baseline_brier"] is not None
        and full["mean_brier"] < full["mean_baseline_brier"]
    )
    auc_ok = bool(full["mean_auc"] is not None and full["mean_auc"] >= experiment.promote_min_auc)
    folds_ok = enough_folds(full["fold_count"])
    atr_auc_ok = bool(auc_delta is not None and auc_delta >= experiment.atr_auc_margin)
    atr_brier_ok = True
    if experiment.require_brier_better_than_atr:
        atr_brier_ok = bool(brier_delta is not None and brier_delta < 0)

    beats_atr = bool(atr_auc_ok and atr_brier_ok)
    promote_criteria = bool(folds_ok and auc_ok and beats_baseline and beats_atr)

    if not full["folds"]:
        status = STATUS_REJECTED
        note = "Not enough dated samples for walk-forward folds."
    elif not folds_ok or not auc_ok or not beats_baseline:
        status = STATUS_REJECTED
        note = (
            f"Failed base promotion gates (folds={full['fold_count']}, "
            f"mean_auc={full['mean_auc']}, mean_brier={full['mean_brier']} vs "
            f"baseline={full['mean_baseline_brier']})."
        )
    elif not beats_atr:
        status = STATUS_INSUFFICIENT_ATR
        note = (
            f"Full model did not add meaningful value beyond ATR-only "
            f"(AUC delta={auc_delta}, required>={experiment.atr_auc_margin}; "
            f"Brier delta={brier_delta}, require_strictly_better={experiment.require_brier_better_than_atr}). "
            f"Marked {STATUS_INSUFFICIENT_ATR}."
        )
    else:
        status = STATUS_EXPERIMENT_PASS
        note = (
            f"Experiment {experiment.experiment_id} passed gates including ATR-only comparison. "
            "Not connected to live trading; not promoted over the bidirectional active model."
        )

    y = full_run["y"]
    blob = b""
    if len(set(y.tolist())) >= 2:
        final = _new_classifier()
        final.fit(full_run["x"], y)
        blob_buf = BytesIO()
        joblib.dump(
            {
                "model": final,
                "feature_names": full_names,
                "target_name": target,
                "experiment_id": experiment.experiment_id,
            },
            blob_buf,
        )
        blob = blob_buf.getvalue()

    comparison = {
        "full": full,
        "atr_only": atr,
        "auc_delta_full_minus_atr": auc_delta,
        "brier_delta_full_minus_atr": brier_delta,
        "full_beats_constant_baseline": beats_baseline,
        "full_beats_atr_auc_margin": atr_auc_ok,
        "full_beats_atr_brier": atr_brier_ok,
        "full_adds_value_beyond_atr": beats_atr,
        "atr_auc_margin_required": experiment.atr_auc_margin,
    }
    metrics = {
        **full,
        "comparison": comparison,
        "lineage": experiment.lineage(),
        "sample_stride_bars": experiment.primary_stride_bars,
        "secondary_target": experiment.secondary_target,
    }
    return {
        "promote": False,  # never auto-activate over live bidirectional model
        "experiment_status": status,
        "passed_experiment_gates": promote_criteria,
        "feature_names": full_names,
        "blob": blob,
        "metrics": metrics,
        "notes": note,
        "experiment_id": experiment.experiment_id,
        "target_name": target,
        "sample_stride": experiment.primary_stride_bars,
        "move_threshold": experiment.move_pct,
        "comparison": comparison,
    }


def predict_proba(payload: dict[str, Any], features: dict) -> float:
    names: list[str] = payload["feature_names"]
    x = np.array([[_to_nan(features.get(name)) for name in names]], dtype=float)
    return float(_positive_proba(payload["model"], x)[0])


def load_payload(blob: bytes) -> dict[str, Any]:
    return joblib.load(BytesIO(blob))


async def labeled_rows(
    db: AsyncSession,
    *,
    target_name: str = "significant_move",
) -> list[ResearchSample]:
    tf = timeframe().value
    hours = horizon_hours()
    col = getattr(ResearchSample, target_name)
    return list(
        (
            await db.execute(
                select(ResearchSample).where(
                    ResearchSample.timeframe == tf,
                    ResearchSample.horizon_hours == hours,
                    col.is_not(None),
                )
            )
        ).scalars().all()
    )


async def active_model(db: AsyncSession) -> ModelVersion | None:
    """Live scanner model: bidirectional active only (excludes experiment statuses)."""
    return (
        await db.execute(
            select(ModelVersion)
            .where(ModelVersion.status == STATUS_ACTIVE)
            .order_by(ModelVersion.created_at.desc())
        )
    ).scalars().first()


async def save_training_run(db: AsyncSession, result: dict[str, Any], previous: ModelVersion | None) -> ModelVersion:
    """Bidirectional live promotion path. Does not touch experiment-only rows."""
    status = STATUS_ACTIVE if result["promote"] else STATUS_REJECTED
    if result["promote"] and previous is not None:
        prev_auc = (previous.metrics or {}).get("mean_auc")
        new_auc = (result["metrics"] or {}).get("mean_auc")
        if prev_auc is not None and new_auc is not None and float(new_auc) + 0.005 < float(prev_auc):
            status = STATUS_REJECTED
            result["notes"] = "Walk-forward passed the baseline but lost AUC vs the live model, so it was not promoted."
            result["promote"] = False
    if status == STATUS_ACTIVE:
        live = (
            await db.execute(select(ModelVersion).where(ModelVersion.status == STATUS_ACTIVE))
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
        blob=result["blob"] if status == STATUS_ACTIVE else None,
        experiment_id="bidirectional_significant_move_v1",
        target_name="significant_move",
        sample_stride=sample_stride(),
        move_threshold=move_pct(),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    log.info("Research model %s status=%s folds=%s", row.id, status, result["metrics"].get("fold_count"))
    return row


async def save_experiment_run(db: AsyncSession, result: dict[str, Any]) -> ModelVersion:
    """Persist an experiment result without retiring the live bidirectional active model."""
    status = result.get("experiment_status") or STATUS_REJECTED
    row = ModelVersion(
        status=status,
        timeframe=timeframe().value,
        horizon_hours=horizon_hours(),
        feature_names=result["feature_names"],
        metrics=result["metrics"],
        notes=result["notes"],
        # Keep blob for lineage/inspection; never mark active for trading.
        blob=result.get("blob") or None,
        experiment_id=result.get("experiment_id"),
        target_name=result.get("target_name"),
        sample_stride=result.get("sample_stride"),
        move_threshold=result.get("move_threshold"),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    log.info(
        "Experiment model %s experiment=%s status=%s folds=%s",
        row.id,
        row.experiment_id,
        status,
        (result.get("metrics") or {}).get("fold_count"),
    )
    return row
