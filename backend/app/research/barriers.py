"""Barrier probability experiment: P(+up% before -down%) within 12h.

Targets are first-touch barrier outcomes from future OHLC — labels only.
Binary training/evaluation uses success vs failure; timeout & ambiguous excluded.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from io import BytesIO
from typing import Any, Sequence

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.experiments import ExperimentSpec
from app.research.features import feature_names
from app.research.history import as_utc
from app.research.opportunity import BARRIER_SPECS, first_touch_barrier
from app.research.ranking import (
    MIN_CROSS_SECTION,
    future_max_drawdown,
    future_max_upside,
    future_return,
    group_by_timestamp,
)
from app.research.spec import embargo

STATUS_REJECTED = "rejected"
STATUS_WEAK = "weak_barrier_signal"
STATUS_PROMISING = "promising_barrier_signal"

CALIBRATION_BUCKETS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))
# Research-only EV diagnostics (not trading returns).
EV_SPECS = {
    "up3_before_down2": (0.03, 0.02),
    "up5_before_down3": (0.05, 0.03),
    "up10_before_down5": (0.10, 0.05),
}
# Conservative hypothetical round-trip cost/slippage applied per event (documented; not optimized).
COST_PER_EVENT = 0.002


def _feat(row: Any, name: str) -> float:
    feats = getattr(row, "features", None) or {}
    value = feats.get(name)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def barrier_state(row: Any, name: str) -> str:
    return str(getattr(row, name, "unknown"))


def barrier_binary(row: Any, name: str) -> int | None:
    """1 success, 0 failure, None for timeout/ambiguous/unknown."""
    state = barrier_state(row, name)
    if state == "up_first":
        return 1
    if state == "down_first":
        return 0
    return None


def barrier_stats(rows: Sequence[Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, up, down in BARRIER_SPECS:
        total = len(rows)
        success = sum(1 for r in rows if barrier_state(r, name) == "up_first")
        failure = sum(1 for r in rows if barrier_state(r, name) == "down_first")
        timeout = sum(1 for r in rows if barrier_state(r, name) == "neither")
        ambiguous = sum(1 for r in rows if barrier_state(r, name) == "ambiguous")
        resolved = success + failure
        out[name] = {
            "up_pct": up,
            "down_pct": down,
            "total": total,
            "success": success,
            "failure": failure,
            "timeout": timeout,
            "ambiguous": ambiguous,
            "resolved_non_ambiguous": resolved,
            "success_rate_resolved": (success / resolved) if resolved else None,
            "timeout_rate": (timeout / total) if total else None,
            "ambiguous_rate": (ambiguous / total) if total else None,
        }
    return out


def calibration_table(y_true: np.ndarray, p: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in CALIBRATION_BUCKETS:
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        n = int(mask.sum())
        if not n:
            rows.append({"bucket": f"{lo:.0%}-{hi:.0%}", "n": 0, "mean_pred": None, "observed": None})
            continue
        rows.append(
            {
                "bucket": f"{lo:.0%}-{hi:.0%}",
                "n": n,
                "mean_pred": float(np.mean(p[mask])),
                "observed": float(np.mean(y_true[mask])),
            }
        )
    return rows


def expected_value(p_success: float, up: float, down: float, *, cost: float = 0.0) -> float:
    """Diagnostic EV per event. cost is subtracted once (round-trip)."""
    return float(p_success) * up - (1.0 - float(p_success)) * down - cost


def _matrix(rows: Sequence[Any], names: Sequence[str]) -> np.ndarray:
    return np.array([[_feat(r, n) for n in names] for r in rows], dtype=float)


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


def _positive_proba(clf: HistGradientBoostingClassifier, x: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(x)
    classes = list(clf.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    if 0 in classes and len(classes) == 1:
        return np.zeros(len(x), dtype=float)
    return proba[:, -1]


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), len(scores)))
    clean = np.where(np.isfinite(scores), scores, -np.inf)
    return np.argsort(clean)[::-1][:k]


def _group_metrics(items: list[dict[str, Any]]) -> dict[str, float | None]:
    if not items:
        return {"n": 0}

    def _nan_agg(values: list[float], fn) -> float | None:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if not len(arr):
            return None
        return float(fn(arr))

    succ = np.array([1.0 if i["state"] == "up_first" else 0.0 for i in items if i["state"] in ("up_first", "down_first")], dtype=float)
    timeout = np.array([1.0 if i["state"] == "neither" else 0.0 for i in items], dtype=float)
    amb = np.array([1.0 if i["state"] == "ambiguous" else 0.0 for i in items], dtype=float)
    raw = [i["raw_upside"] for i in items]
    ret = [i["future_return"] for i in items]
    dd = [i["drawdown"] for i in items]
    return {
        "n": int(len(items)),
        "success_rate": float(np.mean(succ)) if len(succ) else None,
        "resolved_n": int(len(succ)),
        "timeout_rate": float(np.mean(timeout)),
        "ambiguous_rate": float(np.mean(amb)),
        "mean_future_return": _nan_agg(ret, np.mean),
        "mean_raw_upside": _nan_agg(raw, np.mean),
        "mean_drawdown": _nan_agg(dd, np.mean),
        "median_drawdown": _nan_agg(dd, np.median),
        "worst_drawdown": _nan_agg(dd, np.min),
        "frac_dd_worse_3": _nan_agg([1.0 if v <= -0.03 else 0.0 for v in dd], np.mean),
        "frac_dd_worse_5": _nan_agg([1.0 if v <= -0.05 else 0.0 for v in dd], np.mean),
        "frac_dd_worse_10": _nan_agg([1.0 if v <= -0.10 else 0.0 for v in dd], np.mean),
    }


def evaluate_barrier_timestamp(rows: Sequence[Any], scores: np.ndarray, barrier: str) -> dict[str, Any]:
    n = len(rows)
    scored = []
    for row, score in zip(rows, scores):
        scored.append(
            {
                "symbol": row.symbol,
                "score": float(score) if np.isfinite(score) else float("-inf"),
                "state": barrier_state(row, barrier),
                "raw_upside": float(future_max_upside(row) or np.nan),
                "future_return": float(future_return(row) or np.nan),
                "drawdown": float(future_max_drawdown(row) or np.nan),
            }
        )
    order = _top_k_indices(np.array([s["score"] for s in scored], dtype=float), n)
    ranked = [scored[i] for i in order]
    k10 = max(1, int(np.ceil(0.10 * n)))
    k20 = max(1, int(np.ceil(0.20 * n)))

    def take(k: int) -> list[dict[str, Any]]:
        return ranked[: min(k, n)]

    return {
        "n": n,
        "all": _group_metrics(scored),
        "top_1": _group_metrics(take(1)),
        "top_3": _group_metrics(take(3)),
        "top_5": _group_metrics(take(5)),
        "top_10pct": _group_metrics(take(k10)),
        "top_20pct": _group_metrics(take(k20)),
    }


def _mean_path(results: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    vals = []
    for r in results:
        cur: Any = r
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None and np.isfinite(cur):
            vals.append(float(cur))
    return float(np.mean(vals)) if vals else None


def aggregate_barrier_ts(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"timestamps": 0}
    out: dict[str, Any] = {"timestamps": len(results)}
    for group in ("all", "top_1", "top_3", "top_5", "top_10pct", "top_20pct"):
        sample = results[0].get(group) or {}
        for metric in sample:
            if metric == "n":
                continue
            out[f"{group}_{metric}"] = _mean_path(results, (group, metric))
    return out


def evaluate_barrier_by_timestamp(
    rows: Sequence[Any], scores: np.ndarray, barrier: str
) -> tuple[list[dict], dict]:
    by_ts: dict[datetime, list[tuple[Any, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_ts[as_utc(row.ts)].append((row, float(score)))
    ts_results = []
    for _ts, pairs in sorted(by_ts.items(), key=lambda kv: kv[0]):
        bucket_rows = [p[0] for p in pairs]
        bucket_scores = np.array([p[1] for p in pairs], dtype=float)
        ts_results.append(evaluate_barrier_timestamp(bucket_rows, bucket_scores, barrier))
    return ts_results, aggregate_barrier_ts(ts_results)


def binary_classification_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)) if len(y_true) else None,
    }
    if len(y_true) == 0:
        return out
    out["brier"] = float(brier_score_loss(y_true, p))
    out["baseline_brier"] = float(brier_score_loss(y_true, np.full(len(y_true), np.mean(y_true))))
    try:
        out["log_loss"] = float(log_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7)))
        out["baseline_log_loss"] = float(
            log_loss(y_true, np.full(len(y_true), np.clip(np.mean(y_true), 1e-7, 1 - 1e-7)))
        )
    except ValueError:
        out["log_loss"] = None
        out["baseline_log_loss"] = None
    if len(set(y_true.tolist())) > 1:
        out["auc"] = float(roc_auc_score(y_true, p))
        out["pr_auc"] = float(average_precision_score(y_true, p))
        if np.nanstd(p) > 0 and np.nanstd(y_true.astype(float)) > 0:
            rho, _ = spearmanr(p, y_true)
            out["spearman_prob_outcome"] = float(rho) if np.isfinite(rho) else None
        else:
            out["spearman_prob_outcome"] = None
    else:
        out["auc"] = None
        out["pr_auc"] = None
        out["spearman_prob_outcome"] = None
    out["calibration"] = calibration_table(y_true, p)
    return out


def classify_barrier_experiment(
    per_barrier: dict[str, dict[str, Any]],
    *,
    leakage_ok: bool,
) -> tuple[str, str, dict[str, Any]]:
    """Conservative classification; do not loosen to pass."""
    barriers = list(per_barrier)
    beats_atr_top5 = 0
    folds_consistent = 0
    calib_better = 0
    no_atr_retains = 0
    bucket_ok = 0
    topk_visible = 0
    beats_all = 0

    for name, summary in per_barrier.items():
        full = summary.get("full") or {}
        atr = summary.get("atr_only") or {}
        no_atr = summary.get("full_without_atr") or {}
        buckets = summary.get("volatility_buckets") or {}
        folds = summary.get("folds") or []

        f_top5 = full.get("top_5_success_rate")
        a_top5 = atr.get("top_5_success_rate")
        all_top5 = full.get("all_success_rate")
        if f_top5 is not None and a_top5 is not None and f_top5 > a_top5:
            beats_atr_top5 += 1
        if f_top5 is not None and all_top5 is not None and f_top5 > all_top5:
            beats_all += 1
            if f_top5 - all_top5 >= 0.05:
                topk_visible += 1
        fold_sp = [
            (fr.get("full") or {}).get("spearman_prob_outcome")
            for fr in folds
        ]
        finite = [s for s in fold_sp if s is not None and np.isfinite(s)]
        if len(finite) >= 2 and all(s > 0 for s in finite):
            folds_consistent += 1
        if (
            full.get("brier") is not None
            and full.get("baseline_brier") is not None
            and full["brier"] < full["baseline_brier"]
            and full.get("log_loss") is not None
            and full.get("baseline_log_loss") is not None
            and full["log_loss"] < full["baseline_log_loss"]
        ):
            calib_better += 1
        if (
            no_atr.get("spearman_prob_outcome") is not None
            and no_atr["spearman_prob_outcome"] > 0.03
        ):
            no_atr_retains += 1
        flags = []
        for bname in ("low", "medium", "high"):
            b = buckets.get(bname) or {}
            if b.get("full_top5_success_rate") is not None and b.get("all_success_rate") is not None:
                flags.append(b["full_top5_success_rate"] > b["all_success_rate"])
        if flags and all(flags):
            bucket_ok += 1

    n_barriers = max(1, len(barriers))
    evidence = {
        "leakage_audit_passed": leakage_ok,
        "barriers_full_beats_atr_top5": beats_atr_top5,
        "barriers_full_beats_all_assets_top5": beats_all,
        "barriers_consistent_positive_folds": folds_consistent,
        "barriers_calibration_better_than_baseline": calib_better,
        "barriers_no_atr_retains_signal": no_atr_retains,
        "barriers_positive_across_vol_buckets": bucket_ok,
        "barriers_topk_material_vs_all": topk_visible,
        "n_barriers": n_barriers,
    }

    promising = (
        leakage_ok
        and beats_all == n_barriers
        and beats_atr_top5 >= 2
        and folds_consistent == n_barriers
        and calib_better == n_barriers
        and no_atr_retains == n_barriers
        and bucket_ok == n_barriers
        and topk_visible >= 2
    )
    if promising:
        return (
            STATUS_PROMISING,
            "promising_barrier_signal: full model beats baselines on top-5 across barriers with "
            "supporting calibration; research-only, not trading-ready.",
            evidence,
        )
    weak = (
        leakage_ok
        and beats_all >= 1
        and (beats_atr_top5 >= 1 or no_atr_retains >= 1)
        and folds_consistent >= 1
    )
    if weak:
        return (
            STATUS_WEAK,
            "weak_barrier_signal: some barrier ranking ability, but not consistently better than "
            "ATR/momentum across barriers, folds, and regimes.",
            evidence,
        )
    return (
        STATUS_REJECTED,
        "rejected: insufficient evidence that the model predicts barrier-first-touch beyond baselines.",
        evidence,
    )


def fit_barrier_walk_forward(
    rows: Sequence[Any],
    experiment: ExperimentSpec,
    *,
    min_cross_section: int = MIN_CROSS_SECTION,
    leakage_ok: bool = True,
) -> dict[str, Any]:
    # Keep only rows with complete outcomes and valid cross-sections.
    groups = group_by_timestamp(rows)
    kept: list[Any] = []
    considered = 0
    excluded = 0
    sizes: list[int] = []
    for _ts, bucket in groups.items():
        considered += 1
        eligible = [
            r
            for r in bucket
            if future_max_upside(r) is not None
            and future_max_drawdown(r) is not None
            and future_return(r) is not None
        ]
        if len(eligible) < min_cross_section:
            excluded += 1
            continue
        sizes.append(len(eligible))
        kept.extend(eligible)

    cs_stats = {
        "timestamps_considered": considered,
        "timestamps_excluded_below_min": excluded,
        "timestamps_used": considered - excluded,
        "average_cross_section_size": float(np.mean(sizes)) if sizes else None,
        "min_cross_section_size": int(min(sizes)) if sizes else None,
        "max_cross_section_size": int(max(sizes)) if sizes else None,
        "asset_observations_used": len(kept),
    }

    ordered = sorted(kept, key=lambda r: (as_utc(r.ts), r.symbol))
    full_names = feature_names()
    no_atr_names = [n for n in full_names if n != "atr_pct"]
    atr_names = ["atr_pct"]
    mom_feature = experiment.momentum_feature

    unique_ts = sorted({as_utc(r.ts) for r in ordered})
    folds = walk_forward_folds(unique_ts, min_train=60, min_test=14)
    gap = embargo()

    x_full = _matrix(ordered, full_names)
    x_atr = _matrix(ordered, atr_names)
    x_mom = _matrix(ordered, [mom_feature])
    x_no_atr = _matrix(ordered, no_atr_names)
    row_ts = [as_utc(r.ts) for r in ordered]
    atr_pcts = np.array([_feat(r, "atr_pct") for r in ordered], dtype=float)

    per_barrier: dict[str, dict[str, Any]] = {}
    blobs: dict[str, bytes] = {}

    for barrier, up_pct, down_pct in BARRIER_SPECS:
        # Binary subset: success vs failure only.
        y_all = np.array(
            [barrier_binary(r, barrier) if barrier_binary(r, barrier) is not None else -1 for r in ordered],
            dtype=int,
        )
        binary_mask = y_all >= 0
        y_bin = y_all[binary_mask].astype(int)
        idx_bin = np.where(binary_mask)[0]

        fold_reports: list[dict[str, Any]] = []
        fold_spearmans: list[float | None] = []
        all_test_rows: list[Any] = []
        all_full_scores: list[float] = []
        all_atr_scores: list[float] = []
        all_mom_scores: list[float] = []
        all_no_atr_scores: list[float] = []
        train_atr_for_buckets: list[float] = []
        pooled_y: list[int] = []
        pooled_full_p: list[float] = []
        pooled_atr_p: list[float] = []
        pooled_mom_p: list[float] = []
        pooled_no_atr_p: list[float] = []

        for train_ts_idx, test_ts_idx in folds:
            train_ts = {unique_ts[i] for i in train_ts_idx}
            test_ts = {unique_ts[i] for i in test_ts_idx}
            if not fold_is_causal(unique_ts, train_ts_idx, test_ts_idx, gap):
                continue
            train_idx_all = [i for i, t in enumerate(row_ts) if t in train_ts]
            test_idx_all = [i for i, t in enumerate(row_ts) if t in test_ts]
            train_idx = [i for i in train_idx_all if binary_mask[i]]
            test_idx = [i for i in test_idx_all if binary_mask[i]]
            if len(train_idx) < 50 or len(test_idx) < 20:
                continue
            y_train = y_all[train_idx].astype(int)
            if len(set(y_train.tolist())) < 2:
                continue

            train_atr_for_buckets.extend(atr_pcts[train_idx_all].tolist())

            def _fit_predict(x_all: np.ndarray) -> np.ndarray:
                clf = _new_classifier()
                clf.fit(x_all[train_idx], y_train)
                return _positive_proba(clf, x_all[test_idx])

            p_full = _fit_predict(x_full)
            p_atr = _fit_predict(x_atr)
            p_mom = _fit_predict(x_mom)
            p_no_atr = _fit_predict(x_no_atr)

            y_test = y_all[test_idx].astype(int)
            pooled_y.extend(y_test.tolist())
            pooled_full_p.extend(p_full.tolist())
            pooled_atr_p.extend(p_atr.tolist())
            pooled_mom_p.extend(p_mom.tolist())
            pooled_no_atr_p.extend(p_no_atr.tolist())

            test_rows_all = [ordered[i] for i in test_idx_all]
            # For ranking/top-K we score all test rows (including timeout) but success rate uses resolved subset.
            def _predict_all(x_all: np.ndarray) -> np.ndarray:
                clf = _new_classifier()
                clf.fit(x_all[train_idx], y_train)
                return _positive_proba(clf, x_all[test_idx_all])

            pf_all = _predict_all(x_full)
            pa_all = _predict_all(x_atr)
            pm_all = _predict_all(x_mom)
            pn_all = _predict_all(x_no_atr)

            all_test_rows.extend(test_rows_all)
            all_full_scores.extend(pf_all.tolist())
            all_atr_scores.extend(pa_all.tolist())
            all_mom_scores.extend(pm_all.tolist())
            all_no_atr_scores.extend(pn_all.tolist())

            _, full_rank = evaluate_barrier_by_timestamp(test_rows_all, pf_all, barrier)
            _, atr_rank = evaluate_barrier_by_timestamp(test_rows_all, pa_all, barrier)
            _, mom_rank = evaluate_barrier_by_timestamp(test_rows_all, pm_all, barrier)
            _, no_atr_rank = evaluate_barrier_by_timestamp(test_rows_all, pn_all, barrier)

            full_cls = binary_classification_metrics(y_test, p_full)
            atr_cls = binary_classification_metrics(y_test, p_atr)
            mom_cls = binary_classification_metrics(y_test, p_mom)
            no_atr_cls = binary_classification_metrics(y_test, p_no_atr)

            fold_spearmans.append(full_cls.get("spearman_prob_outcome"))
            fold_reports.append(
                {
                    "train_start": min(train_ts).isoformat(),
                    "train_end": max(train_ts).isoformat(),
                    "test_start": min(test_ts).isoformat(),
                    "test_end": max(test_ts).isoformat(),
                    "train_binary_observations": len(train_idx),
                    "test_binary_observations": len(test_idx),
                    "test_asset_observations": len(test_idx_all),
                    "test_timestamps": full_rank.get("timestamps"),
                    "average_cross_section_size": (
                        float(np.mean([len(g) for g in group_by_timestamp(test_rows_all).values()]))
                        if test_rows_all
                        else None
                    ),
                    "full": {**full_cls, **{k: v for k, v in full_rank.items() if k != "timestamps"}},
                    "atr_only": {**atr_cls, **{k: v for k, v in atr_rank.items() if k != "timestamps"}},
                    "momentum": {**mom_cls, **{k: v for k, v in mom_rank.items() if k != "timestamps"}},
                    "full_without_atr": {
                        **no_atr_cls,
                        **{k: v for k, v in no_atr_rank.items() if k != "timestamps"},
                    },
                }
            )

        def _agg(path: tuple[str, ...]) -> float | None:
            vals = []
            for fr in fold_reports:
                cur: Any = fr
                ok = True
                for key in path:
                    if not isinstance(cur, dict) or key not in cur:
                        ok = False
                        break
                    cur = cur[key]
                if ok and cur is not None and isinstance(cur, (int, float)) and np.isfinite(cur):
                    vals.append(float(cur))
            return float(np.mean(vals)) if vals else None

        def _std(path: tuple[str, ...]) -> float | None:
            vals = []
            for fr in fold_reports:
                cur: Any = fr
                ok = True
                for key in path:
                    if not isinstance(cur, dict) or key not in cur:
                        ok = False
                        break
                    cur = cur[key]
                if ok and cur is not None and isinstance(cur, (int, float)) and np.isfinite(cur):
                    vals.append(float(cur))
            if len(vals) < 2:
                return 0.0 if vals else None
            return float(np.std(vals, ddof=1))

        def summarize(prefix: str) -> dict[str, Any]:
            keys = [
                "auc",
                "pr_auc",
                "brier",
                "baseline_brier",
                "log_loss",
                "baseline_log_loss",
                "spearman_prob_outcome",
                "base_rate",
                "all_success_rate",
                "all_mean_future_return",
                "all_mean_drawdown",
                "top_1_success_rate",
                "top_3_success_rate",
                "top_5_success_rate",
                "top_10pct_success_rate",
                "top_20pct_success_rate",
                "top_5_mean_future_return",
                "top_5_mean_raw_upside",
                "top_5_mean_drawdown",
                "top_5_median_drawdown",
                "top_5_worst_drawdown",
                "top_5_frac_dd_worse_3",
                "top_5_frac_dd_worse_5",
                "top_5_frac_dd_worse_10",
                "top_5_timeout_rate",
                "top_5_ambiguous_rate",
            ]
            out = {k: _agg((prefix, k)) for k in keys}
            out["auc_std"] = _std((prefix, "auc"))
            out["spearman_std"] = _std((prefix, "spearman_prob_outcome"))
            return out

        full_summary = summarize("full")
        atr_summary = summarize("atr_only")
        mom_summary = summarize("momentum")
        no_atr_summary = summarize("full_without_atr")

        # Pooled calibration for full model
        pooled = {}
        if pooled_y:
            pooled = {
                "full": binary_classification_metrics(np.asarray(pooled_y), np.asarray(pooled_full_p)),
                "atr_only": binary_classification_metrics(np.asarray(pooled_y), np.asarray(pooled_atr_p)),
            }

        # Volatility buckets from training ATR tertiles
        bucket_summary: dict[str, Any] = {}
        if train_atr_for_buckets and all_test_rows:
            finite_train = [a for a in train_atr_for_buckets if np.isfinite(a)]
            if len(finite_train) >= 30:
                q33, q66 = np.quantile(finite_train, [1 / 3, 2 / 3])
                buckets: dict[str, list[tuple[Any, float, float, float, float]]] = {
                    "low": [],
                    "medium": [],
                    "high": [],
                }
                for row, fs, as_, ms, ns in zip(
                    all_test_rows,
                    all_full_scores,
                    all_atr_scores,
                    all_mom_scores,
                    all_no_atr_scores,
                ):
                    atr = _feat(row, "atr_pct")
                    if not np.isfinite(atr):
                        continue
                    lab = "low" if atr <= q33 else ("medium" if atr <= q66 else "high")
                    buckets[lab].append((row, fs, as_, ms, ns))
                for lab, items in buckets.items():
                    if len(items) < 30:
                        bucket_summary[lab] = {
                            "n": len(items),
                            "note": "Too few observations.",
                            "q33": float(q33),
                            "q66": float(q66),
                        }
                        continue
                    brows = [i[0] for i in items]
                    _, fagg = evaluate_barrier_by_timestamp(brows, np.array([i[1] for i in items]), barrier)
                    _, aagg = evaluate_barrier_by_timestamp(brows, np.array([i[2] for i in items]), barrier)
                    _, magg = evaluate_barrier_by_timestamp(brows, np.array([i[3] for i in items]), barrier)
                    # Binary AUC/Brier within bucket for full
                    by = np.array(
                        [barrier_binary(r, barrier) if barrier_binary(r, barrier) is not None else -1 for r in brows],
                        dtype=int,
                    )
                    bmask = by >= 0
                    bfull = np.array([i[1] for i in items], dtype=float)[bmask]
                    by_bin = by[bmask]
                    bcls = binary_classification_metrics(by_bin, bfull) if len(by_bin) else {}
                    bucket_summary[lab] = {
                        "n": len(items),
                        "q33": float(q33),
                        "q66": float(q66),
                        "base_success_rate": bcls.get("base_rate"),
                        "full_top5_success_rate": fagg.get("top_5_success_rate"),
                        "atr_top5_success_rate": aagg.get("top_5_success_rate"),
                        "momentum_top5_success_rate": magg.get("top_5_success_rate"),
                        "all_success_rate": fagg.get("all_success_rate"),
                        "full_auc": bcls.get("auc"),
                        "full_brier": bcls.get("brier"),
                    }

        # Expected value diagnostics (research-only)
        ev = {}
        for label, (up, down) in EV_SPECS.items():
            if label != barrier:
                continue
            f_rate = full_summary.get("top_5_success_rate")
            a_rate = atr_summary.get("top_5_success_rate")
            ev = {
                "assumption": f"success=+{up:.0%}, failure=-{down:.0%}; diagnostic only; no leverage/compounding/sizing.",
                "cost_per_event": COST_PER_EVENT,
                "full_top5_ev_gross": (
                    expected_value(f_rate, up, down) if f_rate is not None else None
                ),
                "atr_top5_ev_gross": (
                    expected_value(a_rate, up, down) if a_rate is not None else None
                ),
                "full_top5_ev_net_of_cost": (
                    expected_value(f_rate, up, down, cost=COST_PER_EVENT) if f_rate is not None else None
                ),
                "atr_top5_ev_net_of_cost": (
                    expected_value(a_rate, up, down, cost=COST_PER_EVENT) if a_rate is not None else None
                ),
            }

        # Final blob (train on all binary rows) for lineage only; never activated.
        blob = b""
        if len(idx_bin) >= 50 and len(set(y_bin.tolist())) >= 2:
            final = _new_classifier()
            final.fit(x_full[idx_bin], y_bin)
            buf = BytesIO()
            joblib.dump(
                {
                    "model": final,
                    "feature_names": full_names,
                    "target_name": barrier,
                    "experiment_id": experiment.experiment_id,
                },
                buf,
            )
            blob = buf.getvalue()
        blobs[barrier] = blob

        per_barrier[barrier] = {
            "folds": fold_reports,
            "fold_count": len(fold_reports),
            "full": full_summary,
            "atr_only": atr_summary,
            "momentum": mom_summary,
            "full_without_atr": no_atr_summary,
            "pooled_calibration": pooled,
            "volatility_buckets": bucket_summary,
            "expected_value": ev,
            "fold_spearmans": fold_spearmans,
            "downside": {
                "full_top5": {
                    "mean_drawdown": full_summary.get("top_5_mean_drawdown"),
                    "median_drawdown": full_summary.get("top_5_median_drawdown"),
                    "worst_drawdown": full_summary.get("top_5_worst_drawdown"),
                    "frac_worse_3": full_summary.get("top_5_frac_dd_worse_3"),
                    "frac_worse_5": full_summary.get("top_5_frac_dd_worse_5"),
                    "frac_worse_10": full_summary.get("top_5_frac_dd_worse_10"),
                },
                "atr_top5": {
                    "mean_drawdown": atr_summary.get("top_5_mean_drawdown"),
                    "median_drawdown": atr_summary.get("top_5_median_drawdown"),
                    "worst_drawdown": atr_summary.get("top_5_worst_drawdown"),
                    "frac_worse_3": atr_summary.get("top_5_frac_dd_worse_3"),
                    "frac_worse_5": atr_summary.get("top_5_frac_dd_worse_5"),
                    "frac_worse_10": atr_summary.get("top_5_frac_dd_worse_10"),
                },
                "all_assets": {"mean_drawdown": full_summary.get("all_mean_drawdown")},
            },
        }

    status, note, evidence = classify_barrier_experiment(per_barrier, leakage_ok=leakage_ok)

    metrics = {
        "cross_section": cs_stats,
        "barriers": per_barrier,
        "barrier_stats": barrier_stats(ordered),
        "evidence": evidence,
        "lineage": experiment.lineage(),
        "model_type": "HistGradientBoostingClassifier",
        "binary_subset": "success vs failure; timeout & ambiguous excluded",
        "cost_per_event": COST_PER_EVENT,
    }
    return {
        "promote": False,
        "experiment_status": status,
        "passed_experiment_gates": status == STATUS_PROMISING,
        "feature_names": full_names,
        "blob": blobs.get(BARRIER_SPECS[0][0], b""),
        "metrics": metrics,
        "notes": note,
        "experiment_id": experiment.experiment_id,
        "target_name": "barrier_first_touch",
        "sample_stride": experiment.primary_stride_bars,
        "move_threshold": experiment.move_pct,
    }
