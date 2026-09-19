"""Cross-sectional relative upside ranking experiment.

Hypothesis: at timestamp T, can features available at T rank symbols by
future_max_upside_12h better than ATR-only or simple momentum?

Future ranking is the evaluation target only — never a feature.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.experiments import ExperimentSpec
from app.research.features import feature_names
from app.research.history import as_utc
from app.research.spec import embargo

MIN_CROSS_SECTION = 10
STATUS_REJECTED = "rejected"
STATUS_WEAK = "weak_ranking_signal"
STATUS_PROMISING = "promising_ranking_signal"

HIT_THRESHOLDS = (0.03, 0.05, 0.10)


def _feat(row: Any, name: str) -> float:
    feats = getattr(row, "features", None) or {}
    value = feats.get(name)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def future_max_upside(row: Any) -> float | None:
    value = getattr(row, "max_upside", None)
    if value is None:
        return None
    return float(value)


def future_max_drawdown(row: Any) -> float | None:
    value = getattr(row, "max_drawdown", None)
    if value is None:
        return None
    return float(value)


def future_return(row: Any) -> float | None:
    value = getattr(row, "fwd_return", None)
    if value is None:
        return None
    return float(value)


def rank_percentiles(values: Sequence[float]) -> np.ndarray:
    """Map values to [0, 1] within a cross-section. 1.0 = largest."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    if n == 1:
        return np.array([1.0])
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    # Average ties
    sorted_vals = arr[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg = (i + j - 1) / 2.0
            ranks[order[i:j]] = avg
        i = j
    return ranks / (n - 1)


def group_by_timestamp(rows: Sequence[Any]) -> dict[datetime, list[Any]]:
    groups: dict[datetime, list[Any]] = defaultdict(list)
    for row in rows:
        if future_max_upside(row) is None:
            continue
        groups[as_utc(row.ts)].append(row)
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


def annotate_cross_sectional_ranks(
    rows: Sequence[Any],
    *,
    min_cross_section: int = MIN_CROSS_SECTION,
) -> tuple[list[Any], dict[str, Any]]:
    """Attach future_upside_rank_percentile for timestamps with enough assets."""
    groups = group_by_timestamp(rows)
    considered = 0
    excluded = 0
    sizes: list[int] = []
    kept: list[Any] = []
    for _ts, bucket in groups.items():
        considered += 1
        if len(bucket) < min_cross_section:
            excluded += 1
            continue
        upsides = [float(future_max_upside(r)) for r in bucket]
        percentiles = rank_percentiles(upsides)
        sizes.append(len(bucket))
        for row, pct in zip(bucket, percentiles):
            row.future_max_upside_12h = float(future_max_upside(row))
            row.future_return_12h = future_return(row)
            row.future_max_drawdown_12h = future_max_drawdown(row)
            row.future_upside_rank_percentile = float(pct)
            kept.append(row)
    stats = {
        "timestamps_considered": considered,
        "timestamps_excluded_below_min": excluded,
        "timestamps_used": considered - excluded,
        "min_cross_section_required": min_cross_section,
        "average_cross_section_size": float(np.mean(sizes)) if sizes else None,
        "min_cross_section_size": int(min(sizes)) if sizes else None,
        "max_cross_section_size": int(max(sizes)) if sizes else None,
        "asset_observations_used": len(kept),
    }
    return kept, stats


def _matrix(rows: Sequence[Any], names: Sequence[str]) -> np.ndarray:
    return np.array([[_feat(r, n) for n in names] for r in rows], dtype=float)


def _new_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.06,
        max_iter=140,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=42,
    )


def score_rows(rows: Sequence[Any], scores: np.ndarray) -> list[dict[str, Any]]:
    """Pair rows with scores for one timestamp bucket."""
    out = []
    for row, score in zip(rows, scores):
        out.append(
            {
                "symbol": row.symbol,
                "score": float(score) if np.isfinite(score) else float("-inf"),
                "upside": float(future_max_upside(row)),
                "drawdown": float(future_max_drawdown(row) or 0.0),
                "atr_pct": _feat(row, "atr_pct"),
                "ret_24": _feat(row, "ret_24"),
                "actual_rank_pct": float(getattr(row, "future_upside_rank_percentile", float("nan"))),
            }
        )
    return out


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), len(scores)))
    # nan -> -inf so they sort last
    clean = np.where(np.isfinite(scores), scores, -np.inf)
    return np.argsort(clean)[::-1][:k]


def group_stats(items: list[dict[str, Any]]) -> dict[str, float | None]:
    if not items:
        return {
            "n": 0,
            "mean_upside": None,
            "median_upside": None,
            "max_upside": None,
            "mean_drawdown": None,
            "mean_atr_pct": None,
            "hit_3pct": None,
            "hit_5pct": None,
            "hit_10pct": None,
        }
    ups = np.array([i["upside"] for i in items], dtype=float)
    dds = np.array([i["drawdown"] for i in items], dtype=float)
    atrs = np.array([i["atr_pct"] for i in items], dtype=float)
    atrs = atrs[np.isfinite(atrs)]
    return {
        "n": int(len(items)),
        "mean_upside": float(np.mean(ups)),
        "median_upside": float(np.median(ups)),
        "max_upside": float(np.max(ups)),
        "mean_drawdown": float(np.mean(dds)),
        "mean_atr_pct": float(np.mean(atrs)) if len(atrs) else None,
        "hit_3pct": float(np.mean(ups >= 0.03)),
        "hit_5pct": float(np.mean(ups >= 0.05)),
        "hit_10pct": float(np.mean(ups >= 0.10)),
    }


def evaluate_timestamp_ranking(scored: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate one timestamp's predicted scores vs realized upside ranks."""
    n = len(scored)
    scores = np.array([s["score"] for s in scored], dtype=float)
    upsides = np.array([s["upside"] for s in scored], dtype=float)
    actual_pct = np.array([s["actual_rank_pct"] for s in scored], dtype=float)
    pred_pct = rank_percentiles(scores)

    spearman = None
    if n >= 3 and np.nanstd(scores) > 0 and np.nanstd(upsides) > 0:
        rho, _ = spearmanr(scores, upsides)
        spearman = float(rho) if np.isfinite(rho) else None

    order = top_k_indices(scores, n)
    ranked = [scored[i] for i in order]
    k20 = max(1, int(np.ceil(0.20 * n)))

    def take(k: int) -> list[dict[str, Any]]:
        return ranked[: min(k, n)]

    return {
        "n": n,
        "spearman": spearman,
        "all": group_stats(scored),
        "top_1": group_stats(take(1)),
        "top_3": group_stats(take(3)),
        "top_5": group_stats(take(5)),
        "top_20pct": group_stats(take(k20)),
        "pred_rank_pct": pred_pct.tolist(),
        "actual_rank_pct": actual_pct.tolist(),
    }


def _mean_key(results: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
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


def aggregate_timestamp_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "timestamps": 0,
            "spearman": None,
            "top_1_mean_upside": None,
            "top_3_mean_upside": None,
            "top_5_mean_upside": None,
            "top_20pct_mean_upside": None,
            "all_mean_upside": None,
            "top_1_hit_3pct": None,
            "top_5_hit_3pct": None,
            "top_5_hit_5pct": None,
            "top_5_hit_10pct": None,
            "top_5_mean_atr_pct": None,
            "top_5_mean_drawdown": None,
            "top_1": None,
            "top_5": None,
            "top_20pct": None,
        }
    return {
        "timestamps": len(results),
        "spearman": _mean_key(results, ("spearman",)),
        "top_1_mean_upside": _mean_key(results, ("top_1", "mean_upside")),
        "top_3_mean_upside": _mean_key(results, ("top_3", "mean_upside")),
        "top_5_mean_upside": _mean_key(results, ("top_5", "mean_upside")),
        "top_20pct_mean_upside": _mean_key(results, ("top_20pct", "mean_upside")),
        "all_mean_upside": _mean_key(results, ("all", "mean_upside")),
        "top_1_hit_3pct": _mean_key(results, ("top_1", "hit_3pct")),
        "top_5_hit_3pct": _mean_key(results, ("top_5", "hit_3pct")),
        "top_5_hit_5pct": _mean_key(results, ("top_5", "hit_5pct")),
        "top_5_hit_10pct": _mean_key(results, ("top_5", "hit_10pct")),
        "top_5_mean_atr_pct": _mean_key(results, ("top_5", "mean_atr_pct")),
        "top_5_mean_drawdown": _mean_key(results, ("top_5", "mean_drawdown")),
        "top_1": {
            "mean_upside": _mean_key(results, ("top_1", "mean_upside")),
            "median_upside": _mean_key(results, ("top_1", "median_upside")),
            "max_upside": _mean_key(results, ("top_1", "max_upside")),
            "hit_3pct": _mean_key(results, ("top_1", "hit_3pct")),
            "hit_5pct": _mean_key(results, ("top_1", "hit_5pct")),
            "hit_10pct": _mean_key(results, ("top_1", "hit_10pct")),
        },
        "top_5": {
            "mean_upside": _mean_key(results, ("top_5", "mean_upside")),
            "median_upside": _mean_key(results, ("top_5", "median_upside")),
            "max_upside": _mean_key(results, ("top_5", "max_upside")),
            "hit_3pct": _mean_key(results, ("top_5", "hit_3pct")),
            "hit_5pct": _mean_key(results, ("top_5", "hit_5pct")),
            "hit_10pct": _mean_key(results, ("top_5", "hit_10pct")),
            "mean_atr_pct": _mean_key(results, ("top_5", "mean_atr_pct")),
            "mean_drawdown": _mean_key(results, ("top_5", "mean_drawdown")),
        },
        "top_20pct": {
            "mean_upside": _mean_key(results, ("top_20pct", "mean_upside")),
            "median_upside": _mean_key(results, ("top_20pct", "median_upside")),
            "max_upside": _mean_key(results, ("top_20pct", "max_upside")),
            "hit_3pct": _mean_key(results, ("top_20pct", "hit_3pct")),
            "hit_5pct": _mean_key(results, ("top_20pct", "hit_5pct")),
            "hit_10pct": _mean_key(results, ("top_20pct", "hit_10pct")),
        },
        "all": {
            "mean_upside": _mean_key(results, ("all", "mean_upside")),
            "mean_atr_pct": _mean_key(results, ("all", "mean_atr_pct")),
            "mean_drawdown": _mean_key(results, ("all", "mean_drawdown")),
            "hit_3pct": _mean_key(results, ("all", "hit_3pct")),
            "hit_5pct": _mean_key(results, ("all", "hit_5pct")),
            "hit_10pct": _mean_key(results, ("all", "hit_10pct")),
        },
    }


def evaluate_scored_rows_by_timestamp(
    rows: Sequence[Any],
    scores_by_index: dict[int, float],
    row_index: dict[int, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Group scored rows by timestamp and evaluate ranking metrics."""
    # Build id(row) -> score map via positional indices on the provided list.
    by_ts: dict[datetime, list[tuple[Any, float]]] = defaultdict(list)
    for i, row in enumerate(rows):
        score = scores_by_index.get(i)
        if score is None:
            continue
        by_ts[as_utc(row.ts)].append((row, float(score)))

    ts_results = []
    for _ts, pairs in sorted(by_ts.items(), key=lambda kv: kv[0]):
        bucket_rows = [p[0] for p in pairs]
        bucket_scores = np.array([p[1] for p in pairs], dtype=float)
        scored = score_rows(bucket_rows, bucket_scores)
        ts_results.append(evaluate_timestamp_ranking(scored))
    return ts_results, aggregate_timestamp_results(ts_results)


def score_with_feature(rows: Sequence[Any], feature: str) -> np.ndarray:
    return np.array([_feat(r, feature) for r in rows], dtype=float)


def classify_ranking_result(
    full: dict[str, Any],
    atr: dict[str, Any],
    momentum: dict[str, Any],
    fold_spearmans: list[float | None],
    *,
    leakage_ok: bool,
) -> tuple[str, str, dict[str, Any]]:
    """Classify experiment without inventing pass-forcing thresholds.

    Evidence checklist is reported explicitly; status follows the checklist.
    """
    spearman = full.get("spearman")
    top5 = full.get("top_5_mean_upside")
    all_mean = full.get("all_mean_upside")
    atr_top5 = atr.get("top_5_mean_upside")
    mom_top5 = momentum.get("top_5_mean_upside")
    atr_sp = atr.get("spearman")
    mom_sp = momentum.get("spearman")

    finite_folds = [s for s in fold_spearmans if s is not None and np.isfinite(s)]
    positive_spearman = spearman is not None and spearman > 0
    consistent = bool(finite_folds) and all(s > 0 for s in finite_folds) and len(finite_folds) >= 2
    beats_all = (
        top5 is not None and all_mean is not None and all_mean != 0 and top5 > all_mean
    )
    beats_atr = (
        top5 is not None
        and atr_top5 is not None
        and spearman is not None
        and atr_sp is not None
        and top5 > atr_top5
        and spearman > atr_sp
    )
    beats_mom = (
        top5 is not None
        and mom_top5 is not None
        and spearman is not None
        and mom_sp is not None
        and top5 > mom_top5
        and spearman > mom_sp
    )
    multi_fold = len(finite_folds) >= 2

    evidence = {
        "positive_aggregate_spearman": positive_spearman,
        "consistent_positive_spearman_across_folds": consistent,
        "top5_beats_all_assets": beats_all,
        "full_beats_atr_on_spearman_and_top5": beats_atr,
        "full_beats_momentum_on_spearman_and_top5": beats_mom,
        "effect_not_single_fold": multi_fold and consistent,
        "leakage_audit_passed": leakage_ok,
        "fold_spearmans": fold_spearmans,
        "deltas": {
            "spearman_full_minus_atr": (None if spearman is None or atr_sp is None else spearman - atr_sp),
            "spearman_full_minus_momentum": (
                None if spearman is None or mom_sp is None else spearman - mom_sp
            ),
            "top5_upside_full_minus_all": (None if top5 is None or all_mean is None else top5 - all_mean),
            "top5_upside_full_minus_atr": (None if top5 is None or atr_top5 is None else top5 - atr_top5),
            "top5_upside_full_minus_momentum": (
                None if top5 is None or mom_top5 is None else top5 - mom_top5
            ),
        },
    }

    checks = [
        evidence["positive_aggregate_spearman"],
        evidence["consistent_positive_spearman_across_folds"],
        evidence["top5_beats_all_assets"],
        evidence["full_beats_atr_on_spearman_and_top5"],
        evidence["full_beats_momentum_on_spearman_and_top5"],
        evidence["effect_not_single_fold"],
        evidence["leakage_audit_passed"],
    ]
    passed = sum(1 for c in checks if c)

    if all(checks):
        status = STATUS_PROMISING
        note = (
            "promising_ranking_signal: full model shows positive, multi-fold Spearman, "
            "top-5 upside above all-assets, and beats ATR-only and momentum ranking."
        )
    elif positive_spearman and beats_all and leakage_ok and passed >= 3:
        status = STATUS_WEAK
        note = (
            "weak_ranking_signal: some ranking lift vs all-assets, but not a clear win "
            "over ATR-only and/or momentum across folds."
        )
    else:
        status = STATUS_REJECTED
        note = (
            "rejected: insufficient evidence that full-model ranking identifies relative "
            "future upside beyond simple baselines."
        )
    return status, note, evidence


def fit_ranking_walk_forward(
    rows: Sequence[Any],
    experiment: ExperimentSpec,
    *,
    min_cross_section: int = MIN_CROSS_SECTION,
    leakage_ok: bool = True,
) -> dict[str, Any]:
    """Train regressor on future_max_upside_12h; evaluate cross-sectional ranks."""
    annotated, cs_stats = annotate_cross_sectional_ranks(rows, min_cross_section=min_cross_section)
    if not annotated:
        return {
            "promote": False,
            "experiment_status": STATUS_REJECTED,
            "notes": "No timestamps met minimum cross-section size.",
            "metrics": {"cross_section": cs_stats},
            "experiment_id": experiment.experiment_id,
            "target_name": "future_upside_rank_percentile",
            "sample_stride": experiment.primary_stride_bars,
            "feature_names": feature_names(),
            "blob": b"",
        }

    ordered = sorted(annotated, key=lambda r: (as_utc(r.ts), r.symbol))
    full_names = feature_names()
    atr_names = [n for n in experiment.atr_feature_names if n in full_names]
    mom_feature = experiment.momentum_feature

    # Walk-forward on unique timestamps (chronological), then map to row indices.
    # min_train/min_test are timestamp counts (not asset-rows). Each timestamp carries
    # a full cross-section (~40+ assets), so requiring 200 timestamps would skip all folds
    # on a ~90-day 12h-stride history.
    unique_ts = sorted({as_utc(r.ts) for r in ordered})
    folds = walk_forward_folds(
        unique_ts,
        min_train=60,  # ~30 days of 12h timestamps
        min_test=14,  # ~7 days of 12h timestamps
    )
    gap = embargo()

    fold_reports: list[dict[str, Any]] = []
    fold_spearmans: list[float | None] = []

    x_full = _matrix(ordered, full_names)
    y = np.array([float(r.future_max_upside_12h) for r in ordered], dtype=float)
    row_ts = [as_utc(r.ts) for r in ordered]

    for train_ts_idx, test_ts_idx in folds:
        train_ts = {unique_ts[i] for i in train_ts_idx}
        test_ts = {unique_ts[i] for i in test_ts_idx}
        if not fold_is_causal(unique_ts, train_ts_idx, test_ts_idx, gap):
            continue
        train_idx = [i for i, t in enumerate(row_ts) if t in train_ts]
        test_idx = [i for i, t in enumerate(row_ts) if t in test_ts]
        if len(train_idx) < 50 or len(test_idx) < 20:
            continue

        clf = _new_regressor()
        clf.fit(x_full[train_idx], y[train_idx])
        pred = clf.predict(x_full[test_idx])

        test_rows = [ordered[i] for i in test_idx]
        scores_full = {j: float(pred[j]) for j in range(len(test_rows))}
        scores_atr = {j: float(_feat(test_rows[j], atr_names[0])) for j in range(len(test_rows))}
        scores_mom = {j: float(_feat(test_rows[j], mom_feature)) for j in range(len(test_rows))}

        full_ts, full_agg = evaluate_scored_rows_by_timestamp(test_rows, scores_full)
        atr_ts, atr_agg = evaluate_scored_rows_by_timestamp(test_rows, scores_atr)
        mom_ts, mom_agg = evaluate_scored_rows_by_timestamp(test_rows, scores_mom)

        fold_spearmans.append(full_agg.get("spearman"))
        fold_reports.append(
            {
                "train_start": min(train_ts).isoformat(),
                "train_end": max(train_ts).isoformat(),
                "test_start": min(test_ts).isoformat(),
                "test_end": max(test_ts).isoformat(),
                "train_asset_observations": len(train_idx),
                "test_asset_observations": len(test_idx),
                "test_timestamps": full_agg.get("timestamps"),
                "average_cross_section_size": (
                    float(np.mean([r["n"] for r in full_ts])) if full_ts else None
                ),
                "full": full_agg,
                "atr_only": atr_agg,
                "momentum": mom_agg,
                "all_assets": {
                    "mean_upside": full_agg.get("all_mean_upside"),
                    "hit_3pct": (full_agg.get("all") or {}).get("hit_3pct"),
                    "hit_5pct": (full_agg.get("all") or {}).get("hit_5pct"),
                    "hit_10pct": (full_agg.get("all") or {}).get("hit_10pct"),
                    "mean_atr_pct": (full_agg.get("all") or {}).get("mean_atr_pct"),
                },
            }
        )

    def _agg_field(path: tuple[str, ...]) -> float | None:
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

    full_summary = {
        "spearman": _agg_field(("full", "spearman")),
        "top_1_mean_upside": _agg_field(("full", "top_1_mean_upside")),
        "top_3_mean_upside": _agg_field(("full", "top_3_mean_upside")),
        "top_5_mean_upside": _agg_field(("full", "top_5_mean_upside")),
        "top_20pct_mean_upside": _agg_field(("full", "top_20pct_mean_upside")),
        "all_mean_upside": _agg_field(("full", "all_mean_upside")),
        "top_1_hit_3pct": _agg_field(("full", "top_1_hit_3pct")),
        "top_5_hit_3pct": _agg_field(("full", "top_5_hit_3pct")),
        "top_5_hit_5pct": _agg_field(("full", "top_5_hit_5pct")),
        "top_5_hit_10pct": _agg_field(("full", "top_5_hit_10pct")),
        "top_5_mean_atr_pct": _agg_field(("full", "top_5_mean_atr_pct")),
        "top_5_mean_drawdown": _agg_field(("full", "top_5_mean_drawdown")),
        "top_1": {
            "mean_upside": _agg_field(("full", "top_1", "mean_upside")),
            "median_upside": _agg_field(("full", "top_1", "median_upside")),
            "max_upside": _agg_field(("full", "top_1", "max_upside")),
            "hit_3pct": _agg_field(("full", "top_1", "hit_3pct")),
            "hit_5pct": _agg_field(("full", "top_1", "hit_5pct")),
            "hit_10pct": _agg_field(("full", "top_1", "hit_10pct")),
        },
        "top_5": {
            "mean_upside": _agg_field(("full", "top_5", "mean_upside")),
            "median_upside": _agg_field(("full", "top_5", "median_upside")),
            "max_upside": _agg_field(("full", "top_5", "max_upside")),
            "hit_3pct": _agg_field(("full", "top_5", "hit_3pct")),
            "hit_5pct": _agg_field(("full", "top_5", "hit_5pct")),
            "hit_10pct": _agg_field(("full", "top_5", "hit_10pct")),
            "mean_atr_pct": _agg_field(("full", "top_5", "mean_atr_pct")),
            "mean_drawdown": _agg_field(("full", "top_5", "mean_drawdown")),
        },
        "top_20pct": {
            "mean_upside": _agg_field(("full", "top_20pct", "mean_upside")),
            "median_upside": _agg_field(("full", "top_20pct", "median_upside")),
            "max_upside": _agg_field(("full", "top_20pct", "max_upside")),
            "hit_3pct": _agg_field(("full", "top_20pct", "hit_3pct")),
            "hit_5pct": _agg_field(("full", "top_20pct", "hit_5pct")),
            "hit_10pct": _agg_field(("full", "top_20pct", "hit_10pct")),
        },
    }
    atr_summary = {
        "spearman": _agg_field(("atr_only", "spearman")),
        "top_1_mean_upside": _agg_field(("atr_only", "top_1_mean_upside")),
        "top_3_mean_upside": _agg_field(("atr_only", "top_3_mean_upside")),
        "top_5_mean_upside": _agg_field(("atr_only", "top_5_mean_upside")),
        "top_20pct_mean_upside": _agg_field(("atr_only", "top_20pct_mean_upside")),
        "all_mean_upside": _agg_field(("atr_only", "all_mean_upside")),
        "top_1_hit_3pct": _agg_field(("atr_only", "top_1_hit_3pct")),
        "top_5_hit_3pct": _agg_field(("atr_only", "top_5_hit_3pct")),
        "top_5_hit_5pct": _agg_field(("atr_only", "top_5_hit_5pct")),
        "top_5_hit_10pct": _agg_field(("atr_only", "top_5_hit_10pct")),
        "top_5_mean_atr_pct": _agg_field(("atr_only", "top_5_mean_atr_pct")),
        "top_5_mean_drawdown": _agg_field(("atr_only", "top_5_mean_drawdown")),
        "top_5": {
            "mean_upside": _agg_field(("atr_only", "top_5", "mean_upside")),
            "mean_atr_pct": _agg_field(("atr_only", "top_5", "mean_atr_pct")),
            "mean_drawdown": _agg_field(("atr_only", "top_5", "mean_drawdown")),
            "hit_3pct": _agg_field(("atr_only", "top_5", "hit_3pct")),
            "hit_5pct": _agg_field(("atr_only", "top_5", "hit_5pct")),
            "hit_10pct": _agg_field(("atr_only", "top_5", "hit_10pct")),
        },
    }
    mom_summary = {
        "spearman": _agg_field(("momentum", "spearman")),
        "top_1_mean_upside": _agg_field(("momentum", "top_1_mean_upside")),
        "top_3_mean_upside": _agg_field(("momentum", "top_3_mean_upside")),
        "top_5_mean_upside": _agg_field(("momentum", "top_5_mean_upside")),
        "top_20pct_mean_upside": _agg_field(("momentum", "top_20pct_mean_upside")),
        "all_mean_upside": _agg_field(("momentum", "all_mean_upside")),
        "top_1_hit_3pct": _agg_field(("momentum", "top_1_hit_3pct")),
        "top_5_hit_3pct": _agg_field(("momentum", "top_5_hit_3pct")),
        "top_5_hit_5pct": _agg_field(("momentum", "top_5_hit_5pct")),
        "top_5_hit_10pct": _agg_field(("momentum", "top_5_hit_10pct")),
        "top_5": {
            "mean_upside": _agg_field(("momentum", "top_5", "mean_upside")),
            "hit_3pct": _agg_field(("momentum", "top_5", "hit_3pct")),
            "hit_5pct": _agg_field(("momentum", "top_5", "hit_5pct")),
            "hit_10pct": _agg_field(("momentum", "top_5", "hit_10pct")),
        },
    }
    all_assets = {
        "mean_upside": full_summary.get("all_mean_upside"),
        "spearman": 0.0,  # random ranking expectation
        "top_1_mean_upside": full_summary.get("all_mean_upside"),
        "top_3_mean_upside": full_summary.get("all_mean_upside"),
        "top_5_mean_upside": full_summary.get("all_mean_upside"),
        "top_20pct_mean_upside": full_summary.get("all_mean_upside"),
        "top_1_hit_3pct": _agg_field(("all_assets", "hit_3pct")),
        "top_5_hit_3pct": _agg_field(("all_assets", "hit_3pct")),
        "top_5_hit_5pct": _agg_field(("all_assets", "hit_5pct")),
        "top_5_hit_10pct": _agg_field(("all_assets", "hit_10pct")),
        "note": "Random/all-assets: expected top-K mean upside equals universe mean upside.",
    }

    comparison_table = {
        "spearman": {
            "full_model": full_summary.get("spearman"),
            "atr_only": atr_summary.get("spearman"),
            "momentum_ret_24": mom_summary.get("spearman"),
            "random_all_assets": 0.0,
        },
        "top_1_mean_upside": {
            "full_model": full_summary.get("top_1_mean_upside"),
            "atr_only": atr_summary.get("top_1_mean_upside"),
            "momentum_ret_24": mom_summary.get("top_1_mean_upside"),
            "random_all_assets": all_assets.get("top_1_mean_upside"),
        },
        "top_3_mean_upside": {
            "full_model": full_summary.get("top_3_mean_upside"),
            "atr_only": atr_summary.get("top_3_mean_upside"),
            "momentum_ret_24": mom_summary.get("top_3_mean_upside"),
            "random_all_assets": all_assets.get("top_3_mean_upside"),
        },
        "top_5_mean_upside": {
            "full_model": full_summary.get("top_5_mean_upside"),
            "atr_only": atr_summary.get("top_5_mean_upside"),
            "momentum_ret_24": mom_summary.get("top_5_mean_upside"),
            "random_all_assets": all_assets.get("top_5_mean_upside"),
        },
        "top_20pct_mean_upside": {
            "full_model": full_summary.get("top_20pct_mean_upside"),
            "atr_only": atr_summary.get("top_20pct_mean_upside"),
            "momentum_ret_24": mom_summary.get("top_20pct_mean_upside"),
            "random_all_assets": all_assets.get("top_20pct_mean_upside"),
        },
        "top_1_hit_3pct": {
            "full_model": full_summary.get("top_1_hit_3pct"),
            "atr_only": atr_summary.get("top_1_hit_3pct"),
            "momentum_ret_24": mom_summary.get("top_1_hit_3pct"),
            "random_all_assets": all_assets.get("top_1_hit_3pct"),
        },
        "top_5_hit_3pct": {
            "full_model": full_summary.get("top_5_hit_3pct"),
            "atr_only": atr_summary.get("top_5_hit_3pct"),
            "momentum_ret_24": mom_summary.get("top_5_hit_3pct"),
            "random_all_assets": all_assets.get("top_5_hit_3pct"),
        },
        "top_5_hit_5pct": {
            "full_model": full_summary.get("top_5_hit_5pct"),
            "atr_only": atr_summary.get("top_5_hit_5pct"),
            "momentum_ret_24": mom_summary.get("top_5_hit_5pct"),
            "random_all_assets": all_assets.get("top_5_hit_5pct"),
        },
        "top_5_hit_10pct": {
            "full_model": full_summary.get("top_5_hit_10pct"),
            "atr_only": atr_summary.get("top_5_hit_10pct"),
            "momentum_ret_24": mom_summary.get("top_5_hit_10pct"),
            "random_all_assets": all_assets.get("top_5_hit_10pct"),
        },
    }

    concentration = {
        "full_model_top5": {
            "mean_atr_pct": full_summary.get("top_5_mean_atr_pct"),
            "mean_future_max_upside": full_summary.get("top_5_mean_upside"),
            "mean_future_max_drawdown": full_summary.get("top_5_mean_drawdown"),
        },
        "atr_only_top5": {
            "mean_atr_pct": atr_summary.get("top_5_mean_atr_pct"),
            "mean_future_max_upside": atr_summary.get("top_5_mean_upside"),
            "mean_future_max_drawdown": atr_summary.get("top_5_mean_drawdown"),
        },
        "all_assets": {
            "mean_atr_pct": _agg_field(("all_assets", "mean_atr_pct")),
            "mean_future_max_upside": full_summary.get("all_mean_upside"),
            "mean_future_max_drawdown": _mean_all_drawdown(fold_reports),
        },
    }

    status, note, evidence = classify_ranking_result(
        full_summary, atr_summary, mom_summary, fold_spearmans, leakage_ok=leakage_ok
    )

    blob = b""
    if len(ordered) >= 50:
        final = _new_regressor()
        final.fit(x_full, y)
        from io import BytesIO
        import joblib

        buf = BytesIO()
        joblib.dump(
            {
                "model": final,
                "feature_names": full_names,
                "target_name": "future_max_upside_12h",
                "evaluation_target": "future_upside_rank_percentile",
                "experiment_id": experiment.experiment_id,
            },
            buf,
        )
        blob = buf.getvalue()

    metrics = {
        "fold_count": len(fold_reports),
        "folds": fold_reports,
        "cross_section": cs_stats,
        "full": full_summary,
        "atr_only": atr_summary,
        "momentum": mom_summary,
        "all_assets_random": all_assets,
        "comparison_table": comparison_table,
        "concentration": concentration,
        "evidence": evidence,
        "lineage": experiment.lineage(),
        "training_target": "future_max_upside_12h",
        "evaluation_target": "future_upside_rank_percentile",
        "model_type": "HistGradientBoostingRegressor",
    }
    return {
        "promote": False,
        "experiment_status": status,
        "passed_experiment_gates": status == STATUS_PROMISING,
        "feature_names": full_names,
        "blob": blob,
        "metrics": metrics,
        "notes": note,
        "experiment_id": experiment.experiment_id,
        "target_name": "future_upside_rank_percentile",
        "sample_stride": experiment.primary_stride_bars,
        "move_threshold": experiment.move_pct,
        "comparison_table": comparison_table,
    }


def _mean_all_drawdown(fold_reports: list[dict[str, Any]]) -> float | None:
    vals = []
    for fr in fold_reports:
        # Recompute from nested full.all if present in fold aggregations
        all_block = (fr.get("full") or {}).get("all") or {}
        v = all_block.get("mean_drawdown")
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None
