"""Volatility-adjusted upside experiment: information beyond current ATR?

Primary target: cross-sectional residual of future upside given current ATR.
Future values are used ONLY for targets / evaluation — never as features.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from io import BytesIO
from typing import Any, Sequence

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.experiments import ExperimentSpec
from app.research.features import feature_names
from app.research.history import as_utc
from app.research.ranking import (
    MIN_CROSS_SECTION,
    future_max_drawdown,
    future_max_upside,
    future_return,
    group_by_timestamp,
    rank_percentiles,
)
from app.research.spec import embargo

ATR_EPS = 1e-6
STATUS_REJECTED = "rejected"
STATUS_WEAK = "weak_volatility_adjusted_signal"
STATUS_PROMISING = "promising_volatility_adjusted_signal"


def _feat(row: Any, name: str) -> float:
    feats = getattr(row, "features", None) or {}
    value = feats.get(name)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def atr_fraction(row: Any) -> float:
    """Convert stored atr_pct (percent units, e.g. 2.5) to a fraction (0.025)."""
    atr = _feat(row, "atr_pct")
    if not np.isfinite(atr) or atr <= 0:
        return float("nan")
    return float(atr) / 100.0


def atr_normalized_upside(max_upside: float, atr_frac: float, *, eps: float = ATR_EPS) -> float:
    """future_upside_atr_multiple = future_max_upside_12h / max(atr_frac, eps)."""
    return float(max_upside) / max(float(atr_frac), eps)


def cross_section_vol_residuals(
    upsides: Sequence[float],
    atr_fracs: Sequence[float],
    *,
    eps: float = ATR_EPS,
) -> tuple[np.ndarray, dict[str, float]]:
    """OLS residual of log-upside on log-ATR within one timestamp (target only).

    Definitions (explicit):
      atr_frac_i = atr_pct_i / 100
      log_up_i   = log(1 + max(future_max_upside_12h_i, 0))
      log_atr_i  = log(max(atr_frac_i, eps))
      Fit within the timestamp:  log_up = α + β * log_atr   (OLS)
      residual_i = log_up_i - (α̂ + β̂ * log_atr_i)

    residual > 0 ⇒ more upside than the cross-sectional volatility schedule implies.
    Features never see residual / future upside — only evaluation & training labels.
    """
    up = np.asarray(upsides, dtype=float)
    atr = np.asarray(atr_fracs, dtype=float)
    n = len(up)
    residuals = np.full(n, np.nan, dtype=float)
    meta = {"alpha": float("nan"), "beta": float("nan"), "n_fit": 0.0}
    mask = np.isfinite(up) & np.isfinite(atr) & (atr > 0)
    if int(mask.sum()) < 3:
        return residuals, meta
    log_up = np.log1p(np.maximum(up[mask], 0.0))
    log_atr = np.log(np.maximum(atr[mask], eps))
    x = log_atr.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, log_up)
    pred = model.predict(x)
    residuals[mask] = log_up - pred
    meta = {
        "alpha": float(model.intercept_),
        "beta": float(model.coef_[0]),
        "n_fit": float(int(mask.sum())),
    }
    return residuals, meta


def annotate_vol_adjusted_targets(
    rows: Sequence[Any],
    *,
    min_cross_section: int = MIN_CROSS_SECTION,
    eps: float = ATR_EPS,
) -> tuple[list[Any], dict[str, Any]]:
    """Attach ATR-normalized and CS residual targets; drop small cross-sections."""
    groups = group_by_timestamp(rows)
    considered = 0
    excluded = 0
    sizes: list[int] = []
    kept: list[Any] = []
    alphas: list[float] = []
    betas: list[float] = []

    for _ts, bucket in groups.items():
        considered += 1
        # Need valid ATR for vol-adjusted targets.
        eligible = []
        for row in bucket:
            atr_f = atr_fraction(row)
            up = future_max_upside(row)
            if up is None or not np.isfinite(atr_f):
                continue
            eligible.append(row)
        if len(eligible) < min_cross_section:
            excluded += 1
            continue

        upsides = [float(future_max_upside(r)) for r in eligible]
        atrs = [atr_fraction(r) for r in eligible]
        residuals, meta = cross_section_vol_residuals(upsides, atrs, eps=eps)
        if not np.isfinite(meta["alpha"]):
            excluded += 1
            continue
        alphas.append(meta["alpha"])
        betas.append(meta["beta"])
        percentiles = rank_percentiles(residuals.tolist())
        sizes.append(len(eligible))
        for row, residual, pct, up, atr_f in zip(eligible, residuals, percentiles, upsides, atrs):
            row.future_max_upside_12h = float(up)
            row.future_return_12h = future_return(row)
            row.future_max_drawdown_12h = future_max_drawdown(row)
            row.atr_frac = float(atr_f)
            row.future_upside_atr_multiple = atr_normalized_upside(up, atr_f, eps=eps)
            row.vol_adj_residual = float(residual)
            row.vol_adj_rank_percentile = float(pct)
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
        "mean_ols_alpha": float(np.mean(alphas)) if alphas else None,
        "mean_ols_beta": float(np.mean(betas)) if betas else None,
        "target_definition": {
            "atr_normalized": "future_max_upside_12h / max(atr_pct/100, eps)",
            "primary_residual": (
                "Within each timestamp: log_up=log1p(max(upside,0)), "
                "log_atr=log(max(atr_frac,eps)); OLS log_up~log_atr; "
                "residual=log_up-(α+β*log_atr). Rank by residual."
            ),
            "eps": eps,
        },
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


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), len(scores)))
    clean = np.where(np.isfinite(scores), scores, -np.inf)
    return np.argsort(clean)[::-1][:k]


def _group_metrics(items: list[dict[str, Any]]) -> dict[str, float | None]:
    if not items:
        return {
            "n": 0,
            "mean_raw_upside": None,
            "median_raw_upside": None,
            "max_raw_upside": None,
            "mean_vol_adj": None,
            "median_vol_adj": None,
            "mean_atr_multiple": None,
            "hit_3pct": None,
            "hit_5pct": None,
            "hit_10pct": None,
            "mean_atr_pct": None,
        }
    raw = np.array([i["raw_upside"] for i in items], dtype=float)
    adj = np.array([i["vol_adj"] for i in items], dtype=float)
    mult = np.array([i["atr_multiple"] for i in items], dtype=float)
    atr = np.array([i["atr_pct"] for i in items], dtype=float)
    atr = atr[np.isfinite(atr)]
    return {
        "n": int(len(items)),
        "mean_raw_upside": float(np.mean(raw)),
        "median_raw_upside": float(np.median(raw)),
        "max_raw_upside": float(np.max(raw)),
        "mean_vol_adj": float(np.nanmean(adj)),
        "median_vol_adj": float(np.nanmedian(adj)),
        "mean_atr_multiple": float(np.nanmean(mult)),
        "hit_3pct": float(np.mean(raw >= 0.03)),
        "hit_5pct": float(np.mean(raw >= 0.05)),
        "hit_10pct": float(np.mean(raw >= 0.10)),
        "mean_atr_pct": float(np.mean(atr)) if len(atr) else None,
    }


def evaluate_timestamp_scores(
    rows: Sequence[Any],
    scores: np.ndarray,
) -> dict[str, Any]:
    n = len(rows)
    scored = []
    actual = []
    for row, score in zip(rows, scores):
        scored.append(
            {
                "symbol": row.symbol,
                "score": float(score) if np.isfinite(score) else float("-inf"),
                "raw_upside": float(row.future_max_upside_12h),
                "vol_adj": float(row.vol_adj_residual),
                "atr_multiple": float(row.future_upside_atr_multiple),
                "atr_pct": _feat(row, "atr_pct"),
            }
        )
        actual.append(float(row.vol_adj_residual))
    scores_arr = np.array([s["score"] for s in scored], dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    spearman = None
    if n >= 3 and np.nanstd(scores_arr) > 0 and np.nanstd(actual_arr) > 0:
        rho, _ = spearmanr(scores_arr, actual_arr)
        spearman = float(rho) if np.isfinite(rho) else None

    order = _top_k_indices(scores_arr, n)
    ranked = [scored[i] for i in order]
    k20 = max(1, int(np.ceil(0.20 * n)))

    def take(k: int) -> list[dict[str, Any]]:
        return ranked[: min(k, n)]

    return {
        "n": n,
        "spearman": spearman,
        "all": _group_metrics(scored),
        "top_1": _group_metrics(take(1)),
        "top_3": _group_metrics(take(3)),
        "top_5": _group_metrics(take(5)),
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


def aggregate_ts_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"timestamps": 0, "spearman": None}
    out: dict[str, Any] = {
        "timestamps": len(results),
        "spearman": _mean_path(results, ("spearman",)),
        "top_1_mean_raw_upside": _mean_path(results, ("top_1", "mean_raw_upside")),
        "top_3_mean_raw_upside": _mean_path(results, ("top_3", "mean_raw_upside")),
        "top_5_mean_raw_upside": _mean_path(results, ("top_5", "mean_raw_upside")),
        "top_20pct_mean_raw_upside": _mean_path(results, ("top_20pct", "mean_raw_upside")),
        "top_1_mean_vol_adj": _mean_path(results, ("top_1", "mean_vol_adj")),
        "top_3_mean_vol_adj": _mean_path(results, ("top_3", "mean_vol_adj")),
        "top_5_mean_vol_adj": _mean_path(results, ("top_5", "mean_vol_adj")),
        "top_20pct_mean_vol_adj": _mean_path(results, ("top_20pct", "mean_vol_adj")),
        "top_1_median_raw_upside": _mean_path(results, ("top_1", "median_raw_upside")),
        "top_5_median_raw_upside": _mean_path(results, ("top_5", "median_raw_upside")),
        "top_5_median_vol_adj": _mean_path(results, ("top_5", "median_vol_adj")),
        "top_1_hit_3pct": _mean_path(results, ("top_1", "hit_3pct")),
        "top_5_hit_3pct": _mean_path(results, ("top_5", "hit_3pct")),
        "top_5_hit_5pct": _mean_path(results, ("top_5", "hit_5pct")),
        "top_5_hit_10pct": _mean_path(results, ("top_5", "hit_10pct")),
        "top_20pct_hit_3pct": _mean_path(results, ("top_20pct", "hit_3pct")),
        "top_20pct_hit_5pct": _mean_path(results, ("top_20pct", "hit_5pct")),
        "top_20pct_hit_10pct": _mean_path(results, ("top_20pct", "hit_10pct")),
        "all_mean_raw_upside": _mean_path(results, ("all", "mean_raw_upside")),
        "all_mean_vol_adj": _mean_path(results, ("all", "mean_vol_adj")),
        "all_hit_3pct": _mean_path(results, ("all", "hit_3pct")),
        "all_hit_5pct": _mean_path(results, ("all", "hit_5pct")),
        "all_hit_10pct": _mean_path(results, ("all", "hit_10pct")),
        "top_1_max_raw_upside": _mean_path(results, ("top_1", "max_raw_upside")),
        "top_5_max_raw_upside": _mean_path(results, ("top_5", "max_raw_upside")),
        "top_20pct_max_raw_upside": _mean_path(results, ("top_20pct", "max_raw_upside")),
    }
    return out


def evaluate_by_timestamp(rows: Sequence[Any], scores: np.ndarray) -> tuple[list[dict], dict]:
    by_ts: dict[datetime, list[tuple[Any, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_ts[as_utc(row.ts)].append((row, float(score)))
    ts_results = []
    for _ts, pairs in sorted(by_ts.items(), key=lambda kv: kv[0]):
        bucket_rows = [p[0] for p in pairs]
        bucket_scores = np.array([p[1] for p in pairs], dtype=float)
        ts_results.append(evaluate_timestamp_scores(bucket_rows, bucket_scores))
    return ts_results, aggregate_ts_results(ts_results)


def vol_bucket_label(atr_pct: float, q33: float, q66: float) -> str:
    if not np.isfinite(atr_pct):
        return "unknown"
    if atr_pct <= q33:
        return "low"
    if atr_pct <= q66:
        return "medium"
    return "high"


def matched_volatility_analysis(
    rows: Sequence[Any],
    scores: np.ndarray,
    *,
    atr_tol_frac: float = 0.15,
) -> dict[str, Any]:
    """For each timestamp, compare model top-5 picks vs nearest-ATR non-top peers.

    Matching uses ONLY current atr_pct (no future). If no peer within
    |Δatr|/atr <= atr_tol_frac, that pick is skipped.
    """
    by_ts: dict[datetime, list[tuple[Any, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_ts[as_utc(row.ts)].append((row, float(score)))

    diffs: list[float] = []
    selected_ups: list[float] = []
    matched_ups: list[float] = []
    skipped = 0

    for _ts, pairs in by_ts.items():
        if len(pairs) < 10:
            continue
        pairs_sorted = sorted(pairs, key=lambda p: (-p[1] if np.isfinite(p[1]) else float("inf")))
        top5 = pairs_sorted[:5]
        top_ids = {id(p[0]) for p in top5}
        pool = [p[0] for p in pairs if id(p[0]) not in top_ids]
        for row, _score in top5:
            atr = atr_fraction(row)
            if not np.isfinite(atr) or atr <= 0:
                skipped += 1
                continue
            best = None
            best_rel = None
            for peer in pool:
                peer_atr = atr_fraction(peer)
                if not np.isfinite(peer_atr) or peer_atr <= 0:
                    continue
                rel = abs(peer_atr - atr) / atr
                if rel <= atr_tol_frac and (best_rel is None or rel < best_rel):
                    best = peer
                    best_rel = rel
            if best is None:
                skipped += 1
                continue
            su = float(row.future_max_upside_12h)
            mu = float(best.future_max_upside_12h)
            selected_ups.append(su)
            matched_ups.append(mu)
            diffs.append(su - mu)

    if not diffs:
        return {
            "n_matched": 0,
            "n_skipped": skipped,
            "mean_selected_upside": None,
            "mean_matched_upside": None,
            "mean_difference": None,
            "median_difference": None,
            "frac_selected_higher": None,
            "note": "Insufficient matched pairs within ATR tolerance.",
            "atr_tol_frac": atr_tol_frac,
        }
    diffs_a = np.asarray(diffs, dtype=float)
    return {
        "n_matched": int(len(diffs)),
        "n_skipped": skipped,
        "mean_selected_upside": float(np.mean(selected_ups)),
        "mean_matched_upside": float(np.mean(matched_ups)),
        "mean_difference": float(np.mean(diffs_a)),
        "median_difference": float(np.median(diffs_a)),
        "frac_selected_higher": float(np.mean(diffs_a > 0)),
        "atr_tol_frac": atr_tol_frac,
        "note": (
            f"For each model top-5 asset, matched nearest non-top peer with "
            f"|Δatr_frac|/atr_frac <= {atr_tol_frac}."
        ),
    }


def classify_vol_adjusted(
    full: dict[str, Any],
    atr: dict[str, Any],
    no_atr: dict[str, Any],
    fold_spearmans: list[float | None],
    bucket_summary: dict[str, Any],
    matched: dict[str, Any],
    *,
    leakage_ok: bool,
) -> tuple[str, str, dict[str, Any]]:
    sp = full.get("spearman")
    atr_sp = atr.get("spearman")
    no_atr_sp = no_atr.get("spearman")
    top5_adj = full.get("top_5_mean_vol_adj")
    atr_top5_adj = atr.get("top_5_mean_vol_adj")
    finite = [s for s in fold_spearmans if s is not None and np.isfinite(s)]
    positive = sp is not None and sp > 0
    consistent = bool(finite) and len(finite) >= 2 and all(s > 0 for s in finite)
    beats_atr = (
        sp is not None
        and atr_sp is not None
        and top5_adj is not None
        and atr_top5_adj is not None
        and sp > atr_sp
        and top5_adj > atr_top5_adj
    )
    # Material: Spearman margin >= 0.02 AND top5 residual margin > 0
    material = bool(
        beats_atr
        and atr_sp is not None
        and sp is not None
        and (sp - atr_sp) >= 0.02
    )
    no_atr_meaningful = no_atr_sp is not None and no_atr_sp > 0.05
    buckets_ok = False
    if bucket_summary:
        lows = []
        for name in ("low", "medium", "high"):
            b = bucket_summary.get(name) or {}
            bsp = b.get("full_spearman")
            if bsp is not None:
                lows.append(bsp > 0)
        buckets_ok = bool(lows) and all(lows)
    matched_ok = (
        matched.get("n_matched", 0) >= 20
        and matched.get("mean_difference") is not None
        and matched["mean_difference"] > 0
    )

    evidence = {
        "positive_aggregate_spearman": positive,
        "consistent_positive_across_folds": consistent,
        "full_beats_atr_on_spearman_and_top5_vol_adj": beats_atr,
        "material_margin_vs_atr": material,
        "full_without_atr_retains_ranking": no_atr_meaningful,
        "positive_across_vol_buckets": buckets_ok,
        "matched_vol_selected_higher": matched_ok,
        "leakage_audit_passed": leakage_ok,
        "fold_spearmans": fold_spearmans,
        "deltas": {
            "spearman_full_minus_atr": None if sp is None or atr_sp is None else sp - atr_sp,
            "spearman_full_without_atr_minus_atr": (
                None if no_atr_sp is None or atr_sp is None else no_atr_sp - atr_sp
            ),
            "top5_vol_adj_full_minus_atr": (
                None if top5_adj is None or atr_top5_adj is None else top5_adj - atr_top5_adj
            ),
            "top5_raw_full_minus_atr": (
                None
                if full.get("top_5_mean_raw_upside") is None or atr.get("top_5_mean_raw_upside") is None
                else full["top_5_mean_raw_upside"] - atr["top_5_mean_raw_upside"]
            ),
        },
    }

    if (
        leakage_ok
        and consistent
        and material
        and no_atr_meaningful
        and buckets_ok
        and matched_ok
    ):
        return (
            STATUS_PROMISING,
            "promising_volatility_adjusted_signal: full model beats ATR on vol-adjusted "
            "ranking, retains signal without atr_pct, and persists across buckets/matched pairs.",
            evidence,
        )
    if leakage_ok and positive and consistent and (beats_atr or no_atr_meaningful or matched_ok):
        return (
            STATUS_WEAK,
            "weak_volatility_adjusted_signal: positive ranking on vol-adjusted target, "
            "but improvement over ATR-only is small and/or still volatility-linked.",
            evidence,
        )
    return (
        STATUS_REJECTED,
        "rejected: insufficient evidence that features contain upside information beyond volatility.",
        evidence,
    )


def fit_vol_adjusted_walk_forward(
    rows: Sequence[Any],
    experiment: ExperimentSpec,
    *,
    min_cross_section: int = MIN_CROSS_SECTION,
    leakage_ok: bool = True,
) -> dict[str, Any]:
    annotated, cs_stats = annotate_vol_adjusted_targets(rows, min_cross_section=min_cross_section)
    if not annotated:
        return {
            "promote": False,
            "experiment_status": STATUS_REJECTED,
            "notes": "No timestamps met minimum cross-section for vol-adjusted targets.",
            "metrics": {"cross_section": cs_stats},
            "experiment_id": experiment.experiment_id,
            "target_name": "vol_adj_residual",
            "sample_stride": experiment.primary_stride_bars,
            "feature_names": feature_names(),
            "blob": b"",
        }

    ordered = sorted(annotated, key=lambda r: (as_utc(r.ts), r.symbol))
    full_names = feature_names()
    atr_names = ["atr_pct"]
    mom_feature = experiment.momentum_feature
    no_atr_names = [n for n in full_names if n != "atr_pct"]

    unique_ts = sorted({as_utc(r.ts) for r in ordered})
    folds = walk_forward_folds(unique_ts, min_train=60, min_test=14)
    gap = embargo()

    x_full = _matrix(ordered, full_names)
    x_atr = _matrix(ordered, atr_names)
    x_mom = _matrix(ordered, [mom_feature])
    x_no_atr = _matrix(ordered, no_atr_names)
    y = np.array([float(r.vol_adj_residual) for r in ordered], dtype=float)
    # Diagnostic baseline uses future ATR multiple (not a predictive model).
    y_mult = np.array([float(r.future_upside_atr_multiple) for r in ordered], dtype=float)
    row_ts = [as_utc(r.ts) for r in ordered]
    atr_pcts = np.array([_feat(r, "atr_pct") for r in ordered], dtype=float)

    fold_reports: list[dict[str, Any]] = []
    fold_spearmans: list[float | None] = []
    # Collect test predictions across folds for bucket / matched analysis
    all_test_rows: list[Any] = []
    all_full_scores: list[float] = []
    all_atr_scores: list[float] = []
    all_mom_scores: list[float] = []
    all_no_atr_scores: list[float] = []
    train_atr_for_buckets: list[float] = []

    for train_ts_idx, test_ts_idx in folds:
        train_ts = {unique_ts[i] for i in train_ts_idx}
        test_ts = {unique_ts[i] for i in test_ts_idx}
        if not fold_is_causal(unique_ts, train_ts_idx, test_ts_idx, gap):
            continue
        train_idx = [i for i, t in enumerate(row_ts) if t in train_ts]
        test_idx = [i for i, t in enumerate(row_ts) if t in test_ts]
        if len(train_idx) < 50 or len(test_idx) < 20:
            continue

        train_atr_for_buckets.extend(atr_pcts[train_idx].tolist())

        def _fit_predict(x_all: np.ndarray) -> np.ndarray:
            model = _new_regressor()
            model.fit(x_all[train_idx], y[train_idx])
            return model.predict(x_all[test_idx])

        pred_full = _fit_predict(x_full)
        pred_atr = _fit_predict(x_atr)
        pred_mom = _fit_predict(x_mom)
        pred_no_atr = _fit_predict(x_no_atr)

        test_rows = [ordered[i] for i in test_idx]
        # Model D diagnostic: rank by realized ATR multiple (oracle on secondary target).
        diag_scores = y_mult[test_idx]

        full_ts, full_agg = evaluate_by_timestamp(test_rows, pred_full)
        atr_ts, atr_agg = evaluate_by_timestamp(test_rows, pred_atr)
        mom_ts, mom_agg = evaluate_by_timestamp(test_rows, pred_mom)
        no_atr_ts, no_atr_agg = evaluate_by_timestamp(test_rows, pred_no_atr)
        _, diag_agg = evaluate_by_timestamp(test_rows, diag_scores)

        fold_spearmans.append(full_agg.get("spearman"))
        all_test_rows.extend(test_rows)
        all_full_scores.extend(pred_full.tolist())
        all_atr_scores.extend(pred_atr.tolist())
        all_mom_scores.extend(pred_mom.tolist())
        all_no_atr_scores.extend(pred_no_atr.tolist())

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
                "full_without_atr": no_atr_agg,
                "atr_multiple_diagnostic": diag_agg,
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

    def _std_field(path: tuple[str, ...]) -> float | None:
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
        return {
            "spearman": _agg_field((prefix, "spearman")),
            "spearman_std": _std_field((prefix, "spearman")),
            "top_1_mean_raw_upside": _agg_field((prefix, "top_1_mean_raw_upside")),
            "top_3_mean_raw_upside": _agg_field((prefix, "top_3_mean_raw_upside")),
            "top_5_mean_raw_upside": _agg_field((prefix, "top_5_mean_raw_upside")),
            "top_20pct_mean_raw_upside": _agg_field((prefix, "top_20pct_mean_raw_upside")),
            "top_1_mean_vol_adj": _agg_field((prefix, "top_1_mean_vol_adj")),
            "top_5_mean_vol_adj": _agg_field((prefix, "top_5_mean_vol_adj")),
            "top_20pct_mean_vol_adj": _agg_field((prefix, "top_20pct_mean_vol_adj")),
            "top_1_median_raw_upside": _agg_field((prefix, "top_1_median_raw_upside")),
            "top_5_median_raw_upside": _agg_field((prefix, "top_5_median_raw_upside")),
            "top_5_median_vol_adj": _agg_field((prefix, "top_5_median_vol_adj")),
            "top_1_hit_3pct": _agg_field((prefix, "top_1_hit_3pct")),
            "top_5_hit_3pct": _agg_field((prefix, "top_5_hit_3pct")),
            "top_5_hit_5pct": _agg_field((prefix, "top_5_hit_5pct")),
            "top_5_hit_10pct": _agg_field((prefix, "top_5_hit_10pct")),
            "top_20pct_hit_3pct": _agg_field((prefix, "top_20pct_hit_3pct")),
            "top_20pct_hit_5pct": _agg_field((prefix, "top_20pct_hit_5pct")),
            "top_20pct_hit_10pct": _agg_field((prefix, "top_20pct_hit_10pct")),
            "all_mean_raw_upside": _agg_field((prefix, "all_mean_raw_upside")),
            "all_mean_vol_adj": _agg_field((prefix, "all_mean_vol_adj")),
            "all_hit_3pct": _agg_field((prefix, "all_hit_3pct")),
            "all_hit_5pct": _agg_field((prefix, "all_hit_5pct")),
            "all_hit_10pct": _agg_field((prefix, "all_hit_10pct")),
            "top_1_max_raw_upside": _agg_field((prefix, "top_1_max_raw_upside")),
            "top_5_max_raw_upside": _agg_field((prefix, "top_5_max_raw_upside")),
            "top_20pct_max_raw_upside": _agg_field((prefix, "top_20pct_max_raw_upside")),
        }

    full_summary = summarize("full")
    atr_summary = summarize("atr_only")
    mom_summary = summarize("momentum")
    no_atr_summary = summarize("full_without_atr")
    diag_summary = summarize("atr_multiple_diagnostic")
    all_assets = {
        "spearman": 0.0,
        "top_5_mean_raw_upside": full_summary.get("all_mean_raw_upside"),
        "top_5_mean_vol_adj": full_summary.get("all_mean_vol_adj"),
        "top_5_hit_3pct": full_summary.get("all_hit_3pct"),
        "top_5_hit_5pct": full_summary.get("all_hit_5pct"),
        "top_5_hit_10pct": full_summary.get("all_hit_10pct"),
        "note": "Random/all-assets expectation equals universe means on the same timestamps.",
    }

    # Volatility buckets from training-period ATR tertiles (no test leakage into cuts).
    bucket_summary: dict[str, Any] = {}
    if train_atr_for_buckets and all_test_rows:
        finite_train = [a for a in train_atr_for_buckets if np.isfinite(a)]
        if len(finite_train) >= 30:
            q33, q66 = np.quantile(finite_train, [1 / 3, 2 / 3])
            buckets: dict[str, list[tuple[Any, float, float, float]]] = {
                "low": [],
                "medium": [],
                "high": [],
            }
            for row, fs, as_, ms in zip(
                all_test_rows, all_full_scores, all_atr_scores, all_mom_scores
            ):
                lab = vol_bucket_label(_feat(row, "atr_pct"), float(q33), float(q66))
                if lab in buckets:
                    buckets[lab].append((row, fs, as_, ms))

            for lab, items in buckets.items():
                if len(items) < 30:
                    bucket_summary[lab] = {
                        "n": len(items),
                        "note": "Too few observations for reliable bucket metrics.",
                        "q33": float(q33),
                        "q66": float(q66),
                    }
                    continue
                brows = [i[0] for i in items]
                # Evaluate within each timestamp still for fairness
                _, fagg = evaluate_by_timestamp(brows, np.array([i[1] for i in items]))
                _, aagg = evaluate_by_timestamp(brows, np.array([i[2] for i in items]))
                _, magg = evaluate_by_timestamp(brows, np.array([i[3] for i in items]))
                bucket_summary[lab] = {
                    "n": len(items),
                    "q33": float(q33),
                    "q66": float(q66),
                    "full_spearman": fagg.get("spearman"),
                    "atr_spearman": aagg.get("spearman"),
                    "momentum_spearman": magg.get("spearman"),
                    "full_top5_raw_upside": fagg.get("top_5_mean_raw_upside"),
                    "atr_top5_raw_upside": aagg.get("top_5_mean_raw_upside"),
                    "full_top5_vol_adj": fagg.get("top_5_mean_vol_adj"),
                    "atr_top5_vol_adj": aagg.get("top_5_mean_vol_adj"),
                }

    matched = (
        matched_volatility_analysis(all_test_rows, np.asarray(all_full_scores, dtype=float))
        if all_test_rows
        else {"n_matched": 0, "note": "No test rows."}
    )

    comparison_table = {
        "spearman": {
            "full_model": full_summary.get("spearman"),
            "atr_only": atr_summary.get("spearman"),
            "momentum_ret_24": mom_summary.get("spearman"),
            "full_without_atr": no_atr_summary.get("spearman"),
            "atr_multiple_diagnostic": diag_summary.get("spearman"),
            "random_all_assets": 0.0,
        },
        "top_5_mean_raw_upside": {
            "full_model": full_summary.get("top_5_mean_raw_upside"),
            "atr_only": atr_summary.get("top_5_mean_raw_upside"),
            "momentum_ret_24": mom_summary.get("top_5_mean_raw_upside"),
            "full_without_atr": no_atr_summary.get("top_5_mean_raw_upside"),
            "random_all_assets": all_assets.get("top_5_mean_raw_upside"),
        },
        "top_5_mean_vol_adj": {
            "full_model": full_summary.get("top_5_mean_vol_adj"),
            "atr_only": atr_summary.get("top_5_mean_vol_adj"),
            "momentum_ret_24": mom_summary.get("top_5_mean_vol_adj"),
            "full_without_atr": no_atr_summary.get("top_5_mean_vol_adj"),
            "random_all_assets": all_assets.get("top_5_mean_vol_adj"),
        },
        "top_5_hit_3pct": {
            "full_model": full_summary.get("top_5_hit_3pct"),
            "atr_only": atr_summary.get("top_5_hit_3pct"),
            "momentum_ret_24": mom_summary.get("top_5_hit_3pct"),
            "full_without_atr": no_atr_summary.get("top_5_hit_3pct"),
            "random_all_assets": all_assets.get("top_5_hit_3pct"),
        },
        "top_5_hit_5pct": {
            "full_model": full_summary.get("top_5_hit_5pct"),
            "atr_only": atr_summary.get("top_5_hit_5pct"),
            "momentum_ret_24": mom_summary.get("top_5_hit_5pct"),
            "full_without_atr": no_atr_summary.get("top_5_hit_5pct"),
            "random_all_assets": all_assets.get("top_5_hit_5pct"),
        },
        "top_5_hit_10pct": {
            "full_model": full_summary.get("top_5_hit_10pct"),
            "atr_only": atr_summary.get("top_5_hit_10pct"),
            "momentum_ret_24": mom_summary.get("top_5_hit_10pct"),
            "full_without_atr": no_atr_summary.get("top_5_hit_10pct"),
            "random_all_assets": all_assets.get("top_5_hit_10pct"),
        },
    }

    status, note, evidence = classify_vol_adjusted(
        full_summary,
        atr_summary,
        no_atr_summary,
        fold_spearmans,
        bucket_summary,
        matched,
        leakage_ok=leakage_ok,
    )

    blob = b""
    if len(ordered) >= 50:
        final = _new_regressor()
        final.fit(x_full, y)
        buf = BytesIO()
        joblib.dump(
            {
                "model": final,
                "feature_names": full_names,
                "target_name": "vol_adj_residual",
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
        "full_without_atr": no_atr_summary,
        "atr_multiple_diagnostic": diag_summary,
        "all_assets_random": all_assets,
        "comparison_table": comparison_table,
        "volatility_buckets": bucket_summary,
        "matched_volatility": matched,
        "evidence": evidence,
        "lineage": experiment.lineage(),
        "training_target": "vol_adj_residual",
        "secondary_target": "future_upside_atr_multiple",
        "model_type": "HistGradientBoostingRegressor",
        "full_minus_atr_spearman": evidence["deltas"]["spearman_full_minus_atr"],
        "full_without_atr_vs_atr_spearman": evidence["deltas"]["spearman_full_without_atr_minus_atr"],
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
        "target_name": "vol_adj_residual",
        "sample_stride": experiment.primary_stride_bars,
        "move_threshold": experiment.move_pct,
        "comparison_table": comparison_table,
    }
