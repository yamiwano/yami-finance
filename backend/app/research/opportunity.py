"""Risk-adjusted opportunity experiment: upside quality vs downside.

Primary modeling target: log1p(future_max_upside / max(|future_max_drawdown|, eps)).
Future outcomes are targets/evaluation only — never features.
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

from app.domain import Bar
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

OPPORTUNITY_EPS = 0.001
STATUS_REJECTED = "rejected"
STATUS_WEAK = "weak_opportunity_signal"
STATUS_PROMISING = "promising_opportunity_signal"

BARRIER_SPECS = (
    ("up3_before_down2", 0.03, 0.02),
    ("up5_before_down3", 0.05, 0.03),
    ("up10_before_down5", 0.10, 0.05),
)


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
    atr = _feat(row, "atr_pct")
    if not np.isfinite(atr) or atr <= 0:
        return float("nan")
    return float(atr) / 100.0


def risk_adjusted_opportunity(
    max_upside: float,
    max_drawdown: float,
    *,
    eps: float = OPPORTUNITY_EPS,
) -> float:
    """raw ratio = upside / max(|drawdown|, eps)."""
    return float(max_upside) / max(abs(float(max_drawdown)), eps)


def log_risk_adjusted_opportunity(
    max_upside: float,
    max_drawdown: float,
    *,
    eps: float = OPPORTUNITY_EPS,
) -> float:
    """Training target: log1p(risk_adjusted_opportunity)."""
    return float(np.log1p(risk_adjusted_opportunity(max_upside, max_drawdown, eps=eps)))


def upside_downside_spread(max_upside: float, max_drawdown: float) -> float:
    return float(max_upside) - abs(float(max_drawdown))


def first_touch_barrier(
    future_bars: Sequence[Bar],
    entry_close: float,
    up_pct: float,
    down_pct: float,
    *,
    horizon_bars: int | None = None,
) -> str:
    """Return up_first | down_first | neither | ambiguous.

    Ambiguous when the same candle touches both barriers (OHLC cannot order events).
    Does not invent intrabar path.
    """
    if entry_close <= 0:
        return "neither"
    window = list(future_bars[: horizon_bars if horizon_bars is not None else len(future_bars)])
    up_level = entry_close * (1.0 + up_pct)
    down_level = entry_close * (1.0 - abs(down_pct))
    for bar in window:
        hit_up = bar.high >= up_level
        hit_down = bar.low <= down_level
        if hit_up and hit_down:
            return "ambiguous"
        if hit_up:
            return "up_first"
        if hit_down:
            return "down_first"
    return "neither"


def barrier_flags_from_bars(
    future_bars: Sequence[Bar],
    entry_close: float,
    *,
    horizon_bars: int,
) -> dict[str, str]:
    out = {}
    for name, up_pct, down_pct in BARRIER_SPECS:
        out[name] = first_touch_barrier(
            future_bars, entry_close, up_pct, down_pct, horizon_bars=horizon_bars
        )
    return out


def annotate_opportunity_targets(
    rows: Sequence[Any],
    *,
    min_cross_section: int = MIN_CROSS_SECTION,
    eps: float = OPPORTUNITY_EPS,
) -> tuple[list[Any], dict[str, Any]]:
    """Attach opportunity targets; require complete upside/drawdown and CS size."""
    groups = group_by_timestamp(rows)
    considered = 0
    excluded = 0
    sizes: list[int] = []
    kept: list[Any] = []

    for _ts, bucket in groups.items():
        considered += 1
        eligible = []
        for row in bucket:
            up = future_max_upside(row)
            dd = future_max_drawdown(row)
            if up is None or dd is None:
                continue
            if not np.isfinite(up) or not np.isfinite(dd):
                continue
            eligible.append(row)
        if len(eligible) < min_cross_section:
            excluded += 1
            continue

        opportunities = []
        for row in eligible:
            up = float(future_max_upside(row))
            dd = float(future_max_drawdown(row))
            opp = risk_adjusted_opportunity(up, dd, eps=eps)
            log_opp = log_risk_adjusted_opportunity(up, dd, eps=eps)
            spread = upside_downside_spread(up, dd)
            atr_f = atr_fraction(row)
            row.future_max_upside_12h = up
            row.future_return_12h = future_return(row)
            row.future_max_drawdown_12h = dd
            row.risk_adjusted_opportunity = opp
            row.log_risk_adjusted_opportunity = log_opp
            row.upside_downside_spread = spread
            row.atr_frac = atr_f if np.isfinite(atr_f) else float("nan")
            # ATR-normalized diagnostics (evaluation only)
            if np.isfinite(atr_f) and atr_f > 0:
                row.norm_upside = up / atr_f
                row.norm_drawdown = abs(dd) / atr_f
                row.norm_opportunity = up / max(abs(dd), eps)  # same as raw opp; also /atr optional
                row.norm_upside_over_atr = up / atr_f
                row.norm_drawdown_over_atr = abs(dd) / atr_f
                row.norm_opp_over_atr = (up / max(abs(dd), eps)) / atr_f
            else:
                row.norm_upside_over_atr = float("nan")
                row.norm_drawdown_over_atr = float("nan")
                row.norm_opp_over_atr = float("nan")
            # Barrier defaults if not attached yet
            for name, _, _ in BARRIER_SPECS:
                if not hasattr(row, name):
                    setattr(row, name, "unknown")
            opportunities.append(opp)

        percentiles = rank_percentiles(opportunities)
        sizes.append(len(eligible))
        for row, pct in zip(eligible, percentiles):
            row.opportunity_rank_percentile = float(pct)
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
        "eps": eps,
        "target_definition": {
            "risk_adjusted_opportunity": "future_max_upside_12h / max(|future_max_drawdown_12h|, eps)",
            "log_risk_adjusted_opportunity": "log1p(risk_adjusted_opportunity)",
            "upside_downside_spread": "future_max_upside_12h - |future_max_drawdown_12h|",
            "eps": eps,
            "barriers": [
                {"name": n, "up": u, "down": d} for n, u, d in BARRIER_SPECS
            ],
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


def _barrier_rate(items: list[dict[str, Any]], key: str) -> float | None:
    vals = [i.get(key) for i in items]
    usable = [v for v in vals if v in ("up_first", "down_first", "neither")]
    if not usable:
        return None
    return float(np.mean([1.0 if v == "up_first" else 0.0 for v in usable]))


def _group_metrics(items: list[dict[str, Any]]) -> dict[str, float | None]:
    if not items:
        return {"n": 0}
    opp = np.array([i["opportunity"] for i in items], dtype=float)
    log_opp = np.array([i["log_opportunity"] for i in items], dtype=float)
    spread = np.array([i["spread"] for i in items], dtype=float)
    raw = np.array([i["raw_upside"] for i in items], dtype=float)
    ret = np.array([i["future_return"] for i in items], dtype=float)
    dd = np.array([i["drawdown"] for i in items], dtype=float)
    nu = np.array([i["norm_upside"] for i in items], dtype=float)
    nd = np.array([i["norm_drawdown"] for i in items], dtype=float)
    no = np.array([i["norm_opp"] for i in items], dtype=float)
    return {
        "n": int(len(items)),
        "mean_opportunity": float(np.nanmean(opp)),
        "median_opportunity": float(np.nanmedian(opp)),
        "mean_log_opportunity": float(np.nanmean(log_opp)),
        "mean_spread": float(np.nanmean(spread)),
        "mean_raw_upside": float(np.nanmean(raw)),
        "median_raw_upside": float(np.nanmedian(raw)),
        "max_raw_upside": float(np.nanmax(raw)),
        "mean_future_return": float(np.nanmean(ret)),
        "mean_drawdown": float(np.nanmean(dd)),
        "median_drawdown": float(np.nanmedian(dd)),
        "worst_drawdown": float(np.nanmin(dd)),
        "frac_dd_worse_3": float(np.mean(dd <= -0.03)),
        "frac_dd_worse_5": float(np.mean(dd <= -0.05)),
        "frac_dd_worse_10": float(np.mean(dd <= -0.10)),
        "hit_3pct": float(np.mean(raw >= 0.03)),
        "hit_5pct": float(np.mean(raw >= 0.05)),
        "hit_10pct": float(np.mean(raw >= 0.10)),
        "up3_before_down2": _barrier_rate(items, "up3_before_down2"),
        "up5_before_down3": _barrier_rate(items, "up5_before_down3"),
        "up10_before_down5": _barrier_rate(items, "up10_before_down5"),
        "mean_norm_upside": float(np.nanmean(nu)) if np.any(np.isfinite(nu)) else None,
        "mean_norm_drawdown": float(np.nanmean(nd)) if np.any(np.isfinite(nd)) else None,
        "mean_norm_opp": float(np.nanmean(no)) if np.any(np.isfinite(no)) else None,
        "mean_atr_pct": float(np.nanmean([i["atr_pct"] for i in items])),
    }


def evaluate_timestamp_scores(rows: Sequence[Any], scores: np.ndarray) -> dict[str, Any]:
    n = len(rows)
    scored = []
    for row, score in zip(rows, scores):
        scored.append(
            {
                "symbol": row.symbol,
                "score": float(score) if np.isfinite(score) else float("-inf"),
                "opportunity": float(row.risk_adjusted_opportunity),
                "log_opportunity": float(row.log_risk_adjusted_opportunity),
                "spread": float(row.upside_downside_spread),
                "raw_upside": float(row.future_max_upside_12h),
                "future_return": float(row.future_return_12h if row.future_return_12h is not None else np.nan),
                "drawdown": float(row.future_max_drawdown_12h),
                "norm_upside": float(getattr(row, "norm_upside_over_atr", np.nan)),
                "norm_drawdown": float(getattr(row, "norm_drawdown_over_atr", np.nan)),
                "norm_opp": float(getattr(row, "norm_opp_over_atr", np.nan)),
                "atr_pct": _feat(row, "atr_pct"),
                "up3_before_down2": getattr(row, "up3_before_down2", "unknown"),
                "up5_before_down3": getattr(row, "up5_before_down3", "unknown"),
                "up10_before_down5": getattr(row, "up10_before_down5", "unknown"),
            }
        )
    scores_arr = np.array([s["score"] for s in scored], dtype=float)
    actual_opp = np.array([s["opportunity"] for s in scored], dtype=float)
    actual_up = np.array([s["raw_upside"] for s in scored], dtype=float)
    actual_ret = np.array([s["future_return"] for s in scored], dtype=float)
    actual_dd = np.array([s["drawdown"] for s in scored], dtype=float)

    def _sp(a: np.ndarray, b: np.ndarray) -> float | None:
        if len(a) < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
            return None
        rho, _ = spearmanr(a, b, nan_policy="omit")
        return float(rho) if np.isfinite(rho) else None

    order = _top_k_indices(scores_arr, n)
    ranked = [scored[i] for i in order]
    k20 = max(1, int(np.ceil(0.20 * n)))

    def take(k: int) -> list[dict[str, Any]]:
        return ranked[: min(k, n)]

    return {
        "n": n,
        "spearman_opportunity": _sp(scores_arr, actual_opp),
        "spearman_upside": _sp(scores_arr, actual_up),
        "spearman_return": _sp(scores_arr, actual_ret),
        "spearman_drawdown": _sp(scores_arr, actual_dd),
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
        return {"timestamps": 0}
    keys = [
        ("spearman_opportunity", ("spearman_opportunity",)),
        ("spearman_upside", ("spearman_upside",)),
        ("spearman_return", ("spearman_return",)),
        ("spearman_drawdown", ("spearman_drawdown",)),
    ]
    out: dict[str, Any] = {"timestamps": len(results)}
    for name, path in keys:
        out[name] = _mean_path(results, path)

    for group in ("all", "top_1", "top_3", "top_5", "top_20pct"):
        sample = results[0].get(group) or {}
        for metric in sample:
            if metric == "n":
                continue
            out[f"{group}_{metric}"] = _mean_path(results, (group, metric))
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


def classify_opportunity(
    full: dict[str, Any],
    atr: dict[str, Any],
    no_atr: dict[str, Any],
    fold_spearmans: list[float | None],
    bucket_summary: dict[str, Any],
    *,
    leakage_ok: bool,
) -> tuple[str, str, dict[str, Any]]:
    sp = full.get("spearman_opportunity")
    atr_sp = atr.get("spearman_opportunity")
    no_atr_sp = no_atr.get("spearman_opportunity")
    top5_opp = full.get("top_5_mean_opportunity")
    atr_top5_opp = atr.get("top_5_mean_opportunity")
    top5_dd = full.get("top_5_mean_drawdown")
    atr_top5_dd = atr.get("top_5_mean_drawdown")
    top5_barrier = full.get("top_5_up3_before_down2")
    atr_barrier = atr.get("top_5_up3_before_down2")

    finite = [s for s in fold_spearmans if s is not None and np.isfinite(s)]
    positive = sp is not None and sp > 0
    consistent = bool(finite) and len(finite) >= 2 and all(s > 0 for s in finite)
    beats_atr_sp = sp is not None and atr_sp is not None and sp > atr_sp
    beats_atr_top5 = (
        top5_opp is not None and atr_top5_opp is not None and top5_opp > atr_top5_opp
    )
    material = bool(beats_atr_sp and atr_sp is not None and sp is not None and (sp - atr_sp) >= 0.02)
    no_atr_ok = no_atr_sp is not None and no_atr_sp > 0.05
    # Downside not materially worse: full mean drawdown >= atr mean drawdown - 0.005
    # (drawdowns are negative; "worse" means more negative)
    downside_ok = True
    if top5_dd is not None and atr_top5_dd is not None:
        downside_ok = top5_dd >= (atr_top5_dd - 0.005)
    barrier_ok = (
        top5_barrier is not None
        and atr_barrier is not None
        and top5_barrier > atr_barrier
    )
    buckets_ok = False
    if bucket_summary:
        flags = []
        for name in ("low", "medium", "high"):
            b = bucket_summary.get(name) or {}
            bsp = b.get("full_spearman")
            if bsp is not None:
                flags.append(bsp > 0)
        buckets_ok = bool(flags) and all(flags)

    evidence = {
        "positive_aggregate_spearman": positive,
        "consistent_positive_across_folds": consistent,
        "full_beats_atr_spearman": beats_atr_sp,
        "full_beats_atr_top5_opportunity": beats_atr_top5,
        "material_spearman_margin_vs_atr": material,
        "full_without_atr_retains_signal": no_atr_ok,
        "downside_not_materially_worse_vs_atr": downside_ok,
        "barrier_improves_vs_atr": barrier_ok,
        "positive_across_vol_buckets": buckets_ok,
        "leakage_audit_passed": leakage_ok,
        "fold_spearmans": fold_spearmans,
        "deltas": {
            "spearman_full_minus_atr": None if sp is None or atr_sp is None else sp - atr_sp,
            "top5_opp_full_minus_atr": (
                None if top5_opp is None or atr_top5_opp is None else top5_opp - atr_top5_opp
            ),
            "top5_raw_upside_full_minus_atr": (
                None
                if full.get("top_5_mean_raw_upside") is None or atr.get("top_5_mean_raw_upside") is None
                else full["top_5_mean_raw_upside"] - atr["top_5_mean_raw_upside"]
            ),
            "top5_return_full_minus_atr": (
                None
                if full.get("top_5_mean_future_return") is None or atr.get("top_5_mean_future_return") is None
                else full["top_5_mean_future_return"] - atr["top_5_mean_future_return"]
            ),
            "top5_drawdown_full_minus_atr": (
                None if top5_dd is None or atr_top5_dd is None else top5_dd - atr_top5_dd
            ),
        },
    }

    if (
        leakage_ok
        and consistent
        and material
        and beats_atr_top5
        and no_atr_ok
        and downside_ok
        and barrier_ok
        and buckets_ok
    ):
        return (
            STATUS_PROMISING,
            "promising_opportunity_signal: full model consistently beats ATR on opportunity "
            "quality with acceptable downside and barrier improvement; research-only.",
            evidence,
        )
    if leakage_ok and positive and consistent and (beats_atr_sp or beats_atr_top5 or no_atr_ok):
        return (
            STATUS_WEAK,
            "weak_opportunity_signal: some useful opportunity ranking, but improvement over "
            "ATR/simple baselines is small and/or regime-dependent.",
            evidence,
        )
    return (
        STATUS_REJECTED,
        "rejected: insufficient evidence that the model identifies better risk-adjusted "
        "opportunities beyond simple volatility baselines.",
        evidence,
    )


def fit_opportunity_walk_forward(
    rows: Sequence[Any],
    experiment: ExperimentSpec,
    *,
    min_cross_section: int = MIN_CROSS_SECTION,
    leakage_ok: bool = True,
) -> dict[str, Any]:
    annotated, cs_stats = annotate_opportunity_targets(rows, min_cross_section=min_cross_section)
    if not annotated:
        return {
            "promote": False,
            "experiment_status": STATUS_REJECTED,
            "notes": "No timestamps met minimum cross-section for opportunity targets.",
            "metrics": {"cross_section": cs_stats},
            "experiment_id": experiment.experiment_id,
            "target_name": "log_risk_adjusted_opportunity",
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
    y = np.array([float(r.log_risk_adjusted_opportunity) for r in ordered], dtype=float)
    atr_rank_scores = np.array([_feat(r, "atr_pct") for r in ordered], dtype=float)
    row_ts = [as_utc(r.ts) for r in ordered]

    fold_reports: list[dict[str, Any]] = []
    fold_spearmans: list[float | None] = []
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

        train_atr_for_buckets.extend(atr_rank_scores[train_idx].tolist())

        def _fit_predict(x_all: np.ndarray) -> np.ndarray:
            model = _new_regressor()
            model.fit(x_all[train_idx], y[train_idx])
            return model.predict(x_all[test_idx])

        pred_full = _fit_predict(x_full)
        pred_atr = _fit_predict(x_atr)
        pred_mom = _fit_predict(x_mom)
        pred_no_atr = _fit_predict(x_no_atr)
        atr_direct = atr_rank_scores[test_idx]

        test_rows = [ordered[i] for i in test_idx]
        full_ts, full_agg = evaluate_by_timestamp(test_rows, pred_full)
        atr_ts, atr_agg = evaluate_by_timestamp(test_rows, pred_atr)
        mom_ts, mom_agg = evaluate_by_timestamp(test_rows, pred_mom)
        no_atr_ts, no_atr_agg = evaluate_by_timestamp(test_rows, pred_no_atr)
        _, atr_rank_agg = evaluate_by_timestamp(test_rows, atr_direct)

        fold_spearmans.append(full_agg.get("spearman_opportunity"))
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
                "atr_ranking": atr_rank_agg,
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
        metrics = [
            "spearman_opportunity",
            "spearman_upside",
            "spearman_return",
            "spearman_drawdown",
            "top_1_mean_opportunity",
            "top_5_mean_opportunity",
            "top_20pct_mean_opportunity",
            "top_5_median_opportunity",
            "top_5_mean_log_opportunity",
            "top_5_mean_spread",
            "top_5_mean_raw_upside",
            "top_5_median_raw_upside",
            "top_5_mean_future_return",
            "top_5_mean_drawdown",
            "top_5_median_drawdown",
            "top_5_worst_drawdown",
            "top_5_frac_dd_worse_3",
            "top_5_frac_dd_worse_5",
            "top_5_frac_dd_worse_10",
            "top_5_hit_3pct",
            "top_5_hit_5pct",
            "top_5_hit_10pct",
            "top_5_up3_before_down2",
            "top_5_up5_before_down3",
            "top_5_up10_before_down5",
            "top_1_up3_before_down2",
            "top_1_up5_before_down3",
            "top_1_up10_before_down5",
            "top_20pct_up3_before_down2",
            "top_20pct_up5_before_down3",
            "top_20pct_up10_before_down5",
            "top_5_mean_norm_upside",
            "top_5_mean_norm_drawdown",
            "top_5_mean_norm_opp",
            "all_mean_opportunity",
            "all_mean_raw_upside",
            "all_mean_future_return",
            "all_mean_drawdown",
            "all_up3_before_down2",
            "all_up5_before_down3",
            "all_up10_before_down5",
            "all_hit_3pct",
            "all_hit_5pct",
            "all_hit_10pct",
        ]
        out = {
            "spearman_opportunity_std": _std_field((prefix, "spearman_opportunity")),
        }
        for m in metrics:
            out[m] = _agg_field((prefix, m))
        return out

    full_summary = summarize("full")
    atr_summary = summarize("atr_only")
    mom_summary = summarize("momentum")
    no_atr_summary = summarize("full_without_atr")
    atr_rank_summary = summarize("atr_ranking")
    all_assets = {
        "spearman_opportunity": 0.0,
        "top_5_mean_opportunity": full_summary.get("all_mean_opportunity"),
        "top_5_mean_raw_upside": full_summary.get("all_mean_raw_upside"),
        "top_5_mean_future_return": full_summary.get("all_mean_future_return"),
        "top_5_mean_drawdown": full_summary.get("all_mean_drawdown"),
        "top_5_up3_before_down2": full_summary.get("all_up3_before_down2"),
        "top_5_up5_before_down3": full_summary.get("all_up5_before_down3"),
        "top_5_up10_before_down5": full_summary.get("all_up10_before_down5"),
        "note": "All-assets / random expectation uses universe means on the same timestamps.",
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
                lab = vol_bucket_label(_feat(row, "atr_pct"), float(q33), float(q66))
                if lab in buckets:
                    buckets[lab].append((row, fs, as_, ms, ns))
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
                _, fagg = evaluate_by_timestamp(brows, np.array([i[1] for i in items]))
                _, aagg = evaluate_by_timestamp(brows, np.array([i[2] for i in items]))
                _, magg = evaluate_by_timestamp(brows, np.array([i[3] for i in items]))
                _, nagg = evaluate_by_timestamp(brows, np.array([i[4] for i in items]))
                bucket_summary[lab] = {
                    "n": len(items),
                    "q33": float(q33),
                    "q66": float(q66),
                    "full_spearman": fagg.get("spearman_opportunity"),
                    "atr_spearman": aagg.get("spearman_opportunity"),
                    "momentum_spearman": magg.get("spearman_opportunity"),
                    "full_without_atr_spearman": nagg.get("spearman_opportunity"),
                    "full_top5_raw_upside": fagg.get("top_5_mean_raw_upside"),
                    "full_top5_return": fagg.get("top_5_mean_future_return"),
                    "full_top5_drawdown": fagg.get("top_5_mean_drawdown"),
                    "full_top5_opportunity": fagg.get("top_5_mean_opportunity"),
                    "full_top5_up3_before_down2": fagg.get("top_5_up3_before_down2"),
                    "full_top5_up5_before_down3": fagg.get("top_5_up5_before_down3"),
                    "full_top5_up10_before_down5": fagg.get("top_5_up10_before_down5"),
                }

    comparison_table = {
        "spearman_opportunity": {
            "full_model": full_summary.get("spearman_opportunity"),
            "atr_only": atr_summary.get("spearman_opportunity"),
            "momentum_ret_24": mom_summary.get("spearman_opportunity"),
            "full_without_atr": no_atr_summary.get("spearman_opportunity"),
            "atr_ranking": atr_rank_summary.get("spearman_opportunity"),
            "random_all_assets": 0.0,
        },
        "top_5_mean_opportunity": {
            "full_model": full_summary.get("top_5_mean_opportunity"),
            "atr_only": atr_summary.get("top_5_mean_opportunity"),
            "momentum_ret_24": mom_summary.get("top_5_mean_opportunity"),
            "full_without_atr": no_atr_summary.get("top_5_mean_opportunity"),
            "atr_ranking": atr_rank_summary.get("top_5_mean_opportunity"),
            "random_all_assets": all_assets.get("top_5_mean_opportunity"),
        },
        "top_5_mean_raw_upside": {
            "full_model": full_summary.get("top_5_mean_raw_upside"),
            "atr_only": atr_summary.get("top_5_mean_raw_upside"),
            "momentum_ret_24": mom_summary.get("top_5_mean_raw_upside"),
            "full_without_atr": no_atr_summary.get("top_5_mean_raw_upside"),
            "random_all_assets": all_assets.get("top_5_mean_raw_upside"),
        },
        "top_5_mean_future_return": {
            "full_model": full_summary.get("top_5_mean_future_return"),
            "atr_only": atr_summary.get("top_5_mean_future_return"),
            "momentum_ret_24": mom_summary.get("top_5_mean_future_return"),
            "full_without_atr": no_atr_summary.get("top_5_mean_future_return"),
            "random_all_assets": all_assets.get("top_5_mean_future_return"),
        },
        "top_5_mean_drawdown": {
            "full_model": full_summary.get("top_5_mean_drawdown"),
            "atr_only": atr_summary.get("top_5_mean_drawdown"),
            "momentum_ret_24": mom_summary.get("top_5_mean_drawdown"),
            "full_without_atr": no_atr_summary.get("top_5_mean_drawdown"),
            "random_all_assets": all_assets.get("top_5_mean_drawdown"),
        },
        "top_5_up3_before_down2": {
            "full_model": full_summary.get("top_5_up3_before_down2"),
            "atr_only": atr_summary.get("top_5_up3_before_down2"),
            "momentum_ret_24": mom_summary.get("top_5_up3_before_down2"),
            "full_without_atr": no_atr_summary.get("top_5_up3_before_down2"),
            "random_all_assets": all_assets.get("top_5_up3_before_down2"),
        },
        "top_5_up5_before_down3": {
            "full_model": full_summary.get("top_5_up5_before_down3"),
            "atr_only": atr_summary.get("top_5_up5_before_down3"),
            "momentum_ret_24": mom_summary.get("top_5_up5_before_down3"),
            "full_without_atr": no_atr_summary.get("top_5_up5_before_down3"),
            "random_all_assets": all_assets.get("top_5_up5_before_down3"),
        },
        "top_5_up10_before_down5": {
            "full_model": full_summary.get("top_5_up10_before_down5"),
            "atr_only": atr_summary.get("top_5_up10_before_down5"),
            "momentum_ret_24": mom_summary.get("top_5_up10_before_down5"),
            "full_without_atr": no_atr_summary.get("top_5_up10_before_down5"),
            "random_all_assets": all_assets.get("top_5_up10_before_down5"),
        },
    }

    downside_control = {
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
        "all_assets": {
            "mean_drawdown": full_summary.get("all_mean_drawdown"),
        },
    }

    atr_norm = {
        "full_top5_norm_upside": full_summary.get("top_5_mean_norm_upside"),
        "atr_top5_norm_upside": atr_summary.get("top_5_mean_norm_upside"),
        "full_top5_norm_drawdown": full_summary.get("top_5_mean_norm_drawdown"),
        "atr_top5_norm_drawdown": atr_summary.get("top_5_mean_norm_drawdown"),
        "full_top5_norm_opp": full_summary.get("top_5_mean_norm_opp"),
        "atr_top5_norm_opp": atr_summary.get("top_5_mean_norm_opp"),
    }

    status, note, evidence = classify_opportunity(
        full_summary,
        atr_summary,
        no_atr_summary,
        fold_spearmans,
        bucket_summary,
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
                "target_name": "log_risk_adjusted_opportunity",
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
        "atr_ranking": atr_rank_summary,
        "all_assets_random": all_assets,
        "comparison_table": comparison_table,
        "volatility_buckets": bucket_summary,
        "downside_control": downside_control,
        "atr_normalized_opportunity": atr_norm,
        "evidence": evidence,
        "lineage": experiment.lineage(),
        "training_target": "log_risk_adjusted_opportunity",
        "evaluation_target": "risk_adjusted_opportunity",
        "model_type": "HistGradientBoostingRegressor",
        "full_minus_atr_spearman": evidence["deltas"]["spearman_full_minus_atr"],
        "full_minus_atr_top5_opportunity": evidence["deltas"]["top5_opp_full_minus_atr"],
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
        "target_name": "log_risk_adjusted_opportunity",
        "sample_stride": experiment.primary_stride_bars,
        "move_threshold": experiment.move_pct,
        "comparison_table": comparison_table,
    }
