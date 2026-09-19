"""Regime filter out-of-sample experiment.

Frozen base model + frozen regime thresholds from regime_analysis_v1.
Filters sit outside the model; OOS period is strictly after discovery data.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from app.research.barriers import COST_PER_EVENT, barrier_binary, binary_classification_metrics
from app.research.experiments import ExperimentSpec
from app.research.features import feature_names
from app.research.history import as_utc
from app.research.opportunity import BARRIER_SPECS
from app.research.portfolio import (
    RANDOM_SEED,
    RANDOM_SIMS,
    _matrix,
    _new_classifier,
    _positive_proba,
    run_portfolio_backtest,
)
from app.research.ranking import MIN_CROSS_SECTION, group_by_timestamp
from app.research.regime import momentum_dispersion_regime, volume_regime
from app.research.robustness import evaluate_period_predictions, fit_models_on_rows, predict_with_models
from app.research.spec import embargo

STATUS_GENERALIZES = "filter_generalizes"
STATUS_MIXED = "filter_mixed"
STATUS_FAILS = "filter_fails"
STATUS_INSUFFICIENT = "insufficient_oos_data"

FILTERS = ("baseline", "high_dispersion", "normal_volume", "high_dispersion_plus_normal_volume")


def _feat(row: Any, name: str) -> float:
    feats = getattr(row, "features", None) or {}
    value = feats.get(name)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def filter_passes(
    rows_at_ts: Sequence[Any],
    filter_name: str,
    thresholds: dict[str, Any],
) -> bool:
    """Evaluate a regime filter for one timestamp using only data <= T."""
    if filter_name == "baseline":
        return True
    disp = momentum_dispersion_regime(rows_at_ts, *thresholds["momentum_dispersion"])
    vol = volume_regime(rows_at_ts, *thresholds["volume"])
    if filter_name == "high_dispersion":
        return disp == "high"
    if filter_name == "normal_volume":
        return vol == "normal"
    if filter_name == "high_dispersion_plus_normal_volume":
        return disp == "high" and vol == "normal"
    return False


def apply_filter_to_rows(
    rows: Sequence[Any],
    filter_name: str,
    thresholds: dict[str, Any],
) -> list[Any]:
    """Keep only rows whose timestamp passes the filter."""
    by_ts = group_by_timestamp(rows)
    keep_ts = set()
    for ts, bucket in by_ts.items():
        if filter_passes(bucket, filter_name, thresholds):
            keep_ts.add(ts)
    return [r for r in rows if as_utc(r.ts) in keep_ts]


def exposure_stats(
    rows: Sequence[Any],
    filtered_rows: Sequence[Any],
) -> dict[str, Any]:
    by_ts = group_by_timestamp(rows)
    filt_ts = {as_utc(r.ts) for r in filtered_rows}
    return {
        "timestamps_total": len(by_ts),
        "timestamps_allowed": len(filt_ts),
        "pct_timestamps_allowed": (len(filt_ts) / len(by_ts)) if by_ts else None,
        "candidate_signals": len(rows),
        "signals_after_filter": len(filtered_rows),
    }


def run_filtered_backtest(
    rows: Sequence[Any],
    bars_by_symbol: dict[str, list],
    scores: np.ndarray,
    *,
    filter_name: str,
    thresholds: dict[str, Any],
    barrier: str,
    up_pct: float,
    down_pct: float,
    top_k: int,
    horizon_bars: int,
    cost: float = COST_PER_EVENT,
    slippage: float = 0.0,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run portfolio backtest on filtered timestamps only."""
    filtered = apply_filter_to_rows(rows, filter_name, thresholds)
    if not filtered:
        return {
            "filter": filter_name,
            "n_trades": 0,
            "note": "No timestamps passed filter.",
            "exposure": exposure_stats(rows, filtered),
        }
    # Map scores to filtered rows by identity
    score_by_id = {id(r): float(s) for r, s in zip(rows, scores)}
    filt_scores = np.array([score_by_id[id(r)] for r in filtered], dtype=float)
    res = run_portfolio_backtest(
        filtered,
        bars_by_symbol,
        filt_scores,
        barrier=barrier,
        up_pct=up_pct,
        down_pct=down_pct,
        top_k=top_k,
        horizon_bars=horizon_bars,
        cost=cost,
        slippage=slippage,
        strategy=filter_name,
        seed=seed,
    )
    res["filter"] = filter_name
    res["exposure"] = exposure_stats(rows, filtered)
    return res


def random_filtered_benchmark(
    rows: Sequence[Any],
    bars_by_symbol: dict[str, list],
    *,
    filter_name: str,
    thresholds: dict[str, Any],
    barrier: str,
    up_pct: float,
    down_pct: float,
    top_k: int,
    horizon_bars: int,
    n_sims: int = RANDOM_SIMS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    filtered = apply_filter_to_rows(rows, filter_name, thresholds)
    if not filtered:
        return {"filter": filter_name, "note": "No timestamps passed filter."}
    nets = []
    for s in range(n_sims):
        res = run_portfolio_backtest(
            filtered,
            bars_by_symbol,
            np.zeros(len(filtered)),
            barrier=barrier,
            up_pct=up_pct,
            down_pct=down_pct,
            top_k=top_k,
            horizon_bars=horizon_bars,
            cost=COST_PER_EVENT,
            strategy="random",
            seed=seed + s,
        )
        nets.append(res["cumulative_net_return"])
    return {
        "filter": filter_name,
        "n_sims": n_sims,
        "cumulative_net_return_mean": float(np.mean(nets)),
        "cumulative_net_return_median": float(np.median(nets)),
        "cumulative_net_return_p5": float(np.quantile(nets, 0.05)),
        "cumulative_net_return_p95": float(np.quantile(nets, 0.95)),
    }


def classify_filter_oos(
    baseline: dict[str, Any],
    filtered: dict[str, Any],
) -> str:
    """Objective classification based on OOS net improvement across barriers."""
    if not baseline or not filtered:
        return STATUS_INSUFFICIENT
    improvements = 0
    total = 0
    for barrier, _, _ in BARRIER_SPECS:
        b = (baseline.get(barrier) or {}).get("top_5") or {}
        f = (filtered.get(barrier) or {}).get("top_5") or {}
        if b.get("cumulative_net_return") is None or f.get("cumulative_net_return") is None:
            continue
        total += 1
        if f["cumulative_net_return"] > b["cumulative_net_return"]:
            improvements += 1
    if total == 0:
        return STATUS_INSUFFICIENT
    if improvements == total:
        return STATUS_GENERALIZES
    if improvements >= 1:
        return STATUS_MIXED
    return STATUS_FAILS


def run_regime_filter_oos(
    rows: Sequence[Any],
    bars_by_symbol: dict[str, list],
    experiment: ExperimentSpec,
    *,
    train_end: datetime,
    thresholds: dict[str, Any],
    min_cross_section: int = MIN_CROSS_SECTION,
    leakage_ok: bool = True,
) -> dict[str, Any]:
    """Evaluate frozen filters on OOS period (rows > train_end)."""
    train_end = as_utc(train_end)
    oos_rows = [r for r in rows if as_utc(r.ts) > train_end]
    if not oos_rows:
        return {
            "promote": False,
            "experiment_status": STATUS_INSUFFICIENT,
            "notes": "No OOS rows after train_end.",
            "metrics": {},
            "experiment_id": experiment.experiment_id,
            "target_name": "barrier_first_touch",
            "sample_stride": experiment.primary_stride_bars,
            "feature_names": feature_names(),
            "blob": b"",
        }

    # Cross-sectional eligibility
    groups = group_by_timestamp(oos_rows)
    kept: list[Any] = []
    for _ts, bucket in groups.items():
        eligible = [
            r
            for r in bucket
            if getattr(r, "max_upside", None) is not None
            and getattr(r, "max_drawdown", None) is not None
            and getattr(r, "fwd_return", None) is not None
        ]
        if len(eligible) >= min_cross_section:
            kept.extend(eligible)

    if not kept:
        return {
            "promote": False,
            "experiment_status": STATUS_INSUFFICIENT,
            "notes": "No OOS rows met cross-section minimum.",
            "metrics": {},
            "experiment_id": experiment.experiment_id,
            "target_name": "barrier_first_touch",
            "sample_stride": experiment.primary_stride_bars,
            "feature_names": feature_names(),
            "blob": b"",
        }

    train_rows = [r for r in rows if as_utc(r.ts) <= train_end - embargo()]
    h_bars = 12
    per_barrier: dict[str, Any] = {}
    baseline_portfolio: dict[str, Any] = {}
    filtered_portfolio: dict[str, Any] = {}

    for barrier, up_pct, down_pct in BARRIER_SPECS:
        models = fit_models_on_rows(train_rows, barrier)
        if not models:
            continue
        scores = predict_with_models(models, kept)
        full_scores = scores["full"]

        barrier_result: dict[str, Any] = {
            "predictive": {},
            "portfolio": {},
            "exposure": {},
            "random_benchmark": {},
            "slippage": {},
        }

        # Predictive metrics per filter (same model, filtered evaluation)
        for filt in FILTERS:
            filt_rows = apply_filter_to_rows(kept, filt, thresholds)
            if len(filt_rows) < 20:
                barrier_result["predictive"][filt] = {"note": "insufficient_sample", "n": len(filt_rows)}
                continue
            score_by_id = {id(r): float(s) for r, s in zip(kept, full_scores)}
            filt_scores = np.array([score_by_id[id(r)] for r in filt_rows], dtype=float)
            barrier_result["predictive"][filt] = evaluate_period_predictions(filt_rows, filt_scores, barrier)
            barrier_result["exposure"][filt] = exposure_stats(kept, filt_rows)

        # Portfolio per filter (top-5 primary)
        for filt in FILTERS:
            res = run_filtered_backtest(
                kept,
                bars_by_symbol,
                full_scores,
                filter_name=filt,
                thresholds=thresholds,
                barrier=barrier,
                up_pct=up_pct,
                down_pct=down_pct,
                top_k=5,
                horizon_bars=h_bars,
            )
            barrier_result["portfolio"][filt] = res
            if filt == "baseline":
                baseline_portfolio[barrier] = res
            else:
                filtered_portfolio.setdefault(barrier, {})[filt] = res

        # Benchmarks (unfiltered)
        for strat in ("atr_only", "momentum", "full_without_atr"):
            res = run_portfolio_backtest(
                kept,
                bars_by_symbol,
                scores[strat],
                barrier=barrier,
                up_pct=up_pct,
                down_pct=down_pct,
                top_k=5,
                horizon_bars=h_bars,
                cost=COST_PER_EVENT,
                strategy=strat,
            )
            barrier_result["portfolio"][strat] = res

        # Random filtered benchmark
        for filt in FILTERS:
            barrier_result["random_benchmark"][filt] = random_filtered_benchmark(
                kept,
                bars_by_symbol,
                filter_name=filt,
                thresholds=thresholds,
                barrier=barrier,
                up_pct=up_pct,
                down_pct=down_pct,
                top_k=5,
                horizon_bars=h_bars,
            )

        # Slippage sensitivity on top-5
        for filt in FILTERS:
            slip = {}
            for case, extra in {"base": 0.0, "plus_0.1pct": 0.001, "plus_0.25pct": 0.0025}.items():
                res = run_filtered_backtest(
                    kept,
                    bars_by_symbol,
                    full_scores,
                    filter_name=filt,
                    thresholds=thresholds,
                    barrier=barrier,
                    up_pct=up_pct,
                    down_pct=down_pct,
                    top_k=5,
                    horizon_bars=h_bars,
                    slippage=extra,
                )
                slip[case] = {
                    "cumulative_net_return": res.get("cumulative_net_return"),
                    "n_trades": res.get("n_trades"),
                    "win_rate": res.get("win_rate"),
                }
            barrier_result["slippage"][filt] = slip

        per_barrier[barrier] = barrier_result

    # Classification: does combined filter improve baseline on all barriers?
    combined_vs_baseline = {}
    for barrier, _, _ in BARRIER_SPECS:
        b = (per_barrier.get(barrier) or {}).get("portfolio", {}).get("baseline") or {}
        f = (per_barrier.get(barrier) or {}).get("portfolio", {}).get("high_dispersion_plus_normal_volume") or {}
        if b.get("cumulative_net_return") is not None and f.get("cumulative_net_return") is not None:
            combined_vs_baseline[barrier] = {
                "baseline_net": b["cumulative_net_return"],
                "combined_net": f["cumulative_net_return"],
                "delta": f["cumulative_net_return"] - b["cumulative_net_return"],
            }
    status = classify_filter_oos(
        {b: {"top_5": v} for b, v in ((k, {"cumulative_net_return": v["baseline_net"]}) for k, v in combined_vs_baseline.items())},
        {b: {"top_5": v} for b, v in ((k, {"cumulative_net_return": v["combined_net"]}) for k, v in combined_vs_baseline.items())},
    )
    if not leakage_ok:
        status = STATUS_FAILS

    metrics = {
        "train_end": train_end.isoformat(),
        "oos_rows": len(kept),
        "oos_timestamps": len({as_utc(r.ts) for r in kept}),
        "thresholds": thresholds,
        "barriers": per_barrier,
        "combined_vs_baseline": combined_vs_baseline,
        "lineage": experiment.lineage(),
        "model_type": "HistGradientBoostingClassifier",
        "cost_per_event": COST_PER_EVENT,
        "random_seed": RANDOM_SEED,
        "random_sims": RANDOM_SIMS,
    }
    return {
        "promote": False,
        "experiment_status": status,
        "passed_experiment_gates": status == STATUS_GENERALIZES,
        "feature_names": feature_names(),
        "blob": b"",
        "metrics": metrics,
        "notes": f"Regime filter OOS classification: {status}.",
        "experiment_id": experiment.experiment_id,
        "target_name": "barrier_first_touch",
        "sample_stride": experiment.primary_stride_bars,
        "move_threshold": experiment.move_pct,
    }
