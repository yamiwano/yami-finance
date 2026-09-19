"""Multi-period historical robustness experiment.

Frozen methodology = barrier_probability_v2 + portfolio_backtest_v1.
Expands history, segments chronologically, and evaluates each unseen period
with walk-forward training on prior periods only.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr

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
from app.research.spec import embargo


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

STATUS_ROBUST = "robust_across_periods"
STATUS_MIXED = "mixed_regime_signal"
STATUS_PERIOD_SPECIFIC = "period_specific_signal"
STATUS_FAILS = "fails_robustness_test"
STATUS_INSUFFICIENT = "insufficient_history"


def segment_periods(
    timestamps: Sequence[datetime],
    *,
    n_periods: int = 4,
) -> list[dict[str, Any]]:
    """Mechanically split unique timestamps into n_periods chronological periods."""
    ts = sorted({as_utc(t) for t in timestamps})
    n = len(ts)
    if n < n_periods * 10:
        return []
    bounds = np.linspace(0, n, n_periods + 1).astype(int)
    periods = []
    for i in range(n_periods):
        start_idx, end_idx = bounds[i], bounds[i + 1]
        periods.append(
            {
                "period": i + 1,
                "start": ts[start_idx].isoformat(),
                "end": ts[end_idx - 1].isoformat(),
                "timestamps": end_idx - start_idx,
                "start_ts": ts[start_idx],
                "end_ts": ts[end_idx - 1],
            }
        )
    return periods


def rows_in_period(rows: Sequence[Any], start: datetime, end: datetime) -> list[Any]:
    start = as_utc(start)
    end = as_utc(end)
    return [r for r in rows if start <= as_utc(r.ts) <= end]


def evaluate_period_predictions(
    rows: Sequence[Any],
    scores: np.ndarray,
    barrier: str,
) -> dict[str, Any]:
    """Predictive metrics for one period (binary subset + ranking)."""
    y = np.array(
        [barrier_binary(r, barrier) if barrier_binary(r, barrier) is not None else -1 for r in rows],
        dtype=int,
    )
    mask = y >= 0
    out: dict[str, Any] = {"observations": len(rows), "resolved": int(mask.sum())}
    if mask.sum() >= 3 and len(set(y[mask].tolist())) > 1:
        out.update(binary_classification_metrics(y[mask].astype(int), scores[mask]))
    else:
        out.update({"auc": None, "brier": None, "log_loss": None, "spearman_prob_outcome": None})

    # Top-K success on resolved subset
    by_ts = group_by_timestamp(rows)
    top5_rates = []
    top10_rates = []
    all_rates = []
    for _ts, bucket in by_ts.items():
        resolved = [r for r in bucket if barrier_binary(r, barrier) is not None]
        if len(resolved) < 5:
            continue
        order = np.argsort([scores[rows.index(r)] for r in resolved])[::-1]
        succ = np.array([barrier_binary(r, barrier) for r in resolved], dtype=float)
        top5_rates.append(float(np.mean(succ[order[:5]])))
        k10 = max(1, int(np.ceil(0.10 * len(resolved))))
        top10_rates.append(float(np.mean(succ[order[:k10]])))
        all_rates.append(float(np.mean(succ)))
    out["top_5_success_rate"] = float(np.mean(top5_rates)) if top5_rates else None
    out["top_10pct_success_rate"] = float(np.mean(top10_rates)) if top10_rates else None
    out["all_success_rate"] = float(np.mean(all_rates)) if all_rates else None
    return out


def fit_models_on_rows(rows: Sequence[Any], barrier: str) -> dict[str, Any]:
    full_names = feature_names()
    no_atr_names = [n for n in full_names if n != "atr_pct"]
    y = np.array(
        [barrier_binary(r, barrier) if barrier_binary(r, barrier) is not None else -1 for r in rows],
        dtype=int,
    )
    mask = y >= 0
    if mask.sum() < 50 or len(set(y[mask].tolist())) < 2:
        return {}
    idx = np.where(mask)[0]
    y_bin = y[mask].astype(int)

    def _fit(names: list[str]):
        x = _matrix(rows, names)
        clf = _new_classifier()
        clf.fit(x[idx], y_bin)
        return clf, names

    return {
        "full": _fit(full_names),
        "atr_only": _fit(["atr_pct"]),
        "momentum": _fit(["ret_24"]),
        "full_without_atr": _fit(no_atr_names),
    }


def predict_with_models(models: dict[str, Any], rows: Sequence[Any]) -> dict[str, np.ndarray]:
    out = {}
    for name, (clf, names) in models.items():
        x = _matrix(rows, names)
        out[name] = _positive_proba(clf, x)
    return out


def run_multi_period_robustness(
    rows: Sequence[Any],
    bars_by_symbol: dict[str, list],
    experiment: ExperimentSpec,
    *,
    n_periods: int = 4,
    min_cross_section: int = MIN_CROSS_SECTION,
    leakage_ok: bool = True,
) -> dict[str, Any]:
    # Cross-sectional eligibility
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
            if getattr(r, "max_upside", None) is not None
            and getattr(r, "max_drawdown", None) is not None
            and getattr(r, "fwd_return", None) is not None
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
    timestamps = sorted({as_utc(r.ts) for r in ordered})
    periods = segment_periods(timestamps, n_periods=n_periods)
    if not periods:
        return {
            "promote": False,
            "experiment_status": STATUS_INSUFFICIENT,
            "notes": "Insufficient timestamps for multi-period segmentation.",
            "metrics": {"cross_section": cs_stats},
            "experiment_id": experiment.experiment_id,
            "target_name": "barrier_first_touch",
            "sample_stride": experiment.primary_stride_bars,
            "feature_names": feature_names(),
            "blob": b"",
        }

    gap = embargo()
    per_period: list[dict[str, Any]] = []
    combined_trades: list[dict[str, Any]] = []

    for i, period in enumerate(periods):
        train_end = as_utc(period["start_ts"]) - gap
        train_rows = [r for r in ordered if as_utc(r.ts) <= train_end]
        test_rows = rows_in_period(ordered, period["start_ts"], period["end_ts"])
        if len(train_rows) < 100 or len(test_rows) < 50:
            per_period.append(
                {
                    **{k: v for k, v in period.items() if k not in ("start_ts", "end_ts")},
                    "note": "Insufficient train/test rows.",
                }
            )
            continue

        period_result: dict[str, Any] = {
            **{k: v for k, v in period.items() if k not in ("start_ts", "end_ts")},
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "barriers": {},
        }

        for barrier, up_pct, down_pct in BARRIER_SPECS:
            models = fit_models_on_rows(train_rows, barrier)
            if not models:
                period_result["barriers"][barrier] = {"note": "Insufficient binary training data."}
                continue
            test_scores = predict_with_models(models, test_rows)
            barrier_metrics: dict[str, Any] = {}
            for strat, scores in test_scores.items():
                barrier_metrics[strat] = evaluate_period_predictions(test_rows, scores, barrier)
            # Portfolio backtest top-5 for full/atr/momentum/no_atr/random/equal_weight
            portfolio: dict[str, Any] = {}
            for strat, scores in test_scores.items():
                portfolio[strat] = {
                    "top_5": run_portfolio_backtest(
                        test_rows,
                        bars_by_symbol,
                        scores,
                        barrier=barrier,
                        up_pct=up_pct,
                        down_pct=down_pct,
                        top_k=5,
                        horizon_bars=12,
                        cost=COST_PER_EVENT,
                        strategy=strat,
                    )
                }
            # Random sims
            rand = []
            for s in range(RANDOM_SIMS):
                res = run_portfolio_backtest(
                    test_rows,
                    bars_by_symbol,
                    np.zeros(len(test_rows)),
                    barrier=barrier,
                    up_pct=up_pct,
                    down_pct=down_pct,
                    top_k=5,
                    horizon_bars=12,
                    cost=COST_PER_EVENT,
                    strategy="random",
                    seed=RANDOM_SEED + s,
                )
                rand.append(res["cumulative_net_return"])
            portfolio["random"] = {
                "top_5": {
                    "cumulative_net_return_mean": float(np.mean(rand)),
                    "cumulative_net_return_median": float(np.median(rand)),
                    "cumulative_net_return_p5": float(np.quantile(rand, 0.05)),
                    "cumulative_net_return_p95": float(np.quantile(rand, 0.95)),
                    "n_sims": RANDOM_SIMS,
                }
            }
            portfolio["equal_weight_all"] = {
                "top_5": run_portfolio_backtest(
                    test_rows,
                    bars_by_symbol,
                    np.zeros(len(test_rows)),
                    barrier=barrier,
                    up_pct=up_pct,
                    down_pct=down_pct,
                    top_k=max(1, len(test_rows) // max(1, len(set(r.ts for r in test_rows)))),
                    horizon_bars=12,
                    cost=COST_PER_EVENT,
                    strategy="equal_weight_all",
                )
            }
            barrier_metrics["portfolio"] = portfolio
            period_result["barriers"][barrier] = barrier_metrics

        per_period.append(period_result)

    # Combined chronological portfolio across evaluation periods (full model, top-5).
    # Per-period returns are normalized to equal starting capital; combined equity
    # compounds period-level net returns (not raw trade compounding across periods).
    combined: dict[str, Any] = {}
    for barrier, up_pct, down_pct in BARRIER_SPECS:
        period_nets = []
        all_trades = []
        for pr in per_period:
            port = ((pr.get("barriers") or {}).get(barrier) or {}).get("portfolio") or {}
            full5 = (port.get("full") or {}).get("top_5") or {}
            net = full5.get("cumulative_net_return")
            if net is not None:
                period_nets.append(float(net))
            all_trades.extend(full5.get("trade_ledger") or [])
        if not period_nets:
            continue
        equity = np.cumprod(1.0 + np.asarray(period_nets, dtype=float))
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1.0
        rets = np.array([t["net_return"] for t in all_trades], dtype=float)
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        combined[barrier] = {
            "n_trades": len(all_trades),
            "cumulative_net_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
            "max_drawdown": float(np.min(dd)) if len(dd) else 0.0,
            "sharpe": float(np.mean(rets) / np.std(rets, ddof=1)) if len(rets) > 1 and np.std(rets, ddof=1) > 0 else None,
            "sortino": (
                float(np.mean(rets) / np.std(losses, ddof=1))
                if len(losses) > 1 and np.std(losses, ddof=1) > 0
                else None
            ),
            "profit_factor": (
                float(np.sum(wins) / abs(np.sum(losses))) if len(losses) and np.sum(losses) != 0 else None
            ),
            "win_rate": float(np.mean(rets > 0)) if len(rets) else None,
            "avg_trade_return": float(np.mean(rets)) if len(rets) else None,
            "period_net_returns": period_nets,
            "note": "Compounds period-level net returns; per-period portfolios are equal-capital normalized.",
        }

    # Classification: count periods where full beats ATR/momentum on AUC and top-5
    robust_counts = {"auc_gt_050": 0, "auc_gt_055": 0, "full_gt_atr": 0, "full_gt_mom": 0, "full_gt_all": 0, "periods_evaluated": 0}
    for pr in per_period:
        barriers = pr.get("barriers") or {}
        if not barriers:
            continue
        robust_counts["periods_evaluated"] += 1
        for barrier, _, _ in BARRIER_SPECS:
            b = barriers.get(barrier) or {}
            full = b.get("full") or {}
            atr = b.get("atr_only") or {}
            mom = b.get("momentum") or {}
            if full.get("auc") is not None and full["auc"] > 0.50:
                robust_counts["auc_gt_050"] += 1
            if full.get("auc") is not None and full["auc"] > 0.55:
                robust_counts["auc_gt_055"] += 1
            if full.get("auc") is not None and atr.get("auc") is not None and full["auc"] > atr["auc"]:
                robust_counts["full_gt_atr"] += 1
            if full.get("auc") is not None and mom.get("auc") is not None and full["auc"] > mom["auc"]:
                robust_counts["full_gt_mom"] += 1
            if (
                full.get("top_5_success_rate") is not None
                and full.get("all_success_rate") is not None
                and full["top_5_success_rate"] > full["all_success_rate"]
            ):
                robust_counts["full_gt_all"] += 1

    n_eval = robust_counts["periods_evaluated"]
    if n_eval == 0:
        status = STATUS_INSUFFICIENT
    elif robust_counts["auc_gt_050"] == 0:
        status = STATUS_FAILS
    elif robust_counts["full_gt_atr"] >= 2 * n_eval and robust_counts["full_gt_mom"] >= 2 * n_eval:
        status = STATUS_ROBUST
    elif robust_counts["full_gt_atr"] >= n_eval or robust_counts["full_gt_mom"] >= n_eval:
        status = STATUS_MIXED
    else:
        status = STATUS_PERIOD_SPECIFIC
    if not leakage_ok:
        status = STATUS_FAILS

    metrics = {
        "cross_section": cs_stats,
        "periods": per_period,
        "combined": combined,
        "robust_counts": robust_counts,
        "lineage": experiment.lineage(),
        "model_type": "HistGradientBoostingClassifier",
        "random_seed": RANDOM_SEED,
        "random_sims": RANDOM_SIMS,
        "cost_per_event": COST_PER_EVENT,
    }
    return {
        "promote": False,
        "experiment_status": status,
        "passed_experiment_gates": status == STATUS_ROBUST,
        "feature_names": feature_names(),
        "blob": b"",
        "metrics": _jsonable(metrics),
        "notes": f"Multi-period classification: {status}.",
        "experiment_id": experiment.experiment_id,
        "target_name": "barrier_first_touch",
        "sample_stride": experiment.primary_stride_bars,
        "move_threshold": experiment.move_pct,
    }
