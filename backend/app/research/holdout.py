"""Untouched out-of-sample holdout experiment.

Frozen spec = barrier_probability_v2 (same features, targets, model, universe).
Holdout is the latest chronological slice; development is the earlier portion.
No holdout rows are used for training, thresholds, or decisions.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier

from app.research.barriers import (
    COST_PER_EVENT,
    EV_SPECS,
    barrier_binary,
    barrier_state,
    binary_classification_metrics,
    expected_value,
)
from app.research.experiments import ExperimentSpec
from app.research.features import feature_names
from app.research.history import as_utc
from app.research.opportunity import BARRIER_SPECS
from app.research.ranking import (
    MIN_CROSS_SECTION,
    future_max_drawdown,
    future_max_upside,
    future_return,
    group_by_timestamp,
)
from app.research.spec import embargo

RANDOM_SEED = 42
BOOTSTRAP_SEED = 42
BOOTSTRAP_N = 1000

STATUS_GENERALIZES = "generalizes"
STATUS_WEAK = "weak_generalization"
STATUS_FAILS = "fails_to_generalize"
STATUS_INSUFFICIENT = "insufficient_holdout"


def _feat(row: Any, name: str) -> float:
    feats = getattr(row, "features", None) or {}
    value = feats.get(name)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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


def choose_holdout_cutoff(
    timestamps: Sequence[datetime],
    *,
    holdout_frac: float = 0.25,
    min_holdout: int = 20,
) -> tuple[datetime, dict[str, Any]]:
    """Chronological cutoff: latest ~holdout_frac of unique timestamps is holdout."""
    ts = sorted({as_utc(t) for t in timestamps})
    n = len(ts)
    if n < min_holdout + 10:
        cutoff = ts[-1] if ts else None
        return cutoff, {
            "cutoff": cutoff.isoformat() if cutoff else None,
            "holdout_frac_requested": holdout_frac,
            "unique_timestamps": n,
            "holdout_timestamps": 0,
            "development_timestamps": n,
            "note": "Insufficient timestamps for holdout.",
        }
    n_holdout = max(min_holdout, int(round(n * holdout_frac)))
    n_holdout = min(n_holdout, n - 10)
    cutoff = ts[n - n_holdout]
    return cutoff, {
        "cutoff": cutoff.isoformat(),
        "holdout_frac_requested": holdout_frac,
        "unique_timestamps": n,
        "holdout_timestamps": n_holdout,
        "development_timestamps": n - n_holdout,
        "note": (
            f"Latest {n_holdout}/{n} timestamps (~{holdout_frac:.0%}) form the untouched holdout; "
            "cutoff chosen chronologically before any holdout evaluation."
        ),
    }


def split_development_holdout(
    rows: Sequence[Any], cutoff: datetime
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    cutoff = as_utc(cutoff)
    dev = [r for r in rows if as_utc(r.ts) < cutoff]
    hold = [r for r in rows if as_utc(r.ts) >= cutoff]
    dev_ts = sorted({as_utc(r.ts) for r in dev})
    hold_ts = sorted({as_utc(r.ts) for r in hold})
    meta = {
        "cutoff": cutoff.isoformat(),
        "development_start": dev_ts[0].isoformat() if dev_ts else None,
        "development_end": dev_ts[-1].isoformat() if dev_ts else None,
        "holdout_start": hold_ts[0].isoformat() if hold_ts else None,
        "holdout_end": hold_ts[-1].isoformat() if hold_ts else None,
        "development_rows": len(dev),
        "holdout_rows": len(hold),
        "development_timestamps": len(dev_ts),
        "holdout_timestamps": len(hold_ts),
    }
    return dev, hold, meta


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), len(scores)))
    clean = np.where(np.isfinite(scores), scores, -np.inf)
    return np.argsort(clean)[::-1][:k]


def _group_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"n": 0}

    def _nan_agg(values: list[float], fn) -> float | None:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if not len(arr):
            return None
        return float(fn(arr))

    resolved = [i for i in items if i["state"] in ("up_first", "down_first")]
    succ = np.array([1.0 if i["state"] == "up_first" else 0.0 for i in resolved], dtype=float)
    timeout = np.array([1.0 if i["state"] == "neither" else 0.0 for i in items], dtype=float)
    amb = np.array([1.0 if i["state"] == "ambiguous" else 0.0 for i in items], dtype=float)
    ret = [i["future_return"] for i in items]
    raw = [i["raw_upside"] for i in items]
    dd = [i["drawdown"] for i in items]
    return {
        "n": int(len(items)),
        "resolved_n": int(len(resolved)),
        "success_rate": float(np.mean(succ)) if len(succ) else None,
        "timeout_rate": float(np.mean(timeout)),
        "ambiguous_rate": float(np.mean(amb)),
        "mean_future_return": _nan_agg(ret, np.mean),
        "median_future_return": _nan_agg(ret, np.median),
        "mean_raw_upside": _nan_agg(raw, np.mean),
        "mean_drawdown": _nan_agg(dd, np.mean),
        "median_drawdown": _nan_agg(dd, np.median),
        "worst_drawdown": _nan_agg(dd, np.min),
        "frac_dd_worse_3": _nan_agg([1.0 if v <= -0.03 else 0.0 for v in dd], np.mean),
        "frac_dd_worse_5": _nan_agg([1.0 if v <= -0.05 else 0.0 for v in dd], np.mean),
        "frac_dd_worse_10": _nan_agg([1.0 if v <= -0.10 else 0.0 for v in dd], np.mean),
    }


def evaluate_timestamp(rows: Sequence[Any], scores: np.ndarray, barrier: str) -> dict[str, Any]:
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

    def take(k: int) -> list[dict[str, Any]]:
        return ranked[: min(k, n)]

    return {
        "n": n,
        "all": _group_metrics(scored),
        "top_1": _group_metrics(take(1)),
        "top_3": _group_metrics(take(3)),
        "top_5": _group_metrics(take(5)),
        "top_10pct": _group_metrics(take(k10)),
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


def aggregate_ts(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"timestamps": 0}
    out: dict[str, Any] = {"timestamps": len(results)}
    for group in ("all", "top_1", "top_3", "top_5", "top_10pct"):
        sample = results[0].get(group) or {}
        for metric in sample:
            if metric == "n":
                continue
            out[f"{group}_{metric}"] = _mean_path(results, (group, metric))
    return out


def evaluate_by_timestamp(rows: Sequence[Any], scores: np.ndarray, barrier: str) -> tuple[list[dict], dict]:
    by_ts: dict[datetime, list[tuple[Any, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_ts[as_utc(row.ts)].append((row, float(score)))
    ts_results = []
    for _ts, pairs in sorted(by_ts.items(), key=lambda kv: kv[0]):
        bucket_rows = [p[0] for p in pairs]
        bucket_scores = np.array([p[1] for p in pairs], dtype=float)
        ts_results.append(evaluate_timestamp(bucket_rows, bucket_scores, barrier))
    return ts_results, aggregate_ts(ts_results)


def random_topk_benchmark(
    rows: Sequence[Any],
    barrier: str,
    *,
    k: int = 5,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Fixed-seed random top-K success rate per timestamp, averaged."""
    rng = np.random.default_rng(seed)
    by_ts: dict[datetime, list[Any]] = defaultdict(list)
    for row in rows:
        by_ts[as_utc(row.ts)].append(row)
    rates = []
    for _ts, bucket in by_ts.items():
        if len(bucket) < 2:
            continue
        idx = rng.permutation(len(bucket))[: min(k, len(bucket))]
        picked = [bucket[i] for i in idx]
        resolved = [r for r in picked if barrier_state(r, barrier) in ("up_first", "down_first")]
        if resolved:
            rates.append(float(np.mean([1.0 if barrier_state(r, barrier) == "up_first" else 0.0 for r in resolved])))
    return {
        "k": k,
        "seed": seed,
        "timestamps": len(rates),
        "random_topk_success_rate": float(np.mean(rates)) if rates else None,
        "note": "Fixed-seed random benchmark; not a Monte Carlo distribution.",
    }


def block_bootstrap_ci(
    rows: Sequence[Any],
    scores: np.ndarray,
    barrier: str,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    block_size: int = 5,
) -> dict[str, Any]:
    """Block bootstrap over timestamps for top-5/top-10% success and Spearman."""
    by_ts: dict[datetime, list[tuple[Any, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_ts[as_utc(row.ts)].append((row, float(score)))
    ts_keys = sorted(by_ts)
    if not ts_keys:
        return {"note": "No timestamps."}
    blocks = [ts_keys[i : i + block_size] for i in range(0, len(ts_keys), block_size)]
    rng = np.random.default_rng(seed)

    def _stat(sample_keys: list[datetime]) -> tuple[float | None, float | None, float | None]:
        rs = []
        ss = []
        for k in sample_keys:
            rs.extend(by_ts[k])
        if not rs:
            return None, None, None
        rows_s = [p[0] for p in rs]
        scores_s = np.array([p[1] for p in rs], dtype=float)
        _, agg = evaluate_by_timestamp(rows_s, scores_s, barrier)
        y = np.array(
            [barrier_binary(r, barrier) if barrier_binary(r, barrier) is not None else -1 for r in rows_s],
            dtype=int,
        )
        mask = y >= 0
        sp = None
        if mask.sum() >= 3 and len(set(y[mask].tolist())) > 1 and np.nanstd(scores_s[mask]) > 0:
            rho, _ = spearmanr(scores_s[mask], y[mask])
            sp = float(rho) if np.isfinite(rho) else None
        return agg.get("top_5_success_rate"), agg.get("top_10pct_success_rate"), sp

    top5 = []
    top10 = []
    sps = []
    for _ in range(n_boot):
        sampled_blocks = rng.choice(len(blocks), size=len(blocks), replace=True)
        keys = [ts for b in sampled_blocks for ts in blocks[b]]
        t5, t10, sp = _stat(keys)
        if t5 is not None:
            top5.append(t5)
        if t10 is not None:
            top10.append(t10)
        if sp is not None:
            sps.append(sp)

    def _ci(vals: list[float]) -> dict[str, float | None]:
        if not vals:
            return {"mean": None, "ci_low": None, "ci_high": None}
        arr = np.asarray(vals, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
        }

    return {
        "n_boot": n_boot,
        "seed": seed,
        "block_size_timestamps": block_size,
        "top_5_success_rate": _ci(top5),
        "top_10pct_success_rate": _ci(top10),
        "spearman": _ci(sps),
        "note": "Block bootstrap over timestamps; not a formal significance test.",
    }


def economic_selection(
    rows: Sequence[Any],
    scores: np.ndarray,
    barrier: str,
    *,
    cost: float = COST_PER_EVENT,
) -> dict[str, Any]:
    """Research-only top-5 selection economics using realized fwd_return."""
    by_ts: dict[datetime, list[tuple[Any, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_ts[as_utc(row.ts)].append((row, float(score)))
    returns = []
    n_sel = 0
    for _ts, pairs in by_ts.items():
        if len(pairs) < 5:
            continue
        order = _top_k_indices(np.array([p[1] for p in pairs], dtype=float), 5)
        for i in order:
            r = future_return(pairs[i][0])
            if r is None or not np.isfinite(r):
                continue
            returns.append(float(r))
            n_sel += 1
    if not returns:
        return {"n_selections": 0, "note": "No selections."}
    arr = np.asarray(returns, dtype=float)
    gross = float(np.mean(arr))
    net = gross - cost
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    equity = np.cumsum(arr - cost)
    peak = np.maximum.accumulate(equity) if len(equity) else np.array([0.0])
    max_dd = float(np.min(equity - peak)) if len(equity) else None
    return {
        "n_selections": n_sel,
        "gross_avg_return": gross,
        "transaction_cost_per_selection": cost,
        "net_avg_return": net,
        "median_return": float(np.median(arr)),
        "win_rate": float(np.mean(arr > 0)),
        "avg_winner": float(np.mean(wins)) if len(wins) else None,
        "avg_loser": float(np.mean(losses)) if len(losses) else None,
        "max_drawdown_cumulative": max_dd,
        "turnover": 1.0,
        "note": "Hypothetical top-5 selection; no leverage/sizing; research-only.",
    }


def classify_generalization(dev: dict[str, Any], hold: dict[str, Any]) -> str:
    """Objective classification of generalization decay."""
    barriers = list(hold)
    if not barriers:
        return STATUS_INSUFFICIENT
    positive = 0
    weak = 0
    for name in barriers:
        h = (hold.get(name) or {}).get("full") or {}
        d = (dev.get(name) or {}).get("full") or {}
        sp_h = h.get("spearman_prob_outcome")
        t5_h = h.get("top_5_success_rate")
        all_h = h.get("all_success_rate")
        sp_d = d.get("spearman_prob_outcome")
        t5_d = d.get("top_5_success_rate")
        if sp_h is None or t5_h is None or all_h is None:
            return STATUS_INSUFFICIENT
        if sp_h > 0 and t5_h > all_h:
            positive += 1
            if sp_d is not None and t5_d is not None:
                if sp_h >= 0.5 * sp_d and t5_h >= 0.8 * t5_d:
                    weak += 1
    if positive == 0:
        return STATUS_FAILS
    if positive == len(barriers) and weak == len(barriers):
        return STATUS_GENERALIZES
    if positive >= 1:
        return STATUS_WEAK
    return STATUS_FAILS


def fit_holdout_experiment(
    rows: Sequence[Any],
    experiment: ExperimentSpec,
    *,
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

    timestamps = sorted({as_utc(r.ts) for r in ordered})
    cutoff, cutoff_meta = choose_holdout_cutoff(timestamps)
    dev_rows, hold_rows, split_meta = split_development_holdout(ordered, cutoff)
    if not dev_rows or not hold_rows:
        return {
            "promote": False,
            "experiment_status": STATUS_INSUFFICIENT,
            "notes": "Insufficient development or holdout rows.",
            "metrics": {"cross_section": cs_stats, "cutoff": cutoff_meta, "split": split_meta},
            "experiment_id": experiment.experiment_id,
            "target_name": "barrier_first_touch",
            "sample_stride": experiment.primary_stride_bars,
            "feature_names": full_names,
            "blob": b"",
        }

    # Enforce embargo: drop development rows whose label window overlaps holdout start.
    gap = embargo()
    hold_start = min(as_utc(r.ts) for r in hold_rows)
    dev_cutoff = hold_start - gap
    dev_rows_embargoed = [r for r in dev_rows if as_utc(r.ts) <= dev_cutoff]
    embargo_dropped = len(dev_rows) - len(dev_rows_embargoed)

    x_full_dev = _matrix(dev_rows_embargoed, full_names)
    x_atr_dev = _matrix(dev_rows_embargoed, atr_names)
    x_mom_dev = _matrix(dev_rows_embargoed, [mom_feature])
    x_no_atr_dev = _matrix(dev_rows_embargoed, no_atr_names)

    x_full_hold = _matrix(hold_rows, full_names)
    x_atr_hold = _matrix(hold_rows, atr_names)
    x_mom_hold = _matrix(hold_rows, [mom_feature])
    x_no_atr_hold = _matrix(hold_rows, no_atr_names)

    per_barrier: dict[str, dict[str, Any]] = {}
    dev_metrics: dict[str, Any] = {}
    hold_metrics: dict[str, Any] = {}

    for barrier, up_pct, down_pct in BARRIER_SPECS:
        y_dev_all = np.array(
            [barrier_binary(r, barrier) if barrier_binary(r, barrier) is not None else -1 for r in dev_rows_embargoed],
            dtype=int,
        )
        y_hold_all = np.array(
            [barrier_binary(r, barrier) if barrier_binary(r, barrier) is not None else -1 for r in hold_rows],
            dtype=int,
        )
        dev_mask = y_dev_all >= 0
        hold_mask = y_hold_all >= 0
        if dev_mask.sum() < 50 or hold_mask.sum() < 20:
            per_barrier[barrier] = {"note": "Insufficient binary observations."}
            continue

        y_dev = y_dev_all[dev_mask].astype(int)
        if len(set(y_dev.tolist())) < 2:
            per_barrier[barrier] = {"note": "Single-class development labels."}
            continue

        dev_idx = np.where(dev_mask)[0]
        hold_idx = np.where(hold_mask)[0]

        def _fit(x_all: np.ndarray, idx: np.ndarray) -> HistGradientBoostingClassifier:
            clf = _new_classifier()
            clf.fit(x_all[idx], y_dev)
            return clf

        clf_full = _fit(x_full_dev, dev_idx)
        clf_atr = _fit(x_atr_dev, dev_idx)
        clf_mom = _fit(x_mom_dev, dev_idx)
        clf_no_atr = _fit(x_no_atr_dev, dev_idx)

        # Holdout predictions (all holdout rows for ranking; binary metrics on resolved subset)
        p_full_hold = _positive_proba(clf_full, x_full_hold)
        p_atr_hold = _positive_proba(clf_atr, x_atr_hold)
        p_mom_hold = _positive_proba(clf_mom, x_mom_hold)
        p_no_atr_hold = _positive_proba(clf_no_atr, x_no_atr_hold)

        y_hold_bin = y_hold_all[hold_mask].astype(int)
        full_cls = binary_classification_metrics(y_hold_bin, p_full_hold[hold_mask])
        atr_cls = binary_classification_metrics(y_hold_bin, p_atr_hold[hold_mask])
        mom_cls = binary_classification_metrics(y_hold_bin, p_mom_hold[hold_mask])
        no_atr_cls = binary_classification_metrics(y_hold_bin, p_no_atr_hold[hold_mask])

        _, full_rank = evaluate_by_timestamp(hold_rows, p_full_hold, barrier)
        _, atr_rank = evaluate_by_timestamp(hold_rows, p_atr_hold, barrier)
        _, mom_rank = evaluate_by_timestamp(hold_rows, p_mom_hold, barrier)
        _, no_atr_rank = evaluate_by_timestamp(hold_rows, p_no_atr_hold, barrier)

        # Development in-sample-ish reference (walk-forward style would be ideal; here we
        # report development fit metrics for context only, clearly labeled).
        p_full_dev = _positive_proba(clf_full, x_full_dev[dev_idx])
        dev_cls = binary_classification_metrics(y_dev, p_full_dev)
        _, dev_rank = evaluate_by_timestamp(
            [dev_rows_embargoed[i] for i in dev_idx], p_full_dev, barrier
        )

        random_bench = random_topk_benchmark(hold_rows, barrier)
        bootstrap = block_bootstrap_ci(hold_rows, p_full_hold, barrier)
        econ_full = economic_selection(hold_rows, p_full_hold, barrier)
        econ_atr = economic_selection(hold_rows, p_atr_hold, barrier)

        # Volatility buckets on holdout using development ATR tertiles
        dev_atrs = np.array([_feat(r, "atr_pct") for r in dev_rows_embargoed], dtype=float)
        finite = dev_atrs[np.isfinite(dev_atrs)]
        buckets: dict[str, Any] = {}
        if len(finite) >= 30:
            q33, q66 = np.quantile(finite, [1 / 3, 2 / 3])
            by_bucket: dict[str, list[tuple[Any, float, float, float]]] = {
                "low": [],
                "medium": [],
                "high": [],
            }
            for row, fs, as_, ms in zip(hold_rows, p_full_hold, p_atr_hold, p_mom_hold):
                atr = _feat(row, "atr_pct")
                if not np.isfinite(atr):
                    continue
                lab = "low" if atr <= q33 else ("medium" if atr <= q66 else "high")
                by_bucket[lab].append((row, fs, as_, ms))
            for lab, items in by_bucket.items():
                if len(items) < 30:
                    buckets[lab] = {"n": len(items), "note": "Insufficient."}
                    continue
                brows = [i[0] for i in items]
                _, fagg = evaluate_by_timestamp(brows, np.array([i[1] for i in items]), barrier)
                _, aagg = evaluate_by_timestamp(brows, np.array([i[2] for i in items]), barrier)
                _, magg = evaluate_by_timestamp(brows, np.array([i[3] for i in items]), barrier)
                buckets[lab] = {
                    "n": len(items),
                    "full_top5_success": fagg.get("top_5_success_rate"),
                    "atr_top5_success": aagg.get("top_5_success_rate"),
                    "mom_top5_success": magg.get("top_5_success_rate"),
                    "all_success": fagg.get("all_success_rate"),
                }

        ev = {}
        for label, (up, down) in EV_SPECS.items():
            if label != barrier:
                continue
            f_rate = full_rank.get("top_5_success_rate")
            a_rate = atr_rank.get("top_5_success_rate")
            ev = {
                "assumption": f"success=+{up:.0%}, failure=-{down:.0%}; diagnostic only.",
                "cost_per_event": COST_PER_EVENT,
                "full_top5_ev_net": expected_value(f_rate, up, down, cost=COST_PER_EVENT) if f_rate is not None else None,
                "atr_top5_ev_net": expected_value(a_rate, up, down, cost=COST_PER_EVENT) if a_rate is not None else None,
            }

        per_barrier[barrier] = {
            "full": {**full_cls, **{k: v for k, v in full_rank.items() if k != "timestamps"}},
            "atr_only": {**atr_cls, **{k: v for k, v in atr_rank.items() if k != "timestamps"}},
            "momentum": {**mom_cls, **{k: v for k, v in mom_rank.items() if k != "timestamps"}},
            "full_without_atr": {**no_atr_cls, **{k: v for k, v in no_atr_rank.items() if k != "timestamps"}},
            "development_full": {**dev_cls, **{k: v for k, v in dev_rank.items() if k != "timestamps"}},
            "random_benchmark": random_bench,
            "bootstrap": bootstrap,
            "economic_full": econ_full,
            "economic_atr": econ_atr,
            "volatility_buckets": buckets,
            "expected_value": ev,
            "holdout_binary_observations": int(hold_mask.sum()),
            "development_binary_observations": int(dev_mask.sum()),
        }
        dev_metrics[barrier] = {"full": per_barrier[barrier]["development_full"]}
        hold_metrics[barrier] = {"full": per_barrier[barrier]["full"]}

    status = classify_generalization(dev_metrics, hold_metrics)
    if not leakage_ok:
        status = STATUS_FAILS

    metrics = {
        "cross_section": cs_stats,
        "cutoff": cutoff_meta,
        "split": split_meta,
        "embargo_dropped_development_rows": embargo_dropped,
        "barriers": per_barrier,
        "lineage": experiment.lineage(),
        "model_type": "HistGradientBoostingClassifier",
        "random_seed": RANDOM_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "cost_per_event": COST_PER_EVENT,
    }
    return {
        "promote": False,
        "experiment_status": status,
        "passed_experiment_gates": status == STATUS_GENERALIZES,
        "feature_names": full_names,
        "blob": b"",
        "metrics": metrics,
        "notes": f"Holdout classification: {status}.",
        "experiment_id": experiment.experiment_id,
        "target_name": "barrier_first_touch",
        "sample_stride": experiment.primary_stride_bars,
        "move_threshold": experiment.move_pct,
    }
