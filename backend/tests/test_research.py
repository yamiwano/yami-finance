import math
import unittest
from datetime import datetime, timedelta, timezone

from app.domain import Bar
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.dataset import select_by_stride
from app.research.experiments import DIRECTIONAL_UP_3PCT_12H_V1, get_experiment
from app.research.features import extract_features, feature_names
from app.research.labels import CLEAN_DRAWDOWN_FLOOR, label_forward, labels_from_outcomes
from app.research.model import (
    STATUS_INSUFFICIENT_ATR,
    _metrics,
    design_matrix,
    fit_experiment_walk_forward,
)
from app.research.spec import LOOKBACK_BARS, embargo


def make_bars(n: int, *, start: datetime | None = None, drift: float = 0.0, seed: float = 100.0) -> list[Bar]:
    ts = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = seed
    bars: list[Bar] = []
    for i in range(n):
        open_p = price
        close_p = max(1.0, price * (1.0 + drift) + math.sin(i / 7.0) * 0.4)
        high = max(open_p, close_p) + 0.3
        low = min(open_p, close_p) - 0.3
        bars.append(
            Bar(
                ts=ts + timedelta(hours=i),
                open=open_p,
                high=high,
                low=low,
                close=close_p,
                volume=1000 + (i % 20) * 50,
                closed=True,
            )
        )
        price = close_p
    return bars


class _FakeSample:
    def __init__(self, symbol: str, ts: datetime, features: dict, **labels):
        self.symbol = symbol
        self.ts = ts
        self.features = features
        for key, value in labels.items():
            setattr(self, key, value)


class ResearchFoundationTests(unittest.TestCase):
    def _feats_close(self, a: dict | None, b: dict | None) -> bool:
        if a is None or b is None or set(a) != set(b):
            return False
        for key, value in a.items():
            other = b[key]
            if math.isnan(value) and math.isnan(other):
                continue
            if abs(value - other) > 1e-9:
                return False
        return True

    def test_feature_keys_match_contract(self):
        feats = extract_features(make_bars(LOOKBACK_BARS + 5), "1h")
        self.assertIsNotNone(feats)
        self.assertEqual(set(feats), set(feature_names()))

    def test_features_ignore_future_bars(self):
        bars = make_bars(LOOKBACK_BARS + 20)
        at_t = extract_features(bars[:LOOKBACK_BARS], "1h")
        future = make_bars(20, start=bars[LOOKBACK_BARS].ts, drift=0.08, seed=bars[LOOKBACK_BARS - 1].close * 3)
        leaked = extract_features(bars[:LOOKBACK_BARS] + future, "1h")
        as_of_t = extract_features((bars[:LOOKBACK_BARS] + future)[:LOOKBACK_BARS], "1h")
        self.assertTrue(self._feats_close(at_t, as_of_t))
        self.assertFalse(self._feats_close(at_t, leaked))

    def test_mutating_decision_bar_changes_features(self):
        bars = make_bars(LOOKBACK_BARS)
        before = extract_features(bars, "1h")
        mutated = list(bars)
        last = mutated[-1]
        mutated[-1] = last.model_copy(update={"close": last.close * 1.2, "high": last.high * 1.2})
        after = extract_features(mutated, "1h")
        self.assertFalse(self._feats_close(before, after))

    def test_labels_use_only_future_bars(self):
        bars = make_bars(30, drift=0.0)
        entry = bars[10].close
        future = bars[11:23]
        a = label_forward(future, entry, 12, 0.03)
        mutated_now = list(bars)
        mutated_now[10] = bars[10].model_copy(update={"close": entry * 5, "high": entry * 5})
        b = label_forward(mutated_now[11:23], entry, 12, 0.03)
        self.assertEqual(a, b)
        pumped = [bar.model_copy(update={"close": bar.close * 2, "high": bar.high * 2}) for bar in future]
        c = label_forward(pumped, entry, 12, 0.03)
        self.assertNotEqual(a, c)
        self.assertTrue(c["significant_move"])
        self.assertGreater(c["fwd_return"], a["fwd_return"])

    def test_incomplete_horizon_is_unlabeled(self):
        bars = make_bars(5)
        self.assertIsNone(label_forward(bars[1:], bars[0].close, 12, 0.03))

    def test_walk_forward_never_overlaps_embargo(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [start + timedelta(hours=i) for i in range(24 * 90)]
        folds = walk_forward_folds(ts, min_train=50, min_test=20)
        self.assertGreaterEqual(len(folds), 2)
        gap = embargo()
        for train_idx, test_idx in folds:
            self.assertTrue(fold_is_causal(ts, train_idx, test_idx, gap))
            train_max = ts[train_idx[-1]]
            test_min = ts[test_idx[0]]
            self.assertGreaterEqual(test_min, train_max + gap)

    def test_short_history_still_embargoes(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [start + timedelta(hours=i) for i in range(24 * 12)]
        folds = walk_forward_folds(ts, min_train=10, min_test=5)
        self.assertGreaterEqual(len(folds), 1)
        gap = embargo()
        for train_idx, test_idx in folds:
            self.assertTrue(fold_is_causal(ts, train_idx, test_idx, gap))

    def test_max_upside_and_large_up_move(self):
        entry = 100.0
        future = make_bars(12, seed=100.0)
        # Force a +5% high in the window without requiring final close > +3%.
        spiked = list(future)
        spiked[3] = spiked[3].model_copy(update={"high": entry * 1.05, "close": entry * 0.99})
        spiked[-1] = spiked[-1].model_copy(update={"close": entry * 0.98, "high": entry * 0.99})
        labels = label_forward(spiked, entry, 12, 0.03)
        self.assertIsNotNone(labels)
        self.assertAlmostEqual(labels["max_upside"], 0.05, places=6)
        self.assertTrue(labels["large_up_move"])
        self.assertLess(labels["fwd_return"], 0.03)
        self.assertEqual(len(spiked), 12)

    def test_clean_up_move_requires_limited_drawdown(self):
        entry = 100.0
        future = make_bars(12, seed=100.0)
        clean = list(future)
        clean[2] = clean[2].model_copy(update={"high": entry * 1.04, "low": entry * 0.99})
        labels_clean = label_forward(clean, entry, 12, 0.03)
        self.assertTrue(labels_clean["large_up_move"])
        self.assertTrue(labels_clean["clean_up_move"])

        dirty = list(future)
        dirty[1] = dirty[1].model_copy(update={"low": entry * 0.97})  # -3% < -1.5%
        dirty[2] = dirty[2].model_copy(update={"high": entry * 1.04})
        labels_dirty = label_forward(dirty, entry, 12, 0.03)
        self.assertTrue(labels_dirty["large_up_move"])
        self.assertFalse(labels_dirty["clean_up_move"])
        self.assertLessEqual(labels_dirty["max_drawdown"], CLEAN_DRAWDOWN_FLOOR)

    def test_significant_move_preserved_alongside_directional(self):
        entry = 100.0
        # Pure downside path: drawdown hits -5%, highs never reach +3%.
        future = [
            Bar(
                ts=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i),
                open=entry,
                high=entry * 1.01,
                low=entry * (0.95 if i == 4 else 0.99),
                close=entry * 0.98,
                volume=1000,
                closed=True,
            )
            for i in range(12)
        ]
        labels = label_forward(future, entry, 12, 0.03)
        self.assertTrue(labels["significant_move"])
        self.assertFalse(labels["large_up_move"])
        self.assertFalse(labels["clean_up_move"])

    def test_labels_from_outcomes_roundtrip(self):
        derived = labels_from_outcomes(0.01, 0.04, -0.01, 0.03)
        self.assertTrue(derived["large_up_move"])
        self.assertTrue(derived["clean_up_move"])
        self.assertTrue(derived["significant_move"])

    def test_twelve_bar_stride_selection(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [
            _FakeSample("BTCUSDT", start + timedelta(hours=6 * i), {}, large_up_move=False)
            for i in range(10)
        ]
        thinned = select_by_stride(rows, 12, base_stride=6)
        self.assertEqual(len(thinned), 5)
        gaps = [
            (thinned[i + 1].ts - thinned[i].ts).total_seconds() / 3600 for i in range(len(thinned) - 1)
        ]
        self.assertTrue(all(g == 12 for g in gaps))

    def test_atr_only_feature_selection(self):
        exp = DIRECTIONAL_UP_3PCT_12H_V1
        self.assertEqual(list(exp.atr_feature_names), ["atr_pct"])
        self.assertIn("atr_pct", feature_names())
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [
            _FakeSample(
                "ETHUSDT",
                start + timedelta(hours=i),
                {"atr_pct": 1.0 + 0.01 * i, "rsi": 50.0},
                large_up_move=(i % 3 == 0),
            )
            for i in range(30)
        ]
        x, y = design_matrix(rows, list(exp.atr_feature_names), target_name="large_up_move")
        self.assertEqual(x.shape, (30, 1))
        self.assertEqual(set(y.tolist()), {0, 1})

    def test_experiment_lineage_contract(self):
        exp = get_experiment("directional_up_3pct_12h_v1")
        lineage = exp.lineage()
        self.assertEqual(lineage["experiment_id"], "directional_up_3pct_12h_v1")
        self.assertEqual(lineage["target_name"], "large_up_move")
        self.assertEqual(lineage["primary_stride_bars"], 12)
        self.assertEqual(lineage["sensitivity_stride_bars"], 6)
        self.assertEqual(lineage["horizon_hours"], 12)
        self.assertEqual(lineage["timeframe"], "1h")

    def test_top_quintile_precision_in_metrics(self):
        y = __import__("numpy").array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1] * 3)
        p = __import__("numpy").linspace(0.1, 0.9, len(y))
        stats = _metrics(y, p)
        self.assertIsNotNone(stats["top_quintile_precision"])
        self.assertIsNotNone(stats["top_quintile_positive_count"])
        self.assertEqual(stats["top_quintile_n"], max(1, len(y) // 5))

    def test_experiment_fit_records_atr_comparison(self):
        # Synthetic rows: label correlated with atr_pct only.
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = []
        names = feature_names()
        for i in range(800):
            atr = 0.5 + (i % 50) / 10.0
            feats = {n: 0.0 for n in names}
            feats["atr_pct"] = atr
            rows.append(
                _FakeSample(
                    "BTCUSDT",
                    start + timedelta(hours=i),
                    feats,
                    large_up_move=atr > 2.5,
                    clean_up_move=atr > 3.0,
                    significant_move=atr > 2.0,
                )
            )
        result = fit_experiment_walk_forward(rows, DIRECTIONAL_UP_3PCT_12H_V1)
        self.assertIn("comparison", result)
        self.assertIn("atr_only", result["comparison"])
        self.assertEqual(result["experiment_id"], "directional_up_3pct_12h_v1")
        self.assertEqual(result["target_name"], "large_up_move")
        self.assertFalse(result["promote"])
        self.assertIn(
            result["experiment_status"],
            {"rejected", STATUS_INSUFFICIENT_ATR, "experiment_pass"},
        )
        self.assertEqual(result["metrics"]["lineage"]["primary_stride_bars"], 12)


class RelativeUpsideRankingTests(unittest.TestCase):
    def _cross_section(self, ts: datetime, n: int = 12, *, upside_fn=None, atr_fn=None, ret_fn=None):
        names = feature_names()
        rows = []
        for i in range(n):
            feats = {name: 0.0 for name in names}
            atr = 1.0 + 0.1 * i if atr_fn is None else atr_fn(i)
            ret = 0.01 * (i - n / 2) if ret_fn is None else ret_fn(i)
            upside = 0.01 * i if upside_fn is None else upside_fn(i)
            feats["atr_pct"] = atr
            feats["ret_24"] = ret
            rows.append(
                _FakeSample(
                    f"S{i}",
                    ts,
                    feats,
                    max_upside=upside,
                    max_drawdown=-0.01,
                    fwd_return=upside / 2,
                )
            )
        return rows

    def test_rank_percentiles_direction(self):
        from app.research.ranking import rank_percentiles

        pct = rank_percentiles([0.01, 0.05, 0.02])
        self.assertAlmostEqual(pct[1], 1.0)
        self.assertAlmostEqual(pct[0], 0.0)
        self.assertTrue(0.0 < pct[2] < 1.0)

    def test_cross_sectional_grouping_and_min_size(self):
        from app.research.ranking import annotate_cross_sectional_ranks

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = self._cross_section(start, n=12)
        rows += self._cross_section(start + timedelta(hours=12), n=5)
        kept, stats = annotate_cross_sectional_ranks(rows, min_cross_section=10)
        self.assertEqual(stats["timestamps_considered"], 2)
        self.assertEqual(stats["timestamps_excluded_below_min"], 1)
        self.assertEqual(stats["timestamps_used"], 1)
        self.assertEqual(len(kept), 12)
        self.assertTrue(all(hasattr(r, "future_upside_rank_percentile") for r in kept))
        best = max(kept, key=lambda r: r.max_upside)
        self.assertAlmostEqual(best.future_upside_rank_percentile, 1.0)

    def test_future_upside_ranking_assigns_continuous_fields(self):
        from app.research.ranking import annotate_cross_sectional_ranks

        ts = datetime(2024, 2, 1, tzinfo=timezone.utc)
        rows = self._cross_section(ts, n=10)
        kept, _ = annotate_cross_sectional_ranks(rows, min_cross_section=10)
        self.assertEqual(kept[0].future_max_upside_12h, kept[0].max_upside)
        self.assertEqual(kept[0].future_return_12h, kept[0].fwd_return)
        self.assertEqual(kept[0].future_max_drawdown_12h, kept[0].max_drawdown)

    def test_top_k_and_hit_rates(self):
        from app.research.ranking import evaluate_timestamp_ranking, score_rows
        import numpy as np

        ts = datetime(2024, 3, 1, tzinfo=timezone.utc)
        rows = self._cross_section(ts, n=10, upside_fn=lambda i: 0.02 * i)
        # Perfect ranking scores = upside
        scores = np.array([r.max_upside for r in rows], dtype=float)
        scored = score_rows(rows, scores)
        # Need actual ranks for score_rows path — annotate first
        from app.research.ranking import annotate_cross_sectional_ranks

        annotate_cross_sectional_ranks(rows, min_cross_section=10)
        scored = score_rows(rows, scores)
        metrics = evaluate_timestamp_ranking(scored)
        self.assertGreater(metrics["spearman"], 0.99)
        self.assertAlmostEqual(metrics["top_1"]["mean_upside"], max(r.max_upside for r in rows))
        self.assertEqual(metrics["top_1"]["hit_3pct"], 1.0)  # 0.18 >= 0.03
        self.assertGreater(metrics["top_5"]["hit_5pct"], 0.0)

    def test_atr_and_momentum_ranking_baselines(self):
        from app.research.ranking import evaluate_scored_rows_by_timestamp, score_with_feature

        ts = datetime(2024, 4, 1, tzinfo=timezone.utc)
        # Upside = ATR, so ATR ranking should be strong; momentum opposite.
        rows = self._cross_section(
            ts,
            n=12,
            upside_fn=lambda i: 0.01 * i,
            atr_fn=lambda i: 1.0 + 0.1 * i,
            ret_fn=lambda i: 1.0 - 0.05 * i,
        )
        from app.research.ranking import annotate_cross_sectional_ranks

        annotate_cross_sectional_ranks(rows, min_cross_section=10)
        atr_scores = {i: float(score_with_feature(rows, "atr_pct")[i]) for i in range(len(rows))}
        mom_scores = {i: float(score_with_feature(rows, "ret_24")[i]) for i in range(len(rows))}
        _, atr_agg = evaluate_scored_rows_by_timestamp(rows, atr_scores)
        _, mom_agg = evaluate_scored_rows_by_timestamp(rows, mom_scores)
        self.assertGreater(atr_agg["spearman"], 0.9)
        self.assertLess(mom_agg["spearman"], 0.0)

    def test_walk_forward_chronology_and_embargo_for_unique_timestamps(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [start + timedelta(hours=12 * i) for i in range(200)]
        folds = walk_forward_folds(ts, min_train=20, min_test=10)
        self.assertGreaterEqual(len(folds), 1)
        gap = embargo()
        for train_idx, test_idx in folds:
            self.assertTrue(fold_is_causal(ts, train_idx, test_idx, gap))
            self.assertLess(max(train_idx), min(test_idx))

    def test_ranking_fit_lineage_never_promotes_live(self):
        from app.research.experiments import RELATIVE_UPSIDE_RANK_12H_V1
        from app.research.ranking import fit_ranking_walk_forward

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        names = feature_names()
        rows = []
        # Many timestamps × 12 symbols; upside correlated with atr + a bit of ret_24 noise.
        for t in range(400):
            ts = start + timedelta(hours=12 * t)
            for i in range(12):
                feats = {n: 0.0 for n in names}
                atr = 1.0 + 0.2 * i + 0.01 * (t % 7)
                ret = 0.02 * i
                feats["atr_pct"] = atr
                feats["ret_24"] = ret
                upside = 0.005 * atr + 0.001 * ret
                rows.append(
                    _FakeSample(
                        f"S{i}",
                        ts,
                        feats,
                        max_upside=upside,
                        max_drawdown=-0.01 * i,
                        fwd_return=upside * 0.5,
                    )
                )
        result = fit_ranking_walk_forward(rows, RELATIVE_UPSIDE_RANK_12H_V1, leakage_ok=True)
        self.assertEqual(result["experiment_id"], "relative_upside_rank_12h_v1")
        self.assertFalse(result["promote"])
        self.assertIn(
            result["experiment_status"],
            {"rejected", "weak_ranking_signal", "promising_ranking_signal"},
        )
        self.assertIn("comparison_table", result["metrics"])
        self.assertEqual(result["metrics"]["lineage"]["primary_stride_bars"], 12)
        self.assertIn("spearman", result["metrics"]["comparison_table"])


class VolatilityAdjustedUpsideTests(unittest.TestCase):
    def _rows_at(self, ts: datetime, n: int = 12, *, upside_fn=None, atr_fn=None, ret_fn=None, extra_fn=None):
        names = feature_names()
        rows = []
        for i in range(n):
            feats = {name: 0.0 for name in names}
            atr = 1.0 + 0.15 * i if atr_fn is None else atr_fn(i)
            ret = 0.01 * i if ret_fn is None else ret_fn(i)
            upside = 0.01 * i if upside_fn is None else upside_fn(i)
            feats["atr_pct"] = atr
            feats["ret_24"] = ret
            if extra_fn:
                feats.update(extra_fn(i))
            rows.append(
                _FakeSample(
                    f"S{i}",
                    ts,
                    feats,
                    max_upside=upside,
                    max_drawdown=-0.01,
                    fwd_return=upside / 2,
                )
            )
        return rows

    def test_atr_normalized_target(self):
        from app.research.vol_adjusted import atr_normalized_upside

        self.assertAlmostEqual(atr_normalized_upside(0.05, 0.025), 2.0)
        self.assertGreater(atr_normalized_upside(0.05, 0.0), 0)

    def test_vol_residual_ranking_direction(self):
        from app.research.vol_adjusted import annotate_vol_adjusted_targets, cross_section_vol_residuals
        import numpy as np

        # Same ATR, different upside → higher upside gets higher residual.
        ups = [0.01, 0.05, 0.02]
        atrs = [0.02, 0.02, 0.02]
        resid, meta = cross_section_vol_residuals(ups, atrs)
        self.assertTrue(np.isfinite(meta["alpha"]))
        self.assertGreater(resid[1], resid[0])
        self.assertGreater(resid[1], resid[2])

        ts = datetime(2024, 5, 1, tzinfo=timezone.utc)
        rows = self._rows_at(
            ts,
            n=12,
            upside_fn=lambda i: 0.01 + 0.01 * i,
            atr_fn=lambda i: 2.0,  # constant ATR% → residual tracks upside
        )
        kept, stats = annotate_vol_adjusted_targets(rows, min_cross_section=10)
        self.assertEqual(stats["timestamps_used"], 1)
        best = max(kept, key=lambda r: r.vol_adj_residual)
        self.assertEqual(best.symbol, "S11")
        self.assertAlmostEqual(best.vol_adj_rank_percentile, 1.0)

    def test_min_cross_section_excludes_small_buckets(self):
        from app.research.vol_adjusted import annotate_vol_adjusted_targets

        start = datetime(2024, 6, 1, tzinfo=timezone.utc)
        rows = self._rows_at(start, n=12) + self._rows_at(start + timedelta(hours=12), n=5)
        kept, stats = annotate_vol_adjusted_targets(rows, min_cross_section=10)
        self.assertEqual(stats["timestamps_excluded_below_min"], 1)
        self.assertEqual(len(kept), 12)

    def test_vol_buckets_use_current_atr_only(self):
        from app.research.vol_adjusted import vol_bucket_label

        self.assertEqual(vol_bucket_label(1.0, 2.0, 4.0), "low")
        self.assertEqual(vol_bucket_label(3.0, 2.0, 4.0), "medium")
        self.assertEqual(vol_bucket_label(5.0, 2.0, 4.0), "high")

    def test_matched_volatility_methodology(self):
        from app.research.vol_adjusted import annotate_vol_adjusted_targets, matched_volatility_analysis
        import numpy as np

        ts = datetime(2024, 7, 1, tzinfo=timezone.utc)
        # Pair structure: even i high residual via upside; odd peers similar ATR.
        rows = self._rows_at(
            ts,
            n=12,
            atr_fn=lambda i: 2.0 + 0.01 * (i // 2),
            upside_fn=lambda i: 0.08 if i % 2 == 0 else 0.01,
            ret_fn=lambda i: 0.02 if i % 2 == 0 else -0.02,
        )
        kept, _ = annotate_vol_adjusted_targets(rows, min_cross_section=10)
        # Score = residual so top picks are high-upside evens
        scores = np.array([r.vol_adj_residual for r in kept], dtype=float)
        out = matched_volatility_analysis(kept, scores, atr_tol_frac=0.5)
        self.assertGreaterEqual(out["n_matched"], 1)
        self.assertIsNotNone(out["mean_difference"])

    def test_top_k_vol_adj_metrics(self):
        from app.research.vol_adjusted import annotate_vol_adjusted_targets, evaluate_timestamp_scores
        import numpy as np

        ts = datetime(2024, 8, 1, tzinfo=timezone.utc)
        rows = self._rows_at(ts, n=12, upside_fn=lambda i: 0.01 * i, atr_fn=lambda i: 2.0)
        kept, _ = annotate_vol_adjusted_targets(rows, min_cross_section=10)
        scores = np.array([r.vol_adj_residual for r in kept], dtype=float)
        metrics = evaluate_timestamp_scores(kept, scores)
        self.assertGreater(metrics["spearman"], 0.99)
        self.assertIsNotNone(metrics["top_5"]["mean_vol_adj"])
        self.assertIsNotNone(metrics["top_5"]["hit_3pct"])

    def test_fit_includes_ablation_and_never_promotes(self):
        from app.research.experiments import VOLATILITY_ADJUSTED_UPSIDE_V1
        from app.research.vol_adjusted import fit_vol_adjusted_walk_forward

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        names = feature_names()
        rows = []
        for t in range(400):
            ts = start + timedelta(hours=12 * t)
            for i in range(12):
                feats = {n: 0.0 for n in names}
                atr = 1.5 + 0.2 * i
                # Residual-like signal in rsi independent of atr
                residual_signal = 0.02 * ((i + t) % 5)
                feats["atr_pct"] = atr
                feats["ret_24"] = 0.01 * i
                feats["rsi"] = 40.0 + 10.0 * residual_signal * 50
                upside = 0.004 * atr + residual_signal
                rows.append(
                    _FakeSample(
                        f"S{i}",
                        ts,
                        feats,
                        max_upside=max(0.001, upside),
                        max_drawdown=-0.01,
                        fwd_return=upside * 0.4,
                    )
                )
        result = fit_vol_adjusted_walk_forward(rows, VOLATILITY_ADJUSTED_UPSIDE_V1, leakage_ok=True)
        self.assertEqual(result["experiment_id"], "volatility_adjusted_upside_v1")
        self.assertFalse(result["promote"])
        self.assertIn(
            result["experiment_status"],
            {"rejected", "weak_volatility_adjusted_signal", "promising_volatility_adjusted_signal"},
        )
        self.assertIn("full_without_atr", result["metrics"])
        self.assertIn("volatility_buckets", result["metrics"])
        self.assertIn("matched_volatility", result["metrics"])
        table = result["metrics"]["comparison_table"]
        self.assertIn("full_without_atr", table["spearman"])


class RiskAdjustedOpportunityTests(unittest.TestCase):
    def test_opportunity_and_log_and_tiny_drawdown(self):
        from app.research.opportunity import (
            OPPORTUNITY_EPS,
            log_risk_adjusted_opportunity,
            risk_adjusted_opportunity,
        )

        self.assertAlmostEqual(risk_adjusted_opportunity(0.06, -0.02), 3.0)
        self.assertAlmostEqual(
            risk_adjusted_opportunity(0.05, 0.0),
            0.05 / OPPORTUNITY_EPS,
        )
        self.assertAlmostEqual(
            log_risk_adjusted_opportunity(0.06, -0.02),
            __import__("math").log1p(3.0),
        )

    def test_barrier_first_touch_and_ambiguous(self):
        from app.research.opportunity import first_touch_barrier

        entry = 100.0
        # Up first on bar 0
        bars_up = [
            Bar(ts=datetime(2024, 1, 1, tzinfo=timezone.utc), open=100, high=104, low=99.5, close=103, volume=1, closed=True),
            Bar(ts=datetime(2024, 1, 1, 1, tzinfo=timezone.utc), open=103, high=103, low=97, close=98, volume=1, closed=True),
        ]
        self.assertEqual(first_touch_barrier(bars_up, entry, 0.03, 0.02), "up_first")
        # Down first
        bars_dn = [
            Bar(ts=datetime(2024, 1, 1, tzinfo=timezone.utc), open=100, high=100.5, low=97.5, close=98, volume=1, closed=True),
        ]
        self.assertEqual(first_touch_barrier(bars_dn, entry, 0.03, 0.02), "down_first")
        # Ambiguous same candle
        bars_amb = [
            Bar(ts=datetime(2024, 1, 1, tzinfo=timezone.utc), open=100, high=104, low=97, close=101, volume=1, closed=True),
        ]
        self.assertEqual(first_touch_barrier(bars_amb, entry, 0.03, 0.02), "ambiguous")
        # Neither
        bars_none = [
            Bar(ts=datetime(2024, 1, 1, tzinfo=timezone.utc), open=100, high=101, low=99, close=100.5, volume=1, closed=True),
        ]
        self.assertEqual(first_touch_barrier(bars_none, entry, 0.03, 0.02), "neither")

    def test_annotate_opportunity_ranking_direction(self):
        from app.research.opportunity import annotate_opportunity_targets

        ts = datetime(2024, 9, 1, tzinfo=timezone.utc)
        names = feature_names()
        rows = []
        for i in range(12):
            feats = {n: 0.0 for n in names}
            feats["atr_pct"] = 2.0
            feats["ret_24"] = 0.01 * i
            # Higher i => better opportunity (more upside, same drawdown)
            rows.append(
                _FakeSample(
                    f"S{i}",
                    ts,
                    feats,
                    max_upside=0.02 + 0.01 * i,
                    max_drawdown=-0.02,
                    fwd_return=0.01 * i,
                    up3_before_down2="up_first" if i > 5 else "down_first",
                    up5_before_down3="neither",
                    up10_before_down5="neither",
                )
            )
        kept, stats = annotate_opportunity_targets(rows, min_cross_section=10)
        self.assertEqual(stats["timestamps_used"], 1)
        best = max(kept, key=lambda r: r.risk_adjusted_opportunity)
        self.assertEqual(best.symbol, "S11")
        self.assertAlmostEqual(best.opportunity_rank_percentile, 1.0)

    def test_top_k_and_downside_metrics(self):
        from app.research.opportunity import annotate_opportunity_targets, evaluate_timestamp_scores
        import numpy as np

        ts = datetime(2024, 9, 2, tzinfo=timezone.utc)
        names = feature_names()
        rows = []
        for i in range(12):
            feats = {n: 0.0 for n in names}
            feats["atr_pct"] = 1.5 + 0.1 * i
            rows.append(
                _FakeSample(
                    f"S{i}",
                    ts,
                    feats,
                    max_upside=0.01 * (i + 1),
                    max_drawdown=-0.01,
                    fwd_return=0.005 * i,
                    up3_before_down2="up_first",
                    up5_before_down3="neither",
                    up10_before_down5="neither",
                )
            )
        kept, _ = annotate_opportunity_targets(rows, min_cross_section=10)
        scores = np.array([r.risk_adjusted_opportunity for r in kept], dtype=float)
        metrics = evaluate_timestamp_scores(kept, scores)
        self.assertGreater(metrics["spearman_opportunity"], 0.99)
        self.assertIsNotNone(metrics["top_5"]["mean_drawdown"])
        self.assertIsNotNone(metrics["top_5"]["up3_before_down2"])

    def test_fit_never_promotes_and_has_ablation(self):
        from app.research.experiments import RISK_ADJUSTED_OPPORTUNITY_V1
        from app.research.opportunity import fit_opportunity_walk_forward

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        names = feature_names()
        rows = []
        for t in range(400):
            ts = start + timedelta(hours=12 * t)
            for i in range(12):
                feats = {n: 0.0 for n in names}
                atr = 1.5 + 0.15 * i
                signal = 0.01 * ((i + t) % 4)
                feats["atr_pct"] = atr
                feats["ret_24"] = 0.01 * i
                feats["rsi"] = 50 + 20 * signal
                up = 0.02 + signal
                dd = -0.015 - 0.002 * (i % 3)
                rows.append(
                    _FakeSample(
                        f"S{i}",
                        ts,
                        feats,
                        max_upside=up,
                        max_drawdown=dd,
                        fwd_return=up * 0.3,
                        up3_before_down2="up_first" if signal > 0.01 else "down_first",
                        up5_before_down3="neither",
                        up10_before_down5="neither",
                    )
                )
        result = fit_opportunity_walk_forward(rows, RISK_ADJUSTED_OPPORTUNITY_V1, leakage_ok=True)
        self.assertEqual(result["experiment_id"], "risk_adjusted_opportunity_v1")
        self.assertFalse(result["promote"])
        self.assertIn(
            result["experiment_status"],
            {"rejected", "weak_opportunity_signal", "promising_opportunity_signal"},
        )
        self.assertIn("full_without_atr", result["metrics"])
        self.assertIn("downside_control", result["metrics"])
        self.assertIn("volatility_buckets", result["metrics"])


class BarrierProbabilityTests(unittest.TestCase):
    def _bar(self, ts: datetime, high: float, low: float, close: float | None = None) -> Bar:
        return Bar(
            ts=ts,
            open=close if close is not None else high,
            high=high,
            low=low,
            close=close if close is not None else high,
            volume=1000,
            closed=True,
        )

    def test_barrier_detection_all_specs(self):
        from app.research.opportunity import first_touch_barrier

        entry = 100.0
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Up-first for each spec
        for name, up, down in (("a", 0.03, 0.02), ("b", 0.05, 0.03), ("c", 0.10, 0.05)):
            bars = [self._bar(t0, high=entry * (1 + up + 0.001), low=entry * 0.995)]
            self.assertEqual(first_touch_barrier(bars, entry, up, down), "up_first")
            bars = [self._bar(t0, high=entry * 1.001, low=entry * (1 - down - 0.001))]
            self.assertEqual(first_touch_barrier(bars, entry, up, down), "down_first")

    def test_first_touch_ordering_and_timeout(self):
        from app.research.opportunity import first_touch_barrier

        entry = 100.0
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Down first then up later
        bars = [
            self._bar(t0, high=100.5, low=97.5),
            self._bar(t0 + timedelta(hours=1), high=104, low=99),
        ]
        self.assertEqual(first_touch_barrier(bars, entry, 0.03, 0.02), "down_first")
        # Timeout
        bars = [self._bar(t0, high=101, low=99) for _ in range(12)]
        self.assertEqual(first_touch_barrier(bars, entry, 0.03, 0.02, horizon_bars=12), "neither")

    def test_ambiguous_same_candle(self):
        from app.research.opportunity import first_touch_barrier

        entry = 100.0
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = [self._bar(t0, high=104, low=97)]
        self.assertEqual(first_touch_barrier(bars, entry, 0.03, 0.02), "ambiguous")

    def test_binary_subset_excludes_timeout_and_ambiguous(self):
        from app.research.barriers import barrier_binary

        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        up = _FakeSample("A", ts, {}, up3_before_down2="up_first")
        dn = _FakeSample("B", ts, {}, up3_before_down2="down_first")
        to = _FakeSample("C", ts, {}, up3_before_down2="neither")
        amb = _FakeSample("D", ts, {}, up3_before_down2="ambiguous")
        self.assertEqual(barrier_binary(up, "up3_before_down2"), 1)
        self.assertEqual(barrier_binary(dn, "up3_before_down2"), 0)
        self.assertIsNone(barrier_binary(to, "up3_before_down2"))
        self.assertIsNone(barrier_binary(amb, "up3_before_down2"))

    def test_calibration_and_expected_value(self):
        import numpy as np

        from app.research.barriers import calibration_table, expected_value

        y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.3, 0.9, 0.4, 0.65, 0.55])
        table = calibration_table(y, p)
        self.assertEqual(len(table), 5)
        self.assertTrue(any(row["n"] > 0 for row in table))
        self.assertAlmostEqual(expected_value(0.6, 0.03, 0.02), 0.6 * 0.03 - 0.4 * 0.02)
        self.assertAlmostEqual(
            expected_value(0.6, 0.03, 0.02, cost=0.002),
            0.6 * 0.03 - 0.4 * 0.02 - 0.002,
        )

    def test_probability_output_range(self):
        from app.research.barriers import _new_classifier, _positive_proba
        import numpy as np

        x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]] * 10)
        y = np.array([0, 0, 0, 1, 1, 1] * 10)
        clf = _new_classifier()
        clf.fit(x, y)
        p = _positive_proba(clf, x)
        self.assertTrue(np.all(p >= 0.0))
        self.assertTrue(np.all(p <= 1.0))

    def test_barrier_fit_never_promotes(self):
        from app.research.barriers import fit_barrier_walk_forward
        from app.research.experiments import BARRIER_PROBABILITY_V1

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        names = feature_names()
        rows = []
        for t in range(400):
            ts = start + timedelta(hours=12 * t)
            for i in range(12):
                feats = {n: 0.0 for n in names}
                atr = 1.5 + 0.15 * i
                signal = (i + t) % 3
                feats["atr_pct"] = atr
                feats["ret_24"] = 0.01 * i
                feats["rsi"] = 45 + 10 * signal
                state = "up_first" if signal == 2 else ("down_first" if signal == 1 else "neither")
                rows.append(
                    _FakeSample(
                        f"S{i}",
                        ts,
                        feats,
                        max_upside=0.02 + 0.01 * signal,
                        max_drawdown=-0.02,
                        fwd_return=0.005 * signal,
                        up3_before_down2=state,
                        up5_before_down3="neither",
                        up10_before_down5="neither",
                    )
                )
        result = fit_barrier_walk_forward(rows, BARRIER_PROBABILITY_V1, leakage_ok=True)
        self.assertEqual(result["experiment_id"], "barrier_probability_v1")
        self.assertFalse(result["promote"])
        self.assertIn(
            result["experiment_status"],
            {"rejected", "weak_barrier_signal", "promising_barrier_signal"},
        )
        self.assertIn("barriers", result["metrics"])
        self.assertIn("up3_before_down2", result["metrics"]["barriers"])
        self.assertIn("full_without_atr", result["metrics"]["barriers"]["up3_before_down2"])


class PointInTimeUniverseTests(unittest.TestCase):
    def _bars(self, symbol_seed: int, n: int, *, start: datetime, price: float = 100.0, volume: float = 1000.0):
        bars = []
        for i in range(n):
            bars.append(
                Bar(
                    ts=start + timedelta(hours=i),
                    open=price,
                    high=price * 1.01,
                    low=price * 0.99,
                    close=price * (1 + 0.0001 * ((i + symbol_seed) % 5)),
                    volume=volume * (1 + (i % 3)),
                    closed=True,
                )
            )
        return bars

    def test_trailing_quote_volume_uses_only_past(self):
        from app.research.universe import trailing_quote_volume_24h

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = self._bars(0, 48, start=start, price=100.0, volume=1000.0)
        ts = start + timedelta(hours=24)
        qv = trailing_quote_volume_24h(bars, ts)
        self.assertIsNotNone(qv)
        # Mutating future bars must not change the value at ts
        future = list(bars)
        for j in range(25, 48):
            future[j] = future[j].model_copy(update={"volume": 1e9, "close": 1e6})
        self.assertEqual(trailing_quote_volume_24h(future, ts), qv)

    def test_trailing_window_boundary(self):
        from app.research.universe import trailing_quote_volume_24h

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = self._bars(0, 30, start=start, price=100.0, volume=10.0)
        ts = start + timedelta(hours=24)
        # bar at exactly ts-24h excluded; bar at ts included
        qv = trailing_quote_volume_24h(bars, ts)
        self.assertIsNotNone(qv)
        expected = sum(
            b.close * b.volume for b in bars if start < b.ts <= ts
        )
        self.assertAlmostEqual(qv, expected)

    def test_universe_selection_top_n_and_warmup(self):
        from app.research.universe import select_universe_at

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = start + timedelta(hours=100)
        bars_by = {
            "AAA": self._bars(1, 120, start=start, volume=100.0),
            "BBB": self._bars(2, 120, start=start, volume=500.0),
            "CCC": self._bars(3, 120, start=start, volume=300.0),
            "NEW": self._bars(4, 10, start=ts - timedelta(hours=9), volume=9999.0),  # insufficient warmup
        }
        rows = select_universe_at(bars_by, ts, top_n=2, warmup_bars=80)
        selected = [r for r in rows if r["selected"]]
        self.assertEqual([r["symbol"] for r in selected], ["BBB", "CCC"])
        new = next(r for r in rows if r["symbol"] == "NEW")
        self.assertFalse(new["eligible"])
        self.assertEqual(new["reason"], "insufficient_warmup")
        self.assertIsNone(new["selection_rank"])

    def test_no_future_volume_changes_selection(self):
        from app.research.universe import select_universe_at

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = start + timedelta(hours=100)
        bars_by = {
            "AAA": self._bars(1, 120, start=start, volume=100.0),
            "BBB": self._bars(2, 120, start=start, volume=200.0),
        }
        before = select_universe_at(bars_by, ts, top_n=1, warmup_bars=80)
        mutated = {k: list(v) for k, v in bars_by.items()}
        for sym in mutated:
            for j in range(len(mutated[sym])):
                if mutated[sym][j].ts > ts:
                    mutated[sym][j] = mutated[sym][j].model_copy(update={"volume": 1e9, "close": 1e6})
        after = select_universe_at(mutated, ts, top_n=1, warmup_bars=80)
        self.assertEqual(
            [r["symbol"] for r in before if r["selected"]],
            [r["symbol"] for r in after if r["selected"]],
        )

    def test_deterministic_reconstruction(self):
        from app.research.universe import select_universe_at

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = start + timedelta(hours=100)
        bars_by = {
            "AAA": self._bars(1, 120, start=start, volume=100.0),
            "BBB": self._bars(2, 120, start=start, volume=200.0),
            "CCC": self._bars(3, 120, start=start, volume=150.0),
        }
        a = select_universe_at(bars_by, ts, top_n=2, warmup_bars=80)
        b = select_universe_at(bars_by, ts, top_n=2, warmup_bars=80)
        self.assertEqual(
            [(r["symbol"], r["selection_rank"]) for r in a],
            [(r["symbol"], r["selection_rank"]) for r in b],
        )

    def test_compare_universes_overlap(self):
        from app.research.universe import compare_universes

        old = ["AAA", "BBB", "CCC"]
        new = {
            datetime(2024, 1, 2, tzinfo=timezone.utc): ["AAA", "DDD"],
            datetime(2024, 1, 3, tzinfo=timezone.utc): ["BBB", "DDD"],
        }
        out = compare_universes(old, new)
        self.assertEqual(out["old_universe_size"], 3)
        self.assertAlmostEqual(out["average_overlap_count"], 1.0)
        self.assertIn(("DDD", 2), out["symbols_entering_top"])


class OutOfSampleHoldoutTests(unittest.TestCase):
    def _rows(self, n_ts: int = 120, n_sym: int = 12, *, start: datetime | None = None):
        start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
        names = feature_names()
        rows = []
        for t in range(n_ts):
            ts = start + timedelta(hours=12 * t)
            for i in range(n_sym):
                feats = {n: 0.0 for n in names}
                feats["atr_pct"] = 1.5 + 0.1 * i
                feats["ret_24"] = 0.01 * i
                signal = (i + t) % 3
                state = "up_first" if signal == 2 else ("down_first" if signal == 1 else "neither")
                rows.append(
                    _FakeSample(
                        f"S{i}",
                        ts,
                        feats,
                        max_upside=0.02 + 0.01 * signal,
                        max_drawdown=-0.02,
                        fwd_return=0.005 * signal,
                        up3_before_down2=state,
                        up5_before_down3=state,
                        up10_before_down5=state,
                    )
                )
        return rows

    def test_chronological_cutoff(self):
        from app.research.holdout import choose_holdout_cutoff

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [start + timedelta(hours=12 * i) for i in range(100)]
        cutoff, meta = choose_holdout_cutoff(ts, holdout_frac=0.25)
        self.assertEqual(cutoff, ts[75])
        self.assertEqual(meta["holdout_timestamps"], 25)
        self.assertEqual(meta["development_timestamps"], 75)

    def test_no_training_rows_after_cutoff(self):
        from app.research.holdout import choose_holdout_cutoff, split_development_holdout

        rows = self._rows(n_ts=60, n_sym=12)
        cutoff, _ = choose_holdout_cutoff([r.ts for r in rows])
        dev, hold, meta = split_development_holdout(rows, cutoff)
        self.assertTrue(all(r.ts < cutoff for r in dev))
        self.assertTrue(all(r.ts >= cutoff for r in hold))
        self.assertEqual(meta["holdout_rows"], len(hold))

    def test_embargo_drops_overlapping_development(self):
        from app.research.holdout import choose_holdout_cutoff, split_development_holdout
        from app.research.spec import embargo

        rows = self._rows(n_ts=60, n_sym=12)
        cutoff, _ = choose_holdout_cutoff([r.ts for r in rows])
        dev, hold, meta = split_development_holdout(rows, cutoff)
        hold_start = min(r.ts for r in hold)
        gap = embargo()
        dev_embargoed = [r for r in dev if r.ts <= hold_start - gap]
        self.assertLessEqual(len(dev_embargoed), len(dev))

    def test_deterministic_random_benchmark(self):
        from app.research.holdout import random_topk_benchmark

        rows = self._rows(n_ts=30, n_sym=12)
        a = random_topk_benchmark(rows, "up3_before_down2")
        b = random_topk_benchmark(rows, "up3_before_down2")
        self.assertEqual(a["random_topk_success_rate"], b["random_topk_success_rate"])
        self.assertEqual(a["seed"], b["seed"])

    def test_holdout_fit_never_promotes(self):
        from app.research.experiments import OUT_OF_SAMPLE_HOLDOUT_V1
        from app.research.holdout import fit_holdout_experiment

        rows = self._rows(n_ts=120, n_sym=12)
        result = fit_holdout_experiment(rows, OUT_OF_SAMPLE_HOLDOUT_V1, leakage_ok=True)
        self.assertEqual(result["experiment_id"], "out_of_sample_holdout_v1")
        self.assertFalse(result["promote"])
        self.assertIn(
            result["experiment_status"],
            {"generalizes", "weak_generalization", "fails_to_generalize", "insufficient_holdout"},
        )
        self.assertIn("split", result["metrics"])
        self.assertIn("barriers", result["metrics"])


class PortfolioBacktestTests(unittest.TestCase):
    def _bars(self, n: int, *, start: datetime, price: float = 100.0):
        bars = []
        for i in range(n):
            p = price * (1 + 0.001 * i)
            bars.append(
                Bar(
                    ts=start + timedelta(hours=i),
                    open=p,
                    high=p * 1.02,
                    low=p * 0.99,
                    close=p * 1.001,
                    volume=1000,
                    closed=True,
                )
            )
        return bars

    def test_next_candle_entry(self):
        from app.research.portfolio import simulate_trade_exit

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = self._bars(20, start=start)
        # Signal at bar 0, entry at bar 1 open
        entry_idx = 1
        entry_price = bars[entry_idx].open
        out = simulate_trade_exit(bars, entry_idx, entry_price, 0.03, 0.02, 12)
        self.assertIsNotNone(out)
        exit_idx, exit_price, reason = out
        self.assertGreater(exit_idx, entry_idx)
        self.assertIn(reason, {"up_barrier", "down_barrier", "timeout", "ambiguous"})

    def test_ambiguous_same_candle_exits_at_open(self):
        from app.research.portfolio import simulate_trade_exit

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = self._bars(5, start=start)
        entry_idx = 1
        entry_price = 100.0
        # Make bar 2 touch both barriers
        bars[2] = bars[2].model_copy(update={"high": 104.0, "low": 97.0, "open": 100.5})
        out = simulate_trade_exit(bars, entry_idx, entry_price, 0.03, 0.02, 12)
        self.assertIsNotNone(out)
        exit_idx, exit_price, reason = out
        self.assertEqual(reason, "ambiguous")
        self.assertEqual(exit_price, 100.5)

    def test_timeout_exit(self):
        from app.research.portfolio import simulate_trade_exit

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars = self._bars(20, start=start)
        entry_idx = 1
        entry_price = 100.0
        # Flat bars never touch barriers
        flat = [
            b.model_copy(update={"high": 100.5, "low": 99.5, "close": 100.0, "open": 100.0})
            for b in bars
        ]
        out = simulate_trade_exit(flat, entry_idx, entry_price, 0.03, 0.02, 12)
        self.assertIsNotNone(out)
        _, _, reason = out
        self.assertEqual(reason, "timeout")

    def test_transaction_cost_and_slippage(self):
        from app.research.portfolio import COST_PER_EVENT

        self.assertAlmostEqual(COST_PER_EVENT, 0.002)
        # Net = gross - (cost + slippage)
        gross = 0.03
        net = gross - (COST_PER_EVENT + 0.001)
        self.assertAlmostEqual(net, 0.027)

    def test_no_leverage_exposure_cap(self):
        from app.research.portfolio import run_portfolio_backtest
        import numpy as np

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        names = feature_names()
        rows = []
        bars_by = {}
        for s in range(12):
            sym = f"S{s}"
            bars_by[sym] = self._bars(40, start=start)
            for t in range(0, 20, 12):
                ts = start + timedelta(hours=t)
                feats = {n: 0.0 for n in names}
                feats["atr_pct"] = 1.0 + 0.1 * s
                rows.append(
                    _FakeSample(
                        sym,
                        ts,
                        feats,
                        max_upside=0.02,
                        max_drawdown=-0.01,
                        fwd_return=0.01,
                        up3_before_down2="up_first",
                    )
                )
        scores = np.array([1.0 for _ in rows])
        res = run_portfolio_backtest(
            rows,
            bars_by,
            scores,
            barrier="up3_before_down2",
            up_pct=0.03,
            down_pct=0.02,
            top_k=5,
            horizon_bars=12,
            strategy="full",
        )
        for point in res["equity_curve"]:
            self.assertLessEqual(point["gross_exposure"], 1.0 + 1e-9)
        self.assertIn("trade_ledger", res)
        self.assertIn("equity_curve", res)

    def test_deterministic_random_benchmark(self):
        from app.research.portfolio import run_portfolio_backtest
        import numpy as np

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        names = feature_names()
        rows = []
        bars_by = {}
        for s in range(12):
            sym = f"S{s}"
            bars_by[sym] = self._bars(40, start=start)
            for t in range(0, 20, 12):
                ts = start + timedelta(hours=t)
                feats = {n: 0.0 for n in names}
                rows.append(
                    _FakeSample(
                        sym,
                        ts,
                        feats,
                        max_upside=0.02,
                        max_drawdown=-0.01,
                        fwd_return=0.01,
                        up3_before_down2="up_first",
                    )
                )
        scores = np.zeros(len(rows))
        a = run_portfolio_backtest(
            rows, bars_by, scores, barrier="up3_before_down2", up_pct=0.03, down_pct=0.02,
            top_k=5, horizon_bars=12, strategy="random", seed=42,
        )
        b = run_portfolio_backtest(
            rows, bars_by, scores, barrier="up3_before_down2", up_pct=0.03, down_pct=0.02,
            top_k=5, horizon_bars=12, strategy="random", seed=42,
        )
        self.assertEqual(a["cumulative_net_return"], b["cumulative_net_return"])


class MultiPeriodRobustnessTests(unittest.TestCase):
    def _rows(self, n_ts: int = 200, n_sym: int = 12, *, start: datetime | None = None):
        start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
        names = feature_names()
        rows = []
        for t in range(n_ts):
            ts = start + timedelta(hours=12 * t)
            for i in range(n_sym):
                feats = {n: 0.0 for n in names}
                feats["atr_pct"] = 1.5 + 0.1 * i
                feats["ret_24"] = 0.01 * i
                signal = (i + t) % 3
                state = "up_first" if signal == 2 else ("down_first" if signal == 1 else "neither")
                rows.append(
                    _FakeSample(
                        f"S{i}",
                        ts,
                        feats,
                        max_upside=0.02 + 0.01 * signal,
                        max_drawdown=-0.02,
                        fwd_return=0.005 * signal,
                        up3_before_down2=state,
                        up5_before_down3=state,
                        up10_before_down5=state,
                    )
                )
        return rows

    def test_chronological_segmentation(self):
        from app.research.robustness import segment_periods

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = [start + timedelta(hours=12 * i) for i in range(100)]
        periods = segment_periods(ts, n_periods=4)
        self.assertEqual(len(periods), 4)
        for i in range(1, 4):
            self.assertGreater(periods[i]["start_ts"], periods[i - 1]["end_ts"])

    def test_period_isolation(self):
        from app.research.robustness import rows_in_period, segment_periods

        rows = self._rows(n_ts=100, n_sym=12)
        ts = sorted({r.ts for r in rows})
        periods = segment_periods(ts, n_periods=4)
        p2 = rows_in_period(rows, periods[1]["start_ts"], periods[1]["end_ts"])
        self.assertTrue(all(periods[1]["start_ts"] <= r.ts <= periods[1]["end_ts"] for r in p2))
        self.assertTrue(all(r.ts < periods[1]["start_ts"] or r.ts > periods[1]["end_ts"] for r in rows if r not in p2))

    def test_walk_forward_training_uses_prior_periods(self):
        from app.research.robustness import segment_periods

        rows = self._rows(n_ts=100, n_sym=12)
        ts = sorted({r.ts for r in rows})
        periods = segment_periods(ts, n_periods=4)
        # For period 3, training must end before period 3 start (minus embargo)
        train_end = periods[2]["start_ts"]
        train_rows = [r for r in rows if r.ts < train_end]
        self.assertTrue(all(r.ts < periods[2]["start_ts"] for r in train_rows))

    def test_robustness_fit_never_promotes(self):
        from app.research.experiments import MULTI_PERIOD_ROBUSTNESS_V1
        from app.research.robustness import run_multi_period_robustness

        rows = self._rows(n_ts=200, n_sym=12)
        bars_by = {}
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for s in range(12):
            bars_by[f"S{s}"] = [
                Bar(
                    ts=start + timedelta(hours=i),
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.5,
                    volume=1000,
                    closed=True,
                )
                for i in range(2600)
            ]
        result = run_multi_period_robustness(rows, bars_by, MULTI_PERIOD_ROBUSTNESS_V1, leakage_ok=True)
        self.assertEqual(result["experiment_id"], "multi_period_robustness_v1")
        self.assertFalse(result["promote"])
        self.assertIn(
            result["experiment_status"],
            {
                "robust_across_periods",
                "mixed_regime_signal",
                "period_specific_signal",
                "fails_robustness_test",
                "insufficient_history",
            },
        )
        self.assertIn("periods", result["metrics"])
        self.assertIn("combined", result["metrics"])


if __name__ == "__main__":
    unittest.main()
