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


if __name__ == "__main__":
    unittest.main()
