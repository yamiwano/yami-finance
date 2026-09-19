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


if __name__ == "__main__":
    unittest.main()
