import math
import unittest
from datetime import datetime, timedelta, timezone

from app.domain import Bar
from app.research.backtest import fold_is_causal, walk_forward_folds
from app.research.features import extract_features, feature_names
from app.research.labels import label_forward
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


if __name__ == "__main__":
    unittest.main()
