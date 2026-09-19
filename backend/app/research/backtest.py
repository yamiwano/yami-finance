"""Time-based walk-forward splits. Future bars never enter training."""

from __future__ import annotations

from datetime import datetime, timedelta

from app.research.history import as_utc
from app.research.spec import MIN_FOLDS, MIN_TEST_SAMPLES, MIN_TRAIN_SAMPLES, TEST_SPAN, TRAIN_SPAN, embargo


def walk_forward_folds(
    timestamps: list[datetime],
    *,
    train_span: timedelta = TRAIN_SPAN,
    test_span: timedelta = TEST_SPAN,
    embargo_span: timedelta | None = None,
    min_train: int = MIN_TRAIN_SAMPLES,
    min_test: int = MIN_TEST_SAMPLES,
) -> list[tuple[list[int], list[int]]]:
    """Expanding train window, then embargo equal to the label horizon, then a test window."""
    if not timestamps:
        return []
    ts = [as_utc(t) for t in timestamps]
    gap = embargo() if embargo_span is None else embargo_span
    start = ts[0]
    end = ts[-1]
    available = end - start
    if available < train_span + gap + test_span:
        train_span = max(timedelta(hours=48), available * 0.55)
        test_span = max(timedelta(hours=18), available * 0.18)
    folds: list[tuple[list[int], list[int]]] = []
    cut = start + train_span
    while True:
        test_start = cut + gap
        test_end = test_start + test_span
        if test_end > end:
            break
        train_idx = [i for i, t in enumerate(ts) if start <= t < cut]
        test_idx = [i for i, t in enumerate(ts) if test_start <= t < test_end]
        if len(train_idx) >= min_train and len(test_idx) >= min_test:
            folds.append((train_idx, test_idx))
        cut += test_span
    return folds


def fold_is_causal(timestamps: list[datetime], train_idx: list[int], test_idx: list[int], embargo_span: timedelta) -> bool:
    if not train_idx or not test_idx:
        return False
    ts = [as_utc(t) for t in timestamps]
    train_max = max(ts[i] for i in train_idx)
    test_min = min(ts[i] for i in test_idx)
    return test_min >= train_max + embargo_span


def enough_folds(n: int) -> bool:
    return n >= MIN_FOLDS
