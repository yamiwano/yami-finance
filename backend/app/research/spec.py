"""Frozen research contracts. Change these only with a new model version."""

from __future__ import annotations

from datetime import timedelta

from app.config import get_settings
from app.domain import TF_SECONDS, Timeframe

LOOKBACK_BARS = 80
MIN_TRAIN_SAMPLES = 200
MIN_TEST_SAMPLES = 40
MIN_FOLDS = 2
PROMOTE_MIN_AUC = 0.55
TRAIN_SPAN = timedelta(days=45)
TEST_SPAN = timedelta(days=14)


def timeframe() -> Timeframe:
    return Timeframe(get_settings().research_timeframe)


def horizon_hours() -> int:
    return int(get_settings().research_horizon_hours)


def move_pct() -> float:
    return float(get_settings().research_move_pct)


def sample_stride() -> int:
    return max(1, int(get_settings().research_sample_stride))


def backfill_days() -> int:
    return int(get_settings().research_backfill_days)


def horizon_bars(tf: Timeframe | str | None = None, hours: int | None = None) -> int:
    tf_value = tf.value if isinstance(tf, Timeframe) else (tf or timeframe().value)
    hours = horizon_hours() if hours is None else hours
    sec = TF_SECONDS[Timeframe(tf_value)]
    return max(1, int(hours * 3600 / sec))


def embargo() -> timedelta:
    return timedelta(hours=horizon_hours())
