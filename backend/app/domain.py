from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AssetType(str, Enum):
    CRYPTO = "crypto"
    STOCK = "stock"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    WATCH = "WATCH"
    AVOID = "AVOID"
    NONE = "NONE"


class Timeframe(str, Enum):
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"


class SignalStatus(str, Enum):
    ACTIVE = "active"
    TARGET_1_HIT = "target_1_hit"
    TARGET_2_HIT = "target_2_hit"
    STOPPED = "stopped"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


class StrategyId(str, Enum):
    VOLUME_BREAKOUT = "volume_confirmed_breakout"
    TREND_PULLBACK = "trend_pullback"
    MOMENTUM_CONTINUATION = "momentum_continuation"
    FAILED_BREAKOUT = "failed_breakout"
    PARABOLIC_EXHAUSTION = "parabolic_exhaustion"
    BREAKDOWN_CONTINUATION = "breakdown_continuation"


STRATEGY_LABELS = {
    StrategyId.VOLUME_BREAKOUT: "Volume-confirmed breakout",
    StrategyId.TREND_PULLBACK: "Trend pullback",
    StrategyId.MOMENTUM_CONTINUATION: "Momentum continuation",
    StrategyId.FAILED_BREAKOUT: "Failed breakout",
    StrategyId.PARABOLIC_EXHAUSTION: "Parabolic exhaustion",
    StrategyId.BREAKDOWN_CONTINUATION: "Breakdown continuation",
}

LONG_STRATEGIES = {
    StrategyId.VOLUME_BREAKOUT,
    StrategyId.TREND_PULLBACK,
    StrategyId.MOMENTUM_CONTINUATION,
}
SHORT_STRATEGIES = {
    StrategyId.FAILED_BREAKOUT,
    StrategyId.PARABOLIC_EXHAUSTION,
    StrategyId.BREAKDOWN_CONTINUATION,
}

TF_SECONDS = {Timeframe.M5: 300, Timeframe.M15: 900, Timeframe.H1: 3600}

CLOSED_STATUSES = {
    SignalStatus.TARGET_2_HIT,
    SignalStatus.STOPPED,
    SignalStatus.INVALIDATED,
    SignalStatus.EXPIRED,
}


class Bar(BaseModel):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True


class IndicatorSnapshot(BaseModel):
    ema9: float | None = None
    ema21: float | None = None
    ema50: float | None = None
    vwap: float | None = None
    rsi: float | None = None
    atr: float | None = None
    atr_pct: float | None = None
    volume_sma20: float | None = None
    relative_volume: float | None = None
    swing_high: float | None = None
    swing_low: float | None = None
    resistance: float | None = None
    support: float | None = None


class ScoreComponents(BaseModel):
    technical_structure: float = 0
    volume_activity: float = 0
    market_context: float = 0
    setup_quality: float = 0
    catalyst_context: float = 0
    penalties: float = 0
    penalty_reasons: list[str] = Field(default_factory=list)
    total: float = 0


class StrategyHit(BaseModel):
    fired: bool = False
    direction: Direction = Direction.NONE
    strategy: StrategyId | None = None
    timeframe: Timeframe
    reasons: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    entry_low: float = 0
    entry_high: float = 0
    invalidation: float = 0
    stop: float = 0
    target_1: float = 0
    target_2: float = 0
    risk_reward: float = 0
    structure_quality: float = 0
    setup_quality: float = 0


class Quote(BaseModel):
    symbol: str
    price: float
    change_pct: float = 0
    volume: float = 0
    spread_bps: float = 8
    bid: float | None = None
    ask: float | None = None
    source: str = "unknown"
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
