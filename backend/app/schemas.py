from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class AssetOut(BaseModel):
    id: UUID
    symbol: str
    name: str
    asset_type: str
    exchange: str
    is_active: bool = True


class QuoteOut(BaseModel):
    symbol: str
    price: float
    change_pct: float
    volume: float
    spread_bps: float
    bid: float | None = None
    ask: float | None = None
    source: str = "unknown"
    ts: datetime


class SignalOut(BaseModel):
    id: UUID
    asset_id: UUID
    symbol: str
    asset_type: str
    direction: str
    strategy: str
    strategy_label: str | None = None
    timeframe: str
    status: str
    detected_at: datetime
    closed_at: datetime | None = None
    score: float
    current_price: float
    entry_low: float
    entry_high: float
    invalidation: float
    stop: float
    target_1: float
    target_2: float
    risk_reward: float
    r_multiple: float | None = None
    exit_price: float | None = None
    score_components: dict[str, Any] = Field(default_factory=dict)
    risk_flags: list[Any] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    snapshot: dict[str, Any] = Field(default_factory=dict)
    ai_explanation: dict[str, Any] | None = None


class CandleOut(BaseModel):
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class ChartOut(BaseModel):
    symbol: str
    timeframe: str
    candles: list[CandleOut]
    ema9: list[float | None]
    ema21: list[float | None]
    ema50: list[float | None]
    vwap: list[float | None]
    rsi: list[float | None]


class SettingsIn(BaseModel):
    min_score: float | None = None
    enabled_strategies: list[str] | None = None
    enabled_timeframes: list[str] | None = None
    learn_from_outcomes: bool | None = None


class WatchlistIn(BaseModel):
    symbol: str
    notes: str = ""
