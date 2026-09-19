import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.types import GUID, JSONType


class PaperAccount(Base):
    __tablename__ = "paper_account"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    cash: Mapped[float] = mapped_column(Float)
    equity: Mapped[float] = mapped_column(Float)
    starting_capital: Mapped[float] = mapped_column(Float)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    total_fees: Mapped[float] = mapped_column(Float, default=0.0)
    total_slippage: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PaperSignal(Base):
    __tablename__ = "paper_signals"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    asset_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("assets.id"), index=True)
    direction: Mapped[str] = mapped_column(String(8))
    model_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("model_versions.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    feature_version: Mapped[str] = mapped_column(String(64))
    experiment_version: Mapped[str] = mapped_column(String(64))
    probability: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)
    barrier: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))
    features: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PaperPosition(Base):
    __tablename__ = "paper_positions"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    signal_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("paper_signals.id"), index=True)
    account_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("paper_account.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    asset_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("assets.id"), index=True)
    direction: Mapped[str] = mapped_column(String(8))
    entry_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    entry_price: Mapped[float] = mapped_column(Float)
    weight: Mapped[float] = mapped_column(Float)
    up_pct: Mapped[float] = mapped_column(Float)
    down_pct: Mapped[float] = mapped_column(Float)
    horizon_bars: Mapped[int] = mapped_column(Integer)
    cost_rate: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="open")
    account_starting_capital: Mapped[float] = mapped_column(Float)
    exit_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    gross_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    signal_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("paper_signals.id"), index=True)
    position_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("paper_positions.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    entry_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    entry_price: Mapped[float] = mapped_column(Float)
    exit_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    exit_price: Mapped[float] = mapped_column(Float)
    exit_reason: Mapped[str] = mapped_column(String(32))
    gross_return: Mapped[float] = mapped_column(Float)
    net_return: Mapped[float] = mapped_column(Float)
    cost: Mapped[float] = mapped_column(Float)
    holding_hours: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
