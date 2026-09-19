import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.types import GUID, JSONType


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("assets.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    asset_type: Mapped[str] = mapped_column(String(16), index=True)
    direction: Mapped[str] = mapped_column(String(16), index=True)
    strategy: Mapped[str] = mapped_column(String(64), index=True)
    timeframe: Mapped[str] = mapped_column(String(8), index=True)
    status: Mapped[str] = mapped_column(String(24), index=True, default="active")
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    score: Mapped[float] = mapped_column(Float)
    current_price: Mapped[float] = mapped_column(Float)
    entry_low: Mapped[float] = mapped_column(Float)
    entry_high: Mapped[float] = mapped_column(Float)
    invalidation: Mapped[float] = mapped_column(Float)
    stop: Mapped[float] = mapped_column(Float)
    target_1: Mapped[float] = mapped_column(Float)
    target_2: Mapped[float] = mapped_column(Float)
    risk_reward: Mapped[float] = mapped_column(Float)
    r_multiple: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    score_components: Mapped[dict] = mapped_column(JSONType, default=dict)
    risk_flags: Mapped[list] = mapped_column(JSONType, default=list)
    reasons: Mapped[list] = mapped_column(JSONType, default=list)
    evidence: Mapped[dict] = mapped_column(JSONType, default=dict)
    snapshot: Mapped[dict] = mapped_column(JSONType, default=dict)
    ai_explanation: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    events: Mapped[list["SignalEvent"]] = relationship(back_populates="signal", cascade="all, delete-orphan")


class SignalEvent(Base):
    __tablename__ = "signal_events"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    signal_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("signals.id"), index=True)
    status: Mapped[str] = mapped_column(String(24))
    price: Mapped[float] = mapped_column(Float)
    note: Mapped[str] = mapped_column(Text, default="")
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    signal: Mapped[Signal] = relationship(back_populates="events")
