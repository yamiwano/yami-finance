import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.types import GUID, JSONType


class ResearchSample(Base):
    """Market state at T plus labels computed from bars after T."""

    __tablename__ = "research_samples"
    __table_args__ = (UniqueConstraint("symbol", "timeframe", "ts", "horizon_hours", name="uq_research_sample"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    asset_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("assets.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    horizon_hours: Mapped[int] = mapped_column(Integer)
    features: Mapped[dict] = mapped_column(JSONType, default=dict)
    fwd_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_upside: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    significant_move: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    labeled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True, default="candidate")
    timeframe: Mapped[str] = mapped_column(String(8))
    horizon_hours: Mapped[int] = mapped_column(Integer)
    feature_names: Mapped[list] = mapped_column(JSONType, default=list)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    notes: Mapped[str] = mapped_column(String(512), default="")
    blob: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("model_id", "symbol", "timeframe", "ts", name="uq_prediction_asof"),)

    id: Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("model_versions.id"), index=True)
    asset_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("assets.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    p_significant: Mapped[float] = mapped_column(Float)
    features: Mapped[dict] = mapped_column(JSONType, default=dict)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    fwd_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_upside: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    significant_move: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
