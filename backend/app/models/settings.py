from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.types import JSONType


class AppSettings(Base):
    __tablename__ = "app_settings"

    id: Mapped[str] = mapped_column(String(16), primary_key=True, default="default")
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
