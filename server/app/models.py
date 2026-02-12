"""SQLAlchemy ORM models for TBVoice."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    """Anonymous user profile (one per device)."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    gender: Mapped[str | None] = mapped_column(String(1))  # 'm' / 'w'
    age: Mapped[int | None] = mapped_column(Integer)
    smoking: Mapped[bool | None] = mapped_column(Boolean)
    complaints: Mapped[str | None] = mapped_column(Text)  # comma-separated
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    analyses: Mapped[list["Analysis"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Analysis(Base):
    """Single voice analysis session."""

    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    # Result
    verdict: Mapped[str] = mapped_column(String(20))  # 'healthy', 'pathological', 'abstain'
    verdict_label: Mapped[str] = mapped_column(String(50))  # user-facing label
    confidence: Mapped[float] = mapped_column(Float)
    calibrated_probability: Mapped[float | None] = mapped_column(Float)
    abstain: Mapped[bool] = mapped_column(Boolean, default=False)
    ood_warning: Mapped[bool] = mapped_column(Boolean, default=False)

    # Category scores (JSON string)
    category_scores: Mapped[str | None] = mapped_column(Text)

    # Recommendation text
    recommendation: Mapped[str | None] = mapped_column(Text)

    # Raw data
    feature_vector: Mapped[bytes | None] = mapped_column(LargeBinary)
    audio_hash: Mapped[str | None] = mapped_column(String(64))

    # Client metadata
    app_version: Mapped[str | None] = mapped_column(String(20))
    device_model: Mapped[str | None] = mapped_column(String(100))

    # Server metadata
    model_version: Mapped[str | None] = mapped_column(String(100))
    processing_time_ms: Mapped[int | None] = mapped_column(Integer)

    user: Mapped["User"] = relationship(back_populates="analyses")
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")


class Feedback(Base):
    """User feedback / correction after visiting a doctor."""

    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(String(36), ForeignKey("analyses.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    actual_diagnosis: Mapped[str | None] = mapped_column(String(50))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    applied: Mapped[bool] = mapped_column(Boolean, default=False)

    analysis: Mapped["Analysis"] = relationship(back_populates="feedback")
    user: Mapped["User"] = relationship(back_populates="feedback")
