"""Pydantic models for API request/response validation."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# ------------------------------------------------------------------
# Requests
# ------------------------------------------------------------------

class UserCreate(BaseModel):
    """Sent once on first app launch."""
    user_id: str = Field(..., description="Device-generated UUID")
    gender: Optional[str] = Field(None, pattern="^[mw]$")
    age: Optional[int] = Field(None, ge=1, le=120)
    smoking: Optional[bool] = None
    complaints: Optional[str] = None


class FeedbackCreate(BaseModel):
    """User submits correction after doctor visit."""
    analysis_id: str
    user_id: str
    actual_diagnosis: Optional[str] = None
    notes: Optional[str] = None


# ------------------------------------------------------------------
# Responses
# ------------------------------------------------------------------

class CategoryScore(BaseModel):
    """One of five voice quality categories."""
    status: str  # "normal", "attention", "concern"
    label: str   # user-facing Russian label
    score: float = Field(..., ge=0.0, le=1.0)


class AnalysisResult(BaseModel):
    """Core prediction result."""
    verdict: str           # "healthy", "pathological", "abstain"
    verdict_label: str     # "Норма", "Внимание", "Неопределённо"
    confidence: float
    confidence_percent: int
    abstain: bool
    calibrated_probability: Optional[float] = None


class AnalysisDetails(BaseModel):
    """Five user-friendly voice quality categories."""
    pitch_stability: CategoryScore
    harmonic_quality: CategoryScore
    voice_steadiness: CategoryScore
    spectral_clarity: CategoryScore
    breath_support: CategoryScore


class AnalysisResponse(BaseModel):
    """Full response to POST /analyze."""
    status: str = "success"
    analysis_id: str
    result: AnalysisResult
    details: AnalysisDetails
    recommendation: str
    ood_warning: bool = False
    disclaimer: str = (
        "Результат носит информационный характер "
        "и не является медицинским диагнозом."
    )
    processing_time_ms: int


class AnalysisHistoryItem(BaseModel):
    """One item in the history list."""
    analysis_id: str
    timestamp: datetime
    verdict: str
    verdict_label: str
    confidence: float
    confidence_percent: int


class HistoryResponse(BaseModel):
    """Response to GET /history/{user_id}."""
    user_id: str
    total: int
    analyses: list[AnalysisHistoryItem]


class FeedbackResponse(BaseModel):
    """Response to POST /feedback."""
    status: str = "recorded"
    feedback_id: str
    message: str = "Спасибо! Ваш отзыв поможет улучшить точность анализа."


class HealthResponse(BaseModel):
    """Response to GET /health."""
    status: str = "ok"
    model_loaded: bool
    model_backend: str
    database: str = "connected"


class StatusResponse(BaseModel):
    """Response to GET /status."""
    version: str
    model_backend: str
    model_mode: str
    model_trained: bool
    total_analyses: int
    total_users: int
    total_feedback: int
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error response."""
    status: str = "error"
    message: str
    detail: Optional[str] = None
