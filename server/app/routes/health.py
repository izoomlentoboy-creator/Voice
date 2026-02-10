"""Health and status endpoints."""

import logging
import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import config
from ..database import get_db
from ..models import Analysis, Feedback, User
from ..schemas import HealthResponse, StatusResponse
from ..services.predictor import predictor

logger = logging.getLogger(__name__)

router = APIRouter()

_start_time = time.monotonic()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
def health_check():
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_backend=config.MODEL_BACKEND,
    )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Server status with statistics",
)
def server_status(db: Session = Depends(get_db)):
    return StatusResponse(
        version="1.0.0",
        model_backend=config.MODEL_BACKEND,
        model_mode=config.MODEL_MODE,
        model_trained=predictor.is_loaded,
        total_analyses=db.query(Analysis).count(),
        total_users=db.query(User).count(),
        total_feedback=db.query(Feedback).count(),
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )
