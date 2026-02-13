"""GET /api/v1/history/{user_id} — analysis history.

Requires X-Device-Key header matching the user_id to prevent
unauthorized access to other users' analysis history (IDOR protection).
"""

import json
import logging
import re

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Analysis
from ..schemas import (
    AnalysisDetails,
    AnalysisHistoryItem,
    AnalysisResponse,
    AnalysisResult,
    CategoryScore,
    HistoryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# UUID v4 pattern for input validation
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _validate_user_id(user_id: str) -> str:
    """Validate user_id format to prevent injection."""
    if not _UUID_RE.match(user_id):
        raise HTTPException(status_code=400, detail="Некорректный формат user_id")
    return user_id


@router.get(
    "/history/{user_id}",
    response_model=HistoryResponse,
    summary="Get analysis history for a user",
)
def get_history(
    user_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    x_device_key: str = Header(None, alias="X-Device-Key"),
    db: Session = Depends(get_db),
):
    _validate_user_id(user_id)

    # Ownership check: the requesting device must match the user_id
    if x_device_key is None or x_device_key != user_id:
        raise HTTPException(
            status_code=403,
            detail="Доступ запрещён. Недействительный ключ устройства.",
        )

    total = db.query(Analysis).filter(Analysis.user_id == user_id).count()
    analyses = (
        db.query(Analysis)
        .filter(Analysis.user_id == user_id)
        .order_by(Analysis.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    items = [
        AnalysisHistoryItem(
            analysis_id=a.id,
            timestamp=a.timestamp,
            verdict=a.verdict,
            verdict_label=a.verdict_label,
            confidence=a.confidence,
            confidence_percent=int(a.confidence * 100),
        )
        for a in analyses
    ]

    return HistoryResponse(user_id=user_id, total=total, analyses=items)


@router.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get a specific analysis result",
)
def get_analysis(
    analysis_id: str,
    x_device_key: str = Header(None, alias="X-Device-Key"),
    db: Session = Depends(get_db),
):
    if not _UUID_RE.match(analysis_id):
        raise HTTPException(status_code=400, detail="Некорректный формат analysis_id")

    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if analysis is None:
        raise HTTPException(status_code=404, detail="Анализ не найден")

    # Ownership check: only the device that created the analysis can access it
    if x_device_key is None or x_device_key != analysis.user_id:
        raise HTTPException(
            status_code=403,
            detail="Доступ запрещён. Недействительный ключ устройства.",
        )

    # Parse stored category scores
    default_cat = CategoryScore(status="normal", label="—", score=0.75)
    if analysis.category_scores:
        try:
            cats = json.loads(analysis.category_scores)
            details = AnalysisDetails(
                pitch_stability=CategoryScore(**cats.get("pitch_stability", {})) if "pitch_stability" in cats else default_cat,
                harmonic_quality=CategoryScore(**cats.get("harmonic_quality", {})) if "harmonic_quality" in cats else default_cat,
                voice_steadiness=CategoryScore(**cats.get("voice_steadiness", {})) if "voice_steadiness" in cats else default_cat,
                spectral_clarity=CategoryScore(**cats.get("spectral_clarity", {})) if "spectral_clarity" in cats else default_cat,
                breath_support=CategoryScore(**cats.get("breath_support", {})) if "breath_support" in cats else default_cat,
            )
        except Exception:
            details = AnalysisDetails(
                pitch_stability=default_cat, harmonic_quality=default_cat,
                voice_steadiness=default_cat, spectral_clarity=default_cat,
                breath_support=default_cat,
            )
    else:
        details = AnalysisDetails(
            pitch_stability=default_cat, harmonic_quality=default_cat,
            voice_steadiness=default_cat, spectral_clarity=default_cat,
            breath_support=default_cat,
        )

    return AnalysisResponse(
        analysis_id=analysis.id,
        result=AnalysisResult(
            verdict=analysis.verdict,
            verdict_label=analysis.verdict_label,
            confidence=analysis.confidence,
            confidence_percent=int(analysis.confidence * 100),
            abstain=analysis.abstain,
            calibrated_probability=analysis.calibrated_probability,
        ),
        details=details,
        recommendation=analysis.recommendation or "",
        ood_warning=analysis.ood_warning,
        processing_time_ms=analysis.processing_time_ms or 0,
    )
