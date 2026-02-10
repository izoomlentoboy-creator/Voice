"""POST /api/v1/feedback — user correction after doctor visit."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Analysis, Feedback
from ..schemas import FeedbackCreate, FeedbackResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback/correction for an analysis",
)
def submit_feedback(
    data: FeedbackCreate,
    db: Session = Depends(get_db),
):
    # Verify analysis exists
    analysis = db.query(Analysis).filter(Analysis.id == data.analysis_id).first()
    if analysis is None:
        raise HTTPException(status_code=404, detail="Анализ не найден")

    feedback = Feedback(
        analysis_id=data.analysis_id,
        user_id=data.user_id,
        actual_diagnosis=data.actual_diagnosis,
        notes=data.notes,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    logger.info(
        "Feedback %s for analysis %s: diagnosis=%s",
        feedback.id, data.analysis_id, data.actual_diagnosis,
    )

    return FeedbackResponse(feedback_id=feedback.id)
