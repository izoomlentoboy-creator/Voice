"""POST /api/v1/analyze — main analysis endpoint.

Accepts 3 audio files (vowels A, I, U), processes them through the
ML pipeline, and returns a user-friendly result.
"""

import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Analysis, User
from ..schemas import (
    AnalysisDetails,
    AnalysisResponse,
    AnalysisResult,
    CategoryScore,
    ErrorResponse,
)
from ..services.audio_processor import AudioValidationError, validate_and_load
from ..services.interpreter import (
    build_recommendation,
    interpret_features,
    verdict_to_label,
)
from ..services.predictor import predictor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid audio"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    summary="Analyze voice recordings",
    description="Upload 3 vowel recordings (A, I, U) for voice disorder screening.",
)
async def analyze_voice(
    audio_a: UploadFile = File(..., description="Vowel A recording (WAV)"),
    audio_i: UploadFile = File(..., description="Vowel I recording (WAV)"),
    audio_u: UploadFile = File(..., description="Vowel U recording (WAV)"),
    user_id: str = Form(..., description="Device-generated UUID"),
    gender: str = Form(None, description="'m' or 'w'"),
    age: int = Form(None, description="User age"),
    app_version: str = Form(None, description="App version string"),
    device_model: str = Form(None, description="Device model"),
    db: Session = Depends(get_db),
):
    # --- Check model is ready ---
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Модель не загружена. Попробуйте позже.",
        )

    # --- Ensure user exists ---
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        user = User(id=user_id, gender=gender, age=age)
        db.add(user)
        db.commit()

    # --- Validate and load audio files ---
    audio_files = [
        ("А", audio_a),
        ("И", audio_i),
        ("У", audio_u),
    ]

    audio_list = []
    combined_hash_parts = []

    for vowel_name, upload in audio_files:
        try:
            file_bytes = await upload.read()
            audio, sr, sha256 = validate_and_load(file_bytes, upload.filename or "audio.wav")
            audio_list.append((audio, sr))
            combined_hash_parts.append(sha256)
        except AudioValidationError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Ошибка в записи «{vowel_name}»: {e}",
            )

    combined_hash = "|".join(combined_hash_parts)

    # --- Run prediction ---
    try:
        result = predictor.predict(audio_list)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка анализа: {e}",
        )

    prediction = result["prediction"]
    feature_vector = result["feature_vector"]
    ood_result = result["ood_result"]
    processing_time_ms = result["processing_time_ms"]

    # --- Determine verdict ---
    label = prediction["label"]
    if prediction["abstain"]:
        verdict = "abstain"
    elif label == 1:
        verdict = "pathological"
    else:
        verdict = "healthy"

    confidence = prediction["confidence"]
    verdict_label = verdict_to_label(verdict)

    # --- Interpret features into categories ---
    ref_stats = predictor.get_ref_stats()
    categories = interpret_features(feature_vector, ref_stats)

    # --- Build recommendation ---
    recommendation = build_recommendation(
        verdict, categories, prediction["abstain"], confidence,
    )

    # --- Build response ---
    details = AnalysisDetails(
        pitch_stability=CategoryScore(
            status=categories["pitch_stability"].status,
            label=categories["pitch_stability"].label,
            score=categories["pitch_stability"].score,
        ),
        harmonic_quality=CategoryScore(
            status=categories["harmonic_quality"].status,
            label=categories["harmonic_quality"].label,
            score=categories["harmonic_quality"].score,
        ),
        voice_steadiness=CategoryScore(
            status=categories["voice_steadiness"].status,
            label=categories["voice_steadiness"].label,
            score=categories["voice_steadiness"].score,
        ),
        spectral_clarity=CategoryScore(
            status=categories["spectral_clarity"].status,
            label=categories["spectral_clarity"].label,
            score=categories["spectral_clarity"].score,
        ),
        breath_support=CategoryScore(
            status=categories["breath_support"].status,
            label=categories["breath_support"].label,
            score=categories["breath_support"].score,
        ),
    )

    ood_warning = bool(ood_result and ood_result.get("ood", False))

    # --- Save to database ---
    category_scores_json = json.dumps({
        cat: {"status": r.status, "label": r.label, "score": r.score}
        for cat, r in categories.items()
    }, ensure_ascii=False)

    analysis = Analysis(
        user_id=user_id,
        verdict=verdict,
        verdict_label=verdict_label,
        confidence=confidence,
        calibrated_probability=prediction["probabilities"].get("1"),
        abstain=prediction["abstain"],
        ood_warning=ood_warning,
        category_scores=category_scores_json,
        recommendation=recommendation,
        feature_vector=feature_vector.tobytes(),
        audio_hash=combined_hash,
        app_version=app_version,
        device_model=device_model,
        model_version=result.get("model_version"),
        processing_time_ms=processing_time_ms,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    logger.info(
        "Analysis %s: verdict=%s confidence=%.2f ood=%s time=%dms",
        analysis.id, verdict, confidence, ood_warning, processing_time_ms,
    )

    return AnalysisResponse(
        analysis_id=analysis.id,
        result=AnalysisResult(
            verdict=verdict,
            verdict_label=verdict_label,
            confidence=round(confidence, 4),
            confidence_percent=int(confidence * 100),
            abstain=prediction["abstain"],
            calibrated_probability=prediction["probabilities"].get("1"),
        ),
        details=details,
        recommendation=recommendation,
        ood_warning=ood_warning,
        processing_time_ms=processing_time_ms,
    )
