"""FastAPI inference server for voice disorder detection.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /predict         Upload audio file for analysis
    GET  /status          System status
    GET  /health          Health check

NOTE: This is a screening tool, not a substitute for professional diagnosis.
"""

import logging
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from voice_disorder_detection import config
from voice_disorder_detection.pipeline import VoiceDisorderPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pipeline = VoiceDisorderPipeline(
    mode=config.MODE_BINARY,
    download_mode="off",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup so first request doesn't fail."""
    try:
        pipeline._ensure_model_loaded()
        logger.info("Model loaded at startup.")
    except RuntimeError:
        logger.warning("No trained model found. Train a model first.")
    yield


app = FastAPI(
    title="Voice Disorder Detection API",
    description=(
        "Detects voice disorders from audio recordings. "
        "This is a screening tool — not a substitute for professional medical diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionResponse(BaseModel):
    label: int
    diagnosis: str
    confidence: float
    probabilities: dict
    abstain: bool
    abstain_reason: str | None = None
    calibrated: bool = False
    disclaimer: str = (
        "This is an automated screening result. "
        "It is NOT a medical diagnosis. Consult a specialist."
    )


class StatusResponse(BaseModel):
    mode: str
    backend: str
    model_trained: bool
    model_file_exists: bool


@app.get("/health")
def health():
    return {"status": "ok" if pipeline.model.is_trained else "degraded"}


@app.get("/status", response_model=StatusResponse)
def status():
    return {
        "mode": pipeline.mode,
        "backend": pipeline.backend,
        "model_trained": pipeline.model.is_trained,
        "model_file_exists": config.model_path(pipeline.mode, pipeline.backend).exists(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Upload a WAV/FLAC/MP3 file for voice disorder screening."""
    import librosa

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        audio, sr = librosa.load(tmp_path, sr=None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")
    finally:
        if tmp_path:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if len(audio) < 1000:
        raise HTTPException(status_code=400, detail="Audio too short for analysis")

    try:
        result = pipeline.predict_from_audio(audio, sr)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    label = result["label"]
    if pipeline.mode == config.MODE_BINARY:
        diagnosis = "PATHOLOGICAL" if label == 1 else "HEALTHY"
    else:
        diagnosis = f"Class {label}"

    return PredictionResponse(
        label=label,
        diagnosis=diagnosis,
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        abstain=result["abstain"],
        abstain_reason=result.get("abstain_reason"),
        calibrated=result.get("calibrated", False),
    )
