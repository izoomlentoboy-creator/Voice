"""TBVoice — FastAPI server application.

Voice disorder screening API that processes uploaded audio recordings
through a trained ML pipeline and returns user-friendly results.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .database import create_tables
from .routes import analyze, feedback, health, history
from .services.predictor import predictor

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tbvoice")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create tables and load ML model. Shutdown: cleanup."""
    logger.info("TBVoice server starting...")
    create_tables()
    logger.info("Database tables created.")

    predictor.load()
    if predictor.is_loaded:
        logger.info("ML model loaded and ready.")
    else:
        logger.warning("ML model NOT available — train the model first.")

    yield

    logger.info("TBVoice server shutting down.")


app = FastAPI(
    title="TBVoice API",
    description=(
        "Voice disorder screening API. Upload vowel recordings "
        "(А, И, У) and receive an AI-powered voice analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS (allow iOS app) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request logging middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed = (time.monotonic() - start) * 1000
    logger.info(
        "%s %s → %d (%.0fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response


# --- Global error handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Внутренняя ошибка сервера. Попробуйте позже.",
        },
    )


# --- Register routes ---
app.include_router(analyze.router, prefix=config.API_PREFIX, tags=["Analysis"])
app.include_router(history.router, prefix=config.API_PREFIX, tags=["History"])
app.include_router(feedback.router, prefix=config.API_PREFIX, tags=["Feedback"])
app.include_router(health.router, prefix=config.API_PREFIX, tags=["Health"])


# --- Root ---
@app.get("/", include_in_schema=False)
def root():
    return {
        "app": "TBVoice",
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"{config.API_PREFIX}/health",
    }
