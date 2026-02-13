"""TBVoice — FastAPI server application.

Voice disorder screening API that processes uploaded audio recordings
through a trained ML pipeline and returns user-friendly results.
"""

import collections
import logging
import os
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

# --- CORS ---
# Configurable via environment variable. Defaults to localhost for dev safety.
_allowed_origins = os.environ.get("TBVOICE_CORS_ORIGINS", "").split(",")
_allowed_origins = [o.strip() for o in _allowed_origins if o.strip()]
if not _allowed_origins:
    _allowed_origins = ["http://localhost:3000", "http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- Rate limiting middleware ---
_rate_limit_window: dict[str, collections.deque] = {}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce per-client rate limiting based on IP address."""
    if request.url.path.endswith("/health"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_seconds = 60.0

    if client_ip not in _rate_limit_window:
        _rate_limit_window[client_ip] = collections.deque()

    timestamps = _rate_limit_window[client_ip]

    while timestamps and (now - timestamps[0]) > window_seconds:
        timestamps.popleft()

    if len(timestamps) >= config.RATE_LIMIT_PER_MINUTE:
        logger.warning("Rate limit exceeded for %s (%d requests)", client_ip, len(timestamps))
        return JSONResponse(
            status_code=429,
            content={
                "status": "error",
                "message": "Слишком много запросов. Попробуйте через минуту.",
            },
        )

    timestamps.append(now)
    return await call_next(request)


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
