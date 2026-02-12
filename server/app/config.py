"""TBVoice server configuration.

All settings are loaded from environment variables with sensible defaults
for local development. In production, set via .env or Docker environment.
"""

import os
from pathlib import Path

# --- Paths ---
SERVER_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SERVER_DIR.parent
ML_DIR = PROJECT_ROOT / "voice_disorder_detection"

# Database
DATABASE_URL = os.environ.get(
    "TBVOICE_DATABASE_URL",
    f"sqlite:///{SERVER_DIR / 'tbvoice.db'}",
)

# Upload storage (temporary WAV files — cleaned up after processing)
UPLOAD_DIR = Path(os.environ.get("TBVOICE_UPLOAD_DIR", SERVER_DIR / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Audio constraints ---
AUDIO_MIN_DURATION_SEC = 0.5
AUDIO_MAX_DURATION_SEC = 10.0
AUDIO_MAX_FILE_SIZE_MB = 5
AUDIO_SAMPLE_RATE = 16000

# --- API ---
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# CORS — comma-separated allowed origins (e.g. "https://myapp.com,http://localhost:3000")
# Empty or unset defaults to localhost-only for development.
_cors_raw = os.environ.get("TBVOICE_CORS_ORIGINS", "")
CORS_ORIGINS: list[str] = [
    o.strip() for o in _cors_raw.split(",") if o.strip()
] or ["http://localhost:3000", "http://localhost:8080"]

# Rate limiting (requests per minute per user)
RATE_LIMIT_PER_MINUTE = int(os.environ.get("TBVOICE_RATE_LIMIT", "10"))

# API key header name
API_KEY_HEADER = "X-Device-Key"

# --- Model ---
# Which ML backend to use for predictions
MODEL_BACKEND = os.environ.get("TBVOICE_MODEL_BACKEND", "ensemble")
MODEL_MODE = os.environ.get("TBVOICE_MODEL_MODE", "binary")

# --- Server ---
HOST = os.environ.get("TBVOICE_HOST", "0.0.0.0")
PORT = int(os.environ.get("TBVOICE_PORT", "8000"))
WORKERS = int(os.environ.get("TBVOICE_WORKERS", "2"))
DEBUG = os.environ.get("TBVOICE_DEBUG", "false").lower() == "true"

# --- Cleanup ---
AUDIO_RETENTION_DAYS = int(os.environ.get("TBVOICE_AUDIO_RETENTION_DAYS", "30"))
