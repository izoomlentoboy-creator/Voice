"""Tests for the API endpoints (using FastAPI TestClient)."""

import io
import struct

import numpy as np
from fastapi.testclient import TestClient

from server.app.database import Base, engine
from server.app.main import app

# Create test tables
Base.metadata.create_all(bind=engine)

client = TestClient(app)

# A valid UUID for testing
_TEST_USER_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


def _make_wav_bytes(duration_sec: float = 2.0, sr: int = 16000) -> bytes:
    """Generate a valid WAV file as bytes (sine wave)."""
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, dtype=np.float32)
    # 220 Hz sine wave with some noise
    audio = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.random.randn(n_samples)).astype(np.float32)

    # Build WAV manually (16-bit PCM)
    audio_int16 = (audio * 32767).astype(np.int16)
    data_bytes = audio_int16.tobytes()

    wav = io.BytesIO()
    # RIFF header
    wav.write(b"RIFF")
    wav.write(struct.pack("<I", 36 + len(data_bytes)))
    wav.write(b"WAVE")
    # fmt chunk
    wav.write(b"fmt ")
    wav.write(struct.pack("<I", 16))           # chunk size
    wav.write(struct.pack("<H", 1))            # PCM
    wav.write(struct.pack("<H", 1))            # mono
    wav.write(struct.pack("<I", sr))           # sample rate
    wav.write(struct.pack("<I", sr * 2))       # byte rate
    wav.write(struct.pack("<H", 2))            # block align
    wav.write(struct.pack("<H", 16))           # bits per sample
    # data chunk
    wav.write(b"data")
    wav.write(struct.pack("<I", len(data_bytes)))
    wav.write(data_bytes)

    return wav.getvalue()


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert "model_loaded" in data

    def test_status_returns_stats(self):
        resp = client.get("/api/v1/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "total_analyses" in data
        assert "uptime_seconds" in data


class TestRootEndpoint:
    def test_root(self):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["app"] == "TBVoice"


class TestHistoryEndpoint:
    def test_empty_history_with_auth(self):
        """History for a valid UUID with correct device key should return empty list."""
        resp = client.get(
            f"/api/v1/history/{_TEST_USER_ID}",
            headers={"X-Device-Key": _TEST_USER_ID},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["analyses"] == []

    def test_history_without_auth_returns_403(self):
        """History without X-Device-Key header should return 403."""
        resp = client.get(f"/api/v1/history/{_TEST_USER_ID}")
        assert resp.status_code == 403

    def test_history_wrong_device_key_returns_403(self):
        """History with mismatched device key should return 403."""
        wrong_key = "00000000-0000-0000-0000-000000000000"
        resp = client.get(
            f"/api/v1/history/{_TEST_USER_ID}",
            headers={"X-Device-Key": wrong_key},
        )
        assert resp.status_code == 403

    def test_history_invalid_user_id_format_returns_400(self):
        """Non-UUID user_id should return 400."""
        resp = client.get(
            "/api/v1/history/not-a-uuid",
            headers={"X-Device-Key": "not-a-uuid"},
        )
        assert resp.status_code == 400


class TestFeedbackEndpoint:
    def test_feedback_nonexistent_analysis(self):
        resp = client.post("/api/v1/feedback", json={
            "analysis_id": "nonexistent",
            "user_id": _TEST_USER_ID,
            "actual_diagnosis": "healthy",
        })
        assert resp.status_code == 404


class TestAnalyzeEndpoint:
    def test_analyze_without_model_returns_503(self):
        """If model is not loaded, /analyze should return 503."""
        from server.app.services.predictor import predictor
        if predictor.is_loaded:
            return  # Skip if model happens to be loaded

        wav = _make_wav_bytes()
        resp = client.post(
            "/api/v1/analyze",
            files={
                "audio_a": ("a.wav", wav, "audio/wav"),
                "audio_i": ("i.wav", wav, "audio/wav"),
                "audio_u": ("u.wav", wav, "audio/wav"),
            },
            data={"user_id": _TEST_USER_ID},
        )
        assert resp.status_code == 503

    def test_analyze_with_invalid_audio(self):
        """Garbage bytes should return 400."""
        from server.app.services.predictor import predictor
        if not predictor.is_loaded:
            return  # Can't test this without model

        resp = client.post(
            "/api/v1/analyze",
            files={
                "audio_a": ("a.wav", b"not audio", "audio/wav"),
                "audio_i": ("i.wav", b"not audio", "audio/wav"),
                "audio_u": ("u.wav", b"not audio", "audio/wav"),
            },
            data={"user_id": _TEST_USER_ID},
        )
        assert resp.status_code == 400
