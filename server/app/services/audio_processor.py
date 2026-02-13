"""Audio upload validation and preprocessing.

Responsibilities:
  - Validate uploaded WAV files (format, duration, signal level)
  - Save to temporary storage
  - Compute SHA-256 hash for deduplication
  - Load and return as numpy array ready for feature extraction
"""

import hashlib
import logging
import tempfile
from pathlib import Path

import librosa
import numpy as np

from .. import config

logger = logging.getLogger(__name__)

# Minimum RMS to consider non-silence
_MIN_RMS_DB = -50.0


class AudioValidationError(Exception):
    """Raised when an uploaded audio file fails validation."""

    pass


def validate_and_load(file_bytes: bytes, filename: str) -> tuple[np.ndarray, int, str]:
    """Validate an uploaded audio file and return (audio, sr, sha256_hash).

    Parameters
    ----------
    file_bytes : bytes
        Raw file content.
    filename : str
        Original filename (for format detection).

    Returns
    -------
    audio : np.ndarray (float32, mono)
    sr : int
    sha256 : str

    Raises
    ------
    AudioValidationError
        If the file fails any validation check.
    """
    # --- Size check ---
    max_bytes = config.AUDIO_MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise AudioValidationError(
            f"Файл слишком большой: {len(file_bytes) / 1024 / 1024:.1f} МБ "
            f"(максимум {config.AUDIO_MAX_FILE_SIZE_MB} МБ)"
        )

    # --- Hash ---
    sha256 = hashlib.sha256(file_bytes).hexdigest()

    # --- Write to temp file and load ---
    suffix = Path(filename).suffix or ".wav"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
    except OSError as e:
        raise AudioValidationError(f"Не удалось сохранить файл: {e}")

    try:
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
    except Exception as e:
        raise AudioValidationError(
            f"Не удалось прочитать аудио. Поддерживаются WAV, FLAC, OGG. Ошибка: {e}"
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # --- Duration check ---
    duration = len(audio) / sr
    if duration < config.AUDIO_MIN_DURATION_SEC:
        raise AudioValidationError(
            f"Запись слишком короткая: {duration:.1f} сек "
            f"(минимум {config.AUDIO_MIN_DURATION_SEC} сек)"
        )
    if duration > config.AUDIO_MAX_DURATION_SEC:
        raise AudioValidationError(
            f"Запись слишком длинная: {duration:.1f} сек "
            f"(максимум {config.AUDIO_MAX_DURATION_SEC} сек)"
        )

    # --- Signal level check (not silence) ---
    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)
    if rms_db < _MIN_RMS_DB:
        raise AudioValidationError(
            f"Запись слишком тихая ({rms_db:.0f} дБ). "
            "Пожалуйста, говорите громче или поднесите телефон ближе."
        )

    # --- Resample to target SR ---
    if sr != config.AUDIO_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.AUDIO_SAMPLE_RATE)
        sr = config.AUDIO_SAMPLE_RATE

    logger.info(
        "Audio validated: %.1f sec, %d Hz, RMS=%.1f dB, hash=%s",
        duration, sr, rms_db, sha256[:12],
    )

    return audio.astype(np.float32), sr, sha256
