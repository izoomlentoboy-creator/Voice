"""Audio augmentation for training data expansion.

Applies noise injection, pitch shifting, and time stretching
to increase training set size and improve model robustness.
"""

import logging

import librosa
import numpy as np

from . import config

logger = logging.getLogger(__name__)


def add_noise(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """Add Gaussian noise to the audio signal."""
    noise = np.random.randn(len(audio)).astype(audio.dtype) * noise_level
    return audio + noise


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float = 1.0) -> np.ndarray:
    """Shift pitch by n_steps semitones."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)


def time_stretch(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """Stretch/compress audio in time without changing pitch."""
    return librosa.effects.time_stretch(y=audio, rate=rate)


def augment_audio(
    audio: np.ndarray,
    sr: int,
    noise_levels: list[float] | None = None,
    pitch_steps: list[float] | None = None,
    stretch_rates: list[float] | None = None,
) -> list[np.ndarray]:
    """Generate augmented versions of an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        Float32 audio signal.
    sr : int
        Sampling rate.
    noise_levels : list[float], optional
        Noise amplitudes. Default: config values.
    pitch_steps : list[float], optional
        Semitone shifts. Default: config values.
    stretch_rates : list[float], optional
        Time stretch ratios. Default: config values.

    Returns
    -------
    list[np.ndarray]
        Augmented audio variants (does NOT include the original).
    """
    if noise_levels is None:
        noise_levels = config.AUGMENT_NOISE_LEVELS
    if pitch_steps is None:
        pitch_steps = config.AUGMENT_PITCH_STEPS
    if stretch_rates is None:
        stretch_rates = config.AUGMENT_TIME_STRETCH

    augmented = []

    for level in noise_levels:
        augmented.append(add_noise(audio, level))

    for steps in pitch_steps:
        try:
            augmented.append(pitch_shift(audio, sr, steps))
        except Exception as e:
            logger.debug("Pitch shift by %s failed: %s", steps, e)

    for rate in stretch_rates:
        try:
            augmented.append(time_stretch(audio, rate))
        except Exception as e:
            logger.debug("Time stretch by %s failed: %s", rate, e)

    return augmented
