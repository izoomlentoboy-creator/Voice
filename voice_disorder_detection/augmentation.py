"""Audio augmentation for training data expansion.

Applies noise injection, pitch shifting, time stretching,
and SpecAugment-style masking to increase training set size
and improve model robustness.
All operations use seeded random state for reproducibility.
"""

import logging

import librosa
import numpy as np

from . import config

logger = logging.getLogger(__name__)


def add_noise(
    audio: np.ndarray,
    noise_level: float = 0.005,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Add Gaussian noise to the audio signal (reproducible)."""
    if rng is None:
        rng = np.random.RandomState(config.RANDOM_STATE)
    noise = rng.randn(len(audio)).astype(audio.dtype) * noise_level
    return audio + noise


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float = 1.0) -> np.ndarray:
    """Shift pitch by n_steps semitones."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)


def time_stretch(audio: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """Stretch/compress audio in time without changing pitch."""
    return librosa.effects.time_stretch(y=audio, rate=rate)


def spec_augment(
    audio: np.ndarray,
    sr: int,
    n_freq_masks: int = 2,
    n_time_masks: int = 2,
    freq_mask_width: int = 10,
    time_mask_width: int = 15,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Apply SpecAugment-style masking in the spectrogram domain.

    Converts audio to STFT, applies frequency and time masks,
    then converts back. This is more realistic than simple noise
    for voice disorder detection tasks.
    """
    if rng is None:
        rng = np.random.RandomState(config.RANDOM_STATE)

    stft = librosa.stft(audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    n_freq, n_time = stft.shape

    for _ in range(n_freq_masks):
        f_width = rng.randint(1, min(freq_mask_width, n_freq // 4) + 1)
        f_start = rng.randint(0, max(1, n_freq - f_width))
        stft[f_start:f_start + f_width, :] *= 0.0

    for _ in range(n_time_masks):
        t_width = rng.randint(1, min(time_mask_width, n_time // 4) + 1)
        t_start = rng.randint(0, max(1, n_time - t_width))
        stft[:, t_start:t_start + t_width] *= 0.0

    return librosa.istft(stft, hop_length=config.HOP_LENGTH, length=len(audio))


def augment_audio(
    audio: np.ndarray,
    sr: int,
    noise_levels: list[float] | None = None,
    pitch_steps: list[float] | None = None,
    stretch_rates: list[float] | None = None,
    use_spec_augment: bool = True,
    seed: int = config.RANDOM_STATE,
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
    use_spec_augment : bool
        Whether to include SpecAugment-style augmentation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[np.ndarray]
        Augmented audio variants (does NOT include the original).
    """
    rng = np.random.RandomState(seed)

    if noise_levels is None:
        noise_levels = config.AUGMENT_NOISE_LEVELS
    if pitch_steps is None:
        pitch_steps = config.AUGMENT_PITCH_STEPS
    if stretch_rates is None:
        stretch_rates = config.AUGMENT_TIME_STRETCH

    augmented = []

    for level in noise_levels:
        augmented.append(add_noise(audio, level, rng=rng))

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

    if use_spec_augment:
        try:
            augmented.append(spec_augment(audio, sr, rng=rng))
        except Exception as e:
            logger.debug("SpecAugment failed: %s", e)

    return augmented
