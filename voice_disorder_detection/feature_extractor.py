"""Audio feature extraction for voice disorder detection.

Converts raw audio signals into feature vectors suitable for classification.
Extracts spectral, cepstral, temporal, and prosodic features.
"""

import numpy as np
import librosa
from typing import Optional

from . import config


def audio_to_float(audio: np.ndarray) -> np.ndarray:
    """Convert int16 audio to float32 in [-1.0, 1.0]."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype in (np.float32, np.float64):
        return audio.astype(np.float32)
    return audio.astype(np.float32) / np.iinfo(audio.dtype).max


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sampling rate."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def trim_silence(audio: np.ndarray, top_db: float = 25.0) -> np.ndarray:
    """Trim leading and trailing silence."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio to [-1.0, 1.0]."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio


def preprocess_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = config.SAMPLE_RATE,
) -> np.ndarray:
    """Full preprocessing pipeline: convert, resample, trim, normalize."""
    audio = audio_to_float(audio)
    audio = resample_audio(audio, orig_sr, target_sr)
    audio = trim_silence(audio)
    audio = normalize_audio(audio)
    return audio


def extract_mfcc_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract MFCC features with delta and delta-delta.

    Returns aggregated statistics over time frames:
    mean, std, min, max, skew, kurtosis for each coefficient.
    """
    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    all_coeffs = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
    return _aggregate_over_time(all_coeffs)


def extract_spectral_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract spectral features: centroid, bandwidth, rolloff, flatness, contrast."""
    features = []

    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    features.append(centroid)

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    features.append(bandwidth)

    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    features.append(rolloff)

    flatness = librosa.feature.spectral_flatness(
        y=audio, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    features.append(flatness)

    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    features.append(contrast)

    combined = np.vstack(features)
    return _aggregate_over_time(combined)


def extract_temporal_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract temporal features: ZCR, RMS energy."""
    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    rms = librosa.feature.rms(
        y=audio, frame_length=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    combined = np.vstack([zcr, rms])
    return _aggregate_over_time(combined)


def extract_pitch_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract pitch-related features: F0, jitter, shimmer approximations."""
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"), sr=sr,
    )

    f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
    if len(f0_clean) == 0:
        f0_clean = np.array([0.0])

    features = []

    # F0 statistics
    features.extend([
        np.mean(f0_clean),
        np.std(f0_clean),
        np.min(f0_clean),
        np.max(f0_clean),
    ])

    # Voiced fraction
    if voiced_flag is not None:
        features.append(np.mean(voiced_flag.astype(float)))
    else:
        features.append(0.0)

    # Jitter approximation (cycle-to-cycle F0 variation)
    if len(f0_clean) > 1:
        periods = 1.0 / np.clip(f0_clean, 1.0, None)
        jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods)
        features.append(jitter)
    else:
        features.append(0.0)

    # Shimmer approximation (amplitude variation between consecutive frames)
    rms = librosa.feature.rms(y=audio, frame_length=config.N_FFT, hop_length=config.HOP_LENGTH)[0]
    if len(rms) > 1:
        shimmer = np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-10)
        features.append(shimmer)
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


def extract_harmonic_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract harmonics-to-noise ratio and harmonic/percussive energy ratio."""
    harmonic, percussive = librosa.effects.hpss(audio)

    h_energy = np.sum(harmonic ** 2)
    p_energy = np.sum(percussive ** 2)
    total_energy = h_energy + p_energy + 1e-10

    hnr = 10 * np.log10(h_energy / (p_energy + 1e-10) + 1e-10)
    h_ratio = h_energy / total_energy
    p_ratio = p_energy / total_energy

    return np.array([hnr, h_ratio, p_ratio], dtype=np.float32)


def extract_all_features(
    audio: np.ndarray,
    sr: int,
    preprocess: bool = True,
    orig_sr: Optional[int] = None,
) -> np.ndarray:
    """Extract the complete feature vector from an audio signal.

    Parameters
    ----------
    audio : np.ndarray
        Raw audio signal (int16 or float).
    sr : int
        Sampling rate of the input audio.
    preprocess : bool
        If True, apply full preprocessing (resample, trim, normalize).
    orig_sr : int, optional
        Original sampling rate (used if different from sr after preprocessing).

    Returns
    -------
    np.ndarray
        1-D feature vector (float32).
    """
    if preprocess:
        audio = preprocess_audio(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        sr = config.SAMPLE_RATE

    # Minimum length check
    min_samples = config.N_FFT + config.HOP_LENGTH
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))

    parts = [
        extract_mfcc_features(audio, sr),
        extract_spectral_features(audio, sr),
        extract_temporal_features(audio, sr),
        extract_pitch_features(audio, sr),
        extract_harmonic_features(audio, sr),
    ]

    feature_vector = np.concatenate(parts)

    # Replace any NaN/Inf with 0
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_vector.astype(np.float32)


def get_feature_names() -> list[str]:
    """Return human-readable names for each feature dimension."""
    names = []
    stats = ["mean", "std", "min", "max", "skew", "kurtosis"]

    # MFCCs (13) + delta (13) + delta2 (13) = 39, each with 6 stats
    for prefix in ["mfcc", "mfcc_d", "mfcc_d2"]:
        for i in range(config.N_MFCC):
            for s in stats:
                names.append(f"{prefix}_{i}_{s}")

    # Spectral: centroid(1) + bandwidth(1) + rolloff(1) + flatness(1) + contrast(7) = 11
    for feat in ["spec_centroid", "spec_bandwidth", "spec_rolloff", "spec_flatness"]:
        for s in stats:
            names.append(f"{feat}_{s}")
    for i in range(7):
        for s in stats:
            names.append(f"spec_contrast_{i}_{s}")

    # Temporal: ZCR(1) + RMS(1) = 2
    for feat in ["zcr", "rms"]:
        for s in stats:
            names.append(f"{feat}_{s}")

    # Pitch: f0_mean, f0_std, f0_min, f0_max, voiced_fraction, jitter, shimmer
    names.extend(["f0_mean", "f0_std", "f0_min", "f0_max",
                   "voiced_fraction", "jitter", "shimmer"])

    # Harmonic: HNR, harmonic_ratio, percussive_ratio
    names.extend(["hnr", "harmonic_ratio", "percussive_ratio"])

    return names


# --- Internal helpers ---

def _aggregate_over_time(features_2d: np.ndarray) -> np.ndarray:
    """Compute statistics over the time axis of a 2D feature matrix.

    For each feature row, computes: mean, std, min, max, skewness, kurtosis.
    """
    from scipy.stats import skew, kurtosis

    result = []
    for row in features_2d:
        result.extend([
            np.mean(row),
            np.std(row),
            np.min(row),
            np.max(row),
            float(skew(row)),
            float(kurtosis(row)),
        ])
    return np.array(result, dtype=np.float32)
