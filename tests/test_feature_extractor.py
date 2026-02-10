"""Tests for feature extraction."""

import numpy as np
import pytest

from voice_disorder_detection.feature_extractor import (
    audio_to_float,
    extract_all_features,
    extract_harmonic_features,
    extract_mfcc_features,
    extract_pitch_features,
    extract_spectral_features,
    extract_temporal_features,
    get_feature_names,
    normalize_audio,
    preprocess_audio,
)


def _make_test_signal(sr=16000, duration=1.0, f0=150.0):
    """Create a synthetic vowel-like signal."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.zeros_like(t, dtype=np.float32)
    for k in range(1, 6):
        signal += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t)
    signal += 0.01 * np.random.randn(len(signal)).astype(np.float32)
    return signal, sr


def _make_int16_signal():
    signal, sr = _make_test_signal()
    return (signal / np.max(np.abs(signal)) * 32767).astype(np.int16), sr


class TestAudioConversion:
    def test_int16_to_float(self):
        audio_int16 = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        result = audio_to_float(audio_int16)
        assert result.dtype == np.float32
        assert -1.0 <= result.min() and result.max() <= 1.0

    def test_float_passthrough(self):
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        result = audio_to_float(audio)
        np.testing.assert_array_equal(result, audio)

    def test_normalize(self):
        audio = np.array([0.0, 0.25, -0.25], dtype=np.float32)
        result = normalize_audio(audio)
        assert np.max(np.abs(result)) == pytest.approx(1.0)


class TestPreprocessing:
    def test_preprocess_int16(self):
        audio, sr = _make_int16_signal()
        result = preprocess_audio(audio, orig_sr=sr)
        assert result.dtype == np.float32
        assert len(result) > 0
        assert np.max(np.abs(result)) <= 1.0 + 1e-6

    def test_preprocess_resamples(self):
        signal, _ = _make_test_signal(sr=44100)
        result = preprocess_audio(signal, orig_sr=44100, target_sr=16000)
        # Should be shorter after downsampling
        assert len(result) < len(signal)


class TestFeatureExtraction:
    def test_mfcc_shape(self):
        signal, sr = _make_test_signal()
        feats = extract_mfcc_features(signal, sr)
        # 13 MFCC + 13 delta + 13 delta2 = 39 coefficients x 6 stats = 234
        assert feats.shape == (234,)

    def test_spectral_shape(self):
        signal, sr = _make_test_signal()
        feats = extract_spectral_features(signal, sr)
        # 4 scalar features + 7 contrast = 11 x 6 stats = 66
        assert feats.shape == (66,)

    def test_temporal_shape(self):
        signal, sr = _make_test_signal()
        feats = extract_temporal_features(signal, sr)
        # ZCR + RMS = 2 x 6 stats = 12
        assert feats.shape == (12,)

    def test_pitch_shape(self):
        signal, sr = _make_test_signal()
        feats = extract_pitch_features(signal, sr)
        assert feats.shape == (7,)

    def test_harmonic_shape(self):
        signal, sr = _make_test_signal()
        feats = extract_harmonic_features(signal, sr)
        assert feats.shape == (3,)

    def test_all_features_shape(self):
        audio, sr = _make_int16_signal()
        feats = extract_all_features(audio, sr)
        assert feats.shape == (322,)
        assert feats.dtype == np.float32

    def test_no_nan_inf(self):
        audio, sr = _make_int16_signal()
        feats = extract_all_features(audio, sr)
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isinf(feats))

    def test_feature_names_match(self):
        names = get_feature_names()
        audio, sr = _make_int16_signal()
        feats = extract_all_features(audio, sr)
        assert len(names) == len(feats)

    def test_short_audio_padded(self):
        # Very short audio should be padded, not crash
        short = np.random.randn(100).astype(np.float32)
        feats = extract_all_features(short, 16000, preprocess=False)
        assert feats.shape == (322,)
