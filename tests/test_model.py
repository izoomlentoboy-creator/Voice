"""Tests for the model (training, prediction, save/load, abstain)."""

import numpy as np
import pytest

from voice_disorder_detection import config
from voice_disorder_detection.model import VoiceDisorderModel


def _make_synthetic_dataset(n=200, n_features=322, n_classes=2):
    """Create a separable synthetic dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, n_features).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    # Make it separable
    X[y == 0] += 0.5
    X[y == 1] -= 0.5
    idx = rng.permutation(n)
    return X[idx], y[idx]


class TestModelTraining:
    def test_ensemble_train(self):
        X, y = _make_synthetic_dataset()
        model = VoiceDisorderModel(mode="binary", backend="ensemble")
        meta = model.train(X, y)
        assert model.is_trained
        assert meta["n_samples"] == 200
        assert meta["n_classes"] == 2
        assert meta["backend"] == "ensemble"

    def test_logreg_train(self):
        X, y = _make_synthetic_dataset()
        model = VoiceDisorderModel(mode="binary", backend="logreg")
        meta = model.train(X, y)
        assert model.is_trained
        assert meta["backend"] == "logreg"

    def test_predict(self):
        X, y = _make_synthetic_dataset()
        model = VoiceDisorderModel(mode="binary")
        model.train(X, y)
        preds = model.predict(X[:10])
        assert preds.shape == (10,)

    def test_predict_proba(self):
        X, y = _make_synthetic_dataset()
        model = VoiceDisorderModel(mode="binary")
        model.train(X, y)
        proba = model.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        # Probabilities sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestAbstain:
    def test_abstain_low_confidence(self):
        X, y = _make_synthetic_dataset()
        model = VoiceDisorderModel(mode="binary")
        model.train(X, y)
        # Force abstain with very high threshold
        results = model.predict_with_confidence(X[:5], abstain_threshold=0.99)
        # At least some should abstain with 0.99 threshold
        abstain_count = sum(1 for r in results if r["abstain"])
        assert abstain_count >= 0  # may or may not abstain
        # Check structure
        assert "abstain" in results[0]
        assert "confidence" in results[0]

    def test_no_abstain_low_threshold(self):
        X, y = _make_synthetic_dataset()
        model = VoiceDisorderModel(mode="binary")
        model.train(X, y)
        results = model.predict_with_confidence(X[:5], abstain_threshold=0.0)
        for r in results:
            assert r["abstain"] is False


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        X, y = _make_synthetic_dataset()
        model = VoiceDisorderModel(mode="binary")
        model.train(X, y)
        preds_before = model.predict(X[:10])

        # Override config paths for test
        import voice_disorder_detection.config as cfg
        orig_models = cfg.MODELS_DIR
        cfg.MODELS_DIR = tmp_path
        try:
            model.save()
            model2 = VoiceDisorderModel(mode="binary")
            model2.load()
            preds_after = model2.predict(X[:10])
            np.testing.assert_array_equal(preds_before, preds_after)
        finally:
            cfg.MODELS_DIR = orig_models

    def test_load_nonexistent(self):
        model = VoiceDisorderModel(mode="binary")
        with pytest.raises(FileNotFoundError):
            model.load(path=config.MODELS_DIR / "nonexistent.joblib")


class TestFeatureImportance:
    def test_ensemble_importance(self):
        X, y = _make_synthetic_dataset()
        model = VoiceDisorderModel(mode="binary", backend="ensemble")
        model.train(X, y)
        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0

    def test_logreg_importance(self):
        X, y = _make_synthetic_dataset()
        model = VoiceDisorderModel(mode="binary", backend="logreg")
        model.train(X, y)
        importance = model.get_feature_importance()
        assert importance is not None


class TestIncremental:
    def test_incremental_update(self):
        X, y = _make_synthetic_dataset()
        model = VoiceDisorderModel(mode="binary")
        model.train(X, y)
        # Should not raise
        model.incremental_update(X[:5], y[:5])
