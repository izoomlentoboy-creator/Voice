"""Tests for domain shift monitoring."""

import numpy as np
import pytest

from voice_disorder_detection.domain_monitor import DomainMonitor


def _make_training_data(n=200, d=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(n, d) * 2 + 1  # mean=1, std=2


class TestDomainMonitorFit:
    def test_fit_basic(self):
        X = _make_training_data()
        monitor = DomainMonitor()
        monitor.fit(X)
        assert monitor.is_fitted
        assert monitor._maha_threshold > 0
        assert monitor._mean is not None
        assert monitor._cov_inv is not None

    def test_fit_stores_ref_stats(self):
        X = _make_training_data()
        monitor = DomainMonitor()
        monitor.fit(X)
        assert monitor._ref_stats is not None
        assert len(monitor._ref_stats) == X.shape[1]
        # Check per-feature stats
        for j in range(X.shape[1]):
            stats = monitor._ref_stats[j]
            assert "mean" in stats
            assert "std" in stats
            assert "ref_sample" in stats


class TestOODDetection:
    def test_inliers_not_flagged(self):
        """Samples from the same distribution should not be OOD."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(300, 30) * 2 + 1
        X_test = rng.randn(20, 30) * 2 + 1  # same distribution

        monitor = DomainMonitor()
        monitor.fit(X_train)
        results = monitor.check_ood(X_test)

        assert len(results) == 20
        ood_count = sum(r["ood"] for r in results)
        # Most should not be OOD (allow a few false positives)
        assert ood_count < 10

    def test_outliers_flagged(self):
        """Samples far from training distribution should be flagged as OOD."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(300, 30)
        # Extreme outliers: mean shifted by 20 std
        X_outliers = rng.randn(10, 30) + 20

        monitor = DomainMonitor()
        monitor.fit(X_train)
        results = monitor.check_ood(X_outliers)

        ood_count = sum(r["ood"] for r in results)
        # Most extreme outliers should be flagged
        assert ood_count >= 7

    def test_single_sample(self):
        """check_ood should work with a single sample."""
        X_train = _make_training_data()
        monitor = DomainMonitor()
        monitor.fit(X_train)

        x = X_train[0:1]
        results = monitor.check_ood(x)
        assert len(results) == 1
        assert "mahalanobis_distance" in results[0]
        assert "iforest_score" in results[0]
        assert "ood" in results[0]

    def test_result_fields(self):
        X_train = _make_training_data()
        monitor = DomainMonitor()
        monitor.fit(X_train)

        results = monitor.check_ood(X_train[:5])
        for r in results:
            assert "mahalanobis_distance" in r
            assert "mahalanobis_ood" in r
            assert "iforest_score" in r
            assert "iforest_ood" in r
            assert "ood" in r


class TestDriftDetection:
    def test_no_drift_same_distribution(self):
        """Same distribution should not trigger drift."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(500, 20)
        X_new = rng.randn(200, 20)  # same distribution, large enough for stable PSI

        monitor = DomainMonitor()
        monitor.fit(X_train)
        result = monitor.check_drift(X_new)

        assert result["drift_detected"] is False

    def test_drift_detected_on_shift(self):
        """A large mean shift should trigger drift detection."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(300, 20)
        # Shift all features by 5 std
        X_shifted = rng.randn(50, 20) + 5

        monitor = DomainMonitor()
        monitor.fit(X_train)
        result = monitor.check_drift(X_shifted)

        assert result["drift_detected"] is True
        assert result["n_features_drifted_ks"] > 0

    def test_small_batch_skipped(self):
        """Batches with < 10 samples should skip drift detection."""
        X_train = _make_training_data()
        monitor = DomainMonitor()
        monitor.fit(X_train)

        result = monitor.check_drift(X_train[:5])
        assert result["drift_detected"] is False
        assert "reason" in result

    def test_drift_result_fields(self):
        X_train = _make_training_data()
        monitor = DomainMonitor()
        monitor.fit(X_train)

        result = monitor.check_drift(X_train[:50])
        assert "n_features_total" in result
        assert "n_features_drifted_ks" in result
        assert "n_features_drifted_psi" in result
        assert "drift_detected" in result
        assert "summary" in result


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        X_train = _make_training_data()
        monitor = DomainMonitor()
        monitor.fit(X_train)

        save_path = tmp_path / "monitor.joblib"
        monitor.save(save_path)

        monitor2 = DomainMonitor()
        monitor2.load(save_path)
        assert monitor2.is_fitted
        assert monitor2._maha_threshold == monitor._maha_threshold

        # Check OOD results are consistent
        x_test = X_train[:3]
        r1 = monitor.check_ood(x_test)
        r2 = monitor2.check_ood(x_test)
        for a, b in zip(r1, r2):
            assert abs(a["mahalanobis_distance"] - b["mahalanobis_distance"]) < 1e-6

    def test_load_nonexistent(self, tmp_path):
        monitor = DomainMonitor()
        with pytest.raises(FileNotFoundError):
            monitor.load(tmp_path / "nonexistent.joblib")

    def test_check_without_fit_raises(self):
        monitor = DomainMonitor()
        with pytest.raises(RuntimeError, match="not fitted"):
            monitor.check_ood(np.zeros((1, 10)))
