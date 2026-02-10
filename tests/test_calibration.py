"""Tests for calibration and threshold optimization."""

import numpy as np

from voice_disorder_detection.calibration import (
    compute_reliability_diagram,
    optimize_threshold,
)


class TestReliabilityDiagram:
    def test_perfect_calibration(self):
        """Perfectly calibrated model should have ECE ~0."""
        rng = np.random.RandomState(42)
        n = 1000
        # Generate well-calibrated probabilities
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.uniform(0, 1, n) < y_prob).astype(int)

        result = compute_reliability_diagram(y_true, y_prob, n_bins=10)
        assert "ece" in result
        assert "mce" in result
        assert "bins" in result
        assert len(result["bins"]) == 10
        # ECE should be low for approximately calibrated predictions
        assert result["ece"] < 0.1

    def test_overconfident_model(self):
        """A model that always predicts 0.9 but is only 50% accurate should have high ECE."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

        result = compute_reliability_diagram(y_true, y_prob, n_bins=10)
        # Overconfident: confidence 0.9, accuracy 0.5 => ECE ~0.4
        assert result["ece"] > 0.3

    def test_empty_bins_handled(self):
        """Bins with no samples should have None for avg values."""
        y_true = np.array([0, 1])
        y_prob = np.array([0.15, 0.85])

        result = compute_reliability_diagram(y_true, y_prob, n_bins=10)
        empty_bins = [b for b in result["bins"] if b["count"] == 0]
        assert len(empty_bins) > 0
        for b in empty_bins:
            assert b["avg_confidence"] is None
            assert b["avg_accuracy"] is None


class TestThresholdOptimization:
    def _make_data(self):
        rng = np.random.RandomState(42)
        n = 500
        y_true = rng.randint(0, 2, n)
        # Create probabilities correlated with true labels
        y_prob = np.where(y_true == 1, rng.beta(5, 2, n), rng.beta(2, 5, n))
        return y_true, y_prob

    def test_all_criteria_present(self):
        y_true, y_prob = self._make_data()
        result = optimize_threshold(y_true, y_prob)

        assert "youden_j" in result
        assert "cost_weighted" in result
        assert "sensitivity_at_0.95" in result
        assert "max_f1" in result
        assert "default_0.5" in result

    def test_youden_has_threshold(self):
        y_true, y_prob = self._make_data()
        result = optimize_threshold(y_true, y_prob)

        youden = result["youden_j"]
        assert 0.0 < youden["threshold"] < 1.0
        assert "sensitivity" in youden
        assert "specificity" in youden
        assert youden["j"] > 0  # Should be positive for informative model

    def test_max_f1_reasonable(self):
        y_true, y_prob = self._make_data()
        result = optimize_threshold(y_true, y_prob)

        assert result["max_f1"]["f1"] > 0.5  # Better than random

    def test_sensitivity_at_target(self):
        y_true, y_prob = self._make_data()
        result = optimize_threshold(y_true, y_prob, sensitivity_target=0.90)

        sens_result = result["sensitivity_at_0.9"]
        # Should achieve at least the target sensitivity
        assert sens_result["sensitivity"] >= 0.90

    def test_cost_weighted(self):
        y_true, y_prob = self._make_data()
        result = optimize_threshold(y_true, y_prob, cost_fp=1.0, cost_fn=10.0)

        # Higher FN cost should push threshold lower (more aggressive detection)
        assert result["cost_weighted"]["threshold"] < 0.5
