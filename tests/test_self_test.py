"""Tests for self-testing with patient-level split and medical metrics."""

import numpy as np

from voice_disorder_detection.model import VoiceDisorderModel
from voice_disorder_detection.self_test import SelfTester


def _make_dataset_with_speakers(n=200, n_features=322):
    """Synthetic dataset with speaker IDs."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, n_features).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    X[y == 0] += 0.5
    X[y == 1] -= 0.5
    # 20 speakers, each with 10 samples
    speaker_ids = list(range(20)) * 10
    speaker_ids = sorted(speaker_ids)[:n]
    idx = rng.permutation(n)
    return X[idx], y[idx], [speaker_ids[i] for i in idx]


class TestPatientLevelSplit:
    def test_full_eval_with_speakers(self):
        X, y, speakers = _make_dataset_with_speakers()
        model = VoiceDisorderModel(mode="binary")
        tester = SelfTester(model)
        result = tester.run_full_evaluation(X, y, speaker_ids=speakers)
        assert result["split"]["method"] == "patient_level"
        assert result["split"]["speaker_overlap"] == 0
        assert "sensitivity" in result
        assert "specificity" in result

    def test_full_eval_without_speakers(self):
        X, y, _ = _make_dataset_with_speakers()
        model = VoiceDisorderModel(mode="binary")
        tester = SelfTester(model)
        result = tester.run_full_evaluation(X, y)
        assert result["split"]["method"] == "stratified"

    def test_cv_with_speakers(self):
        X, y, speakers = _make_dataset_with_speakers()
        model = VoiceDisorderModel(mode="binary")
        tester = SelfTester(model)
        result = tester.run_cross_validation(X, y, speaker_ids=speakers, n_folds=3)
        assert result["split_method"] == "patient_level_group_kfold"
        assert result["n_folds"] == 3
        assert "sensitivity_mean" in result
        assert "specificity_mean" in result


class TestMedicalMetrics:
    def test_binary_metrics(self):
        X, y, speakers = _make_dataset_with_speakers()
        model = VoiceDisorderModel(mode="binary")
        tester = SelfTester(model)
        result = tester.run_full_evaluation(X, y, speaker_ids=speakers)
        # Medical metrics
        assert "sensitivity" in result
        assert "specificity" in result
        assert "ppv" in result
        assert "npv" in result
        assert "false_positive_rate" in result
        assert "false_negative_rate" in result
        # Probabilistic metrics
        assert "auc_roc" in result
        assert "pr_auc" in result
        assert "brier_score" in result
        assert "ece" in result
        # Sanity checks
        assert 0 <= result["sensitivity"] <= 1
        assert 0 <= result["specificity"] <= 1
        assert 0 <= result["brier_score"] <= 1

    def test_ece_computation(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
        ece = SelfTester._compute_ece(y_true, y_prob)
        assert 0 <= ece <= 1


class TestSubgroupAnalysis:
    def test_subgroup_with_metadata(self):
        X, y, speakers = _make_dataset_with_speakers()
        metadata = [
            {"gender": "m" if i % 2 == 0 else "w", "age": 30 + (i % 50)}
            for i in range(len(y))
        ]
        model = VoiceDisorderModel(mode="binary")
        tester = SelfTester(model)
        result = tester.run_subgroup_analysis(X, y, metadata, speaker_ids=speakers)
        assert "overall" in result
        # Should have at least some subgroup results
        assert any(k.startswith("gender_") for k in result)


class TestQuickTest:
    def test_quick_passes(self):
        X, y, _ = _make_dataset_with_speakers()
        model = VoiceDisorderModel(mode="binary")
        model.train(X, y)
        tester = SelfTester(model)
        result = tester.run_quick_test(X, y)
        assert result["passed"] is True
        assert result["accuracy"] > 0.5
