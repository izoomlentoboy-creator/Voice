"""Tests for the interpreter service."""

import numpy as np

from server.app.services.interpreter import (
    CategoryResult,
    build_recommendation,
    interpret_features,
    verdict_to_label,
)


class TestInterpretFeatures:
    def test_with_no_ref_stats(self):
        """Without reference stats, all categories should be neutral."""
        features = np.random.randn(322).astype(np.float32)
        result = interpret_features(features, ref_stats=None)

        assert len(result) == 5
        for cat, r in result.items():
            assert r.status == "normal"
            assert r.score == 0.75

    def test_with_matching_ref_stats(self):
        """Features equal to the mean should score high (normal)."""
        n_features = 322
        ref_stats = {
            "mean": np.zeros(n_features),
            "std": np.ones(n_features),
        }
        # Features exactly at the mean → z-score = 0 → score = 1.0
        features = np.zeros(n_features).astype(np.float32)
        result = interpret_features(features, ref_stats)

        for cat, r in result.items():
            assert r.status == "normal"
            assert r.score >= 0.9

    def test_extreme_features_flag_concern(self):
        """Features 5+ std from mean should flag concern."""
        n_features = 322
        ref_stats = {
            "mean": np.zeros(n_features),
            "std": np.ones(n_features),
        }
        # All features 5 std away
        features = np.full(n_features, 5.0, dtype=np.float32)
        result = interpret_features(features, ref_stats)

        concern_count = sum(1 for r in result.values() if r.status == "concern")
        # At least some categories should be flagged
        assert concern_count >= 2

    def test_all_categories_present(self):
        features = np.random.randn(322).astype(np.float32)
        ref_stats = {
            "mean": np.zeros(322),
            "std": np.ones(322),
        }
        result = interpret_features(features, ref_stats)

        expected = {"pitch_stability", "harmonic_quality", "voice_steadiness",
                    "spectral_clarity", "breath_support"}
        assert set(result.keys()) == expected


class TestBuildRecommendation:
    def _normal_cats(self):
        return {
            "pitch_stability": CategoryResult("normal", "Высота голоса", 0.9),
            "harmonic_quality": CategoryResult("normal", "Гармоничность", 0.85),
            "voice_steadiness": CategoryResult("normal", "Стабильность", 0.88),
            "spectral_clarity": CategoryResult("normal", "Тембр", 0.92),
            "breath_support": CategoryResult("normal", "Дыхание", 0.87),
        }

    def test_healthy_recommendation(self):
        rec = build_recommendation("healthy", self._normal_cats(), False, 0.94)
        assert "не обнаружено" in rec
        assert "1-3 месяца" in rec

    def test_pathological_recommendation(self):
        cats = self._normal_cats()
        cats["pitch_stability"] = CategoryResult("attention", "Высота голоса", 0.5)
        rec = build_recommendation("pathological", cats, False, 0.78)
        assert "специалист" in rec.lower() or "врач" in rec.lower()
        assert "высота голоса" in rec.lower()

    def test_abstain_recommendation(self):
        rec = build_recommendation("abstain", self._normal_cats(), True, 0.47)
        assert "не удалось" in rec.lower()
        assert "47%" in rec


class TestVerdictToLabel:
    def test_healthy(self):
        assert verdict_to_label("healthy") == "Норма"

    def test_pathological(self):
        assert verdict_to_label("pathological") == "Внимание"

    def test_abstain(self):
        assert verdict_to_label("abstain") == "Неопределённо"
