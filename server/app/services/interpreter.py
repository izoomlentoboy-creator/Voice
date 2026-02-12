"""Translate raw 322-dim feature vector into 5 user-friendly voice quality categories.

Categories:
  1. Pitch Stability  (Стабильность высоты голоса) — jitter, shimmer, F0 variation
  2. Harmonic Quality  (Гармоничность)             — HNR, harmonic ratio, spectral flatness
  3. Voice Steadiness  (Стабильность голоса)        — RMS/MFCC variability
  4. Spectral Clarity   (Чистота тембра)            — spectral centroid, bandwidth, rolloff
  5. Breath Support     (Дыхание)                   — RMS level, ZCR, voiced fraction

Scoring approach:
  - For each category, pick 2-4 clinically relevant features.
  - Compute z-score vs healthy population reference (mean/std from training).
  - Aggregate |z-scores| into a 0-1 normality score (1 = perfectly normal).
  - Map score to status: normal (≥0.7), attention (0.4-0.7), concern (<0.4).
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Feature name → index mapping built lazily (thread-safe)
_NAME_TO_IDX: dict[str, int] | None = None
_NAME_TO_IDX_LOCK = threading.Lock()


def _build_name_index() -> dict[str, int]:
    global _NAME_TO_IDX
    if _NAME_TO_IDX is not None:
        return _NAME_TO_IDX
    with _NAME_TO_IDX_LOCK:
        if _NAME_TO_IDX is None:
            from voice_disorder_detection.feature_extractor import get_feature_names
            names = get_feature_names()
            _NAME_TO_IDX = {name: i for i, name in enumerate(names)}
    return _NAME_TO_IDX


# Define which features belong to each category
CATEGORY_FEATURES = {
    "pitch_stability": [
        "f0_std", "jitter", "shimmer", "f0_min", "f0_max",
    ],
    "harmonic_quality": [
        "hnr", "harmonic_ratio", "percussive_ratio",
        "spec_flatness_mean",
    ],
    "voice_steadiness": [
        "rms_std", "rms_kurtosis",
        "mfcc_0_std", "mfcc_d_0_std",
    ],
    "spectral_clarity": [
        "spec_centroid_mean", "spec_bandwidth_mean",
        "spec_rolloff_mean", "spec_contrast_0_mean",
    ],
    "breath_support": [
        "rms_mean", "zcr_mean", "zcr_std", "voiced_fraction",
    ],
}

# User-facing labels (Russian)
CATEGORY_LABELS = {
    "pitch_stability": "Высота голоса",
    "harmonic_quality": "Гармоничность",
    "voice_steadiness": "Стабильность",
    "spectral_clarity": "Тембр",
    "breath_support": "Дыхание",
}

STATUS_LABELS = {
    "normal": "Норма",
    "attention": "Внимание",
    "concern": "Отклонение",
}


@dataclass
class CategoryResult:
    status: str   # "normal", "attention", "concern"
    label: str    # Russian label
    score: float  # 0-1 (1 = perfectly normal)


def interpret_features(
    feature_vector: np.ndarray,
    ref_stats: dict | None,
) -> dict[str, CategoryResult]:
    """Convert a 322-dim feature vector into 5 category scores.

    Parameters
    ----------
    feature_vector : np.ndarray, shape (322,)
    ref_stats : dict with 'mean' and 'std' arrays, or None.
        If None, returns a neutral "no data" result.

    Returns
    -------
    dict mapping category name → CategoryResult
    """
    if ref_stats is None:
        # No reference data — return all neutral
        return {
            cat: CategoryResult(status="normal", label=CATEGORY_LABELS[cat], score=0.75)
            for cat in CATEGORY_FEATURES
        }

    name_idx = _build_name_index()
    ref_mean = ref_stats["mean"]
    ref_std = ref_stats["std"]

    results = {}
    for category, feature_names in CATEGORY_FEATURES.items():
        z_scores = []
        for fname in feature_names:
            idx = name_idx.get(fname)
            if idx is None or idx >= len(ref_mean):
                continue
            std = ref_std[idx]
            if std < 1e-10:
                continue
            z = abs(float(feature_vector[idx]) - float(ref_mean[idx])) / float(std)
            z_scores.append(z)

        if not z_scores:
            score = 0.75  # neutral fallback
        else:
            # Average |z-score|. Convert to 0-1 "normality" score:
            # z=0 → score=1.0 (perfectly normal)
            # z=2 → score=0.5 (borderline)
            # z=4+ → score≈0.0 (clearly abnormal)
            avg_z = np.mean(z_scores)
            score = float(np.clip(1.0 - avg_z / 4.0, 0.0, 1.0))

        if score >= 0.70:
            status = "normal"
        elif score >= 0.40:
            status = "attention"
        else:
            status = "concern"

        results[category] = CategoryResult(
            status=status,
            label=CATEGORY_LABELS[category],
            score=round(score, 2),
        )

    return results


def build_recommendation(
    verdict: str,
    categories: dict[str, CategoryResult],
    abstain: bool,
    confidence: float,
) -> str:
    """Generate a human-readable recommendation in Russian.

    Parameters
    ----------
    verdict : "healthy", "pathological", "abstain"
    categories : per-category results
    abstain : whether the model abstained
    confidence : prediction confidence (0-1)

    Returns
    -------
    str : recommendation text
    """
    if abstain:
        return (
            "Не удалось определить результат с достаточной уверенностью "
            f"({confidence:.0%}). Это может быть связано с фоновым шумом "
            "или нестандартным произношением. Попробуйте записать заново "
            "в тихом помещении."
        )

    if verdict == "healthy":
        return (
            "Признаков нарушений голоса не обнаружено. "
            "Рекомендуем повторить проверку через 1-3 месяца для мониторинга."
        )

    # pathological — give detail based on which categories flagged
    concern_cats = [
        categories[c].label
        for c in categories
        if categories[c].status in ("attention", "concern")
    ]

    if not concern_cats:
        detail = ""
    else:
        detail = (
            " Обратите внимание на: "
            + ", ".join(concern_cats).lower()
            + "."
        )

    return (
        "Обнаружены признаки, требующие внимания специалиста."
        + detail
        + " Рекомендуем обратиться к врачу-отоларингологу или фониатру "
        "для консультации. Возможные причины: нагрузка на голос, "
        "воспалительные процессы, аллергические реакции."
    )


def verdict_to_label(verdict: str) -> str:
    """Convert machine verdict to user-facing Russian label."""
    mapping = {
        "healthy": "Норма",
        "pathological": "Внимание",
        "abstain": "Неопределённо",
    }
    return mapping.get(verdict, verdict)
