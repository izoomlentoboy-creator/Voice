"""Model interpretability using SHAP.

Explains which acoustic features drive predictions,
helping validate clinical plausibility of the model.
"""

import logging
from typing import Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def compute_shap_values(
    model,
    X_train: np.ndarray,
    X_explain: Optional[np.ndarray] = None,
    max_background: int = 100,
) -> dict:
    """Compute SHAP values for model explanations.

    Parameters
    ----------
    model : VoiceDisorderModel
        Trained model (must have .scaler and .model attributes).
    X_train : np.ndarray
        Training data for background distribution.
    X_explain : np.ndarray, optional
        Samples to explain. If None, uses a subsample of X_train.
    max_background : int
        Max samples for SHAP background.

    Returns
    -------
    dict
        SHAP analysis results with feature importance rankings.
    """
    try:
        import shap
    except ImportError:
        return {"error": "shap package not installed. Run: pip install shap"}

    X_train_scaled = model.scaler.transform(X_train)

    # Subsample for efficiency
    if len(X_train_scaled) > max_background:
        idx = np.random.RandomState(config.RANDOM_STATE).choice(
            len(X_train_scaled), max_background, replace=False,
        )
        background = X_train_scaled[idx]
    else:
        background = X_train_scaled

    if X_explain is None:
        n_explain = min(50, len(X_train))
        idx = np.random.RandomState(config.RANDOM_STATE).choice(
            len(X_train), n_explain, replace=False,
        )
        X_explain = X_train[idx]

    X_explain_scaled = model.scaler.transform(X_explain)

    # Use KernelExplainer (model-agnostic)
    try:
        explainer = shap.KernelExplainer(
            model.model.predict_proba, background,
        )
        shap_values = explainer.shap_values(X_explain_scaled, nsamples=100)
    except Exception as e:
        logger.warning("SHAP KernelExplainer failed: %s. Trying TreeExplainer.", e)
        try:
            rf = model.model.named_estimators_.get("rf")
            if rf is None:
                return {"error": f"SHAP computation failed: {e}"}
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_explain_scaled)
        except Exception as e2:
            return {"error": f"SHAP computation failed: {e2}"}

    return _summarize_shap(shap_values)


def _summarize_shap(shap_values) -> dict:
    """Summarize SHAP values into a feature importance ranking."""
    from .feature_extractor import get_feature_names

    names = get_feature_names()

    # shap_values may be a list (one per class) or a single array
    if isinstance(shap_values, list):
        # For binary: use class 1 (pathological)
        if len(shap_values) == 2:
            sv = np.abs(shap_values[1])
        else:
            sv = np.mean([np.abs(s) for s in shap_values], axis=0)
    else:
        sv = np.abs(shap_values)

    # Mean absolute SHAP per feature
    mean_shap = np.mean(sv, axis=0)

    if len(names) != len(mean_shap):
        names = [f"feature_{i}" for i in range(len(mean_shap))]

    ranked = sorted(zip(names, mean_shap), key=lambda x: x[1], reverse=True)

    # Categorize features
    categories = {
        "mfcc": [], "spectral": [], "temporal": [],
        "pitch": [], "harmonic": [],
    }
    for name, importance in ranked:
        imp = round(float(importance), 6)
        if name.startswith("mfcc"):
            categories["mfcc"].append((name, imp))
        elif name.startswith("spec_"):
            categories["spectral"].append((name, imp))
        elif name in ("zcr", "rms") or name.startswith("zcr_") or name.startswith("rms_"):
            categories["temporal"].append((name, imp))
        elif name in ("f0_mean", "f0_std", "f0_min", "f0_max",
                       "voiced_fraction", "jitter", "shimmer"):
            categories["pitch"].append((name, imp))
        elif name in ("hnr", "harmonic_ratio", "percussive_ratio"):
            categories["harmonic"].append((name, imp))

    # Category-level importance
    category_importance = {}
    for cat, features in categories.items():
        if features:
            category_importance[cat] = round(sum(v for _, v in features), 6)

    return {
        "top_20_features": {n: v for n, v in ranked[:20]},
        "category_importance": dict(
            sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
        ),
        "clinical_features": {
            name: round(float(imp), 6)
            for name, imp in ranked
            if name in ("jitter", "shimmer", "hnr", "f0_mean", "f0_std",
                        "voiced_fraction", "harmonic_ratio")
        },
        "n_features_analyzed": len(mean_shap),
        "n_samples_explained": sv.shape[0],
    }
