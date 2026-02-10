"""Singleton ML pipeline wrapper for the server.

Loads the trained model once at startup and provides thread-safe
prediction. Reloads on demand (e.g. after retrain).
"""

import logging
import sys
import threading
import time

import numpy as np

from .. import config

logger = logging.getLogger(__name__)

# Add project root to path so we can import the ML package
_project_root = str(config.PROJECT_ROOT)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class Predictor:
    """Thread-safe wrapper around VoiceDisorderPipeline."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pipeline = None
        self._domain_monitor = None
        self._ref_stats = None  # healthy population stats for interpreter
        self._model_version: str = "not_loaded"

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None and self._pipeline.model.is_trained

    def load(self) -> None:
        """Load model from disk. Call at startup."""
        with self._lock:
            self._load_impl()

    def _load_impl(self) -> None:
        from voice_disorder_detection.pipeline import VoiceDisorderPipeline

        logger.info("Loading ML pipeline (mode=%s, backend=%s)...", config.MODEL_MODE, config.MODEL_BACKEND)
        self._pipeline = VoiceDisorderPipeline(
            mode=config.MODEL_MODE,
            backend=config.MODEL_BACKEND,
        )

        try:
            self._pipeline._ensure_model_loaded()
            meta = self._pipeline.model.training_metadata
            self._model_version = (
                f"{meta.get('backend', '?')}_{meta.get('timestamp', '?')}"
            )
            logger.info("Model loaded: %s", self._model_version)
        except RuntimeError:
            logger.warning("No trained model found — predictor will not work until training.")
            return

        # Load domain monitor if available
        try:
            self._pipeline._ensure_domain_monitor()
            self._domain_monitor = self._pipeline.domain_monitor
            logger.info("Domain monitor loaded.")
        except RuntimeError:
            logger.info("Domain monitor not available (run fit-monitor to enable OOD checks).")

        # Load reference stats for interpreter
        self._load_ref_stats()

    def _load_ref_stats(self) -> None:
        """Load per-feature healthy population statistics for the interpreter."""
        ref_path = config.PROJECT_ROOT / "models" / "healthy_ref_stats.npz"
        if ref_path.exists():
            data = np.load(ref_path)
            self._ref_stats = {
                "mean": data["mean"],
                "std": data["std"],
            }
            logger.info("Healthy reference stats loaded (%d features).", len(data["mean"]))
        else:
            logger.info("No healthy reference stats found. Interpreter will use domain monitor fallback.")
            # Fallback: use domain monitor stats (all-population, not healthy-only)
            if self._domain_monitor is not None and self._domain_monitor._ref_stats:
                n_feat = self._domain_monitor._n_features
                means = np.array([self._domain_monitor._ref_stats[j]["mean"] for j in range(n_feat)])
                stds = np.array([self._domain_monitor._ref_stats[j]["std"] for j in range(n_feat)])
                self._ref_stats = {"mean": means, "std": stds}

    def reload(self) -> None:
        """Reload model (e.g. after retrain)."""
        with self._lock:
            self._pipeline = None
            self._domain_monitor = None
            self._ref_stats = None
            self._load_impl()

    def predict(
        self,
        audio_list: list[tuple[np.ndarray, int]],
    ) -> dict:
        """Run full prediction on a list of vowel recordings.

        Parameters
        ----------
        audio_list : list of (audio_array, sample_rate) tuples
            One per vowel (A, I, U).

        Returns
        -------
        dict with:
            prediction: dict (label, confidence, abstain, probabilities)
            feature_vector: np.ndarray (322,)
            ood_result: dict or None
            processing_time_ms: int
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        start = time.monotonic()

        with self._lock:
            from voice_disorder_detection.feature_extractor import extract_all_features

            # Extract features from each vowel and average
            feature_vectors = []
            for audio, sr in audio_list:
                feats = extract_all_features(audio, sr)
                feature_vectors.append(feats)

            combined = np.mean(feature_vectors, axis=0)
            X = combined.reshape(1, -1)

            # Predict
            results = self._pipeline.model.predict_with_confidence(X)
            prediction = results[0]

            # OOD check
            ood_result = None
            if self._domain_monitor is not None and self._domain_monitor.is_fitted:
                ood_results = self._domain_monitor.check_ood(X)
                ood_result = ood_results[0]

        elapsed_ms = int((time.monotonic() - start) * 1000)

        return {
            "prediction": prediction,
            "feature_vector": combined,
            "ood_result": ood_result,
            "processing_time_ms": elapsed_ms,
            "model_version": self._model_version,
        }

    def get_ref_stats(self) -> dict | None:
        """Return reference statistics for the interpreter."""
        return self._ref_stats


# Module-level singleton
predictor = Predictor()
