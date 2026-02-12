"""Voice disorder detection models.

Supports multiple backends:
  - ensemble: SVM + Random Forest + Gradient Boosting (default)
  - logreg: Logistic Regression baseline
  - cnn: MLP on mel-spectrogram (see cnn_model.py)

Includes abstain mechanism for uncertain predictions.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from . import config

logger = logging.getLogger(__name__)


class VoiceDisorderModel:
    """Classifier for voice disorder detection."""

    def __init__(
        self,
        mode: str = config.MODE_BINARY,
        backend: str = config.BACKEND_ENSEMBLE,
    ):
        """
        Parameters
        ----------
        mode : str
            'binary' or 'multiclass'.
        backend : str
            'ensemble', 'logreg', or 'cnn'.
        """
        self.mode = mode
        self.backend = backend
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False
        self.training_metadata = {}
        self._incremental_model = None

    def _build_model(self, n_classes: int):
        """Build the classifier based on the backend."""
        if self.backend == config.BACKEND_LOGREG:
            return LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=config.RANDOM_STATE,
                solver="lbfgs",
            )

        # Default: ensemble
        svm = SVC(
            kernel="rbf", C=10.0, gamma="scale",
            probability=True, random_state=config.RANDOM_STATE,
            class_weight="balanced",
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=5,
            class_weight="balanced", random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=config.RANDOM_STATE,
        )
        return VotingClassifier(
            estimators=[("svm", svm), ("rf", rf), ("gb", gb)],
            voting="soft", n_jobs=-1,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        session_ids: Optional[list[int]] = None,
        speaker_ids: Optional[list[int]] = None,
    ) -> dict:
        """Train the model on extracted features."""
        logger.info(
            "Training %s model (backend=%s): %d samples, %d features",
            self.mode, self.backend, X.shape[0], X.shape[1],
        )
        start_time = time.time()

        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        X_scaled = self.scaler.fit_transform(X)

        self.model = self._build_model(n_classes)
        logger.info("Fitting %s model, this may take a while...", self.backend)
        self.model.fit(X_scaled, y_encoded)

        # Build incremental learner
        self._incremental_model = SGDClassifier(
            loss="modified_huber", class_weight="balanced",
            random_state=config.RANDOM_STATE, warm_start=True,
        )
        self._incremental_model.fit(X_scaled, y_encoded)

        elapsed = time.time() - start_time
        self.is_trained = True

        self.training_metadata = {
            "mode": self.mode,
            "backend": self.backend,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(n_classes),
            "classes": self.label_encoder.classes_.tolist(),
            "training_time_sec": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if speaker_ids:
            self.training_metadata["n_unique_speakers"] = len(set(speaker_ids))
        if session_ids:
            self.training_metadata["n_sessions"] = len(set(session_ids))

        logger.info("Training complete in %.2fs. Classes: %s", elapsed, self.label_encoder.classes_)
        return self.training_metadata

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._check_trained()
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        self._check_trained()
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict_with_confidence(
        self,
        X: np.ndarray,
        abstain_threshold: Optional[float] = None,
    ) -> list[dict]:
        """Predict with confidence and optional abstain.

        Parameters
        ----------
        X : np.ndarray
        abstain_threshold : float, optional
            Override config.ABSTAIN_THRESHOLD. If max probability is below
            this threshold, the model refuses to make a definitive prediction.
        """
        self._check_trained()
        threshold = abstain_threshold if abstain_threshold is not None else config.ABSTAIN_THRESHOLD
        labels = self.predict(X)
        probas = self.predict_proba(X)

        results = []
        for i in range(len(labels)):
            max_prob = float(np.max(probas[i]))
            class_probs = {
                str(cls): round(float(p), 4)
                for cls, p in zip(self.label_encoder.classes_, probas[i])
            }

            abstained = max_prob < threshold
            result = {
                "label": int(labels[i]) if isinstance(labels[i], np.integer) else labels[i],
                "confidence": round(max_prob, 4),
                "probabilities": class_probs,
                "abstain": abstained,
            }
            if abstained:
                result["abstain_reason"] = (
                    f"Confidence {max_prob:.1%} is below threshold {threshold:.1%}. "
                    "Recommendation: refer to specialist for manual assessment."
                )
            results.append(result)
        return results

    def incremental_update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally update with new labeled data (feedback)."""
        self._check_trained()
        y_encoded = self.label_encoder.transform(y)
        X_scaled = self.scaler.transform(X)
        if self._incremental_model is not None:
            self._incremental_model.partial_fit(X_scaled, y_encoded)
            logger.info("Incremental update with %d samples", X.shape[0])

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model, scaler, and label encoder to disk."""
        self._check_trained()
        model_file = path or config.model_path(self.mode, self.backend)
        scaler_file = config.scaler_path(self.mode, self.backend)
        le_file = config.label_encoder_path(self.mode)
        meta_file = config.metadata_path()

        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        joblib.dump(self.label_encoder, le_file)

        if self._incremental_model is not None:
            inc_file = config.MODELS_DIR / f"incremental_{self.mode}.joblib"
            joblib.dump(self._incremental_model, inc_file)

        with open(meta_file, "w") as f:
            json.dump(self.training_metadata, f, indent=2)

        logger.info("Model saved to %s", model_file)
        return model_file

    def load(self, path: Optional[Path] = None) -> None:
        """Load a previously saved model."""
        model_file = path or config.model_path(self.mode, self.backend)
        scaler_file = config.scaler_path(self.mode, self.backend)
        le_file = config.label_encoder_path(self.mode)
        meta_file = config.metadata_path()

        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")

        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.label_encoder = joblib.load(le_file)

        inc_file = config.MODELS_DIR / f"incremental_{self.mode}.joblib"
        if inc_file.exists():
            self._incremental_model = joblib.load(inc_file)

        if meta_file.exists():
            with open(meta_file) as f:
                self.training_metadata = json.load(f)

        self.is_trained = True
        logger.info("Model loaded from %s", model_file)

    def get_feature_importance(self) -> Optional[dict]:
        """Get feature importance from the Random Forest in the ensemble."""
        if not self.is_trained or self.model is None:
            return None

        try:
            # Ensemble backend
            if hasattr(self.model, "named_estimators_"):
                rf = self.model.named_estimators_.get("rf")
            else:
                rf = None

            importances = None
            if rf is not None and hasattr(rf, "feature_importances_"):
                importances = rf.feature_importances_
            elif hasattr(self.model, "coef_"):
                # LogReg
                importances = np.abs(self.model.coef_).mean(axis=0) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_[0])

            if importances is not None:
                from .feature_extractor import get_feature_names
                names = get_feature_names()
                if len(names) == len(importances):
                    ranked = sorted(
                        zip(names, importances),
                        key=lambda x: x[1], reverse=True,
                    )
                    return {name: round(float(imp), 6) for name, imp in ranked[:30]}
        except Exception as e:
            logger.warning("Could not extract feature importance: %s", e)
        return None

    def _check_trained(self) -> None:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() or load() first.")
