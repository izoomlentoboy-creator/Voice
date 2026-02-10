"""Voice disorder detection model.

Ensemble classifier combining SVM, Random Forest, and Gradient Boosting
for robust voice disorder detection. Supports binary (healthy/pathological)
and multiclass (specific disorder) modes.
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
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from . import config

logger = logging.getLogger(__name__)


class VoiceDisorderModel:
    """Ensemble model for voice disorder detection."""

    def __init__(self, mode: str = config.MODE_BINARY):
        """
        Parameters
        ----------
        mode : str
            'binary' or 'multiclass'.
        """
        self.mode = mode
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False
        self.training_metadata = {}

        # Incremental learner for online feedback
        self._incremental_model = None

    def _build_ensemble(self, n_classes: int) -> VotingClassifier:
        """Build the ensemble classifier."""
        svm = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
        )

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )

        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=config.RANDOM_STATE,
        )

        ensemble = VotingClassifier(
            estimators=[("svm", svm), ("rf", rf), ("gb", gb)],
            voting="soft",
            n_jobs=-1,
        )
        return ensemble

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        session_ids: Optional[list[int]] = None,
    ) -> dict:
        """Train the model on extracted features.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        session_ids : list[int], optional
            For metadata tracking.

        Returns
        -------
        dict
            Training metadata (time, n_samples, n_features, n_classes).
        """
        logger.info(
            "Training %s model: %d samples, %d features",
            self.mode, X.shape[0], X.shape[1],
        )

        start_time = time.time()

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build and train ensemble
        self.model = self._build_ensemble(n_classes)
        self.model.fit(X_scaled, y_encoded)

        # Build incremental learner for future feedback
        self._incremental_model = SGDClassifier(
            loss="modified_huber",  # provides probability estimates
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            warm_start=True,
        )
        self._incremental_model.fit(X_scaled, y_encoded)

        elapsed = time.time() - start_time
        self.is_trained = True

        self.training_metadata = {
            "mode": self.mode,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_classes": int(n_classes),
            "classes": self.label_encoder.classes_.tolist(),
            "training_time_sec": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if session_ids:
            self.training_metadata["n_sessions"] = len(set(session_ids))

        logger.info(
            "Training complete in %.2fs. Classes: %s",
            elapsed, self.label_encoder.classes_,
        )
        return self.training_metadata

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted labels (original label space).
        """
        self._check_trained()
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        self._check_trained()
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict_with_confidence(
        self, X: np.ndarray
    ) -> list[dict]:
        """Predict with detailed confidence information.

        Returns
        -------
        list[dict]
            Each dict contains: 'label', 'confidence', 'probabilities'.
        """
        self._check_trained()
        labels = self.predict(X)
        probas = self.predict_proba(X)

        results = []
        for i in range(len(labels)):
            class_probs = {
                str(cls): round(float(p), 4)
                for cls, p in zip(self.label_encoder.classes_, probas[i])
            }
            results.append({
                "label": int(labels[i]) if isinstance(labels[i], (np.integer,)) else labels[i],
                "confidence": round(float(np.max(probas[i])), 4),
                "probabilities": class_probs,
            })
        return results

    def incremental_update(
        self, X: np.ndarray, y: np.ndarray
    ) -> None:
        """Incrementally update the model with new labeled data (feedback).

        Uses the SGD-based incremental learner for fast online updates
        without full retraining.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        """
        self._check_trained()

        y_encoded = self.label_encoder.transform(y)
        X_scaled = self.scaler.transform(X)

        if self._incremental_model is not None:
            self._incremental_model.partial_fit(X_scaled, y_encoded)
            logger.info("Incremental update with %d samples", X.shape[0])

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model, scaler, and label encoder to disk."""
        self._check_trained()

        model_file = path or config.model_path(self.mode)
        scaler_file = config.scaler_path(self.mode)
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
        model_file = path or config.model_path(self.mode)
        scaler_file = config.scaler_path(self.mode)
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
            rf = self.model.named_estimators_.get("rf")
            if rf is not None:
                from .feature_extractor import get_feature_names
                names = get_feature_names()
                importances = rf.feature_importances_
                if len(names) == len(importances):
                    ranked = sorted(
                        zip(names, importances),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    return {name: round(float(imp), 6) for name, imp in ranked[:30]}
        except Exception as e:
            logger.warning("Could not extract feature importance: %s", e)

        return None

    def _check_trained(self) -> None:
        if not self.is_trained or self.model is None:
            raise RuntimeError(
                "Model is not trained. Call train() or load() first."
            )
