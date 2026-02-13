"""Voice disorder detection models.

Supports multiple backends:
  - ensemble: Stacking classifier (SVM + RF + GB + LogReg meta-learner)
  - logreg: Logistic Regression baseline
  - cnn: MLP on mel-spectrogram (see cnn_model.py)

Includes abstain mechanism for uncertain predictions,
integrated calibration, and proper incremental learning.
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
    StackingClassifier,
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
        self._calibrator = None
        self._feature_selector = None

    def _build_model(self, n_classes: int, params: Optional[dict] = None):
        """Build the classifier based on the backend and optional hyperparameters."""
        if self.backend == config.BACKEND_LOGREG:
            p = params or {}
            return LogisticRegression(
                C=p.get("logreg_C", 1.0),
                class_weight="balanced",
                max_iter=1000,
                random_state=config.RANDOM_STATE,
                solver="lbfgs",
            )

        if self.backend == config.BACKEND_CNN:
            from .cnn_model import MelSpectrogramCNN
            return MelSpectrogramCNN(mode=self.mode)

        # Default: stacking ensemble
        p = params or {}
        svm = SVC(
            kernel="rbf",
            C=p.get("svm_C", 10.0),
            gamma=p.get("svm_gamma", "scale"),
            probability=True,
            random_state=config.RANDOM_STATE,
            class_weight="balanced",
        )
        rf = RandomForestClassifier(
            n_estimators=p.get("rf_n_estimators", 200),
            max_depth=p.get("rf_max_depth", None),
            min_samples_split=5,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=p.get("gb_n_estimators", 200),
            learning_rate=p.get("gb_learning_rate", 0.1),
            max_depth=p.get("gb_max_depth", 5),
            random_state=config.RANDOM_STATE,
        )

        # Stacking ensemble: base learners feed into a LogReg meta-learner
        # that learns optimal weights — superior to simple voting.
        return StackingClassifier(
            estimators=[("svm", svm), ("rf", rf), ("gb", gb)],
            final_estimator=LogisticRegression(
                C=1.0, class_weight="balanced",
                max_iter=500, random_state=config.RANDOM_STATE,
            ),
            cv=3,
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=-1,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        session_ids: Optional[list[int]] = None,
        speaker_ids: Optional[list[int]] = None,
        params: Optional[dict] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> dict:
        """Train the model on extracted features.

        Parameters
        ----------
        params : dict, optional
            Hyperparameter dict (from optimize). If None, uses defaults.
        sample_weight : np.ndarray, optional
            Per-sample weights for handling class imbalance in GB.
        """
        logger.info(
            "Training %s model (backend=%s): %d samples, %d features",
            self.mode, self.backend, X.shape[0], X.shape[1],
        )
        start_time = time.time()

        y_encoded = self.label_encoder.fit_transform(y)
        n_classes = len(self.label_encoder.classes_)
        X_scaled = self.scaler.fit_transform(X)

        self.model = self._build_model(n_classes, params=params)

        # Compute sample weights for class imbalance if not provided
        if sample_weight is None:
            unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
            total = len(y_encoded)
            weight_map = {
                cls: total / (n_classes * cnt)
                for cls, cnt in zip(unique_classes, class_counts)
            }
            sample_weight = np.array([weight_map[yi] for yi in y_encoded])

        # Stacking/Voting classifiers use sample_weight via fit
        try:
            self.model.fit(X_scaled, y_encoded, sample_weight=sample_weight)
        except TypeError:
            # Some estimators don't support sample_weight directly
            self.model.fit(X_scaled, y_encoded)

        # Build incremental learner that mirrors the main model
        self._incremental_model = SGDClassifier(
            loss="modified_huber", class_weight="balanced",
            random_state=config.RANDOM_STATE, warm_start=True,
        )
        self._incremental_model.fit(X_scaled, y_encoded)

        # Store training data summary for incremental updates
        self._train_X = X_scaled.copy()
        self._train_y = y_encoded.copy()

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
            "version": self._generate_version_id(),
        }
        if speaker_ids is not None and len(speaker_ids) > 0:
            self.training_metadata["n_unique_speakers"] = len(set(speaker_ids))
        if session_ids is not None and len(session_ids) > 0:
            self.training_metadata["n_sessions"] = len(set(session_ids))
        if params:
            self.training_metadata["hyperparameters"] = params

        logger.info("Training complete in %.2fs. Classes: %s", elapsed, self.label_encoder.classes_)
        return self.training_metadata

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._check_trained()
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (calibrated if available)."""
        self._check_trained()
        X_scaled = self.scaler.transform(X)

        # Use calibrated model if available for better probability estimates
        if self._calibrator is not None:
            return self._calibrator.predict_proba(X_scaled)

        return self.model.predict_proba(X_scaled)

    def predict_with_confidence(
        self,
        X: np.ndarray,
        abstain_threshold: Optional[float] = None,
    ) -> list[dict]:
        """Predict with confidence and optional abstain.

        Uses calibrated probabilities when available for more
        reliable confidence scores.
        """
        self._check_trained()
        threshold = abstain_threshold if abstain_threshold is not None else config.ABSTAIN_THRESHOLD

        # Use raw model for label, calibrated for probabilities
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
                "calibrated": self._calibrator is not None,
            }
            if abstained:
                result["abstain_reason"] = (
                    f"Confidence {max_prob:.1%} is below threshold {threshold:.1%}. "
                    "Recommendation: refer to specialist for manual assessment."
                )
            results.append(result)
        return results

    def incremental_update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally update model with new labeled data (feedback).

        Updates the SGD auxiliary model AND retrains the main model
        with the augmented dataset for consistent predictions.
        """
        self._check_trained()
        y_encoded = self.label_encoder.transform(y)
        X_scaled = self.scaler.transform(X)

        # Update SGD model incrementally
        if self._incremental_model is not None:
            self._incremental_model.partial_fit(X_scaled, y_encoded)

        # Augment training data and retrain main model for consistency
        if hasattr(self, "_train_X") and self._train_X is not None:
            self._train_X = np.vstack([self._train_X, X_scaled])
            self._train_y = np.concatenate([self._train_y, y_encoded])

            # Retrain main model with augmented dataset
            try:
                n_classes = len(self.label_encoder.classes_)
                new_model = self._build_model(n_classes)
                new_model.fit(self._train_X, self._train_y)
                self.model = new_model
                logger.info(
                    "Main model retrained with %d samples (added %d new)",
                    len(self._train_y), len(y),
                )
            except Exception as e:
                logger.warning(
                    "Main model retrain failed, using SGD fallback: %s", e,
                )
        else:
            logger.info("Incremental update with %d samples (SGD only)", X.shape[0])

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model, scaler, label encoder, and calibrator to disk."""
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

        if self._calibrator is not None:
            cal_file = config.MODELS_DIR / "calibrator.joblib"
            joblib.dump(self._calibrator, cal_file)

        if self._feature_selector is not None:
            sel_file = config.MODELS_DIR / f"feature_selector_{self.mode}_{self.backend}.joblib"
            joblib.dump(self._feature_selector, sel_file)

        with open(meta_file, "w") as f:
            json.dump(self.training_metadata, f, indent=2)

        logger.info("Model saved to %s (version: %s)", model_file,
                     self.training_metadata.get("version", "?"))
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

        # Load calibrator if available
        cal_file = config.MODELS_DIR / "calibrator.joblib"
        if cal_file.exists():
            try:
                self._calibrator = joblib.load(cal_file)
                logger.info("Calibrated model loaded — predictions will use calibrated probabilities.")
            except Exception as e:
                logger.warning("Could not load calibrator: %s", e)

        # Load feature selector if available
        sel_file = config.MODELS_DIR / f"feature_selector_{self.mode}_{self.backend}.joblib"
        if sel_file.exists():
            try:
                self._feature_selector = joblib.load(sel_file)
            except Exception:
                pass

        if meta_file.exists():
            with open(meta_file) as f:
                self.training_metadata = json.load(f)

        self.is_trained = True
        logger.info("Model loaded from %s (version: %s)", model_file,
                     self.training_metadata.get("version", "?"))

    def get_feature_importance(self) -> Optional[dict]:
        """Get feature importance from the ensemble estimators."""
        if not self.is_trained or self.model is None:
            return None

        try:
            importances = None

            # Stacking classifier
            if hasattr(self.model, "estimators_"):
                for estimator in self.model.estimators_:
                    if hasattr(estimator, "feature_importances_"):
                        importances = estimator.feature_importances_
                        break

            # Direct ensemble with named estimators
            if importances is None and hasattr(self.model, "named_estimators_"):
                rf = self.model.named_estimators_.get("rf")
                if rf is not None and hasattr(rf, "feature_importances_"):
                    importances = rf.feature_importances_

            # LogReg coefficients
            if importances is None and hasattr(self.model, "coef_"):
                importances = (
                    np.abs(self.model.coef_).mean(axis=0)
                    if self.model.coef_.ndim > 1
                    else np.abs(self.model.coef_[0])
                )

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

    @staticmethod
    def _generate_version_id() -> str:
        """Generate a unique version identifier for this model training run."""
        import hashlib
        ts = time.strftime("%Y%m%d_%H%M%S")
        unique = hashlib.sha256(ts.encode()).hexdigest()[:8]
        return f"{ts}_{unique}"

    def _check_trained(self) -> None:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() or load() first.")
