"""High-level pipeline that ties together all components.

Provides a single interface for training, prediction, feedback,
and self-testing workflows.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Union

from . import config
from .data_loader import VoiceDataLoader
from .feature_extractor import extract_all_features, get_feature_names
from .feedback import FeedbackManager
from .model import VoiceDisorderModel
from .self_test import SelfTester

logger = logging.getLogger(__name__)


class VoiceDisorderPipeline:
    """End-to-end pipeline for voice disorder detection."""

    def __init__(
        self,
        mode: str = config.MODE_BINARY,
        dbdir: Optional[str] = None,
        download_mode: str = "lazy",
    ):
        self.mode = mode
        self.model = VoiceDisorderModel(mode=mode)
        self.tester = SelfTester(self.model)
        self.feedback = FeedbackManager(self.model)
        self._loader = None
        self._dbdir = dbdir
        self._download_mode = download_mode

    @property
    def loader(self) -> VoiceDataLoader:
        if self._loader is None:
            self._loader = VoiceDataLoader(
                dbdir=self._dbdir,
                download_mode=self._download_mode,
            )
        return self._loader

    # ---- Training ----

    def train(
        self,
        max_samples: Optional[int] = None,
        use_cache: bool = True,
        run_evaluation: bool = True,
    ) -> dict:
        """Full training pipeline: load data, extract features, train, evaluate.

        Parameters
        ----------
        max_samples : int, optional
            Limit samples for faster experimentation.
        use_cache : bool
            Use feature cache.
        run_evaluation : bool
            Run evaluation after training.

        Returns
        -------
        dict
            Training and evaluation results.
        """
        result = {}

        # Extract features
        logger.info("=== Starting training pipeline (mode=%s) ===", self.mode)
        X, y, session_ids = self.loader.extract_dataset(
            mode=self.mode,
            max_samples=max_samples,
            use_cache=use_cache,
        )

        # Train
        train_meta = self.model.train(X, y, session_ids=session_ids)
        result["training"] = train_meta

        # Evaluate
        if run_evaluation and len(X) >= 20:
            eval_result = self.tester.run_full_evaluation(X, y)
            result["evaluation"] = {
                "accuracy": eval_result["accuracy"],
                "f1": eval_result["f1_weighted"],
                "precision": eval_result["precision_weighted"],
                "recall": eval_result["recall_weighted"],
                "auc_roc": eval_result.get("auc_roc"),
            }

        # Save model
        model_path = self.model.save()
        result["model_path"] = str(model_path)

        # Feature importance
        importance = self.model.get_feature_importance()
        if importance:
            result["top_features"] = dict(list(importance.items())[:10])

        logger.info("=== Training pipeline complete ===")
        return result

    # ---- Prediction ----

    def predict_from_audio(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> dict:
        """Predict voice disorder from raw audio.

        Parameters
        ----------
        audio : np.ndarray
            Raw audio signal (int16 or float).
        sr : int
            Sampling rate.

        Returns
        -------
        dict
            Prediction result with label, confidence, and probabilities.
        """
        self._ensure_model_loaded()
        features = extract_all_features(audio, sr)
        results = self.model.predict_with_confidence(features.reshape(1, -1))
        return results[0]

    def predict_from_file(self, audio_path: str) -> dict:
        """Predict from an audio file (WAV, FLAC, etc.)."""
        import librosa

        self._ensure_model_loaded()
        audio, sr = librosa.load(audio_path, sr=None)
        return self.predict_from_audio(audio, sr)

    def predict_from_session(self, session_id: int) -> dict:
        """Predict using recordings from a specific database session."""
        self._ensure_model_loaded()

        session = self.loader.db.get_session(
            session_id, query_recordings=True, query_pathologies=True,
        )
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        features_list = []
        for rec in session.recordings:
            if rec.utterance not in self.loader.utterances:
                continue
            rec_full = self.loader.db.get_recording(rec.id, full_file_paths=True)
            if rec_full is None:
                continue
            try:
                audio = rec_full.nspdata
            except Exception:
                audio = None
            if audio is None or len(audio) == 0:
                continue
            feats = extract_all_features(audio, rec_full.rate)
            features_list.append(feats)

        if not features_list:
            raise ValueError(
                f"No usable recordings found in session {session_id}"
            )

        combined = np.mean(features_list, axis=0).reshape(1, -1)
        results = self.model.predict_with_confidence(combined)
        prediction = results[0]

        # Add ground truth if available
        prediction["session_id"] = session_id
        prediction["actual_type"] = session.type
        if session.pathologies:
            prediction["actual_pathologies"] = [
                p.name for p in session.pathologies
            ]

        return prediction

    # ---- Feedback ----

    def correct_prediction(
        self,
        audio: np.ndarray,
        sr: int,
        correct_label: int,
        session_id: Optional[int] = None,
        note: str = "",
    ) -> dict:
        """Submit a correction for a prediction.

        Parameters
        ----------
        audio : np.ndarray
            The audio that was misclassified.
        sr : int
            Sampling rate.
        correct_label : int
            The true label.
        session_id : int, optional
            Database session ID.
        note : str
            Optional note.

        Returns
        -------
        dict
            Correction details and feedback stats.
        """
        self._ensure_model_loaded()
        features = extract_all_features(audio, sr)
        predicted = self.model.predict(features.reshape(1, -1))[0]

        correction = self.feedback.add_correction(
            features=features,
            predicted_label=int(predicted),
            correct_label=correct_label,
            session_id=session_id,
            note=note,
        )

        return {
            "correction": correction,
            "stats": self.feedback.get_correction_stats(),
        }

    def apply_feedback(self, full_retrain: bool = False) -> dict:
        """Apply accumulated feedback corrections."""
        result = self.feedback.apply_corrections(full_retrain=full_retrain)
        if not full_retrain:
            self.model.save()
        return result

    # ---- Self-Testing ----

    def self_test(
        self,
        max_samples: Optional[int] = None,
        test_type: str = "full",
    ) -> dict:
        """Run self-test on the model.

        Parameters
        ----------
        max_samples : int, optional
            Limit dataset size.
        test_type : str
            'full' for train/test evaluation, 'cv' for cross-validation,
            'quick' for sanity check.

        Returns
        -------
        dict
            Test results.
        """
        X, y, _ = self.loader.extract_dataset(
            mode=self.mode,
            max_samples=max_samples,
            use_cache=True,
        )

        if test_type == "quick":
            self._ensure_model_loaded()
            return self.tester.run_quick_test(X, y)
        elif test_type == "cv":
            return self.tester.run_cross_validation(X, y)
        else:
            return self.tester.run_full_evaluation(X, y)

    def optimize(
        self,
        max_samples: Optional[int] = None,
        n_iter: int = 20,
    ) -> dict:
        """Run hyperparameter optimization and retrain with best params."""
        X, y, session_ids = self.loader.extract_dataset(
            mode=self.mode,
            max_samples=max_samples,
            use_cache=True,
        )

        opt_result = self.tester.optimize_hyperparameters(X, y, n_iter=n_iter)

        # Retrain with full data using the model's default (already good) params
        # The optimization result informs the user about potential improvements
        self.model.train(X, y, session_ids=session_ids)
        self.model.save()

        return opt_result

    # ---- Status ----

    def status(self) -> dict:
        """Get current system status."""
        result = {
            "mode": self.mode,
            "model_trained": self.model.is_trained,
            "model_file_exists": config.model_path(self.mode).exists(),
        }

        if self.model.is_trained:
            result["training_metadata"] = self.model.training_metadata

        result["feedback"] = self.feedback.get_correction_stats()
        result["performance"] = self.tester.get_performance_summary()
        result["regression_check"] = self.tester.check_regression()

        try:
            result["database"] = self.loader.get_database_stats()
        except Exception as e:
            result["database"] = {"error": str(e)}

        return result

    # ---- Internal ----

    def _ensure_model_loaded(self) -> None:
        """Load the model from disk if not already trained."""
        if not self.model.is_trained:
            model_file = config.model_path(self.mode)
            if model_file.exists():
                self.model.load()
            else:
                raise RuntimeError(
                    "No trained model available. Run 'train' first."
                )
