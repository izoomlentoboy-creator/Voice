"""High-level pipeline that ties together all components.

Provides a single interface for training, prediction, feedback,
self-testing, SHAP analysis, baseline comparison, calibration,
domain monitoring, feature selection, and reproducible report generation.
"""

import logging
from typing import Optional

import numpy as np

from . import config
from .data_loader import VoiceDataLoader
from .domain_monitor import DomainMonitor
from .feature_extractor import extract_all_features
from .feedback import FeedbackManager
from .model import VoiceDisorderModel
from .self_test import SelfTester

logger = logging.getLogger(__name__)


class VoiceDisorderPipeline:
    """End-to-end pipeline for voice disorder detection."""

    def __init__(
        self,
        mode: str = config.MODE_BINARY,
        backend: str = config.BACKEND_ENSEMBLE,
        dbdir: Optional[str] = None,
        download_mode: str = "lazy",
    ):
        self.mode = mode
        self.backend = backend
        self.model = VoiceDisorderModel(mode=mode, backend=backend)
        self.tester = SelfTester(self.model)
        self.feedback = FeedbackManager(self.model)
        self.domain_monitor = DomainMonitor()
        self._loader = None
        self._dbdir = dbdir
        self._download_mode = download_mode

    @property
    def loader(self) -> VoiceDataLoader:
        if self._loader is None:
            self._loader = VoiceDataLoader(
                dbdir=self._dbdir, download_mode=self._download_mode,
            )
        return self._loader

    # ---- Training ----

    def train(
        self,
        max_samples: Optional[int] = None,
        use_cache: bool = True,
        run_evaluation: bool = True,
        augment: bool = False,
        params: Optional[dict] = None,
    ) -> dict:
        """Full training pipeline with patient-level evaluation.

        Parameters
        ----------
        params : dict, optional
            Hyperparameters to use (e.g. from optimize()). If None, uses defaults.
        """
        result = {}
        logger.info("=== Training pipeline (mode=%s, backend=%s) ===", self.mode, self.backend)

        X, y, session_ids, speaker_ids, metadata = self.loader.extract_dataset(
            mode=self.mode, max_samples=max_samples,
            use_cache=use_cache, augment=augment,
        )

        # Apply feature selection if dataset is large enough
        if len(X) >= 50 and X.shape[1] > 100:
            X, selector_info = self._apply_feature_selection(X, y)
            result["feature_selection"] = selector_info

        train_meta = self.model.train(
            X, y, session_ids=session_ids,
            speaker_ids=speaker_ids, params=params,
        )
        result["training"] = train_meta

        if run_evaluation and len(X) >= 20:
            eval_result = self.tester.run_full_evaluation(X, y, speaker_ids=speaker_ids)
            result["evaluation"] = {
                k: eval_result.get(k)
                for k in ["accuracy", "sensitivity", "specificity", "f1_weighted",
                           "auc_roc", "pr_auc", "brier_score", "ece", "split"]
            }

        # Fit domain monitor on training distribution
        try:
            self.domain_monitor.fit(X)
            self.domain_monitor.save()
            result["domain_monitor"] = "fitted"
        except Exception as e:
            logger.warning("Failed to fit domain monitor: %s", e)

        model_path = self.model.save()
        result["model_path"] = str(model_path)

        importance = self.model.get_feature_importance()
        if importance:
            result["top_features"] = dict(list(importance.items())[:10])

        logger.info("=== Training pipeline complete ===")
        return result

    def _apply_feature_selection(
        self, X: np.ndarray, y: np.ndarray,
        k: int = 150,
    ) -> tuple[np.ndarray, dict]:
        """Apply feature selection using mutual information.

        Selects the top k most informative features to reduce noise
        and improve generalization.
        """
        from sklearn.feature_selection import SelectKBest, mutual_info_classif

        actual_k = min(k, X.shape[1])
        selector = SelectKBest(mutual_info_classif, k=actual_k)
        X_selected = selector.fit_transform(X, y)

        # Store selector in model for inference
        self.model._feature_selector = selector

        selected_mask = selector.get_support()
        n_selected = int(selected_mask.sum())

        from .feature_extractor import get_feature_names
        names = get_feature_names()
        if len(names) == X.shape[1]:
            selected_names = [n for n, s in zip(names, selected_mask) if s]
        else:
            selected_names = []

        logger.info(
            "Feature selection: %d -> %d features (mutual information)",
            X.shape[1], n_selected,
        )

        return X_selected, {
            "method": "mutual_info",
            "original_features": int(X.shape[1]),
            "selected_features": n_selected,
            "top_selected": selected_names[:20] if selected_names else [],
        }

    # ---- Baseline comparison ----

    def compare_baselines(
        self, max_samples: Optional[int] = None,
    ) -> dict:
        """Compare all backends with patient-level CV."""
        X, y, _, speaker_ids, _ = self.loader.extract_dataset(
            mode=self.mode, max_samples=max_samples, use_cache=True,
        )

        results = {}
        for backend_name in [config.BACKEND_ENSEMBLE, config.BACKEND_LOGREG]:
            logger.info("Evaluating baseline: %s", backend_name)
            m = VoiceDisorderModel(mode=self.mode, backend=backend_name)
            t = SelfTester(m)
            cv = t.run_cross_validation(X, y, speaker_ids=speaker_ids, n_folds=config.CV_FOLDS)
            results[backend_name] = {
                "accuracy": cv["accuracy_mean"],
                "accuracy_std": cv["accuracy_std"],
                "f1": cv["f1_mean"],
                "sensitivity": cv["sensitivity_mean"],
                "specificity": cv["specificity_mean"],
                "auc_roc": cv["auc_roc_mean"],
                "pr_auc": cv["pr_auc_mean"],
                "brier": cv["brier_mean"],
            }

        # CNN baseline with actual evaluation
        logger.info("Evaluating baseline: cnn (MLP on mel-spectrogram)")
        try:
            from sklearn.model_selection import GroupShuffleSplit

            # Use a simple train/test split for CNN
            if speaker_ids is not None and len(speaker_ids) == len(y):
                groups = np.array(speaker_ids)
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
                train_idx, test_idx = next(gss.split(X, y, groups))
            else:
                from sklearn.model_selection import train_test_split
                indices = np.arange(len(y))
                train_idx, test_idx = train_test_split(
                    indices, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y,
                )

            # For CNN we need spectrograms, not pre-extracted features.
            # Since we only have features here, use X directly with an MLP.
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler as SS

            scaler = SS()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            mlp = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu", solver="adam", max_iter=300,
                early_stopping=True, validation_fraction=0.15,
                random_state=config.RANDOM_STATE,
            )
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_test)
            y_proba = mlp.predict_proba(X_test)

            cnn_acc = accuracy_score(y_test, y_pred)
            cnn_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            try:
                cnn_auc = roc_auc_score(y_test, y_proba[:, 1]) if y_proba.shape[1] == 2 else 0.0
            except Exception:
                cnn_auc = 0.0

            results["cnn"] = {
                "accuracy": round(float(cnn_acc), 4),
                "accuracy_std": 0.0,
                "f1": round(float(cnn_f1), 4),
                "sensitivity": 0.0,
                "specificity": 0.0,
                "auc_roc": round(float(cnn_auc), 4),
                "pr_auc": 0.0,
                "brier": 0.0,
                "note": "MLP(256,128) on pre-extracted features",
            }
        except Exception as e:
            logger.warning("CNN baseline evaluation failed: %s", e)
            results["cnn"] = {"error": str(e)}

        return results

    # ---- Prediction ----

    def predict_from_audio(self, audio: np.ndarray, sr: int) -> dict:
        """Predict with abstain support and optional OOD warning."""
        self._ensure_model_loaded()
        features = extract_all_features(audio, sr)
        X = features.reshape(1, -1)

        # Apply feature selector if available
        if self.model._feature_selector is not None:
            try:
                X = self.model._feature_selector.transform(X)
            except Exception:
                pass

        results = self.model.predict_with_confidence(X)
        prediction = results[0]

        # Attach OOD warning if domain monitor is available
        self._attach_ood_warning(prediction, features)

        return prediction

    def predict_from_file(self, audio_path: str) -> dict:
        """Predict from an audio file."""
        import librosa
        self._ensure_model_loaded()
        audio, sr = librosa.load(audio_path, sr=None)
        return self.predict_from_audio(audio, sr)

    def predict_from_session(self, session_id: int) -> dict:
        """Predict using recordings from a database session."""
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
            raise ValueError(f"No usable recordings in session {session_id}")

        combined = np.mean(features_list, axis=0)
        X = combined.reshape(1, -1)

        if self.model._feature_selector is not None:
            try:
                X = self.model._feature_selector.transform(X)
            except Exception:
                pass

        results = self.model.predict_with_confidence(X)
        prediction = results[0]

        prediction["session_id"] = session_id
        prediction["actual_type"] = session.type
        if session.pathologies:
            prediction["actual_pathologies"] = [p.name for p in session.pathologies]
        return prediction

    # ---- Feedback ----

    def correct_prediction(
        self, audio: np.ndarray, sr: int, correct_label: int,
        session_id: Optional[int] = None, note: str = "",
    ) -> dict:
        self._ensure_model_loaded()
        features = extract_all_features(audio, sr)
        predicted = self.model.predict(features.reshape(1, -1))[0]
        correction = self.feedback.add_correction(
            features=features, predicted_label=int(predicted),
            correct_label=correct_label, session_id=session_id, note=note,
        )
        return {"correction": correction, "stats": self.feedback.get_correction_stats()}

    def apply_feedback(self, full_retrain: bool = False) -> dict:
        result = self.feedback.apply_corrections(full_retrain=full_retrain)
        if not full_retrain:
            self.model.save()
        return result

    # ---- Self-Testing ----

    def self_test(
        self, max_samples: Optional[int] = None, test_type: str = "full",
    ) -> dict:
        X, y, _, speaker_ids, metadata = self.loader.extract_dataset(
            mode=self.mode, max_samples=max_samples, use_cache=True,
        )
        if test_type == "quick":
            self._ensure_model_loaded()
            return self.tester.run_quick_test(X, y)
        elif test_type == "cv":
            return self.tester.run_cross_validation(X, y, speaker_ids=speaker_ids)
        elif test_type == "subgroups":
            return self.tester.run_subgroup_analysis(X, y, metadata, speaker_ids=speaker_ids)
        else:
            return self.tester.run_full_evaluation(X, y, speaker_ids=speaker_ids)

    def optimize(self, max_samples: Optional[int] = None, n_iter: int = 20) -> dict:
        """Find optimal hyperparameters and retrain with them."""
        X, y, session_ids, speaker_ids, _ = self.loader.extract_dataset(
            mode=self.mode, max_samples=max_samples, use_cache=True,
        )
        opt_result = self.tester.optimize_hyperparameters(
            X, y, speaker_ids=speaker_ids, n_iter=n_iter,
        )

        # Retrain the model with the best hyperparameters found
        best_params = opt_result.get("best_params", {})
        if best_params:
            logger.info("Retraining with optimized hyperparameters: %s", best_params)
            self.model.train(
                X, y, session_ids=session_ids,
                speaker_ids=speaker_ids, params=best_params,
            )
        else:
            self.model.train(X, y, session_ids=session_ids, speaker_ids=speaker_ids)

        self.model.save()
        return opt_result

    # ---- SHAP ----

    def explain(self, max_samples: Optional[int] = None) -> dict:
        """Run SHAP analysis on the model."""
        from .interpretability import compute_shap_values

        self._ensure_model_loaded()
        X, y, _, _, _ = self.loader.extract_dataset(
            mode=self.mode, max_samples=max_samples, use_cache=True,
        )
        return compute_shap_values(self.model, X)

    # ---- Calibration & Threshold Optimization ----

    def calibrate(
        self,
        max_samples: Optional[int] = None,
        method: str = "isotonic",
    ) -> dict:
        """Run post-hoc calibration and save calibrator for inference."""
        from .calibration import calibrate_model

        X, y, _, speaker_ids, _ = self.loader.extract_dataset(
            mode=self.mode, max_samples=max_samples, use_cache=True,
        )
        result = calibrate_model(
            self.model, X, y, speaker_ids=speaker_ids, method=method,
        )

        # Load the saved calibrator into the model for future predictions
        cal_file = config.MODELS_DIR / "calibrator.joblib"
        if cal_file.exists():
            import joblib
            self.model._calibrator = joblib.load(cal_file)
            self.model.save()
            logger.info("Calibrator integrated into model — future predictions will use calibrated probabilities.")

        return result

    # ---- Domain Monitoring ----

    def fit_domain_monitor(self, max_samples: Optional[int] = None) -> dict:
        """Fit domain monitor on training data distribution."""
        X, y, _, _, _ = self.loader.extract_dataset(
            mode=self.mode, max_samples=max_samples, use_cache=True,
        )
        self.domain_monitor.fit(X)
        monitor_path = self.domain_monitor.save()
        return {
            "status": "fitted",
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "mahalanobis_threshold": self.domain_monitor._maha_threshold,
            "monitor_file": str(monitor_path),
        }

    def check_drift(self, max_samples: Optional[int] = None) -> dict:
        """Check for distribution drift against training data."""
        self._ensure_domain_monitor()
        X, y, _, _, _ = self.loader.extract_dataset(
            mode=self.mode, max_samples=max_samples, use_cache=True,
        )
        return self.domain_monitor.check_drift(X)

    def check_ood_sample(self, audio: np.ndarray, sr: int) -> dict:
        """Check if a single audio sample is out-of-distribution."""
        self._ensure_domain_monitor()
        features = extract_all_features(audio, sr)
        ood_results = self.domain_monitor.check_ood(features.reshape(1, -1))
        return ood_results[0]

    # ---- Reproducible Report ----

    def generate_report(
        self,
        max_samples: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Generate a full reproducible evaluation report (JSON + Markdown)."""
        from .calibration import calibrate_model
        from .report_generator import generate_report as _gen

        X, y, _, speaker_ids, metadata = self.loader.extract_dataset(
            mode=self.mode, max_samples=max_samples, use_cache=True,
        )

        # Train model for evaluation
        self.model.train(X, y, speaker_ids=speaker_ids)

        # Run calibration
        cal_results = None
        try:
            cal_results = calibrate_model(
                self.model, X, y, speaker_ids=speaker_ids,
            )
        except Exception as e:
            logger.warning("Calibration failed during report generation: %s", e)

        from pathlib import Path
        out_dir = Path(output_dir) if output_dir else None

        return _gen(
            model=self.model,
            X=X, y=y,
            speaker_ids=speaker_ids,
            metadata=metadata,
            output_dir=out_dir,
            calibration_results=cal_results,
        )

    # ---- Status ----

    def status(self) -> dict:
        result = {
            "mode": self.mode,
            "backend": self.backend,
            "model_trained": self.model.is_trained,
            "model_file_exists": config.model_path(self.mode, self.backend).exists(),
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

    def _ensure_model_loaded(self) -> None:
        if not self.model.is_trained:
            model_file = config.model_path(self.mode, self.backend)
            if model_file.exists():
                self.model.load()
            else:
                raise RuntimeError("No trained model available. Run 'train' first.")

    def _ensure_domain_monitor(self) -> None:
        if not self.domain_monitor.is_fitted:
            from .domain_monitor import MONITOR_FILE
            if MONITOR_FILE.exists():
                self.domain_monitor.load()
            else:
                raise RuntimeError(
                    "Domain monitor not fitted. Run 'fit-monitor' first."
                )

    def _attach_ood_warning(self, prediction: dict, features: np.ndarray) -> None:
        """Attach OOD warning to prediction if domain monitor is available."""
        if self.domain_monitor.is_fitted:
            ood = self.domain_monitor.check_ood(features.reshape(1, -1))[0]
            prediction["ood_warning"] = ood["ood"]
            if ood["ood"]:
                prediction["ood_detail"] = ood
            return

        # Try loading persisted monitor
        from .domain_monitor import MONITOR_FILE
        if MONITOR_FILE.exists():
            try:
                self.domain_monitor.load()
                ood = self.domain_monitor.check_ood(features.reshape(1, -1))[0]
                prediction["ood_warning"] = ood["ood"]
                if ood["ood"]:
                    prediction["ood_detail"] = ood
            except Exception:
                pass
