"""High-level pipeline that ties together all components.

Provides a single interface for training, prediction, feedback,
self-testing, SHAP analysis, baseline comparison, calibration,
domain monitoring, and reproducible report generation.
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
    ) -> dict:
        """Full training pipeline with patient-level evaluation."""
        result = {}
        logger.info("=== Training pipeline (mode=%s, backend=%s) ===", self.mode, self.backend)

        X, y, session_ids, speaker_ids, metadata = self.loader.extract_dataset(
            mode=self.mode, max_samples=max_samples,
            use_cache=use_cache, augment=augment,
        )

        train_meta = self.model.train(X, y, session_ids=session_ids, speaker_ids=speaker_ids)
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

    # ---- Baseline comparison ----

    def compare_baselines(
        self, max_samples: Optional[int] = None,
    ) -> dict:
        """Compare ensemble vs LogReg baseline with patient-level CV."""
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

        # CNN baseline
        try:

            logger.info("Evaluating baseline: cnn (MLP on mel-spectrogram)")
            # This requires re-extracting spectrograms — skip if too slow
            results["cnn"] = {"note": "CNN baseline requires separate spectrogram extraction. Run with --backend cnn."}
        except Exception:
            pass

        return results

    # ---- Prediction ----

    def predict_from_audio(self, audio: np.ndarray, sr: int) -> dict:
        """Predict with abstain support and optional OOD warning."""
        self._ensure_model_loaded()
        features = extract_all_features(audio, sr)
        results = self.model.predict_with_confidence(features.reshape(1, -1))
        prediction = results[0]

        # Attach OOD warning if domain monitor is available
        if self.domain_monitor.is_fitted:
            ood = self.domain_monitor.check_ood(features.reshape(1, -1))[0]
            prediction["ood_warning"] = ood["ood"]
            if ood["ood"]:
                prediction["ood_detail"] = ood
        elif DomainMonitor is not None:
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

        combined = np.mean(features_list, axis=0).reshape(1, -1)
        results = self.model.predict_with_confidence(combined)
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
        X, y, session_ids, speaker_ids, _ = self.loader.extract_dataset(
            mode=self.mode, max_samples=max_samples, use_cache=True,
        )
        opt_result = self.tester.optimize_hyperparameters(X, y, speaker_ids=speaker_ids, n_iter=n_iter)
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
        """Run post-hoc calibration and threshold optimization."""
        from .calibration import calibrate_model

        X, y, _, speaker_ids, _ = self.loader.extract_dataset(
            mode=self.mode, max_samples=max_samples, use_cache=True,
        )
        return calibrate_model(
            self.model, X, y, speaker_ids=speaker_ids, method=method,
        )

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
        """Check for distribution drift against training data.

        Loads the saved domain monitor and compares current data batch.
        """
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
