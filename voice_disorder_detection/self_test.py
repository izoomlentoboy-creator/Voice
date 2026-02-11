"""Self-testing, evaluation, and optimization.

Key improvements over naive evaluation:
  - Patient-level (GroupKFold) splitting to prevent data leakage
  - Medical metrics: sensitivity, specificity, PR-AUC, Brier score, ECE
  - Subgroup analysis by gender and age
  - Regression detection and performance history
"""

import json
import logging
import time
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    train_test_split,
)

from . import config
from .model import VoiceDisorderModel

logger = logging.getLogger(__name__)


class SelfTester:
    """Automated testing and optimization for the voice disorder model."""

    def __init__(self, model: VoiceDisorderModel):
        self.model = model
        self.history: list[dict] = []
        self._load_history()

    # ------------------------------------------------------------------
    # Patient-level evaluation (P0 fix)
    # ------------------------------------------------------------------

    def run_full_evaluation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        speaker_ids: Optional[list[int]] = None,
        test_size: float = config.TEST_SIZE,
    ) -> dict:
        """Run evaluation with patient-level train/test split.

        When speaker_ids are provided, splits by speaker so that no
        speaker appears in both train and test sets (prevents identity
        leakage). Without speaker_ids, falls back to stratified split.
        """
        logger.info("Running full evaluation (test_size=%.2f)...", test_size)

        if speaker_ids is not None and len(speaker_ids) == len(y):
            groups = np.array(speaker_ids)
            gss = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=config.RANDOM_STATE,
            )
            train_idx, test_idx = next(gss.split(X, y, groups))
            split_method = "patient_level"
        else:
            train_idx, test_idx = train_test_split(
                np.arange(len(y)), test_size=test_size,
                random_state=config.RANDOM_STATE, stratify=y,
            )
            split_method = "stratified"
            logger.warning("No speaker_ids provided — using stratified split (risk of data leakage).")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        self.model.train(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        results = self._compute_metrics(y_test, y_pred, y_proba)
        results["split"] = {
            "method": split_method,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        }
        if speaker_ids is not None:
            train_speakers = set(np.array(speaker_ids)[train_idx])
            test_speakers = set(np.array(speaker_ids)[test_idx])
            results["split"]["train_speakers"] = len(train_speakers)
            results["split"]["test_speakers"] = len(test_speakers)
            results["split"]["speaker_overlap"] = len(train_speakers & test_speakers)

        self._record_result("full_evaluation", results)
        return results

    def run_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        speaker_ids: Optional[list[int]] = None,
        n_folds: int = config.CV_FOLDS,
    ) -> dict:
        """Run cross-validation with patient-level folds (GroupKFold)."""
        logger.info("Running %d-fold cross-validation...", n_folds)

        if speaker_ids is not None and len(speaker_ids) == len(y):
            groups = np.array(speaker_ids)
            unique_groups = len(set(speaker_ids))
            actual_folds = min(n_folds, unique_groups)
            kf = GroupKFold(n_splits=actual_folds)
            split_iter = kf.split(X, y, groups)
            split_method = "patient_level_group_kfold"
        else:
            actual_folds = n_folds
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
            split_iter = kf.split(X, y)
            split_method = "stratified_kfold"
            logger.warning("No speaker_ids — falling back to StratifiedKFold.")

        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_model = VoiceDisorderModel(mode=self.model.mode, backend=self.model.backend)
            fold_model.train(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            y_proba = fold_model.predict_proba(X_test)

            metrics = self._compute_metrics(y_test, y_pred, y_proba)
            metrics["fold"] = fold_idx + 1
            fold_results.append(metrics)

            logger.info(
                "  Fold %d/%d: accuracy=%.4f, sensitivity=%.4f, specificity=%.4f",
                fold_idx + 1, actual_folds,
                metrics["accuracy"],
                metrics.get("sensitivity", 0),
                metrics.get("specificity", 0),
            )

        agg = {
            "n_folds": actual_folds,
            "split_method": split_method,
            "accuracy_mean": float(np.mean([r["accuracy"] for r in fold_results])),
            "accuracy_std": float(np.std([r["accuracy"] for r in fold_results])),
            "f1_mean": float(np.mean([r["f1_weighted"] for r in fold_results])),
            "f1_std": float(np.std([r["f1_weighted"] for r in fold_results])),
            "sensitivity_mean": float(np.mean([r.get("sensitivity", 0) for r in fold_results])),
            "specificity_mean": float(np.mean([r.get("specificity", 0) for r in fold_results])),
            "precision_mean": float(np.mean([r["precision_weighted"] for r in fold_results])),
            "recall_mean": float(np.mean([r["recall_weighted"] for r in fold_results])),
            "auc_roc_mean": float(np.mean([r.get("auc_roc", 0) or 0 for r in fold_results])),
            "pr_auc_mean": float(np.mean([r.get("pr_auc", 0) or 0 for r in fold_results])),
            "brier_mean": float(np.mean([r.get("brier_score", 1) for r in fold_results])),
            "folds": fold_results,
        }

        self._record_result("cross_validation", agg)
        logger.info(
            "CV complete: accuracy=%.4f (+/- %.4f), sensitivity=%.4f, specificity=%.4f",
            agg["accuracy_mean"], agg["accuracy_std"],
            agg["sensitivity_mean"], agg["specificity_mean"],
        )
        return agg

    # ------------------------------------------------------------------
    # Subgroup analysis (P2)
    # ------------------------------------------------------------------

    def run_subgroup_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: list[dict],
        speaker_ids: Optional[list[int]] = None,
    ) -> dict:
        """Evaluate model performance across demographic subgroups.

        Trains on full data, evaluates on subgroup slices.
        """
        logger.info("Running subgroup analysis...")

        if speaker_ids is not None and len(speaker_ids) == len(y):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
            train_idx, test_idx = next(gss.split(X, y, np.array(speaker_ids)))
        else:
            train_idx, test_idx = train_test_split(
                np.arange(len(y)), test_size=0.2,
                random_state=config.RANDOM_STATE, stratify=y,
            )

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        meta_test = [metadata[i] for i in test_idx]

        self.model.train(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        results = {"overall": self._compute_metrics(y_test, y_pred, y_proba)}

        # By gender
        for gender in ["m", "w"]:
            mask = np.array([m.get("gender") == gender for m in meta_test])
            if mask.sum() >= 5:
                results[f"gender_{gender}"] = self._compute_metrics(
                    y_test[mask], y_pred[mask],
                    y_proba[mask] if y_proba is not None else None,
                )
                results[f"gender_{gender}"]["n_samples"] = int(mask.sum())

        # By age groups
        age_groups = [(0, 30, "young"), (30, 50, "middle"), (50, 70, "senior"), (70, 200, "elderly")]
        for lo, hi, name in age_groups:
            mask = np.array([lo <= (m.get("age") or 0) < hi for m in meta_test])
            if mask.sum() >= 5:
                results[f"age_{name}"] = self._compute_metrics(
                    y_test[mask], y_pred[mask],
                    y_proba[mask] if y_proba is not None else None,
                )
                results[f"age_{name}"]["n_samples"] = int(mask.sum())
                results[f"age_{name}"]["age_range"] = f"{lo}-{hi}"

        self._record_result("subgroup_analysis", results)
        return results

    # ------------------------------------------------------------------
    # Quick test, regression, optimization (same as before + medical metrics)
    # ------------------------------------------------------------------

    def run_quick_test(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Quick sanity check: predict on known data."""
        logger.info("Running quick self-test...")
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        result = {
            "type": "quick_test",
            "accuracy": round(float(accuracy), 4),
            "n_samples": int(len(y)),
            "passed": accuracy > 0.5,
        }
        if not result["passed"]:
            logger.warning("Quick test FAILED: accuracy %.4f is below threshold", accuracy)
        else:
            logger.info("Quick test PASSED: accuracy %.4f", accuracy)
        return result

    def check_regression(self) -> dict:
        """Check if recent performance is worse than historical best."""
        if len(self.history) < 2:
            return {"status": "insufficient_history", "message": "Need at least 2 evaluation records."}

        def _get_acc(r):
            res = r.get("results", {})
            return res.get("accuracy", res.get("accuracy_mean", 0))

        best_acc = max(_get_acc(r) for r in self.history)
        current_acc = _get_acc(self.history[-1])
        regression = best_acc - current_acc
        is_regression = regression > 0.02

        result = {
            "status": "regression_detected" if is_regression else "ok",
            "current_accuracy": round(current_acc, 4),
            "best_accuracy": round(best_acc, 4),
            "delta": round(-regression, 4),
        }
        if is_regression:
            logger.warning("Performance regression: current=%.4f, best=%.4f", current_acc, best_acc)
        return result

    def optimize_hyperparameters(
        self, X: np.ndarray, y: np.ndarray,
        speaker_ids: Optional[list[int]] = None, n_iter: int = 20,
    ) -> dict:
        """Random search hyperparameter optimization with patient-level split."""
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
        from sklearn.svm import SVC

        logger.info("Starting hyperparameter optimization (%d iterations)...", n_iter)
        rng = np.random.RandomState(config.RANDOM_STATE)

        if speaker_ids is not None and len(speaker_ids) == len(y):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
            train_idx, val_idx = next(gss.split(X, y, np.array(speaker_ids)))
        else:
            train_idx, val_idx = train_test_split(
                np.arange(len(y)), test_size=0.2, random_state=config.RANDOM_STATE, stratify=y,
            )

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = self.model.scaler
        if not hasattr(scaler, "mean_") or scaler.mean_ is None:
            scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_val_s = scaler.transform(X_val)
        le = self.model.label_encoder
        if not hasattr(le, "classes_") or le.classes_ is None:
            le.fit(y)
        y_train_e = le.transform(y_train)
        y_val_e = le.transform(y_val)

        best_score = 0.0
        best_params = {}

        for i in range(n_iter):
            params = {
                "svm_C": float(rng.choice([0.1, 1.0, 10.0, 100.0])),
                "svm_gamma": str(rng.choice(["scale", "auto"])),
                "rf_n_estimators": int(rng.choice([100, 200, 300, 500])),
                "rf_max_depth": int(rng.choice([5, 10, 20, 0])) or None,
                "gb_n_estimators": int(rng.choice([100, 200, 300])),
                "gb_learning_rate": float(rng.choice([0.01, 0.05, 0.1, 0.2])),
                "gb_max_depth": int(rng.choice([3, 5, 7])),
            }
            try:
                svm = SVC(C=params["svm_C"], gamma=params["svm_gamma"], kernel="rbf",
                          probability=True, class_weight="balanced", random_state=config.RANDOM_STATE)
                rf = RandomForestClassifier(n_estimators=params["rf_n_estimators"],
                                            max_depth=params["rf_max_depth"], class_weight="balanced",
                                            random_state=config.RANDOM_STATE, n_jobs=-1)
                gb = GradientBoostingClassifier(n_estimators=params["gb_n_estimators"],
                                                learning_rate=params["gb_learning_rate"],
                                                max_depth=params["gb_max_depth"],
                                                random_state=config.RANDOM_STATE)
                ensemble = VotingClassifier(estimators=[("svm", svm), ("rf", rf), ("gb", gb)],
                                            voting="soft", n_jobs=-1)
                ensemble.fit(X_train_s, y_train_e)
                score = f1_score(y_val_e, ensemble.predict(X_val_s), average="weighted")
                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info("  Iter %d/%d: new best F1=%.4f", i + 1, n_iter, score)
            except Exception as e:
                logger.warning("  Iter %d failed: %s", i + 1, e)

        result = {"best_f1": round(float(best_score), 4), "best_params": best_params, "iterations": n_iter}
        self._record_result("optimization", result)
        return result

    def get_performance_summary(self) -> dict:
        """Summarize all historical test results."""
        if not self.history:
            return {"message": "No test history available."}
        evaluations = [r for r in self.history if r["test_type"] == "full_evaluation"]
        cv_runs = [r for r in self.history if r["test_type"] == "cross_validation"]
        summary = {
            "total_tests": len(self.history),
            "full_evaluations": len(evaluations),
            "cross_validations": len(cv_runs),
        }
        if evaluations:
            latest = evaluations[-1]["results"]
            summary["latest_evaluation"] = {
                "accuracy": latest.get("accuracy"),
                "sensitivity": latest.get("sensitivity"),
                "specificity": latest.get("specificity"),
                "auc_roc": latest.get("auc_roc"),
                "pr_auc": latest.get("pr_auc"),
                "split_method": latest.get("split", {}).get("method"),
            }
        if cv_runs:
            summary["latest_cv"] = {
                "accuracy": cv_runs[-1]["results"].get("accuracy_mean"),
                "sensitivity": cv_runs[-1]["results"].get("sensitivity_mean"),
                "specificity": cv_runs[-1]["results"].get("specificity_mean"),
            }
        return summary

    # ------------------------------------------------------------------
    # Internal: medical metrics computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute full set of classification metrics including medical ones."""
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        # Binary-specific medical metrics
        unique_classes = sorted(set(y_true))
        if len(unique_classes) == 2:
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            metrics["sensitivity"] = round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4)
            metrics["specificity"] = round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4)
            metrics["ppv"] = round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4)
            metrics["npv"] = round(float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0, 4)
            metrics["false_positive_rate"] = round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4)
            metrics["false_negative_rate"] = round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0, 4)

        # Probabilistic metrics
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    proba_pos = y_proba[:, 1]
                    metrics["auc_roc"] = round(float(roc_auc_score(y_true, proba_pos)), 4)
                    metrics["pr_auc"] = round(float(average_precision_score(y_true, proba_pos)), 4)
                    metrics["brier_score"] = round(float(brier_score_loss(y_true, proba_pos)), 4)

                    # Expected Calibration Error (ECE)
                    metrics["ece"] = round(float(self._compute_ece(y_true, proba_pos)), 4)
                else:
                    metrics["auc_roc"] = round(float(roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted",
                    )), 4)
            except Exception as e:
                logger.debug("Probabilistic metric computation failed: %s", e)
                metrics["auc_roc"] = None

        # Full classification report
        metrics["classification_report"] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0,
        )

        return metrics

    @staticmethod
    def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
        return ece / len(y_true) if len(y_true) > 0 else 0.0

    def _record_result(self, test_type: str, results: dict) -> None:
        record = {
            "test_type": test_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.model.mode,
            "backend": self.model.backend,
            "results": results,
        }
        self.history.append(record)
        self._save_history()

    def _save_history(self) -> None:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.PERFORMANCE_HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def _load_history(self) -> None:
        if config.PERFORMANCE_HISTORY_FILE.exists():
            try:
                with open(config.PERFORMANCE_HISTORY_FILE) as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []
