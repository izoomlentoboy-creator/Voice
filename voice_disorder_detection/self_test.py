"""Self-testing and optimization system.

Provides automated evaluation, cross-validation, performance tracking,
regression detection, and hyperparameter optimization.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
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

    def run_full_evaluation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = config.TEST_SIZE,
    ) -> dict:
        """Run a comprehensive evaluation on a held-out test set.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (full dataset).
        y : np.ndarray
            Labels.
        test_size : float
            Fraction to hold out for testing.

        Returns
        -------
        dict
            Detailed evaluation results.
        """
        logger.info("Running full evaluation (test_size=%.2f)...", test_size)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=config.RANDOM_STATE,
            stratify=y,
        )

        # Train on the train split
        self.model.train(X_train, y_train)

        # Evaluate on the test split
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        results = self._compute_metrics(y_test, y_pred, y_proba)
        results["split"] = {
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        }

        self._record_result("full_evaluation", results)
        return results

    def run_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = config.CV_FOLDS,
    ) -> dict:
        """Run stratified k-fold cross-validation.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        n_folds : int

        Returns
        -------
        dict
            CV results with per-fold and aggregate metrics.
        """
        logger.info("Running %d-fold cross-validation...", n_folds)

        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=config.RANDOM_STATE,
        )

        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_model = VoiceDisorderModel(mode=self.model.mode)
            fold_model.train(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            y_proba = fold_model.predict_proba(X_test)

            metrics = self._compute_metrics(y_test, y_pred, y_proba)
            metrics["fold"] = fold_idx + 1
            fold_results.append(metrics)

            logger.info(
                "  Fold %d/%d: accuracy=%.4f, f1=%.4f",
                fold_idx + 1, n_folds,
                metrics["accuracy"], metrics["f1_weighted"],
            )

        # Aggregate
        agg = {
            "n_folds": n_folds,
            "accuracy_mean": float(np.mean([r["accuracy"] for r in fold_results])),
            "accuracy_std": float(np.std([r["accuracy"] for r in fold_results])),
            "f1_mean": float(np.mean([r["f1_weighted"] for r in fold_results])),
            "f1_std": float(np.std([r["f1_weighted"] for r in fold_results])),
            "precision_mean": float(np.mean([r["precision_weighted"] for r in fold_results])),
            "recall_mean": float(np.mean([r["recall_weighted"] for r in fold_results])),
            "folds": fold_results,
        }

        self._record_result("cross_validation", agg)
        logger.info(
            "CV complete: accuracy=%.4f (+/- %.4f), F1=%.4f (+/- %.4f)",
            agg["accuracy_mean"], agg["accuracy_std"],
            agg["f1_mean"], agg["f1_std"],
        )
        return agg

    def run_quick_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """Quick sanity check: predict on training data.

        Useful for detecting model loading issues or corruption.
        """
        logger.info("Running quick self-test...")
        y_pred = self.model.predict(X)

        accuracy = accuracy_score(y, y_pred)
        result = {
            "type": "quick_test",
            "accuracy": round(float(accuracy), 4),
            "n_samples": int(len(y)),
            "passed": accuracy > 0.5,  # sanity threshold
        }

        if not result["passed"]:
            logger.warning(
                "Quick test FAILED: accuracy %.4f is below threshold",
                accuracy,
            )
        else:
            logger.info("Quick test PASSED: accuracy %.4f", accuracy)

        return result

    def check_regression(self) -> dict:
        """Check if recent performance is worse than historical best.

        Returns
        -------
        dict
            Regression analysis result.
        """
        if len(self.history) < 2:
            return {
                "status": "insufficient_history",
                "message": "Need at least 2 evaluation records.",
            }

        recent = self.history[-1]
        best_acc = max(
            r.get("results", {}).get("accuracy", 0)
            if r["test_type"] == "full_evaluation"
            else r.get("results", {}).get("accuracy_mean", 0)
            for r in self.history
        )
        current_acc = recent.get("results", {}).get(
            "accuracy",
            recent.get("results", {}).get("accuracy_mean", 0),
        )

        regression = best_acc - current_acc
        is_regression = regression > 0.02  # 2% threshold

        result = {
            "status": "regression_detected" if is_regression else "ok",
            "current_accuracy": round(current_acc, 4),
            "best_accuracy": round(best_acc, 4),
            "delta": round(-regression, 4),
        }

        if is_regression:
            logger.warning(
                "Performance regression detected: current=%.4f, best=%.4f",
                current_acc, best_acc,
            )

        return result

    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 20,
    ) -> dict:
        """Simple random search for hyperparameter optimization.

        Tests different SVM/RF/GB configurations and returns the best.
        """
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            RandomForestClassifier,
            VotingClassifier,
        )
        from sklearn.svm import SVC

        logger.info("Starting hyperparameter optimization (%d iterations)...", n_iter)
        rng = np.random.RandomState(config.RANDOM_STATE)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2,
            random_state=config.RANDOM_STATE,
            stratify=y,
        )

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
                svm = SVC(
                    C=params["svm_C"], gamma=params["svm_gamma"],
                    kernel="rbf", probability=True,
                    class_weight="balanced",
                    random_state=config.RANDOM_STATE,
                )
                rf = RandomForestClassifier(
                    n_estimators=params["rf_n_estimators"],
                    max_depth=params["rf_max_depth"],
                    class_weight="balanced",
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                )
                gb = GradientBoostingClassifier(
                    n_estimators=params["gb_n_estimators"],
                    learning_rate=params["gb_learning_rate"],
                    max_depth=params["gb_max_depth"],
                    random_state=config.RANDOM_STATE,
                )
                ensemble = VotingClassifier(
                    estimators=[("svm", svm), ("rf", rf), ("gb", gb)],
                    voting="soft", n_jobs=-1,
                )
                ensemble.fit(X_train_s, y_train_e)
                score = f1_score(
                    y_val_e, ensemble.predict(X_val_s), average="weighted",
                )

                if score > best_score:
                    best_score = score
                    best_params = params
                    logger.info(
                        "  Iter %d/%d: new best F1=%.4f", i + 1, n_iter, score,
                    )
            except Exception as e:
                logger.warning("  Iter %d failed: %s", i + 1, e)

        result = {
            "best_f1": round(float(best_score), 4),
            "best_params": best_params,
            "iterations": n_iter,
        }

        self._record_result("optimization", result)
        logger.info(
            "Optimization complete: best F1=%.4f", best_score,
        )
        return result

    def get_performance_summary(self) -> dict:
        """Summarize all historical test results."""
        if not self.history:
            return {"message": "No test history available."}

        evaluations = [
            r for r in self.history if r["test_type"] == "full_evaluation"
        ]
        cv_runs = [
            r for r in self.history if r["test_type"] == "cross_validation"
        ]

        summary = {
            "total_tests": len(self.history),
            "full_evaluations": len(evaluations),
            "cross_validations": len(cv_runs),
        }

        if evaluations:
            accs = [r["results"]["accuracy"] for r in evaluations]
            summary["eval_accuracy_range"] = {
                "min": round(min(accs), 4),
                "max": round(max(accs), 4),
                "latest": round(accs[-1], 4),
            }

        if cv_runs:
            summary["latest_cv"] = {
                "accuracy": cv_runs[-1]["results"]["accuracy_mean"],
                "f1": cv_runs[-1]["results"]["f1_mean"],
            }

        return summary

    # --- Internal ---

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute a full set of classification metrics."""
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision_weighted": round(
                float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4,
            ),
            "recall_weighted": round(
                float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4,
            ),
            "f1_weighted": round(
                float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True, zero_division=0,
            ),
        }

        # AUC (binary or multi-class)
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:
                    metrics["auc_roc"] = round(
                        float(roc_auc_score(y_true, y_proba[:, 1])), 4,
                    )
                else:
                    metrics["auc_roc"] = round(
                        float(roc_auc_score(
                            y_true, y_proba, multi_class="ovr", average="weighted",
                        )), 4,
                    )
            except Exception:
                metrics["auc_roc"] = None

        return metrics

    def _record_result(self, test_type: str, results: dict) -> None:
        """Save a test result to history."""
        record = {
            "test_type": test_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.model.mode,
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
