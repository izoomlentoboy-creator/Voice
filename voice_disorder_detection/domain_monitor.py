"""Domain-shift and out-of-distribution (OOD) monitoring.

Provides three complementary mechanisms:
  1. **Mahalanobis distance** — parametric OOD detector based on the
     multivariate Gaussian fitted to training features.
  2. **Isolation Forest** — non-parametric anomaly detector.
  3. **Feature drift (KS-test + PSI)** — compare a batch of new inputs
     against the training distribution per-feature.

Usage:
    monitor = DomainMonitor()
    monitor.fit(X_train)          # fit during training
    monitor.save() / .load()      # persist across sessions

    # at prediction time
    ood = monitor.check_ood(X_new)        # per-sample OOD scores
    drift = monitor.check_drift(X_batch)  # batch-level drift report
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

from . import config

logger = logging.getLogger(__name__)

MONITOR_FILE = config.MODELS_DIR / "domain_monitor.joblib"
DRIFT_LOG_FILE = config.LOGS_DIR / "drift_history.json"


class DomainMonitor:
    """Monitors distribution shift and flags out-of-distribution inputs."""

    def __init__(self):
        self.is_fitted = False
        # Parametric (Mahalanobis)
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        # Non-parametric (Isolation Forest)
        self._iforest: Optional[IsolationForest] = None
        # Per-feature reference stats for drift detection
        self._ref_stats: Optional[dict] = None  # {feature_idx: {mean, std, quantiles, values_sample}}
        # Mahalanobis threshold (fitted on training data)
        self._maha_threshold: float = 0.0
        self._n_features: int = 0

    # ------------------------------------------------------------------
    # Fit on training data
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, contamination: float = 0.05) -> None:
        """Fit distribution model on the training feature matrix.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        contamination : float
            Expected fraction of outliers for Isolation Forest.
        """
        n, d = X_train.shape
        self._n_features = d
        logger.info("Fitting domain monitor on %d samples, %d features", n, d)

        # --- Mahalanobis ---
        self._mean = X_train.mean(axis=0)
        cov = np.cov(X_train, rowvar=False)
        # Regularize to avoid singular matrix
        cov += np.eye(d) * 1e-6
        try:
            self._cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self._cov_inv = np.linalg.pinv(cov)
            logger.warning("Covariance matrix singular — using pseudo-inverse.")

        # Set threshold as 99th percentile of training Mahalanobis distances
        train_dists = self._mahalanobis_distances(X_train)
        self._maha_threshold = float(np.percentile(train_dists, 99))

        # --- Isolation Forest ---
        self._iforest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=config.RANDOM_STATE,
            n_jobs=1,
        )
        self._iforest.fit(X_train)

        # --- Per-feature reference stats ---
        self._ref_stats = {}
        # Store a subsample for KS-test (max 500 samples to keep memory bounded)
        subsample_idx = np.random.RandomState(config.RANDOM_STATE).choice(
            n, size=min(n, 500), replace=False,
        )
        for j in range(d):
            col = X_train[:, j]
            self._ref_stats[j] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "q01": float(np.percentile(col, 1)),
                "q25": float(np.percentile(col, 25)),
                "q50": float(np.percentile(col, 50)),
                "q75": float(np.percentile(col, 75)),
                "q99": float(np.percentile(col, 99)),
                "ref_sample": X_train[subsample_idx, j].tolist(),
            }

        self.is_fitted = True
        logger.info(
            "Domain monitor fitted. Mahalanobis threshold (p99): %.2f",
            self._maha_threshold,
        )

    # ------------------------------------------------------------------
    # OOD detection (per-sample)
    # ------------------------------------------------------------------

    def check_ood(self, X: np.ndarray) -> list[dict]:
        """Check if individual samples are out-of-distribution.

        Returns a list of dicts, one per sample:
          - mahalanobis_distance: float
          - mahalanobis_ood: bool (distance > threshold)
          - iforest_score: float (lower = more anomalous)
          - iforest_ood: bool (score < 0 = anomaly)
          - ood: bool (combined — either detector flags it)
        """
        self._check_fitted()
        X = np.atleast_2d(X)

        maha_dists = self._mahalanobis_distances(X)
        iforest_scores = self._iforest.score_samples(X)
        iforest_preds = self._iforest.predict(X)  # 1=inlier, -1=outlier

        results = []
        for i in range(len(X)):
            maha_ood = bool(maha_dists[i] > self._maha_threshold)
            if_ood = bool(iforest_preds[i] == -1)
            results.append({
                "mahalanobis_distance": round(float(maha_dists[i]), 4),
                "mahalanobis_ood": maha_ood,
                "iforest_score": round(float(iforest_scores[i]), 4),
                "iforest_ood": if_ood,
                "ood": maha_ood or if_ood,
            })

        n_ood = sum(r["ood"] for r in results)
        if n_ood > 0:
            logger.warning("%d / %d samples flagged as OOD", n_ood, len(X))
        return results

    # ------------------------------------------------------------------
    # Feature drift detection (batch-level)
    # ------------------------------------------------------------------

    def check_drift(
        self,
        X_new: np.ndarray,
        ks_alpha: float = 0.05,
        psi_threshold: float = 0.2,
    ) -> dict:
        """Compare a batch of new data against the training distribution.

        Uses per-feature two-sample KS test and Population Stability Index (PSI).

        Parameters
        ----------
        X_new : array, shape (n_new, n_features)
        ks_alpha : float
            Significance level for KS test.
        psi_threshold : float
            PSI > this indicates significant drift.

        Returns
        -------
        dict with:
          - n_features_drifted_ks: int
          - n_features_drifted_psi: int
          - drift_detected: bool (more than 10% features drifted)
          - per_feature: list of drifted feature details
          - summary: human-readable string
        """
        self._check_fitted()
        X_new = np.atleast_2d(X_new)
        n_new = X_new.shape[0]
        if n_new < 10:
            return {"drift_detected": False, "reason": "batch too small for drift detection (need >= 10)"}

        drifted_features_ks = []
        drifted_features_psi = []
        per_feature = []

        for j in range(min(X_new.shape[1], self._n_features)):
            ref_sample = np.array(self._ref_stats[j]["ref_sample"])
            new_col = X_new[:, j]

            # KS test
            ks_stat, ks_p = stats.ks_2samp(ref_sample, new_col)
            ks_drifted = ks_p < ks_alpha

            # PSI
            psi_val = self._compute_psi(ref_sample, new_col)
            psi_drifted = psi_val > psi_threshold

            if ks_drifted or psi_drifted:
                info = {
                    "feature_index": j,
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_p_value": round(float(ks_p), 6),
                    "ks_drifted": ks_drifted,
                    "psi": round(float(psi_val), 4),
                    "psi_drifted": psi_drifted,
                    "ref_mean": self._ref_stats[j]["mean"],
                    "new_mean": round(float(np.mean(new_col)), 4),
                }
                per_feature.append(info)

            if ks_drifted:
                drifted_features_ks.append(j)
            if psi_drifted:
                drifted_features_psi.append(j)

        n_total = min(X_new.shape[1], self._n_features)
        ks_frac = len(drifted_features_ks) / n_total if n_total > 0 else 0
        psi_frac = len(drifted_features_psi) / n_total if n_total > 0 else 0
        drift_detected = ks_frac > 0.10 or psi_frac > 0.10

        result = {
            "n_samples": n_new,
            "n_features_total": n_total,
            "n_features_drifted_ks": len(drifted_features_ks),
            "n_features_drifted_psi": len(drifted_features_psi),
            "ks_drift_fraction": round(ks_frac, 4),
            "psi_drift_fraction": round(psi_frac, 4),
            "drift_detected": drift_detected,
            "per_feature": per_feature[:20],  # top 20 to keep report manageable
        }

        if drift_detected:
            result["summary"] = (
                f"DRIFT DETECTED: {len(drifted_features_ks)}/{n_total} features "
                f"failed KS test (alpha={ks_alpha}), "
                f"{len(drifted_features_psi)}/{n_total} exceeded PSI threshold ({psi_threshold})."
            )
            logger.warning(result["summary"])
        else:
            result["summary"] = "No significant distribution drift detected."

        self._log_drift(result)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the fitted monitor to disk."""
        self._check_fitted()
        out = path or MONITOR_FILE
        data = {
            "mean": self._mean,
            "cov_inv": self._cov_inv,
            "iforest": self._iforest,
            "ref_stats": self._ref_stats,
            "maha_threshold": self._maha_threshold,
            "n_features": self._n_features,
        }
        joblib.dump(data, out)
        logger.info("Domain monitor saved to %s", out)
        return out

    def load(self, path: Optional[Path] = None) -> None:
        """Load a previously saved monitor."""
        src = path or MONITOR_FILE
        if not src.exists():
            raise FileNotFoundError(f"Domain monitor not found: {src}")
        data = joblib.load(src)
        self._mean = data["mean"]
        self._cov_inv = data["cov_inv"]
        self._iforest = data["iforest"]
        self._ref_stats = data["ref_stats"]
        self._maha_threshold = data["maha_threshold"]
        self._n_features = data["n_features"]
        self.is_fitted = True
        logger.info("Domain monitor loaded from %s", src)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mahalanobis_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for each row in X."""
        diff = X - self._mean
        left = diff @ self._cov_inv
        return np.sqrt(np.sum(left * diff, axis=1))

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index between two distributions."""
        eps = 1e-8
        # Use reference quantiles as bin edges for consistency
        edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        # Make edges unique
        edges = np.unique(edges)
        if len(edges) < 3:
            return 0.0

        ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
        cur_counts = np.histogram(current, bins=edges)[0].astype(float)

        ref_pct = ref_counts / ref_counts.sum() + eps
        cur_pct = cur_counts / cur_counts.sum() + eps

        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    def _log_drift(self, result: dict) -> None:
        """Append drift check result to history log."""
        import time as _time

        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        history = []
        if DRIFT_LOG_FILE.exists():
            try:
                with open(DRIFT_LOG_FILE) as f:
                    history = json.load(f)
            except Exception:
                pass

        # Keep serializable subset
        entry = {
            "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": result["n_samples"],
            "drift_detected": result["drift_detected"],
            "n_features_drifted_ks": result["n_features_drifted_ks"],
            "n_features_drifted_psi": result["n_features_drifted_psi"],
        }
        history.append(entry)
        # Keep last 100 entries
        history = history[-100:]
        with open(DRIFT_LOG_FILE, "w") as f:
            json.dump(history, f, indent=2)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Domain monitor not fitted. Call fit() or load() first.")
