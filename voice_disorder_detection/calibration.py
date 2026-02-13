"""Probability calibration and decision threshold optimization.

Three responsibilities:
  1. **Reliability diagram data** — bin-level accuracy vs confidence.
  2. **Post-hoc calibration** — Platt scaling and isotonic regression to
     improve probability estimates on a held-out calibration set.
  3. **Threshold optimization** — find the operating point that maximizes
     a chosen clinical criterion (Youden's J, sensitivity @ fixed
     specificity, cost-weighted, etc.).
"""

import logging
from typing import Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit

from . import config
from .model import VoiceDisorderModel

logger = logging.getLogger(__name__)

CALIBRATOR_FILE = config.MODELS_DIR / "calibrator.joblib"


# ------------------------------------------------------------------
# Reliability diagram
# ------------------------------------------------------------------

def compute_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute data for a reliability (calibration) diagram.

    Returns
    -------
    dict with:
      - bins: list of {bin_lower, bin_upper, avg_confidence, avg_accuracy, count}
      - ece: float (Expected Calibration Error)
      - mce: float (Maximum Calibration Error)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    mce = 0.0
    total = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({
                "bin_lower": round(float(lo), 2),
                "bin_upper": round(float(hi), 2),
                "avg_confidence": None,
                "avg_accuracy": None,
                "count": 0,
            })
            continue

        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        gap = abs(avg_acc - avg_conf)

        ece += count * gap
        mce = max(mce, gap)

        bins.append({
            "bin_lower": round(float(lo), 2),
            "bin_upper": round(float(hi), 2),
            "avg_confidence": round(avg_conf, 4),
            "avg_accuracy": round(avg_acc, 4),
            "count": count,
            "gap": round(gap, 4),
        })

    return {
        "bins": bins,
        "ece": round(ece / total, 4) if total > 0 else 0.0,
        "mce": round(float(mce), 4),
        "n_bins": n_bins,
        "n_samples": total,
    }


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE).

    Canonical implementation used by both calibration and self_test modules.
    """
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


# ------------------------------------------------------------------
# Post-hoc calibration
# ------------------------------------------------------------------

def calibrate_model(
    model: VoiceDisorderModel,
    X: np.ndarray,
    y: np.ndarray,
    speaker_ids: Optional[list[int]] = None,
    method: str = "isotonic",
) -> dict:
    """Apply post-hoc calibration (Platt or isotonic) on a held-out set.

    Splits data into train / calibration / test (60/20/20), retrains the
    base model on train, calibrates on calibration set, evaluates on test.

    Parameters
    ----------
    model : VoiceDisorderModel
    X, y : arrays
    speaker_ids : list, optional
    method : str
        'sigmoid' (Platt scaling) or 'isotonic'.

    Returns
    -------
    dict with before/after ECE, reliability diagrams, and the calibrator.
    """
    logger.info("Calibrating model (method=%s)...", method)

    # Three-way split: train (60%) / calibration (20%) / test (20%)
    if speaker_ids is not None and len(speaker_ids) == len(y):
        groups = np.array(speaker_ids)
        # First split: 80% train+cal, 20% test
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
        traincal_idx, test_idx = next(gss1.split(X, y, groups))
        # Second split: 75% train, 25% calibration (of 80% = 60/20)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=config.RANDOM_STATE + 1)
        train_sub, cal_sub = next(gss2.split(
            X[traincal_idx], y[traincal_idx], groups[traincal_idx],
        ))
        train_idx = traincal_idx[train_sub]
        cal_idx = traincal_idx[cal_sub]
    else:
        rng = np.random.RandomState(config.RANDOM_STATE)
        indices = rng.permutation(len(y))
        n_test = max(1, int(len(y) * 0.2))
        n_cal = max(1, int(len(y) * 0.2))
        test_idx = indices[:n_test]
        cal_idx = indices[n_test:n_test + n_cal]
        train_idx = indices[n_test + n_cal:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_cal, y_cal = X[cal_idx], y[cal_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Train base model (preserve speaker_ids for patient-level aware training)
    train_speaker_ids = [speaker_ids[i] for i in train_idx] if speaker_ids else None
    model.train(X_train, y_train, speaker_ids=train_speaker_ids)

    # Probabilities before calibration
    proba_before_test = model.predict_proba(X_test)

    # Encode labels for calibration
    y_cal_enc = model.label_encoder.transform(y_cal)
    y_test_enc = model.label_encoder.transform(y_test)

    # Fit calibrator on calibration set
    X_cal_scaled = model.scaler.transform(X_cal)
    X_test_scaled = model.scaler.transform(X_test)

    calibrated = CalibratedClassifierCV(
        model.model, method=method, cv="prefit",
    )
    calibrated.fit(X_cal_scaled, y_cal_enc)

    # Probabilities after calibration
    proba_after_test = calibrated.predict_proba(X_test_scaled)

    # Binary case: use positive class probability
    if proba_before_test.shape[1] == 2:
        prob_before = proba_before_test[:, 1]
        prob_after = proba_after_test[:, 1]
        y_binary = y_test_enc
    else:
        # Multiclass: use max probability vs correctness
        prob_before = proba_before_test.max(axis=1)
        prob_after = proba_after_test.max(axis=1)
        y_binary = (model.model.predict(X_test_scaled) == y_test_enc).astype(int)

    # Reliability diagrams
    rel_before = compute_reliability_diagram(y_binary, prob_before)
    rel_after = compute_reliability_diagram(y_binary, prob_after)

    # Threshold optimization (only for binary)
    threshold_results = {}
    if proba_before_test.shape[1] == 2:
        threshold_results = optimize_threshold(y_test_enc, prob_after)

    # Save calibrator
    joblib.dump(calibrated, CALIBRATOR_FILE)
    logger.info("Calibrated model saved to %s", CALIBRATOR_FILE)

    result = {
        "method": method,
        "split": {
            "train": len(train_idx),
            "calibration": len(cal_idx),
            "test": len(test_idx),
        },
        "ece_before": rel_before["ece"],
        "mce_before": rel_before["mce"],
        "ece_after": rel_after["ece"],
        "mce_after": rel_after["mce"],
        "ece_improvement": round(rel_before["ece"] - rel_after["ece"], 4),
        "reliability_before": rel_before,
        "reliability_after": rel_after,
        "threshold_optimization": threshold_results,
        "calibrator_file": str(CALIBRATOR_FILE),
    }

    logger.info(
        "Calibration complete. ECE: %.4f -> %.4f (improvement: %.4f)",
        rel_before["ece"], rel_after["ece"], result["ece_improvement"],
    )
    return result


# ------------------------------------------------------------------
# Threshold optimization
# ------------------------------------------------------------------

def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitivity_target: float = 0.95,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
) -> dict:
    """Find optimal decision thresholds under multiple criteria.

    Parameters
    ----------
    y_true : binary labels (0/1)
    y_prob : predicted probability of positive class
    sensitivity_target : float
        Target sensitivity for the 'sensitivity_at_target' criterion.
    cost_fp, cost_fn : float
        Relative costs for cost-weighted optimization.

    Returns
    -------
    dict with optimal thresholds per criterion.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    results = {}

    best_youden = {"j": -1, "threshold": 0.5}
    best_cost = {"cost": np.inf, "threshold": 0.5}
    best_sens_at_target = {"threshold": 0.5, "specificity": 0.0}
    best_f1 = {"f1": 0.0, "threshold": 0.5}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

        # Youden's J = sensitivity + specificity - 1
        j = sens + spec - 1
        if j > best_youden["j"]:
            best_youden = {
                "j": round(j, 4), "threshold": round(float(t), 3),
                "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            }

        # Cost-weighted
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost["cost"]:
            best_cost = {
                "cost": round(float(cost), 2), "threshold": round(float(t), 3),
                "sensitivity": round(sens, 4), "specificity": round(spec, 4),
                "cost_fp": cost_fp, "cost_fn": cost_fn,
            }

        # Sensitivity at target
        if sens >= sensitivity_target and spec > best_sens_at_target["specificity"]:
            best_sens_at_target = {
                "threshold": round(float(t), 3),
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "target": sensitivity_target,
            }

        # Max F1
        if f1 > best_f1["f1"]:
            best_f1 = {
                "f1": round(f1, 4), "threshold": round(float(t), 3),
                "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            }

    results["youden_j"] = best_youden
    results["cost_weighted"] = best_cost
    results[f"sensitivity_at_{sensitivity_target}"] = best_sens_at_target
    results["max_f1"] = best_f1

    # Default threshold (0.5) for reference
    y_default = (y_prob >= 0.5).astype(int)
    tp_d = int(((y_default == 1) & (y_true == 1)).sum())
    tn_d = int(((y_default == 0) & (y_true == 0)).sum())
    fp_d = int(((y_default == 1) & (y_true == 0)).sum())
    fn_d = int(((y_default == 0) & (y_true == 1)).sum())
    sens_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0
    spec_d = tn_d / (tn_d + fp_d) if (tn_d + fp_d) > 0 else 0.0
    results["default_0.5"] = {
        "threshold": 0.5,
        "sensitivity": round(sens_d, 4),
        "specificity": round(spec_d, 4),
    }

    logger.info(
        "Threshold optimization: Youden=%.3f (J=%.4f), cost-weighted=%.3f, max-F1=%.3f",
        best_youden["threshold"], best_youden["j"],
        best_cost["threshold"], best_f1["threshold"],
    )
    return results
