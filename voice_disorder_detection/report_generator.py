"""Reproducible evaluation report generator.

Produces structured JSON and human-readable Markdown reports that capture
the full evaluation context: dataset stats, split method, all metrics,
configuration snapshot, feature importance, subgroup breakdowns,
calibration summary, and git revision.
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .model import VoiceDisorderModel
from .self_test import SelfTester

logger = logging.getLogger(__name__)

REPORTS_DIR = config.PROJECT_ROOT / "reports"


def _get_git_revision() -> str:
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _config_snapshot() -> dict:
    """Capture current configuration as a serializable dict."""
    return {
        "sample_rate": config.SAMPLE_RATE,
        "n_mfcc": config.N_MFCC,
        "n_fft": config.N_FFT,
        "hop_length": config.HOP_LENGTH,
        "n_mels": config.N_MELS,
        "utterances": config.UTTERANCES_FOR_TRAINING,
        "test_size": config.TEST_SIZE,
        "cv_folds": config.CV_FOLDS,
        "abstain_threshold": config.ABSTAIN_THRESHOLD,
        "random_state": config.RANDOM_STATE,
        "augment_noise_levels": config.AUGMENT_NOISE_LEVELS,
        "augment_pitch_steps": config.AUGMENT_PITCH_STEPS,
        "augment_time_stretch": config.AUGMENT_TIME_STRETCH,
    }


def generate_report(
    model: VoiceDisorderModel,
    X: np.ndarray,
    y: np.ndarray,
    speaker_ids: Optional[list[int]] = None,
    metadata: Optional[list[dict]] = None,
    output_dir: Optional[Path] = None,
    calibration_results: Optional[dict] = None,
) -> dict:
    """Generate a full reproducible evaluation report.

    Parameters
    ----------
    model : VoiceDisorderModel
        Trained model to evaluate.
    X, y : arrays
        Full dataset features and labels.
    speaker_ids : list, optional
        For patient-level splitting.
    metadata : list[dict], optional
        Per-sample metadata (gender, age) for subgroup analysis.
    output_dir : Path, optional
        Where to write reports. Defaults to PROJECT_ROOT/reports/.
    calibration_results : dict, optional
        Pre-computed calibration results to include.

    Returns
    -------
    dict : The full report data.
    """
    out = output_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tester = SelfTester(model)

    report = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_revision": _get_git_revision(),
            "mode": model.mode,
            "backend": model.backend,
        },
        "config": _config_snapshot(),
        "dataset": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "class_distribution": {},
            "n_unique_speakers": len(set(speaker_ids)) if speaker_ids else None,
        },
    }

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        report["dataset"]["class_distribution"][str(cls)] = int(cnt)

    # --- Full evaluation (patient-level split) ---
    logger.info("Report: running full evaluation...")
    report["evaluation"] = tester.run_full_evaluation(X, y, speaker_ids=speaker_ids)

    # --- Cross-validation ---
    logger.info("Report: running %d-fold cross-validation...", config.CV_FOLDS)
    report["cross_validation"] = tester.run_cross_validation(
        X, y, speaker_ids=speaker_ids, n_folds=config.CV_FOLDS,
    )

    # --- Subgroup analysis ---
    if metadata:
        logger.info("Report: running subgroup analysis...")
        report["subgroup_analysis"] = tester.run_subgroup_analysis(
            X, y, metadata, speaker_ids=speaker_ids,
        )

    # --- Feature importance ---
    importance = model.get_feature_importance()
    if importance:
        report["feature_importance"] = dict(list(importance.items())[:20])

    # --- Calibration ---
    if calibration_results:
        report["calibration"] = calibration_results

    # --- Regression check ---
    report["regression_check"] = tester.check_regression()

    # Write JSON
    json_path = out / f"report_{model.mode}_{model.backend}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("JSON report saved to %s", json_path)

    # Write Markdown
    md_path = out / f"report_{model.mode}_{model.backend}_{timestamp}.md"
    md = _render_markdown(report)
    with open(md_path, "w") as f:
        f.write(md)
    logger.info("Markdown report saved to %s", md_path)

    report["_files"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def _render_markdown(report: dict) -> str:
    """Render the report dict as a Markdown document."""
    lines = []
    meta = report["meta"]
    lines.append("# Voice Disorder Detection — Evaluation Report")
    lines.append("")
    lines.append(f"**Date:** {meta['timestamp']}  ")
    lines.append(f"**Git revision:** `{meta['git_revision']}`  ")
    lines.append(f"**Mode:** {meta['mode']}  ")
    lines.append(f"**Backend:** {meta['backend']}  ")
    lines.append("")

    # Dataset
    ds = report["dataset"]
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- **Samples:** {ds['n_samples']}")
    lines.append(f"- **Features:** {ds['n_features']}")
    if ds.get("n_unique_speakers"):
        lines.append(f"- **Unique speakers:** {ds['n_unique_speakers']}")
    lines.append(f"- **Class distribution:** {ds['class_distribution']}")
    lines.append("")

    # Evaluation
    ev = report.get("evaluation", {})
    lines.append("## Hold-out Evaluation")
    lines.append("")
    split = ev.get("split", {})
    lines.append(f"- **Split method:** {split.get('method', 'n/a')}")
    lines.append(f"- **Train/Test:** {split.get('train_size', '?')} / {split.get('test_size', '?')}")
    if split.get("speaker_overlap") is not None:
        lines.append(f"- **Speaker overlap:** {split['speaker_overlap']}")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key in ["accuracy", "sensitivity", "specificity", "ppv", "npv",
                 "f1_weighted", "auc_roc", "pr_auc", "brier_score", "ece",
                 "false_positive_rate", "false_negative_rate"]:
        val = ev.get(key)
        if val is not None:
            lines.append(f"| {key} | {val:.4f} |")
    lines.append("")

    # Confusion matrix
    cm = ev.get("confusion_matrix")
    if cm:
        lines.append("**Confusion matrix:**")
        lines.append("```")
        for row in cm:
            lines.append("  " + "  ".join(f"{v:5d}" for v in row))
        lines.append("```")
        lines.append("")

    # Cross-validation
    cv = report.get("cross_validation", {})
    if cv:
        lines.append("## Cross-Validation")
        lines.append("")
        lines.append(f"- **Folds:** {cv.get('n_folds')}")
        lines.append(f"- **Split method:** {cv.get('split_method')}")
        lines.append("")
        lines.append("| Metric | Mean | Std |")
        lines.append("|--------|------|-----|")
        for key_mean, key_std in [
            ("accuracy_mean", "accuracy_std"),
            ("f1_mean", "f1_std"),
            ("sensitivity_mean", None),
            ("specificity_mean", None),
            ("auc_roc_mean", None),
            ("pr_auc_mean", None),
            ("brier_mean", None),
        ]:
            val = cv.get(key_mean)
            std = cv.get(key_std, "") if key_std else ""
            if val is not None:
                std_str = f"{std:.4f}" if isinstance(std, float) else "-"
                lines.append(f"| {key_mean.replace('_mean', '')} | {val:.4f} | {std_str} |")
        lines.append("")

    # Subgroup analysis
    sg = report.get("subgroup_analysis", {})
    if sg:
        lines.append("## Subgroup Analysis")
        lines.append("")
        lines.append("| Subgroup | N | Accuracy | Sensitivity | Specificity | AUC-ROC |")
        lines.append("|----------|---|----------|-------------|-------------|---------|")
        for name, metrics in sg.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                n = metrics.get("n_samples", "?")
                lines.append(
                    f"| {name} | {n} "
                    f"| {metrics.get('accuracy', 0):.4f} "
                    f"| {metrics.get('sensitivity', 0):.4f} "
                    f"| {metrics.get('specificity', 0):.4f} "
                    f"| {metrics.get('auc_roc', 0):.4f} |"
                )
        lines.append("")

    # Calibration
    cal = report.get("calibration", {})
    if cal:
        lines.append("## Calibration")
        lines.append("")
        lines.append(f"- **ECE (before calibration):** {cal.get('ece_before', 'n/a')}")
        lines.append(f"- **ECE (after calibration):** {cal.get('ece_after', 'n/a')}")
        lines.append(f"- **Calibration method:** {cal.get('method', 'n/a')}")
        opt = cal.get("threshold_optimization", {})
        if opt:
            lines.append("")
            lines.append("### Optimal Thresholds")
            lines.append("")
            lines.append("| Criterion | Threshold | Sensitivity | Specificity |")
            lines.append("|-----------|-----------|-------------|-------------|")
            for criterion, vals in opt.items():
                if isinstance(vals, dict):
                    lines.append(
                        f"| {criterion} | {vals.get('threshold', 0):.3f} "
                        f"| {vals.get('sensitivity', 0):.4f} "
                        f"| {vals.get('specificity', 0):.4f} |"
                    )
        lines.append("")

    # Feature importance
    fi = report.get("feature_importance", {})
    if fi:
        lines.append("## Top Features")
        lines.append("")
        lines.append("| Rank | Feature | Importance |")
        lines.append("|------|---------|------------|")
        for rank, (name, imp) in enumerate(fi.items(), 1):
            lines.append(f"| {rank} | {name} | {imp:.6f} |")
        lines.append("")

    # Config
    lines.append("## Configuration Snapshot")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report.get("config", {}), indent=2))
    lines.append("```")
    lines.append("")

    # Regression
    reg = report.get("regression_check", {})
    if reg.get("status") == "regression_detected":
        lines.append("## WARNING: Performance Regression Detected")
        lines.append("")
        lines.append(f"- Current accuracy: {reg.get('current_accuracy')}")
        lines.append(f"- Best accuracy: {reg.get('best_accuracy')}")
        lines.append(f"- Delta: {reg.get('delta')}")
        lines.append("")

    lines.append("---")
    lines.append("*This is a screening tool, not a clinical diagnosis.*")
    lines.append("")

    return "\n".join(lines)
