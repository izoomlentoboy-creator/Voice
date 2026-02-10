"""Feedback and correction system for online learning.

Allows users to correct model predictions, stores corrections,
and triggers incremental or full retraining when enough feedback
has accumulated.
"""

import json
import logging
import time
from typing import Optional

import numpy as np

from . import config
from .model import VoiceDisorderModel

logger = logging.getLogger(__name__)


class FeedbackManager:
    """Manages user corrections and model updates."""

    def __init__(self, model: VoiceDisorderModel):
        self.model = model
        self.corrections: list[dict] = []
        self._load_corrections()

    def add_correction(
        self,
        features: np.ndarray,
        predicted_label: int,
        correct_label: int,
        session_id: Optional[int] = None,
        note: str = "",
    ) -> dict:
        """Record a correction to a model prediction.

        Parameters
        ----------
        features : np.ndarray
            Feature vector for the sample.
        predicted_label : int
            What the model predicted.
        correct_label : int
            The correct label provided by the user.
        session_id : int, optional
            Database session ID for traceability.
        note : str
            Optional user note about the correction.

        Returns
        -------
        dict
            Correction record.
        """
        correction = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "predicted": int(predicted_label),
            "correct": int(correct_label),
            "session_id": session_id,
            "note": note,
            "features": features.tolist(),
            "applied": False,
        }

        self.corrections.append(correction)
        self._save_corrections()

        n_pending = sum(1 for c in self.corrections if not c["applied"])
        logger.info(
            "Correction recorded (predicted=%d, correct=%d). "
            "Pending corrections: %d / %d threshold",
            predicted_label, correct_label,
            n_pending, config.MAX_FEEDBACK_BUFFER,
        )

        # Auto-trigger incremental update if threshold reached
        if n_pending >= config.MAX_FEEDBACK_BUFFER:
            logger.info("Feedback threshold reached, triggering incremental update...")
            self.apply_corrections()

        return correction

    def apply_corrections(self, full_retrain: bool = False) -> dict:
        """Apply accumulated corrections to update the model.

        Parameters
        ----------
        full_retrain : bool
            If True, requires external retraining with full dataset.
            If False, uses incremental (online) update.

        Returns
        -------
        dict
            Summary of applied corrections.
        """
        pending = [c for c in self.corrections if not c["applied"]]
        if not pending:
            logger.info("No pending corrections to apply.")
            return {"applied": 0}

        features = np.array([c["features"] for c in pending], dtype=np.float32)
        labels = np.array([c["correct"] for c in pending], dtype=np.int64)

        if not full_retrain:
            self.model.incremental_update(features, labels)

        # Mark as applied
        for c in pending:
            c["applied"] = True
        self._save_corrections()

        summary = {
            "applied": len(pending),
            "method": "full_retrain" if full_retrain else "incremental",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        logger.info("Applied %d corrections via %s", len(pending), summary["method"])
        return summary

    def get_correction_stats(self) -> dict:
        """Get statistics about accumulated corrections."""
        total = len(self.corrections)
        applied = sum(1 for c in self.corrections if c["applied"])
        pending = total - applied

        # Confusion analysis
        confusion = {}
        for c in self.corrections:
            key = f"{c['predicted']}->{c['correct']}"
            confusion[key] = confusion.get(key, 0) + 1

        return {
            "total_corrections": total,
            "applied": applied,
            "pending": pending,
            "threshold_for_auto_update": config.MAX_FEEDBACK_BUFFER,
            "common_mistakes": dict(
                sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        }

    def clear_applied(self) -> int:
        """Remove already-applied corrections from history."""
        before = len(self.corrections)
        self.corrections = [c for c in self.corrections if not c["applied"]]
        self._save_corrections()
        removed = before - len(self.corrections)
        logger.info("Cleared %d applied corrections", removed)
        return removed

    def _save_corrections(self) -> None:
        """Persist corrections to disk."""
        config.FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(self.corrections, f, indent=2)

    def _load_corrections(self) -> None:
        """Load corrections from disk."""
        if config.FEEDBACK_FILE.exists():
            try:
                with open(config.FEEDBACK_FILE, encoding="utf-8") as f:
                    self.corrections = json.load(f)
                logger.info("Loaded %d corrections from disk", len(self.corrections))
            except Exception as e:
                logger.warning("Could not load corrections: %s", e)
                self.corrections = []
