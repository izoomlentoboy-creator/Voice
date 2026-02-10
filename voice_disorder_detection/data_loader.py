"""Data loading from the Saarbruecken Voice Database (sbvoicedb).

Handles database initialization, audio loading, feature extraction,
caching, and dataset preparation for training/evaluation.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from sbvoicedb import SbVoiceDb, Recording, RecordingSession

from . import config
from .feature_extractor import extract_all_features

logger = logging.getLogger(__name__)


class VoiceDataLoader:
    """Loads voice data from sbvoicedb and extracts features."""

    def __init__(
        self,
        dbdir: Optional[str] = None,
        download_mode: str = "lazy",
        utterances: Optional[list[str]] = None,
    ):
        """
        Parameters
        ----------
        dbdir : str, optional
            Database directory. None uses default.
        download_mode : str
            'lazy', 'immediate', or 'off'.
        utterances : list[str], optional
            Which utterances to use. Default: config.UTTERANCES_FOR_TRAINING.
        """
        self.dbdir = dbdir or config.DB_DIR
        self.download_mode = download_mode
        self.utterances = utterances or config.UTTERANCES_FOR_TRAINING

        db_kwargs = {"download_mode": download_mode}
        if self.dbdir:
            db_kwargs["dbdir"] = self.dbdir

        self.db = SbVoiceDb(**db_kwargs)
        logger.info(
            "Database initialized. Sessions: %d (downloaded: %d)",
            self.db.number_of_all_sessions,
            self.db.number_of_sessions_downloaded,
        )

    def extract_dataset(
        self,
        mode: str = config.MODE_BINARY,
        max_samples: Optional[int] = None,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Extract features and labels from the database.

        Parameters
        ----------
        mode : str
            'binary' for healthy/pathological, 'multiclass' for specific disorders.
        max_samples : int, optional
            Limit the number of samples (for faster experimentation).
        use_cache : bool
            If True, try to load from / save to cache.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y : np.ndarray, shape (n_samples,)
            Labels (0=healthy, 1=pathological for binary; class indices for multiclass).
        session_ids : list[int]
            Session IDs corresponding to each sample.
        """
        cache_file = config.feature_cache_path()
        cache_key = f"{mode}_{max_samples}_{'-'.join(sorted(self.utterances))}"

        if use_cache and cache_file.exists():
            try:
                cached = np.load(cache_file, allow_pickle=True)
                if cached.get("cache_key", "") == cache_key:
                    logger.info("Loading features from cache: %s", cache_file)
                    return (
                        cached["X"],
                        cached["y"],
                        cached["session_ids"].tolist(),
                    )
            except Exception as e:
                logger.warning("Cache load failed: %s", e)

        logger.info("Extracting features (mode=%s)...", mode)
        X_list, y_list, session_ids = [], [], []
        pathology_map = {}  # name -> index, for multiclass

        count = 0
        for session in self.db.iter_sessions():
            if max_samples and count >= max_samples:
                break

            session_full = self.db.get_session(
                session.id,
                query_pathologies=True,
                query_recordings=True,
                query_speaker=True,
            )
            if session_full is None:
                continue

            # Determine label
            if mode == config.MODE_BINARY:
                label = 0 if session_full.type == "n" else 1
            else:
                if session_full.type == "n":
                    label = 0  # healthy
                    pathology_map.setdefault("[Healthy]", 0)
                else:
                    pathologies = session_full.pathologies
                    if not pathologies:
                        continue
                    pname = pathologies[0].name
                    if pname not in pathology_map:
                        pathology_map[pname] = len(pathology_map)
                    label = pathology_map[pname]

            # Extract features from each relevant recording
            session_features = []
            for rec in session_full.recordings:
                if rec.utterance not in self.utterances:
                    continue

                rec_full = self.db.get_recording(
                    rec.id, full_file_paths=True
                )
                if rec_full is None:
                    continue

                try:
                    audio = rec_full.nspdata
                except Exception:
                    audio = None

                if audio is None or len(audio) == 0:
                    continue

                try:
                    feats = extract_all_features(audio, rec_full.rate)
                    session_features.append(feats)
                except Exception as e:
                    logger.warning(
                        "Feature extraction failed for recording %d: %s",
                        rec.id, e,
                    )

            if not session_features:
                continue

            # Average features across utterances for this session
            combined = np.mean(session_features, axis=0)
            X_list.append(combined)
            y_list.append(label)
            session_ids.append(session.id)
            count += 1

            if count % 50 == 0:
                logger.info("Processed %d sessions...", count)

        if not X_list:
            raise RuntimeError(
                "No features extracted. Ensure the database is downloaded "
                "and contains accessible recordings."
            )

        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.int64)

        # Save cache
        if use_cache:
            try:
                np.savez(
                    cache_file,
                    X=X, y=y,
                    session_ids=np.array(session_ids),
                    cache_key=cache_key,
                )
                logger.info("Features cached to %s", cache_file)
            except Exception as e:
                logger.warning("Cache save failed: %s", e)

        # Save pathology map for multiclass
        if mode == config.MODE_MULTICLASS and pathology_map:
            map_file = config.MODELS_DIR / "pathology_map.json"
            with open(map_file, "w", encoding="utf-8") as f:
                json.dump(pathology_map, f, ensure_ascii=False, indent=2)

        logger.info(
            "Extraction complete: %d samples, %d features, %d classes",
            X.shape[0], X.shape[1], len(np.unique(y)),
        )
        return X, y, session_ids

    def extract_single(
        self, audio: np.ndarray, sr: int
    ) -> np.ndarray:
        """Extract features from a single audio signal for prediction."""
        return extract_all_features(audio, sr)

    def get_database_stats(self) -> dict:
        """Return summary statistics about the loaded database."""
        stats = {
            "total_sessions": self.db.number_of_all_sessions,
            "downloaded_sessions": self.db.number_of_sessions_downloaded,
            "missing_datasets": self.db.missing_datasets,
            "has_healthy": self.db.has_healthy_dataset(),
        }

        try:
            stats["total_speakers"] = self.db.get_speaker_count()
            stats["total_pathologies"] = self.db.get_pathology_count()
            stats["pathology_names"] = list(self.db.get_pathology_names())
        except Exception as e:
            logger.warning("Could not fetch full stats: %s", e)

        return stats
