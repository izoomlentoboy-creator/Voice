"""Data loading from the Saarbruecken Voice Database (sbvoicedb).

Handles database initialization, audio loading, feature extraction,
caching, and dataset preparation for training/evaluation.
Key: tracks speaker_id for patient-level splitting.

Memory-optimized: processes in batches with explicit garbage collection
to work on servers with limited RAM (4 GB).
"""

import gc
import json
import logging
from typing import Optional

import numpy as np
from sbvoicedb import SbVoiceDb

from . import config
from .augmentation import augment_audio
from .feature_extractor import extract_all_features, preprocess_audio

logger = logging.getLogger(__name__)


class VoiceDataLoader:
    """Loads voice data from sbvoicedb and extracts features."""

    def __init__(
        self,
        dbdir: Optional[str] = None,
        download_mode: str = "lazy",
        utterances: Optional[list[str]] = None,
    ):
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
        augment: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[dict]]:
        """Extract features and labels from the database.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
            Labels (0=healthy, 1=pathological for binary).
        session_ids : list[int]
        speaker_ids : list[int]
            Speaker IDs for patient-level splitting.
        metadata : list[dict]
            Per-sample metadata (gender, age, pathologies).
        """
        cache_file = config.feature_cache_path()
        cache_key = f"{mode}_{max_samples}_{augment}_{'-'.join(sorted(self.utterances))}"

        if use_cache and cache_file.exists():
            try:
                cached = np.load(cache_file, allow_pickle=True)
                if str(cached.get("cache_key", "")) == cache_key:
                    logger.info("Loading features from cache: %s", cache_file)
                    meta = cached["metadata"].item() if "metadata" in cached else []
                    return (
                        cached["X"],
                        cached["y"],
                        cached["session_ids"].tolist(),
                        cached["speaker_ids"].tolist(),
                        meta if isinstance(meta, list) else [],
                    )
            except Exception as e:
                logger.warning("Cache load failed: %s", e)

        logger.info("Extracting features (mode=%s, augment=%s)...", mode, augment)
        X_list, y_list, session_ids, speaker_ids = [], [], [], []
        metadata_list = []
        pathology_map = {}

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
                    label = 0
                    pathology_map.setdefault("[Healthy]", 0)
                else:
                    pathologies = session_full.pathologies
                    if not pathologies:
                        continue
                    pname = pathologies[0].name
                    if pname not in pathology_map:
                        pathology_map[pname] = len(pathology_map)
                    label = pathology_map[pname]

            # Collect per-sample metadata
            sample_meta = {
                "session_id": session_full.id,
                "speaker_id": session_full.speaker_id,
                "gender": session_full.speaker.gender if session_full.speaker else None,
                "age": session_full.speaker_age,
                "type": session_full.type,
                "pathologies": [p.name for p in session_full.pathologies]
                if session_full.pathologies else [],
            }

            # Extract features from each relevant recording
            session_features = []
            for rec in session_full.recordings:
                if rec.utterance not in self.utterances:
                    continue

                rec_full = self.db.get_recording(rec.id, full_file_paths=True)
                if rec_full is None:
                    continue

                try:
                    audio = rec_full.nspdata
                    # sbvoicedb nspdata returns shape (N, 1) — flatten to 1-D
                    if audio is not None and audio.ndim > 1:
                        audio = audio.squeeze()
                except Exception:
                    audio = None

                if audio is None or len(audio) < 100:
                    continue

                try:
                    rate = rec_full.rate
                    feats = extract_all_features(audio, rate)
                    session_features.append(feats)

                    # Augmentation: process immediately, don't store raw audio
                    if augment:
                        processed = preprocess_audio(audio, rate)
                        try:
                            aug_versions = augment_audio(processed, config.SAMPLE_RATE)
                            for aug_audio in aug_versions[:2]:
                                aug_feats = extract_all_features(
                                    aug_audio, config.SAMPLE_RATE, preprocess=False,
                                )
                                X_list.append(aug_feats)
                                y_list.append(label)
                                session_ids.append(session_full.id)
                                speaker_ids.append(session_full.speaker_id)
                                metadata_list.append({**sample_meta, "augmented": True})
                                del aug_audio, aug_feats
                            del aug_versions, processed
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(
                        "Feature extraction failed for recording %d: %s",
                        rec.id, e,
                    )
                finally:
                    # Free audio memory immediately
                    del audio
                    del rec_full

            if not session_features:
                del session_full
                continue

            # Original sample
            combined = np.mean(session_features, axis=0)
            X_list.append(combined)
            y_list.append(label)
            session_ids.append(session_full.id)
            speaker_ids.append(session_full.speaker_id)
            metadata_list.append(sample_meta)
            count += 1

            # Free session data
            del session_features, session_full, sample_meta

            if count % 50 == 0:
                gc.collect()
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
                    speaker_ids=np.array(speaker_ids),
                    metadata=np.array(metadata_list, dtype=object),
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
            "Extraction complete: %d samples, %d features, %d classes, %d unique speakers",
            X.shape[0], X.shape[1], len(np.unique(y)), len(set(speaker_ids)),
        )
        return X, y, session_ids, speaker_ids, metadata_list

    def extract_synthetic_dataset(
        self,
        mode: str = config.MODE_BINARY,
        n_samples: int = 500,
        use_cache: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[dict]]:
        """Generate a synthetic dataset using real metadata from sbvoicedb.

        Creates realistic feature vectors by generating random audio signals
        and extracting real acoustic features from them. Uses actual session
        metadata (gender, age, pathology type) from the database.

        This is useful when the audio data cannot be downloaded (e.g. proxy
        restrictions) but the SQLite metadata is available.
        """
        cache_file = config.feature_cache_path()
        cache_key = f"synthetic_{mode}_{n_samples}"

        if use_cache and cache_file.exists():
            try:
                cached = np.load(cache_file, allow_pickle=True)
                if str(cached.get("cache_key", "")) == cache_key:
                    logger.info("Loading synthetic features from cache: %s", cache_file)
                    meta = cached["metadata"].item() if "metadata" in cached else []
                    return (
                        cached["X"],
                        cached["y"],
                        cached["session_ids"].tolist(),
                        cached["speaker_ids"].tolist(),
                        meta if isinstance(meta, list) else [],
                    )
            except Exception as e:
                logger.warning("Cache load failed: %s", e)

        logger.info(
            "Generating synthetic dataset (mode=%s, n_samples=%d)...",
            mode, n_samples,
        )
        rng = np.random.RandomState(config.RANDOM_STATE)

        X_list, y_list = [], []
        session_ids, speaker_ids = [], []
        metadata_list = []

        count = 0
        for session in self.db.iter_sessions():
            if count >= n_samples:
                break

            session_full = self.db.get_session(
                session.id,
                query_pathologies=True,
                query_speaker=True,
            )
            if session_full is None:
                continue

            if mode == config.MODE_BINARY:
                label = 0 if session_full.type == "n" else 1
            else:
                continue  # multiclass not supported for synthetic

            sample_meta = {
                "session_id": session_full.id,
                "speaker_id": session_full.speaker_id,
                "gender": session_full.speaker.gender if session_full.speaker else None,
                "age": session_full.speaker_age,
                "type": session_full.type,
                "pathologies": [p.name for p in session_full.pathologies]
                if session_full.pathologies else [],
                "synthetic": True,
            }

            # Generate synthetic audio (1 second of signal at target SR)
            duration = 1.0
            t = np.linspace(0, duration, int(config.SAMPLE_RATE * duration), endpoint=False)
            f0 = rng.uniform(80, 300)
            audio = np.sin(2 * np.pi * f0 * t).astype(np.float32)
            # Add harmonics
            for h in range(2, 5):
                audio += rng.uniform(0.1, 0.4) * np.sin(2 * np.pi * f0 * h * t).astype(np.float32)
            # Add noise (more for pathological)
            noise_level = rng.uniform(0.05, 0.2) if label == 1 else rng.uniform(0.01, 0.05)
            audio += rng.randn(len(audio)).astype(np.float32) * noise_level
            # Add jitter/shimmer for pathological voices
            if label == 1:
                jitter_amount = rng.uniform(0.01, 0.05)
                audio *= (1 + jitter_amount * rng.randn(len(audio)).astype(np.float32))

            audio = audio / (np.max(np.abs(audio)) + 1e-10)

            try:
                feats = extract_all_features(audio, config.SAMPLE_RATE, preprocess=False)
                X_list.append(feats)
                y_list.append(label)
                session_ids.append(session_full.id)
                speaker_ids.append(session_full.speaker_id)
                metadata_list.append(sample_meta)
                count += 1
            except Exception as e:
                logger.warning("Synthetic feature extraction failed for session %d: %s", session.id, e)
                continue

            if count % 100 == 0:
                gc.collect()
                logger.info("Generated %d/%d synthetic samples...", count, n_samples)

        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.int64)

        if use_cache:
            try:
                np.savez(
                    cache_file, X=X, y=y,
                    session_ids=np.array(session_ids),
                    speaker_ids=np.array(speaker_ids),
                    metadata=np.array(metadata_list, dtype=object),
                    cache_key=cache_key,
                )
                logger.info("Synthetic features cached to %s", cache_file)
            except Exception as e:
                logger.warning("Cache save failed: %s", e)

        logger.info(
            "Synthetic extraction complete: %d samples, %d features, %d classes, %d unique speakers",
            X.shape[0], X.shape[1], len(np.unique(y)), len(set(speaker_ids)),
        )
        return X, y, session_ids, speaker_ids, metadata_list

    def extract_single(self, audio: np.ndarray, sr: int) -> np.ndarray:
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
