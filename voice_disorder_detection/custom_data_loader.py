"""Load custom audio data from zip archives for training.

Supports zip files containing audio recordings (WAV, MP3, FLAC, OGG)
organized either by folder labels or with a metadata CSV file.

Expected zip structures:

    Option A — folder-based labels:
        archive.zip/
            healthy/
                file1.wav
                file2.wav
            pathological/
                file3.wav
                file4.wav

    Option B — flat with metadata CSV:
        archive.zip/
            metadata.csv        # columns: filename, label (0/1 or healthy/pathological)
            file1.wav
            file2.wav

    Option C — flat, no labels (all assigned a single label via parameter):
        archive.zip/
            file1.wav
            file2.wav
"""

import csv
import gc
import io
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from . import config
from .augmentation import augment_audio
from .feature_extractor import extract_all_features, preprocess_audio

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

# Folder names recognized as label indicators
HEALTHY_NAMES = {"healthy", "normal", "n", "0", "health"}
PATHOLOGICAL_NAMES = {"pathological", "pathology", "p", "1", "sick", "disorder"}


class CustomZipDataLoader:
    """Extract features from audio files inside a zip archive."""

    def __init__(self, zip_path: str):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {self.zip_path}")
        if not zipfile.is_zipfile(self.zip_path):
            raise ValueError(f"Not a valid zip file: {self.zip_path}")

    def extract_dataset(
        self,
        default_label: Optional[int] = None,
        augment: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[dict]]:
        """Extract features from audio files in the zip archive.

        Parameters
        ----------
        default_label : int, optional
            Label to assign when no label info is available in the archive.
            If None and no labels found, raises an error.
        augment : bool
            If True, apply data augmentation to each audio file.

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        session_ids : list[int]
            Synthetic session IDs (negative to avoid collision with sbvoicedb).
        speaker_ids : list[int]
            Synthetic speaker IDs (negative to avoid collision with sbvoicedb).
        metadata : list[dict]
        """
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            file_labels = self._detect_labels(zf, default_label)
            audio_files = [
                name for name in file_labels
                if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS
            ]

            if not audio_files:
                raise RuntimeError(
                    f"No supported audio files found in {self.zip_path}. "
                    f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )

            logger.info(
                "Found %d audio files in %s", len(audio_files), self.zip_path.name,
            )

            X_list, y_list = [], []
            session_ids, speaker_ids = [], []
            metadata_list = []

            # Use negative IDs to avoid collision with sbvoicedb
            base_id = -100_000

            for idx, audio_name in enumerate(audio_files):
                label = file_labels[audio_name]
                sample_id = base_id - idx

                try:
                    audio_data = zf.read(audio_name)
                    audio, sr = self._load_audio_from_bytes(audio_data, audio_name)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", audio_name, e)
                    continue

                if audio is None or len(audio) < 100:
                    logger.warning("Skipping %s: too short", audio_name)
                    continue

                try:
                    features = extract_all_features(audio, sr)
                    X_list.append(features)
                    y_list.append(label)
                    session_ids.append(sample_id)
                    speaker_ids.append(sample_id)
                    metadata_list.append({
                        "session_id": sample_id,
                        "speaker_id": sample_id,
                        "source": "custom_zip",
                        "zip_file": self.zip_path.name,
                        "audio_file": audio_name,
                        "label": label,
                    })

                    # Augmentation
                    if augment:
                        try:
                            processed = preprocess_audio(audio, sr)
                            aug_versions = augment_audio(processed, config.SAMPLE_RATE)
                            for aug_audio in aug_versions[:2]:
                                aug_feats = extract_all_features(
                                    aug_audio, config.SAMPLE_RATE, preprocess=False,
                                )
                                X_list.append(aug_feats)
                                y_list.append(label)
                                session_ids.append(sample_id)
                                speaker_ids.append(sample_id)
                                metadata_list.append({
                                    "session_id": sample_id,
                                    "speaker_id": sample_id,
                                    "source": "custom_zip",
                                    "zip_file": self.zip_path.name,
                                    "audio_file": audio_name,
                                    "label": label,
                                    "augmented": True,
                                })
                                del aug_audio, aug_feats
                            del aug_versions, processed
                        except Exception:
                            pass

                except Exception as e:
                    logger.warning(
                        "Feature extraction failed for %s: %s", audio_name, e,
                    )

                del audio
                if (idx + 1) % 50 == 0:
                    gc.collect()
                    logger.info("Processed %d / %d files...", idx + 1, len(audio_files))

        if not X_list:
            raise RuntimeError(
                "No features could be extracted from audio files in the archive."
            )

        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.int64)

        logger.info(
            "Custom data extraction complete: %d samples, %d features, "
            "label distribution: %s",
            X.shape[0], X.shape[1],
            dict(zip(*np.unique(y, return_counts=True))),
        )
        return X, y, session_ids, speaker_ids, metadata_list

    def _detect_labels(
        self, zf: zipfile.ZipFile, default_label: Optional[int],
    ) -> dict[str, int]:
        """Detect labeling scheme inside the zip and return {filename: label}."""
        names = zf.namelist()

        # Option B: metadata CSV
        csv_candidates = [n for n in names if Path(n).name.lower() in (
            "metadata.csv", "labels.csv", "annotations.csv",
        )]
        if csv_candidates:
            return self._parse_metadata_csv(zf, csv_candidates[0])

        # Option A: folder-based labels
        folder_labels = self._detect_folder_labels(names)
        if folder_labels:
            return folder_labels

        # Option C: flat structure, use default_label
        if default_label is not None:
            return {
                name: default_label
                for name in names
                if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS
            }

        raise ValueError(
            "Cannot determine labels from the zip archive. "
            "Either organize files into 'healthy'/'pathological' folders, "
            "include a metadata.csv, or pass --extra-data-label (0 or 1)."
        )

    def _detect_folder_labels(self, names: list[str]) -> Optional[dict[str, int]]:
        """Try to detect healthy/pathological folders."""
        result = {}
        for name in names:
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            parts = Path(name).parts
            if len(parts) < 2:
                continue

            folder = parts[-2].lower().strip()
            if folder in HEALTHY_NAMES:
                result[name] = 0
            elif folder in PATHOLOGICAL_NAMES:
                result[name] = 1

        return result if result else None

    def _parse_metadata_csv(
        self, zf: zipfile.ZipFile, csv_name: str,
    ) -> dict[str, int]:
        """Parse a CSV file with filename and label columns."""
        csv_data = zf.read(csv_name).decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_data))

        # Normalize field names
        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV: {csv_name}")

        field_map = {f.lower().strip(): f for f in reader.fieldnames}

        # Find filename column
        fname_col = None
        for candidate in ("filename", "file", "audio", "path", "name"):
            if candidate in field_map:
                fname_col = field_map[candidate]
                break
        if fname_col is None:
            raise ValueError(
                f"CSV {csv_name} must have a 'filename' column. "
                f"Found: {reader.fieldnames}"
            )

        # Find label column
        label_col = None
        for candidate in ("label", "class", "diagnosis", "target", "y"):
            if candidate in field_map:
                label_col = field_map[candidate]
                break
        if label_col is None:
            raise ValueError(
                f"CSV {csv_name} must have a 'label' column. "
                f"Found: {reader.fieldnames}"
            )

        result = {}
        all_names = set(zf.namelist())
        for row in reader:
            fname = row[fname_col].strip()
            label_raw = row[label_col].strip().lower()

            # Parse label
            if label_raw in ("0", "healthy", "normal", "n"):
                label = 0
            elif label_raw in ("1", "pathological", "pathology", "p", "sick"):
                label = 1
            else:
                try:
                    label = int(label_raw)
                except ValueError:
                    logger.warning("Unknown label '%s' for %s, skipping", label_raw, fname)
                    continue

            # Find the file in the archive
            if fname in all_names:
                result[fname] = label
            else:
                # Try searching in subdirectories
                for name in all_names:
                    if name.endswith(fname) or Path(name).name == fname:
                        result[name] = label
                        break

        return result

    @staticmethod
    def _load_audio_from_bytes(
        audio_bytes: bytes, filename: str,
    ) -> tuple[np.ndarray, int]:
        """Load audio from raw bytes using librosa."""
        with tempfile.NamedTemporaryFile(
            suffix=Path(filename).suffix, delete=True,
        ) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            audio, sr = librosa.load(tmp.name, sr=None)
        return audio, sr
