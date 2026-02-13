"""CNN-style model on mel-spectrograms.

Uses mel-spectrogram as a 2D representation of audio, reduces
dimensionality with PCA, and classifies with an MLP.
Falls back to sklearn MLPClassifier (PyTorch optional).

When used through VoiceDisorderModel, the class exposes a
sklearn-compatible interface (fit/predict/predict_proba) that
operates on pre-extracted feature vectors.
"""

import logging

import joblib
import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from . import config

logger = logging.getLogger(__name__)


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_mels: int = config.N_MELS,
    n_fft: int = config.N_FFT,
    hop_length: int = config.HOP_LENGTH,
    max_frames: int = config.MEL_SPEC_MAX_FRAMES,
) -> np.ndarray:
    """Extract a fixed-size mel-spectrogram from audio.

    Returns
    -------
    np.ndarray, shape (n_mels, max_frames)
        Log-mel spectrogram, padded or truncated to max_frames.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Pad or truncate to fixed width
    if log_mel.shape[1] < max_frames:
        pad_width = max_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant",
                         constant_values=log_mel.min())
    else:
        log_mel = log_mel[:, :max_frames]

    return log_mel


class MelSpectrogramCNN:
    """MLP classifier with PCA dimensionality reduction.

    When used through VoiceDisorderModel (the standard path),
    this receives pre-scaled feature vectors and applies
    PCA -> MLP(256, 128).

    Architecture:
        features (N,) -> PCA(200) -> MLP(256, 128) -> classes
    """

    def __init__(self, mode: str = config.MODE_BINARY):
        self.mode = mode
        self._pca = PCA(n_components=config.MEL_PCA_COMPONENTS, random_state=config.RANDOM_STATE)
        self._mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=config.RANDOM_STATE,
            batch_size=32,
        )
        self._pca_fitted = False

    # --- sklearn-compatible interface for VoiceDisorderModel ---

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "MelSpectrogramCNN":
        """Fit PCA + MLP on pre-scaled feature vectors.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Already-scaled feature matrix (scaler is in VoiceDisorderModel).
        y : np.ndarray, shape (n_samples,)
            Encoded labels.
        """
        n_components = min(config.MEL_PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
        if n_components < 1:
            n_components = 1
        self._pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
        X_reduced = self._pca.fit_transform(X)
        self._pca_fitted = True

        self._mlp.fit(X_reduced, y)
        logger.info(
            "CNN/MLP model trained: %d samples, PCA(%d→%d), loss=%.4f",
            X.shape[0], X.shape[1], n_components, self._mlp.loss_,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels from pre-scaled feature vectors."""
        X_reduced = self._pca.transform(X) if self._pca_fitted else X
        return self._mlp.predict(X_reduced)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities from pre-scaled feature vectors."""
        X_reduced = self._pca.transform(X) if self._pca_fitted else X
        return self._mlp.predict_proba(X_reduced)

    # --- Standalone usage (with spectrograms) ---

    def prepare_features(self, spectrograms: np.ndarray, fit: bool = False) -> np.ndarray:
        """Flatten spectrograms and apply PCA + scaling.

        Parameters
        ----------
        spectrograms : np.ndarray, shape (n_samples, n_mels, max_frames)
        fit : bool
            If True, fit PCA and scaler.

        Returns
        -------
        np.ndarray, shape (n_samples, n_components)
        """
        n_samples = spectrograms.shape[0]
        flat = spectrograms.reshape(n_samples, -1)

        if fit:
            n_components = min(config.MEL_PCA_COMPONENTS, flat.shape[0], flat.shape[1])
            self._pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
            reduced = self._pca.fit_transform(flat)
            self._pca_fitted = True
        else:
            reduced = self._pca.transform(flat)

        return reduced

    def save(self) -> None:
        path = config.model_path(self.mode, config.BACKEND_CNN)
        joblib.dump({
            "mlp": self._mlp,
            "pca": self._pca,
            "pca_fitted": self._pca_fitted,
        }, path)
        logger.info("CNN model saved to %s", path)

    def load(self) -> None:
        path = config.model_path(self.mode, config.BACKEND_CNN)
        data = joblib.load(path)
        self._mlp = data["mlp"]
        self._pca = data["pca"]
        self._pca_fitted = data.get("pca_fitted", True)
        logger.info("CNN model loaded from %s", path)
