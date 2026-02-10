"""CNN-style model on mel-spectrograms.

Uses mel-spectrogram as a 2D representation of audio, reduces
dimensionality with PCA, and classifies with an MLP.
Falls back to sklearn MLPClassifier (PyTorch optional).
"""

import logging

import joblib
import librosa
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    """MLP classifier on PCA-reduced mel-spectrograms.

    Architecture:
        mel-spectrogram (128 x 128) -> flatten -> PCA(200) -> MLP(256, 128) -> classes
    """

    def __init__(self, mode: str = config.MODE_BINARY):
        self.mode = mode
        self.pca = PCA(n_components=config.MEL_PCA_COMPONENTS, random_state=config.RANDOM_STATE)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=config.RANDOM_STATE,
            batch_size=32,
        )
        self.is_trained = False

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
            self.pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
            reduced = self.pca.fit_transform(flat)
            reduced = self.scaler.fit_transform(reduced)
        else:
            reduced = self.pca.transform(flat)
            reduced = self.scaler.transform(reduced)

        return reduced

    def train(self, spectrograms: np.ndarray, y: np.ndarray) -> dict:
        """Train the model.

        Parameters
        ----------
        spectrograms : np.ndarray, shape (n_samples, n_mels, max_frames)
        y : np.ndarray, shape (n_samples,)
        """
        y_enc = self.label_encoder.fit_transform(y)
        X = self.prepare_features(spectrograms, fit=True)

        self.model.fit(X, y_enc)
        self.is_trained = True

        logger.info(
            "CNN model trained: %d samples, PCA(%d), MLP loss=%.4f",
            len(y), X.shape[1], self.model.loss_,
        )
        return {
            "backend": "cnn",
            "n_samples": int(len(y)),
            "pca_components": int(X.shape[1]),
            "pca_variance_explained": round(float(sum(self.pca.explained_variance_ratio_)), 4),
            "mlp_loss": round(float(self.model.loss_), 4),
            "mlp_iterations": int(self.model.n_iter_),
        }

    def predict(self, spectrograms: np.ndarray) -> np.ndarray:
        X = self.prepare_features(spectrograms)
        y_pred = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, spectrograms: np.ndarray) -> np.ndarray:
        X = self.prepare_features(spectrograms)
        return self.model.predict_proba(X)

    def save(self) -> None:
        path = config.model_path(self.mode, config.BACKEND_CNN)
        joblib.dump({
            "model": self.model,
            "pca": self.pca,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
        }, path)
        logger.info("CNN model saved to %s", path)

    def load(self) -> None:
        path = config.model_path(self.mode, config.BACKEND_CNN)
        data = joblib.load(path)
        self.model = data["model"]
        self.pca = data["pca"]
        self.scaler = data["scaler"]
        self.label_encoder = data["label_encoder"]
        self.is_trained = True
        logger.info("CNN model loaded from %s", path)
