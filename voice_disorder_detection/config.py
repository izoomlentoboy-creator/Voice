"""Configuration for the voice disorder detection system."""

import os
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FEEDBACK_DIR = PROJECT_ROOT / "feedback"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"

for d in (DATA_DIR, MODELS_DIR, FEEDBACK_DIR, LOGS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Database ---
DB_DIR = os.environ.get("SBVOICEDB_DIR", None)  # None = default platformdirs location

# --- Feature extraction ---
SAMPLE_RATE = 16000          # Resample all audio to this rate
N_MFCC = 13                  # Number of MFCC coefficients
N_FFT = 2048                 # FFT window size
HOP_LENGTH = 512             # Hop length for STFT
N_MELS = 128                 # Number of mel bands

# Utterances to use for feature extraction (sustained vowels are most informative)
# Normal pitch vowels only — sufficient for classification, 3x less memory
UTTERANCES_FOR_TRAINING = [
    "a_n", "i_n", "u_n",     # normal pitch
]

# Extended set (use with --utterances extended for more data, needs more RAM)
UTTERANCES_EXTENDED = [
    "a_n", "i_n", "u_n",     # normal pitch
    "a_l", "i_l", "u_l",     # low pitch
    "a_h", "i_h", "u_h",     # high pitch
]

# --- Model ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Classification modes
MODE_BINARY = "binary"           # healthy vs pathological
MODE_MULTICLASS = "multiclass"   # specific disorder detection

# Model backends
BACKEND_ENSEMBLE = "ensemble"    # SVM + RF + GB
BACKEND_LOGREG = "logreg"       # Logistic regression baseline
BACKEND_CNN = "cnn"             # CNN / MLP on mel-spectrogram

# --- Abstain ---
ABSTAIN_THRESHOLD = 0.6          # Refuse prediction if max_proba < this

# --- Augmentation ---
AUGMENT_NOISE_LEVELS = [0.002, 0.005, 0.01]
AUGMENT_PITCH_STEPS = [-2, -1, 1, 2]  # semitones
AUGMENT_TIME_STRETCH = [0.9, 1.1]

# --- CNN / mel-spectrogram ---
MEL_SPEC_MAX_FRAMES = 128        # Pad/truncate spectrograms to this width
MEL_PCA_COMPONENTS = 200         # PCA reduction for flattened spectrogram

# --- Feedback ---
MAX_FEEDBACK_BUFFER = 100        # Retrain after this many corrections
FEEDBACK_FILE = FEEDBACK_DIR / "corrections.json"

# --- Self-test ---
SELFTEST_RESULTS_FILE = LOGS_DIR / "selftest_results.json"
PERFORMANCE_HISTORY_FILE = LOGS_DIR / "performance_history.json"

# --- Model files ---
def model_path(mode: str, backend: str = BACKEND_ENSEMBLE) -> Path:
    return MODELS_DIR / f"voice_disorder_{mode}_{backend}.joblib"

def scaler_path(mode: str, backend: str = BACKEND_ENSEMBLE) -> Path:
    return MODELS_DIR / f"scaler_{mode}_{backend}.joblib"

def label_encoder_path(mode: str) -> Path:
    return MODELS_DIR / f"label_encoder_{mode}.joblib"

def feature_cache_path() -> Path:
    return CACHE_DIR / "features_cache.npz"

def metadata_path() -> Path:
    return MODELS_DIR / "training_metadata.json"
