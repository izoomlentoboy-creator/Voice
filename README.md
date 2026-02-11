# Voice Disorder Detection

Automated screening of voice disorders from audio recordings using the
[Saarbruecken Voice Database](https://stimmdb.coli.uni-saarland.de/) (sbvoicedb).

> **Disclaimer**: This is a screening tool for research purposes.
> It is **not** a medical diagnosis. Always consult a qualified specialist.

## Task

| | |
|---|---|
| **Input** | Audio recording of a sustained vowel (/a/, /i/, /u/) at normal, low, or high pitch |
| **Output** | Binary classification: `HEALTHY` (0) or `PATHOLOGICAL` (1) + confidence score |
| **Abstain** | If confidence < 60%, the model refuses to predict and recommends specialist referral |
| **Dataset** | Saarbruecken Voice Database: 2043 sessions, 1680 speakers, 71 pathology types |
| **Split** | **Patient-level** (GroupShuffleSplit / GroupKFold) — no speaker appears in both train and test |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train (downloads data automatically on first run)
python scripts/train.py

# Inference on an audio file
python scripts/infer.py path/to/audio.wav

# Inference on a database session
python scripts/infer.py --session 42
```

## Project Structure

```
Voice/
├── main.py                              # Full CLI (all commands)
├── api.py                               # FastAPI inference server (simple)
├── requirements.txt
├── ruff.toml
├── docker-compose.yml                   # Docker orchestration
├── scripts/
│   ├── train.py                         # One-command training
│   └── infer.py                         # One-command inference
├── tests/
│   ├── test_feature_extractor.py
│   ├── test_model.py
│   ├── test_self_test.py
│   ├── test_calibration.py
│   ├── test_domain_monitor.py
│   └── test_report_generator.py
├── voice_disorder_detection/            # Core ML package
│   ├── config.py                        # All configuration
│   ├── feature_extractor.py             # Audio -> 322-dim feature vector
│   ├── augmentation.py                  # Noise, pitch shift, time stretch
│   ├── data_loader.py                   # sbvoicedb interface + speaker tracking
│   ├── model.py                         # Ensemble / LogReg / abstain
│   ├── cnn_model.py                     # MLP on PCA-reduced mel-spectrogram
│   ├── feedback.py                      # Online correction system
│   ├── self_test.py                     # Patient-level eval, medical metrics, subgroups
│   ├── calibration.py                   # Probability calibration + threshold optimization
│   ├── domain_monitor.py               # OOD detection + feature drift monitoring
│   ├── interpretability.py              # SHAP analysis
│   ├── report_generator.py             # Reproducible evaluation reports
│   └── pipeline.py                      # High-level orchestrator
├── server/                              # Production FastAPI server (TBVoice)
│   ├── Dockerfile
│   ├── run.py                           # Local server runner
│   ├── train_model.sh                   # Docker training script
│   ├── generate_ref_stats.py            # Healthy population reference stats
│   ├── app/
│   │   ├── main.py                      # FastAPI app setup
│   │   ├── config.py                    # Server configuration
│   │   ├── models.py                    # SQLAlchemy ORM models
│   │   ├── database.py                  # Database setup
│   │   ├── schemas.py                   # Pydantic request/response schemas
│   │   ├── routes/
│   │   │   ├── analyze.py               # POST /analyze (3 vowel files)
│   │   │   ├── feedback.py              # POST /feedback
│   │   │   ├── health.py                # GET /health, GET /status
│   │   │   └── history.py               # GET /history, GET /analysis
│   │   └── services/
│   │       ├── predictor.py             # ML prediction singleton
│   │       ├── audio_processor.py       # Audio validation
│   │       └── interpreter.py           # Feature → user-friendly categories
│   └── tests/
│       ├── test_api.py
│       └── test_interpreter.py
└── .github/workflows/ci.yml            # Lint + test
```

## Data Format

The sbvoicedb package downloads the Saarbruecken Voice Database automatically.
Each recording session contains:

- **Audio**: 16-bit PCM (`.nsp` format), variable sample rate → resampled to 16 kHz
- **EGG**: Electroglottogram (not used in current models)
- **Metadata**: speaker ID, gender, birthdate, age at recording, pathology names (German)
- **Utterances used**: `a_n`, `i_n`, `u_n`, `a_l`, `i_l`, `u_l`, `a_h`, `i_h`, `u_h` (sustained vowels at 3 pitch levels)

## Features (322 dimensions)

| Group | Count | Features |
|---|---|---|
| MFCC | 234 | 13 coefficients + delta + delta-delta, each with 6 stats (mean/std/min/max/skew/kurtosis) |
| Spectral | 66 | Centroid, bandwidth, rolloff, flatness, contrast (7 bands) × 6 stats |
| Temporal | 12 | Zero-crossing rate, RMS energy × 6 stats |
| Pitch | 7 | F0 (mean/std/min/max), voiced fraction, jitter, shimmer |
| Harmonic | 3 | HNR, harmonic ratio, percussive ratio |

## Models

| Backend | Description | Use case |
|---|---|---|
| `ensemble` (default) | SVM + Random Forest + Gradient Boosting, soft voting | Best accuracy |
| `logreg` | Logistic Regression with balanced weights | Baseline / interpretability |
| `cnn` | MLP on PCA-reduced mel-spectrogram | Alternative temporal representation |

## Evaluation Protocol

**Critical**: All splits are performed at the **patient level** (by `speaker_id`),
not at the recording level. This prevents identity/timbre leakage that inflates metrics.

### Metrics

| Metric | Why |
|---|---|
| Sensitivity (Recall) | Fraction of pathological cases correctly detected — must be high for screening |
| Specificity | Fraction of healthy correctly identified — controls false alarm rate |
| ROC-AUC | Ranking quality independent of threshold |
| PR-AUC | Performance under class imbalance (687 healthy vs 1356 pathological) |
| Brier Score | Calibration — are predicted probabilities meaningful? |
| ECE | Expected Calibration Error — how well probabilities match true frequencies |

### Subgroup Analysis

The model is evaluated separately by:
- **Gender**: male (m) / female (w)
- **Age**: young (0–30), middle (30–50), senior (50–70), elderly (70+)

## CLI Reference

```bash
# Training
python main.py train                                     # default ensemble
python main.py --backend logreg train                    # logistic regression baseline
python main.py train --augment                           # with data augmentation
python main.py train --max-samples 100                   # quick experiment

# Prediction
python main.py predict --file audio.wav
python main.py predict --session 42

# Baselines comparison (patient-level CV)
python main.py compare-baselines

# Evaluation
python main.py self-test --type full                     # train/test with patient split
python main.py self-test --type cv                       # 5-fold GroupKFold
python main.py self-test --type subgroups                # gender/age breakdown
python main.py self-test --type quick                    # sanity check

# Feedback
python main.py feedback --session 42 --label 0           # correct a prediction
python main.py apply-feedback                            # apply accumulated corrections

# Interpretability
python main.py explain                                   # SHAP feature importance

# Calibration & threshold optimization
python main.py calibrate                                 # isotonic calibration (default)
python main.py calibrate --method sigmoid                # Platt scaling

# Domain monitoring
python main.py fit-monitor                               # fit OOD detector on training data
python main.py check-drift                               # check feature distribution drift

# Reproducible evaluation report
python main.py report                                    # generates JSON + Markdown report
python main.py report --output-dir ./my_reports

# Optimization
python main.py optimize --iterations 30

# Simple API server
uvicorn api:app --host 0.0.0.0 --port 8000
# then: curl -X POST http://localhost:8000/predict -F "file=@audio.wav"

# Production server (TBVoice)
python server/run.py                                     # local run
docker compose up --build                                # Docker

# Status
python main.py status
python main.py db-info
```

## Testing

```bash
# Run all tests (core + server)
pytest tests/ server/tests/ -v

# Lint
ruff check .
```

## How to Use Responsibly

1. This system is a **screening tool** — it flags potential voice disorders for further investigation
2. A positive result means "consider specialist referral", not "has disease X"
3. When the model **abstains** (confidence < 60%), the recording quality may be poor or the case is ambiguous
4. Always validate on your target population before clinical deployment
5. The model was trained on German speakers in controlled conditions — generalization to other languages, accents, or recording environments is not guaranteed
