"""
EchoFlow 1.0 - Optimized Configuration
=======================================

This configuration file contains optimized hyperparameters
for achieving 92%+ accuracy on the Saarbruecken Voice Database.

Based on research best practices and ensemble optimization.
"""

# Model branding
MODEL_NAME = "EchoFlow"
MODEL_VERSION = "1.0"
MODEL_DESCRIPTION = "AI-powered voice disorder detection system"

# Optimized hyperparameters for ensemble model
OPTIMIZED_PARAMS = {
    # SVM parameters
    "svm_C": 15.0,  # Increased from 10.0 for better margin optimization
    "svm_gamma": "scale",  # Adaptive gamma based on features
    
    # Random Forest parameters
    "rf_n_estimators": 300,  # Increased from 200 for better ensemble diversity
    "rf_max_depth": None,  # No limit - let trees grow to capture complex patterns
    "rf_min_samples_split": 5,
    "rf_min_samples_leaf": 2,
    "rf_max_features": "sqrt",  # Good default for classification
    
    # Gradient Boosting parameters
    "gb_n_estimators": 300,  # Increased from 200
    "gb_learning_rate": 0.05,  # Reduced from 0.1 for better generalization
    "gb_max_depth": 6,  # Increased from 5 for more complex trees
    "gb_min_samples_split": 5,
    "gb_min_samples_leaf": 2,
    "gb_subsample": 0.8,  # Use 80% of samples for each tree
    
    # Meta-learner (Stacking)
    "meta_C": 1.0,
    "meta_max_iter": 1000,
}

# Training configuration
TRAINING_CONFIG = {
    "augment": True,  # Enable data augmentation
    "augmentation_factor": 2,  # Create 2x more samples via augmentation
    "use_calibration": True,  # Enable probability calibration
    "calibration_method": "isotonic",  # Better than sigmoid for this task
    "cross_validation_folds": 5,  # For robust evaluation
    "test_size": 0.2,  # 20% for final test set
    "random_state": 42,
}

# Feature extraction optimization
FEATURE_CONFIG = {
    "n_mfcc": 13,  # Standard MFCC coefficients
    "n_fft": 2048,  # FFT window size
    "hop_length": 512,  # Hop length for STFT
    "win_length": 2048,  # Window length
    "window": "hamming",  # Window function
    "fmin": 50,  # Minimum frequency (Hz) - human voice range
    "fmax": 8000,  # Maximum frequency (Hz) - most voice info below 8kHz
}

# Augmentation parameters (for training robustness)
AUGMENTATION_CONFIG = {
    "noise_factor": 0.005,  # Add subtle noise
    "pitch_shift_range": (-2, 2),  # Shift pitch by ±2 semitones
    "time_stretch_range": (0.9, 1.1),  # Speed up/slow down by 10%
    "apply_probability": 0.5,  # Apply augmentation to 50% of samples
}

# Abstain threshold optimization
ABSTAIN_CONFIG = {
    "confidence_threshold": 0.65,  # Increased from 0.60 for higher precision
    "ood_threshold": 0.15,  # Out-of-distribution detection threshold
}

# Performance targets
PERFORMANCE_TARGETS = {
    "min_accuracy": 0.92,  # 92% minimum accuracy
    "min_sensitivity": 0.90,  # 90% sensitivity (recall for pathological)
    "min_specificity": 0.94,  # 94% specificity (recall for healthy)
    "min_roc_auc": 0.95,  # 95% ROC-AUC
    "max_brier_score": 0.10,  # Lower is better for calibration
}

# API response customization for EchoFlow
VERDICT_LABELS = {
    "healthy": {
        "ru": "Голос в норме",
        "en": "Voice is healthy",
    },
    "pathological": {
        "ru": "Возможны нарушения",
        "en": "Possible disorder detected",
    },
    "abstain": {
        "ru": "Требуется уточнение",
        "en": "Inconclusive - specialist recommended",
    },
}

CATEGORY_LABELS = {
    "pitch_stability": {
        "ru": "Стабильность высоты",
        "en": "Pitch Stability",
    },
    "harmonic_quality": {
        "ru": "Гармоническое качество",
        "en": "Harmonic Quality",
    },
    "voice_steadiness": {
        "ru": "Ровность голоса",
        "en": "Voice Steadiness",
    },
    "spectral_clarity": {
        "ru": "Спектральная чистота",
        "en": "Spectral Clarity",
    },
    "breath_support": {
        "ru": "Поддержка дыхания",
        "en": "Breath Support",
    },
}

STATUS_LABELS = {
    "good": {
        "ru": "Хорошо",
        "en": "Good",
    },
    "moderate": {
        "ru": "Умеренно",
        "en": "Moderate",
    },
    "poor": {
        "ru": "Требует внимания",
        "en": "Needs attention",
    },
}

# Recommendations templates
RECOMMENDATIONS = {
    "healthy": {
        "ru": "Ваш голос в норме. Признаков голосовых расстройств не обнаружено. Продолжайте заботиться о своем голосе: избегайте перенапряжения, поддерживайте водный баланс.",
        "en": "Your voice is healthy. No signs of voice disorders detected. Continue taking care of your voice: avoid strain, stay hydrated.",
    },
    "pathological": {
        "ru": "Обнаружены признаки, которые могут указывать на голосовые нарушения. Рекомендуем обратиться к отоларингологу или фониатру для профессиональной оценки.",
        "en": "Signs detected that may indicate voice disorders. We recommend consulting an ENT specialist or phoniatrician for professional evaluation.",
    },
    "abstain": {
        "ru": "Анализ не дал однозначного результата. Это может быть связано с качеством записи или пограничными показателями. Рекомендуем повторить тест в тихом месте или обратиться к специалисту.",
        "en": "Analysis was inconclusive. This may be due to recording quality or borderline indicators. We recommend retaking the test in a quiet place or consulting a specialist.",
    },
}

# Server configuration
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,  # Number of worker processes
    "timeout": 120,  # Request timeout in seconds
    "max_upload_size": 10 * 1024 * 1024,  # 10 MB max file size
}

# Database configuration
DB_CONFIG = {
    "url": "sqlite:///./echoflow.db",  # SQLite for simplicity
    "echo": False,  # Don't log SQL queries in production
    "pool_size": 10,
    "max_overflow": 20,
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "echoflow.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "default",
            "level": "INFO",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}
