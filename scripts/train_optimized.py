#!/usr/bin/env python3.11
"""
Optimized Training Script for Voice Disorder Detection
Implements advanced training strategies for achieving 92%+ accuracy
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, recall_score, roc_auc_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_disorder_detection import config
from voice_disorder_detection.data_loader import VoiceDataLoader
from voice_disorder_detection.model import VoiceDisorderModel
from voice_disorder_detection.calibration import calibrate_model
from voice_disorder_detection.evaluation import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('training_optimized.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptimizedTrainer:
    """Advanced trainer with hyperparameter optimization and best practices."""
    
    def __init__(self, backend: str = "ensemble", mode: str = "binary"):
        self.backend = backend
        self.mode = mode
        self.loader = VoiceDataLoader()
        self.best_params = {}
        self.training_history = []
        
    def load_data(self, max_samples: int = None, augment: bool = False):
        """Load and prepare dataset."""
        logger.info(f"Loading data (augment={augment}, max_samples={max_samples})...")
        
        X, y, session_ids, speaker_ids, metadata = self.loader.extract_dataset(
            mode=self.mode,
            max_samples=max_samples,
            use_cache=True,
            augment=augment
        )
        
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y, session_ids, speaker_ids, metadata
    
    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        speaker_ids: list,
        method: str = "random",
        n_iter: int = 30
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using RandomizedSearchCV or GridSearchCV."""
        logger.info(f"Starting hyperparameter optimization (method={method})...")
        
        from sklearn.model_selection import GroupKFold
        
        # Define parameter grids for each backend
        param_grids = {
            "svm": {
                "C": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "kernel": ["rbf", "poly"],
                "class_weight": ["balanced"]
            },
            "random_forest": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": ["balanced", "balanced_subsample"]
            },
            "gradient_boosting": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 10],
                "subsample": [0.8, 0.9, 1.0],
                "min_samples_split": [2, 5, 10]
            },
            "logistic_regression": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
                "class_weight": ["balanced"]
            }
        }
        
        # For ensemble, optimize each component
        if self.backend == "ensemble":
            backend_to_optimize = "random_forest"  # Optimize RF as representative
        else:
            backend_to_optimize = self.backend
        
        if backend_to_optimize not in param_grids:
            logger.warning(f"No parameter grid for {backend_to_optimize}, using defaults")
            return {}
        
        # Create model
        model = VoiceDisorderModel(backend=backend_to_optimize, mode=self.mode)
        
        # Setup cross-validation
        cv = GroupKFold(n_splits=5)
        
        # Define scoring
        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "sensitivity": make_scorer(recall_score, pos_label=1),
            "roc_auc": make_scorer(roc_auc_score, needs_proba=True)
        }
        
        # Run optimization
        if method == "random":
            search = RandomizedSearchCV(
                model.clf,
                param_grids[backend_to_optimize],
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                refit="roc_auc",
                n_jobs=-1,
                verbose=2,
                random_state=config.RANDOM_STATE
            )
        else:
            search = GridSearchCV(
                model.clf,
                param_grids[backend_to_optimize],
                cv=cv,
                scoring=scoring,
                refit="roc_auc",
                n_jobs=-1,
                verbose=2
            )
        
        # Fit
        groups = np.array(speaker_ids) if speaker_ids else None
        search.fit(X, y, groups=groups)
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best ROC-AUC: {search.best_score_:.4f}")
        
        self.best_params = search.best_params_
        
        return search.best_params_
    
    def train_with_best_params(
        self,
        X: np.ndarray,
        y: np.ndarray,
        speaker_ids: list,
        params: Dict[str, Any] = None
    ) -> VoiceDisorderModel:
        """Train model with optimized parameters."""
        logger.info("Training model with optimized parameters...")
        
        model = VoiceDisorderModel(backend=self.backend, mode=self.mode)
        
        # Apply best parameters if available
        if params and self.backend != "ensemble":
            model.clf.set_params(**params)
        
        # Train
        start_time = time.time()
        model.train(X, y, speaker_ids=speaker_ids)
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f}s")
        
        return model
    
    def evaluate_model(
        self,
        model: VoiceDisorderModel,
        X: np.ndarray,
        y: np.ndarray,
        speaker_ids: list,
        metadata: list
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        logger.info("Evaluating model...")
        
        results = evaluate_model(
            model,
            X,
            y,
            speaker_ids=speaker_ids,
            metadata=metadata,
            output_dir=Path("evaluation_results")
        )
        
        # Log key metrics
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Accuracy:     {results['accuracy']:.4f}")
        logger.info(f"Sensitivity:  {results['sensitivity']:.4f}")
        logger.info(f"Specificity:  {results['specificity']:.4f}")
        logger.info(f"ROC-AUC:      {results['roc_auc']:.4f}")
        logger.info(f"PR-AUC:       {results['pr_auc']:.4f}")
        logger.info(f"Brier Score:  {results['brier_score']:.4f}")
        logger.info(f"ECE:          {results['ece']:.4f}")
        logger.info("=" * 60)
        
        return results
    
    def run_full_pipeline(
        self,
        optimize: bool = True,
        augment: bool = True,
        calibrate: bool = True,
        max_samples: int = None
    ):
        """Run complete optimized training pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING OPTIMIZED TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # Stage 1: Load data
        logger.info("\n[STAGE 1/6] Loading data...")
        X, y, session_ids, speaker_ids, metadata = self.load_data(
            max_samples=max_samples,
            augment=False  # Start without augmentation
        )
        
        # Stage 2: Hyperparameter optimization
        best_params = {}
        if optimize and self.backend != "ensemble":
            logger.info("\n[STAGE 2/6] Optimizing hyperparameters...")
            best_params = self.optimize_hyperparameters(X, y, speaker_ids)
        else:
            logger.info("\n[STAGE 2/6] Skipping hyperparameter optimization")
        
        # Stage 3: Train baseline model
        logger.info("\n[STAGE 3/6] Training baseline model...")
        baseline_model = self.train_with_best_params(X, y, speaker_ids, best_params)
        baseline_results = self.evaluate_model(baseline_model, X, y, speaker_ids, metadata)
        
        # Save baseline
        baseline_model.save()
        logger.info(f"Baseline model saved (accuracy: {baseline_results['accuracy']:.4f})")
        
        # Stage 4: Train with augmentation
        if augment:
            logger.info("\n[STAGE 4/6] Training with data augmentation...")
            X_aug, y_aug, _, speaker_ids_aug, metadata_aug = self.load_data(
                max_samples=max_samples,
                augment=True
            )
            augmented_model = self.train_with_best_params(X_aug, y_aug, speaker_ids_aug, best_params)
            augmented_results = self.evaluate_model(augmented_model, X, y, speaker_ids, metadata)
            
            # Keep best model
            if augmented_results['accuracy'] > baseline_results['accuracy']:
                logger.info("Augmented model is better, keeping it")
                best_model = augmented_model
                best_results = augmented_results
                best_model.save()
            else:
                logger.info("Baseline model is better, keeping it")
                best_model = baseline_model
                best_results = baseline_results
        else:
            logger.info("\n[STAGE 4/6] Skipping data augmentation")
            best_model = baseline_model
            best_results = baseline_results
        
        # Stage 5: Calibration
        if calibrate:
            logger.info("\n[STAGE 5/6] Calibrating model...")
            calibration_results = calibrate_model(
                best_model,
                X,
                y,
                speaker_ids=speaker_ids,
                method="isotonic"
            )
            logger.info(f"Calibration: ECE improved from {calibration_results['before_ece']:.4f} to {calibration_results['after_ece']:.4f}")
        else:
            logger.info("\n[STAGE 5/6] Skipping calibration")
        
        # Stage 6: Final evaluation and report
        logger.info("\n[STAGE 6/6] Generating final report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": self.backend,
            "mode": self.mode,
            "dataset_size": len(y),
            "augmentation_used": augment,
            "calibration_used": calibrate,
            "optimization_used": optimize,
            "best_parameters": best_params,
            "results": best_results,
            "training_history": self.training_history
        }
        
        # Save report
        report_path = Path("training_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved to {report_path}")
        
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Final Accuracy: {best_results['accuracy']:.4f}")
        logger.info(f"Final ROC-AUC:  {best_results['roc_auc']:.4f}")
        logger.info("=" * 60)
        
        return best_model, best_results


def main():
    parser = argparse.ArgumentParser(description="Optimized training for Voice Disorder Detection")
    parser.add_argument("--backend", default="ensemble", choices=["svm", "random_forest", "gradient_boosting", "logistic_regression", "ensemble"])
    parser.add_argument("--mode", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate model probabilities")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples (for testing)")
    parser.add_argument("--all", action="store_true", help="Enable all optimizations")
    
    args = parser.parse_args()
    
    if args.all:
        args.optimize = True
        args.augment = True
        args.calibrate = True
    
    # Run training
    trainer = OptimizedTrainer(backend=args.backend, mode=args.mode)
    trainer.run_full_pipeline(
        optimize=args.optimize,
        augment=args.augment,
        calibrate=args.calibrate,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
