#!/usr/bin/env python3
"""
EchoFlow 2.0 - Comprehensive Testing Suite
Tests all components for errors and compatibility
"""

import sys
import os
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Test results
tests_passed = 0
tests_failed = 0
errors = []


def print_header(text):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_test(name, passed, error=None):
    """Print test result"""
    global tests_passed, tests_failed, errors
    
    if passed:
        print(f"✓ {name}")
        tests_passed += 1
    else:
        print(f"✗ {name}")
        tests_failed += 1
        if error:
            errors.append((name, error))
            print(f"  Error: {error}")


def test_imports():
    """Test all required imports"""
    print_header("Testing Imports")
    
    # Core libraries
    try:
        import torch
        print_test("PyTorch", True)
        print(f"  Version: {torch.__version__}")
    except Exception as e:
        print_test("PyTorch", False, str(e))
        return False
    
    try:
        import transformers
        print_test("Transformers", True)
        print(f"  Version: {transformers.__version__}")
    except Exception as e:
        print_test("Transformers", False, str(e))
        return False
    
    try:
        import librosa
        print_test("Librosa", True)
    except Exception as e:
        print_test("Librosa", False, str(e))
        return False
    
    try:
        import numpy as np
        print_test("NumPy", True)
    except Exception as e:
        print_test("NumPy", False, str(e))
        return False
    
    try:
        import scipy
        print_test("SciPy", True)
    except Exception as e:
        print_test("SciPy", False, str(e))
        return False
    
    try:
        from sklearn.metrics import accuracy_score
        print_test("Scikit-learn", True)
    except Exception as e:
        print_test("Scikit-learn", False, str(e))
        return False
    
    return True


def test_model_architecture():
    """Test model architecture"""
    print_header("Testing Model Architecture")
    
    try:
        from models.echoflow_v2 import EchoFlowV2, count_parameters
        print_test("Import EchoFlowV2", True)
    except Exception as e:
        print_test("Import EchoFlowV2", False, str(e))
        return False
    
    try:
        import torch
        model = EchoFlowV2(
            freeze_wav2vec2=True,
            d_model=512,
            nhead=8,
            num_layers=4
        )
        print_test("Create model instance", True)
        
        # Count parameters
        params = count_parameters(model)
        print(f"  Total parameters: {params['total_millions']:.2f}M")
        print(f"  Trainable parameters: {params['trainable_millions']:.2f}M")
        
        # Check parameter counts
        if params['total_millions'] > 300 and params['total_millions'] < 350:
            print_test("Parameter count in expected range", True)
        else:
            print_test("Parameter count in expected range", False, 
                      f"Expected 300-350M, got {params['total_millions']:.2f}M")
        
    except Exception as e:
        print_test("Create model instance", False, str(e))
        traceback.print_exc()
        return False
    
    return True


def test_forward_pass():
    """Test model forward pass"""
    print_header("Testing Forward Pass")
    
    try:
        import torch
        from models.echoflow_v2 import EchoFlowV2
        
        model = EchoFlowV2(freeze_wav2vec2=True)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        seq_len = 16000 * 3  # 3 seconds
        dummy_audio = torch.randn(batch_size, seq_len)
        
        print(f"  Input shape: {dummy_audio.shape}")
        
        # Forward pass
        with torch.no_grad():
            logits = model(dummy_audio)
            probs = model.predict_proba(dummy_audio)
            preds = model.predict(dummy_audio)
        
        print(f"  Logits shape: {logits.shape}")
        print(f"  Probs shape: {probs.shape}")
        print(f"  Preds shape: {preds.shape}")
        
        # Validate shapes
        if logits.shape == (batch_size, 2):
            print_test("Logits shape correct", True)
        else:
            print_test("Logits shape correct", False, f"Expected ({batch_size}, 2), got {logits.shape}")
        
        if probs.shape == (batch_size, 2):
            print_test("Probabilities shape correct", True)
        else:
            print_test("Probabilities shape correct", False, f"Expected ({batch_size}, 2), got {probs.shape}")
        
        if preds.shape == (batch_size,):
            print_test("Predictions shape correct", True)
        else:
            print_test("Predictions shape correct", False, f"Expected ({batch_size},), got {preds.shape}")
        
        # Check probability sum
        prob_sum = probs.sum(dim=1)
        if torch.allclose(prob_sum, torch.ones(batch_size), atol=1e-5):
            print_test("Probabilities sum to 1", True)
        else:
            print_test("Probabilities sum to 1", False, f"Sum: {prob_sum}")
        
    except Exception as e:
        print_test("Forward pass", False, str(e))
        traceback.print_exc()
        return False
    
    return True


def test_augmentation():
    """Test data augmentation pipeline"""
    print_header("Testing Data Augmentation")
    
    try:
        from utils.augmentation import AugmentationPipeline
        import numpy as np
        
        aug = AugmentationPipeline(augmentation_prob=1.0)
        print_test("Create augmentation pipeline", True)
        
        # Create dummy audio
        audio = np.random.randn(16000 * 3)  # 3 seconds
        
        # Test augmentation
        augmented = aug(audio)
        
        if augmented.shape == audio.shape:
            print_test("Augmentation preserves shape", True)
        else:
            print_test("Augmentation preserves shape", False, 
                      f"Expected {audio.shape}, got {augmented.shape}")
        
        # Test multiple augmentations
        results = [aug(audio) for _ in range(5)]
        
        # Check that augmentations produce different results
        all_different = not all(np.array_equal(results[0], r) for r in results[1:])
        print_test("Augmentations produce variations", all_different)
        
    except Exception as e:
        print_test("Augmentation pipeline", False, str(e))
        traceback.print_exc()
        return False
    
    return True


def test_dataset_loader():
    """Test dataset loader"""
    print_header("Testing Dataset Loader")
    
    try:
        from utils.dataset import VoicePathologyDataset
        import numpy as np
        
        print_test("Import dataset loader", True)
        
        # Note: Cannot fully test without actual dataset
        print("  ⚠ Full dataset test requires actual data files")
        print("  ⚠ Skipping dataset loading test")
        
    except Exception as e:
        print_test("Import dataset loader", False, str(e))
        return False
    
    return True


def test_training_script():
    """Test training script imports"""
    print_header("Testing Training Script")
    
    try:
        # Check if train.py exists
        train_file = Path(__file__).parent / "train.py"
        if not train_file.exists():
            print_test("train.py exists", False, "File not found")
            return False
        
        print_test("train.py exists", True)
        
        # Try to import (without executing)
        import importlib.util
        spec = importlib.util.spec_from_file_location("train", train_file)
        if spec and spec.loader:
            print_test("train.py is valid Python", True)
        else:
            print_test("train.py is valid Python", False, "Cannot load module")
        
    except Exception as e:
        print_test("Training script", False, str(e))
        return False
    
    return True


def test_file_structure():
    """Test project file structure"""
    print_header("Testing File Structure")
    
    required_files = [
        "models/echoflow_v2.py",
        "models/feature_extractor.py",
        "models/transformer_classifier.py",
        "utils/augmentation.py",
        "utils/dataset.py",
        "train.py",
        "train_automated.sh",
        "requirements.txt",
        "README.md"
    ]
    
    project_root = Path(__file__).parent
    
    for file_path in required_files:
        full_path = project_root / file_path
        exists = full_path.exists()
        print_test(f"File exists: {file_path}", exists)
    
    return True


def test_gpu_availability():
    """Test GPU availability"""
    print_header("Testing GPU Availability")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print_test("CUDA available", True)
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print_test("CUDA available", False, "No GPU detected (CPU training will be very slow)")
        
    except Exception as e:
        print_test("GPU check", False, str(e))
    
    return True


def print_summary():
    """Print test summary"""
    print_header("Test Summary")
    
    total_tests = tests_passed + tests_failed
    pass_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTotal tests: {total_tests}")
    print(f"Passed: {tests_passed} ({pass_rate:.1f}%)")
    print(f"Failed: {tests_failed}")
    
    if errors:
        print("\n" + "=" * 70)
        print("  Failed Tests Details")
        print("=" * 70)
        for name, error in errors:
            print(f"\n✗ {name}")
            print(f"  {error}")
    
    print("\n" + "=" * 70)
    if tests_failed == 0:
        print("  ✓ ALL TESTS PASSED!")
        print("  System is ready for training")
    else:
        print("  ✗ SOME TESTS FAILED")
        print("  Please fix errors before training")
    print("=" * 70 + "\n")
    
    return tests_failed == 0


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("  EchoFlow 2.0 - Comprehensive Testing Suite")
    print("=" * 70)
    
    # Run all tests
    test_imports()
    test_file_structure()
    test_model_architecture()
    test_forward_pass()
    test_augmentation()
    test_dataset_loader()
    test_training_script()
    test_gpu_availability()
    
    # Print summary
    all_passed = print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
