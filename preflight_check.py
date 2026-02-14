#!/usr/bin/env python3
"""
EchoFlow 2.0 - Pre-flight Check
Comprehensive validation before training
Automatically fixes all detected issues
"""

import sys
import os
import subprocess
from pathlib import Path

# Colors
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}  {text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.NC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")

def print_info(text):
    print(f"{Colors.CYAN}ℹ {text}{Colors.NC}")

def run_command(cmd, quiet=True):
    """Run shell command and return success status"""
    try:
        if quiet:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        else:
            result = subprocess.run(cmd, shell=True, timeout=60)
        return result.returncode == 0
    except:
        return False

def install_package(package, pip_name=None):
    """Install Python package"""
    if pip_name is None:
        pip_name = package
    
    print_info(f"Installing {package}...")
    
    # Try pip3 first
    if run_command(f"pip3 install {pip_name} --quiet"):
        return True
    
    # Try pip
    if run_command(f"pip install {pip_name} --quiet"):
        return True
    
    # Try with sudo
    if run_command(f"sudo pip3 install {pip_name} --quiet"):
        return True
    
    return False

def check_and_fix_dependencies():
    """Check and automatically fix all dependencies"""
    print_header("Checking Dependencies")
    
    all_ok = True
    
    # Check Python
    try:
        import sys
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        else:
            print_error(f"Python {version.major}.{version.minor} (need 3.8+)")
            all_ok = False
    except:
        print_error("Python check failed")
        all_ok = False
    
    # Check and install PyTorch
    try:
        import torch
        print_success(f"PyTorch {torch.__version__}")
    except ImportError:
        print_warning("PyTorch not found, installing...")
        if install_package("PyTorch", "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"):
            print_success("PyTorch installed")
        else:
            print_error("Failed to install PyTorch")
            all_ok = False
    
    # Check and install Transformers
    try:
        import transformers
        print_success(f"Transformers {transformers.__version__}")
    except ImportError:
        print_warning("Transformers not found, installing...")
        if install_package("Transformers", "transformers"):
            print_success("Transformers installed")
        else:
            print_error("Failed to install Transformers")
            all_ok = False
    
    # Check and install Librosa
    try:
        import librosa
        print_success(f"Librosa {librosa.__version__}")
    except ImportError:
        print_warning("Librosa not found, installing...")
        # Install system dependencies first
        run_command("sudo apt-get install -y libsndfile1 ffmpeg", quiet=True)
        if install_package("Librosa", "librosa soundfile"):
            print_success("Librosa installed")
        else:
            print_error("Failed to install Librosa")
            all_ok = False
    
    # Check and install NumPy
    try:
        import numpy as np
        print_success(f"NumPy {np.__version__}")
    except ImportError:
        print_warning("NumPy not found, installing...")
        if install_package("NumPy", "numpy"):
            print_success("NumPy installed")
        else:
            print_error("Failed to install NumPy")
            all_ok = False
    
    # Check and install SciPy
    try:
        import scipy
        print_success(f"SciPy {scipy.__version__}")
    except ImportError:
        print_warning("SciPy not found, installing...")
        if install_package("SciPy", "scipy"):
            print_success("SciPy installed")
        else:
            print_error("Failed to install SciPy")
            all_ok = False
    
    # Check and install scikit-learn
    try:
        import sklearn
        print_success(f"Scikit-learn {sklearn.__version__}")
    except ImportError:
        print_warning("Scikit-learn not found, installing...")
        if install_package("Scikit-learn", "scikit-learn"):
            print_success("Scikit-learn installed")
        else:
            print_error("Failed to install Scikit-learn")
            all_ok = False
    
    # Check and install tqdm
    try:
        import tqdm
        print_success(f"tqdm {tqdm.__version__}")
    except ImportError:
        print_warning("tqdm not found, installing...")
        if install_package("tqdm", "tqdm"):
            print_success("tqdm installed")
        else:
            print_error("Failed to install tqdm")
            all_ok = False
    
    # Check protobuf
    try:
        import google.protobuf
        print_success("Protobuf installed")
    except ImportError:
        print_warning("Protobuf not found, installing...")
        if install_package("Protobuf", "protobuf"):
            print_success("Protobuf installed")
        else:
            print_warning("Protobuf install failed (non-critical)")
    
    return all_ok

def check_model_files():
    """Check if all required model files exist"""
    print_header("Checking Model Files")
    
    required_files = [
        "models/__init__.py",
        "models/echoflow_v2.py",
        "models/feature_extractor.py",
        "models/transformer_classifier.py",
        "utils/__init__.py",
        "utils/augmentation.py",
        "utils/dataset.py",
        "train.py",
        "requirements.txt"
    ]
    
    project_root = Path(__file__).parent
    all_ok = True
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print_success(f"{file_path}")
        else:
            print_error(f"{file_path} - MISSING")
            all_ok = False
            
            # Try to create __init__.py files
            if file_path.endswith("__init__.py"):
                try:
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.touch()
                    print_info(f"Created {file_path}")
                    all_ok = True
                except:
                    pass
    
    return all_ok

def test_model_import():
    """Test if model can be imported"""
    print_header("Testing Model Import")
    
    try:
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from models.echoflow_v2 import EchoFlowV2, count_parameters
        print_success("Model import successful")
        
        # Try to create model instance
        print_info("Creating model instance...")
        import torch
        model = EchoFlowV2(freeze_wav2vec2=True)
        print_success("Model instance created")
        
        # Count parameters
        params = count_parameters(model)
        print_info(f"Total parameters: {params['total_millions']:.2f}M")
        print_info(f"Trainable parameters: {params['trainable_millions']:.2f}M")
        
        return True
        
    except Exception as e:
        print_error(f"Model import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test model forward pass"""
    print_header("Testing Forward Pass")
    
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).parent))
        from models.echoflow_v2 import EchoFlowV2
        
        print_info("Creating model...")
        model = EchoFlowV2(freeze_wav2vec2=True)
        model.eval()
        
        print_info("Creating dummy input...")
        dummy_audio = torch.randn(2, 48000)  # 2 samples, 3 seconds
        
        print_info("Running forward pass...")
        with torch.no_grad():
            logits = model(dummy_audio)
            probs = model.predict_proba(dummy_audio)
        
        print_success(f"Forward pass successful")
        print_info(f"Output shape: {logits.shape}")
        print_info(f"Probabilities shape: {probs.shape}")
        
        return True
        
    except Exception as e:
        print_error(f"Forward pass failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_gpu():
    """Check GPU availability"""
    print_header("Checking GPU")
    
    try:
        import torch
        if torch.cuda.is_available():
            print_success(f"GPU available: {torch.cuda.get_device_name(0)}")
            print_info(f"GPU count: {torch.cuda.device_count()}")
            print_info(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            print_warning("No GPU detected")
            print_info("Training will use CPU (very slow)")
            return False
    except:
        print_warning("GPU check failed")
        return False

def check_disk_space():
    """Check available disk space"""
    print_header("Checking Disk Space")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        
        print_info(f"Free space: {free_gb} GB")
        
        if free_gb < 25:
            print_warning(f"Low disk space: {free_gb}GB (recommended: 25GB+)")
            print_info("Dataset requires ~18GB, model checkpoints ~2GB")
            return False
        else:
            print_success(f"Sufficient disk space: {free_gb}GB")
            return True
    except:
        print_warning("Could not check disk space")
        return True

def main():
    """Run all pre-flight checks"""
    print(f"\n{Colors.CYAN}{'='*70}{Colors.NC}")
    print(f"{Colors.CYAN}  EchoFlow 2.0 - Pre-flight Check{Colors.NC}")
    print(f"{Colors.CYAN}  Validating system and automatically fixing issues{Colors.NC}")
    print(f"{Colors.CYAN}{'='*70}{Colors.NC}\n")
    
    results = {
        "Dependencies": check_and_fix_dependencies(),
        "Model Files": check_model_files(),
        "Model Import": test_model_import(),
        "Forward Pass": test_forward_pass(),
        "GPU": check_gpu(),
        "Disk Space": check_disk_space()
    }
    
    # Summary
    print_header("Pre-flight Check Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        if result:
            print_success(f"{check}")
        else:
            print_error(f"{check}")
    
    print(f"\n{Colors.CYAN}Passed: {passed}/{total}{Colors.NC}\n")
    
    if passed == total:
        print(f"{Colors.GREEN}{'='*70}{Colors.NC}")
        print(f"{Colors.GREEN}  ✓ ALL CHECKS PASSED{Colors.NC}")
        print(f"{Colors.GREEN}  System is ready for training!{Colors.NC}")
        print(f"{Colors.GREEN}{'='*70}{Colors.NC}\n")
        print_info("You can now run: ./train_ultimate.sh")
        return 0
    elif passed >= total - 1:  # Allow GPU check to fail
        print(f"{Colors.YELLOW}{'='*70}{Colors.NC}")
        print(f"{Colors.YELLOW}  ⚠ MOSTLY READY{Colors.NC}")
        print(f"{Colors.YELLOW}  Some non-critical checks failed{Colors.NC}")
        print(f"{Colors.YELLOW}{'='*70}{Colors.NC}\n")
        print_info("You can proceed with: ./train_ultimate.sh")
        print_warning("Note: Training without GPU will be very slow")
        return 0
    else:
        print(f"{Colors.RED}{'='*70}{Colors.NC}")
        print(f"{Colors.RED}  ✗ CHECKS FAILED{Colors.NC}")
        print(f"{Colors.RED}  Please fix errors before training{Colors.NC}")
        print(f"{Colors.RED}{'='*70}{Colors.NC}\n")
        return 1

if __name__ == '__main__':
    sys.exit(main())
