#!/bin/bash
###############################################################################
# EchoFlow 2.0 - Automated Training Script
# Single-command training pipeline for Kaggle or local execution
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASET_URL="https://zenodo.org/records/16874898/files/data.zip"
DATASET_SIZE="17.9 GB"
WORK_DIR="/kaggle/working"
DATA_DIR="${WORK_DIR}/dataset"
CHECKPOINT_DIR="${WORK_DIR}/checkpoints"
BATCH_SIZE=16
EPOCHS=50
LEARNING_RATE=0.0001
NUM_WORKERS=2

# Detect environment
if [ -d "/kaggle" ]; then
    ENV="kaggle"
    echo -e "${BLUE}Detected Kaggle environment${NC}"
else
    ENV="local"
    WORK_DIR="$(pwd)"
    DATA_DIR="${WORK_DIR}/data"
    CHECKPOINT_DIR="${WORK_DIR}/checkpoints"
    echo -e "${BLUE}Detected local environment${NC}"
fi

###############################################################################
# Functions
###############################################################################

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}GPU detected:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        return 0
    else
        print_warning "No GPU detected. Training will use CPU (very slow!)"
        return 1
    fi
}

check_disk_space() {
    local required_gb=25
    local available_gb=$(df -BG "$WORK_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        print_error "Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB"
        exit 1
    else
        print_success "Disk space: ${available_gb}GB available"
    fi
}

install_dependencies() {
    print_header "Step 1: Installing Dependencies"
    
    # Check if already installed
    if python3 -c "import torch, transformers, librosa" 2>/dev/null; then
        print_success "Dependencies already installed"
        return 0
    fi
    
    echo "Installing PyTorch and dependencies..."
    pip install -q torch torchvision torchaudio transformers librosa soundfile scipy scikit-learn tqdm
    
    print_success "Dependencies installed"
}

download_dataset() {
    print_header "Step 2: Downloading Dataset ($DATASET_SIZE)"
    
    # Check if dataset already exists
    if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR)" ]; then
        print_success "Dataset already exists at $DATA_DIR"
        return 0
    fi
    
    mkdir -p "$DATA_DIR"
    
    echo "Downloading Saarbruecken Voice Database..."
    echo "Size: $DATASET_SIZE (this may take 10-15 minutes)"
    
    wget -c "$DATASET_URL" -O "${WORK_DIR}/data.zip" --progress=bar:force 2>&1 | \
        grep --line-buffered -oP '\d+%' | \
        while read percent; do
            echo -ne "\rProgress: $percent"
        done
    echo ""
    
    print_success "Download complete"
    
    echo "Extracting dataset..."
    unzip -q "${WORK_DIR}/data.zip" -d "$DATA_DIR"
    rm "${WORK_DIR}/data.zip"
    
    print_success "Dataset extracted to $DATA_DIR"
}

prepare_dataset() {
    print_header "Step 3: Preparing Dataset Structure"
    
    # Check dataset structure
    if [ ! -d "$DATA_DIR" ]; then
        print_error "Dataset directory not found: $DATA_DIR"
        exit 1
    fi
    
    # Count files
    local total_files=$(find "$DATA_DIR" -name "*.wav" | wc -l)
    
    if [ "$total_files" -eq 0 ]; then
        print_error "No audio files found in dataset"
        exit 1
    fi
    
    print_success "Found $total_files audio files"
    
    # Create organized structure if needed
    if [ ! -d "$DATA_DIR/healthy" ] && [ ! -d "$DATA_DIR/pathological" ]; then
        echo "Organizing dataset structure..."
        python3 - <<EOF
import os
import shutil
from pathlib import Path

data_dir = Path("$DATA_DIR")
healthy_dir = data_dir / "healthy"
pathological_dir = data_dir / "pathological"

healthy_dir.mkdir(exist_ok=True)
pathological_dir.mkdir(exist_ok=True)

# Move files based on naming convention
for wav_file in data_dir.rglob("*.wav"):
    if wav_file.parent.name in ["healthy", "pathological"]:
        continue
    
    # Simple heuristic: files with 'n' (normal) go to healthy
    if '_n_' in wav_file.name.lower() or 'normal' in wav_file.name.lower():
        shutil.move(str(wav_file), str(healthy_dir / wav_file.name))
    else:
        shutil.move(str(wav_file), str(pathological_dir / wav_file.name))

print(f"Healthy: {len(list(healthy_dir.glob('*.wav')))}")
print(f"Pathological: {len(list(pathological_dir.glob('*.wav')))}")
EOF
        print_success "Dataset organized"
    else
        print_success "Dataset structure is correct"
    fi
}

train_model() {
    print_header "Step 4: Training EchoFlow 2.0"
    
    mkdir -p "$CHECKPOINT_DIR"
    
    echo "Training configuration:"
    echo "  Data directory: $DATA_DIR"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Epochs: $EPOCHS"
    echo "  Learning rate: $LEARNING_RATE"
    echo "  Checkpoint directory: $CHECKPOINT_DIR"
    echo ""
    
    # Check if training was already completed
    if [ -f "$CHECKPOINT_DIR/best.pt" ]; then
        print_warning "Found existing checkpoint. Resume training? (y/n)"
        read -t 10 -n 1 resume || resume="y"
        echo ""
        if [ "$resume" != "y" ]; then
            print_success "Using existing checkpoint"
            return 0
        fi
    fi
    
    echo "Starting training (this will take ~20 hours on GPU)..."
    echo ""
    
    python3 train.py \
        --data_dir "$DATA_DIR" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LEARNING_RATE" \
        --save_dir "$CHECKPOINT_DIR" \
        --num_workers "$NUM_WORKERS"
    
    if [ $? -eq 0 ]; then
        print_success "Training completed successfully!"
    else
        print_error "Training failed!"
        exit 1
    fi
}

evaluate_model() {
    print_header "Step 5: Evaluating Model"
    
    if [ ! -f "$CHECKPOINT_DIR/best.pt" ]; then
        print_error "No trained model found at $CHECKPOINT_DIR/best.pt"
        exit 1
    fi
    
    echo "Loading training history..."
    python3 - <<EOF
import json
from pathlib import Path

history_file = Path("$CHECKPOINT_DIR") / "history.json"
if history_file.exists():
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print("\n" + "="*60)
    print("Training Results")
    print("="*60)
    print(f"Best Validation Accuracy: {max(history['val_acc']):.2f}%")
    print(f"Final Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final Val Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print("="*60)
    
    # Check if target achieved
    best_acc = max(history['val_acc'])
    if best_acc >= 92.0:
        print("\n✓ TARGET ACHIEVED: Accuracy >= 92%")
    elif best_acc >= 90.0:
        print("\n⚠ CLOSE TO TARGET: Accuracy >= 90%")
    else:
        print("\n✗ BELOW TARGET: Accuracy < 90%")
else:
    print("No history file found")
EOF
    
    print_success "Evaluation complete"
}

save_artifacts() {
    print_header "Step 6: Saving Artifacts"
    
    # Create deployment package
    local deploy_dir="${WORK_DIR}/echoflow_v2_deployment"
    mkdir -p "$deploy_dir"
    
    echo "Creating deployment package..."
    
    # Copy model
    if [ -f "$CHECKPOINT_DIR/best.pt" ]; then
        cp "$CHECKPOINT_DIR/best.pt" "$deploy_dir/"
        print_success "Model copied: best.pt"
    fi
    
    # Copy history
    if [ -f "$CHECKPOINT_DIR/history.json" ]; then
        cp "$CHECKPOINT_DIR/history.json" "$deploy_dir/"
        print_success "History copied: history.json"
    fi
    
    # Create README
    cat > "$deploy_dir/README.txt" <<EOF
EchoFlow 2.0 - Trained Model Package
====================================

Contents:
- best.pt: Trained model weights (~1.2 GB)
- history.json: Training history and metrics

Deployment Instructions:
1. Copy best.pt to your server
2. Place in: /opt/echoflow/models/checkpoints/
3. Run: python server/app/main.py

For full documentation, see:
https://github.com/izoomlentoboy-creator/Voice
EOF
    
    print_success "Deployment package created: $deploy_dir"
    
    # Show file sizes
    echo ""
    echo "Artifact sizes:"
    du -h "$deploy_dir"/* 2>/dev/null || true
}

###############################################################################
# Main Execution
###############################################################################

main() {
    print_header "EchoFlow 2.0 - Automated Training Pipeline"
    
    echo "Environment: $ENV"
    echo "Working directory: $WORK_DIR"
    echo ""
    
    # System checks
    check_gpu
    check_disk_space
    
    # Execute pipeline
    install_dependencies
    download_dataset
    prepare_dataset
    train_model
    evaluate_model
    save_artifacts
    
    print_header "Training Pipeline Complete! 🎉"
    
    echo "Next steps:"
    echo "1. Download the trained model from: $CHECKPOINT_DIR/best.pt"
    echo "2. Deploy to your server using the deployment guide"
    echo "3. Test with real audio samples"
    echo ""
    print_success "All done!"
}

# Run main function
main "$@"
