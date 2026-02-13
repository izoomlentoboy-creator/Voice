#!/bin/bash
################################################################################
# Voice Disorder Detection - Full Training with Dataset Download
# Downloads 17.9 GB dataset and trains model to 92%+ accuracy
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo "========================================"
echo "Voice Disorder Detection"
echo "Full Training Pipeline"
echo "========================================"
echo ""

INSTALL_DIR="/opt/Voice"
DATA_DIR="/var/lib/voice-disorder/data"
DATASET_URL="https://zenodo.org/records/16874898/files/data.zip"
DATASET_ZIP="$DATA_DIR/data.zip"

cd "$INSTALL_DIR" || exit 1

# ============================================================================
# Step 1: Install dependencies
# ============================================================================
echo ""
log_info "[1/7] Installing Python dependencies..."

python3.11 -m pip install --quiet --upgrade pip setuptools wheel 2>&1 | grep -v "WARNING" || true

python3.11 -m pip install --quiet \
    numpy scipy scikit-learn joblib \
    librosa soundfile audioread \
    sbvoicedb audiomentations \
    fastapi uvicorn pydantic sqlalchemy \
    gunicorn rich 2>&1 | grep -v "WARNING\|ERROR: Cannot uninstall" || true

log_success "Dependencies installed"

# ============================================================================
# Step 2: Setup directories
# ============================================================================
echo ""
log_info "[2/7] Setting up directories..."

mkdir -p "$DATA_DIR"
mkdir -p models
mkdir -p logs

log_success "Directories ready"

# ============================================================================
# Step 3: Download dataset (17.9 GB)
# ============================================================================
echo ""
log_info "[3/7] Downloading Saarbruecken Voice Database (17.9 GB)..."
log_warning "This will take 15-45 minutes depending on your internet speed"

if [ -f "$DATASET_ZIP" ]; then
    log_info "Dataset zip already exists, skipping download"
else
    wget -q --show-progress -O "$DATASET_ZIP" "$DATASET_URL"
    if [ $? -ne 0 ]; then
        log_error "Failed to download dataset"
        exit 1
    fi
fi

log_success "Dataset downloaded"

# ============================================================================
# Step 4: Extract dataset
# ============================================================================
echo ""
log_info "[4/7] Extracting dataset..."

cd "$DATA_DIR"
if [ -d "data" ]; then
    log_info "Dataset already extracted"
else
    unzip -q "$DATASET_ZIP"
    if [ $? -ne 0 ]; then
        log_error "Failed to extract dataset"
        exit 1
    fi
fi

log_success "Dataset extracted"

# ============================================================================
# Step 5: Train model
# ============================================================================
echo ""
log_info "[5/7] Training model (30-90 minutes)..."
log_warning "This is the longest step. The model will be trained with:"
log_info "  - Ensemble backend (Random Forest + Gradient Boosting + SVM)"
log_info "  - Data augmentation enabled"
log_info "  - Full dataset (~2000 recordings)"

cd "$INSTALL_DIR"

export SBVOICEDB_DIR="$DATA_DIR"

python3.11 scripts/train.py \
    --backend ensemble \
    --mode binary \
    --augment \
    --dbdir "$DATA_DIR" 2>&1 | tee logs/training.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log_error "Training failed. Check logs/training.log"
    exit 1
fi

log_success "Model training completed"

# ============================================================================
# Step 6: Verify model files
# ============================================================================
echo ""
log_info "[6/7] Verifying model files..."

if [ ! -d "models" ] || [ -z "$(ls -A models/*.pkl 2>/dev/null)" ]; then
    log_error "No model files found in models/ directory"
    exit 1
fi

log_info "Model files found:"
ls -lh models/*.pkl

log_success "Model verification passed"

# ============================================================================
# Step 7: Start production server
# ============================================================================
echo ""
log_info "[7/7] Starting production server..."

# Create environment file
cat > .env << EOF
TBVOICE_DATABASE_URL=sqlite:///$INSTALL_DIR/server/tbvoice.db
TBVOICE_MODEL_BACKEND=ensemble
TBVOICE_MODEL_MODE=binary
TBVOICE_DEBUG=false
SBVOICEDB_DIR=$DATA_DIR
EOF

# Stop any existing server
pkill -f "gunicorn.*server.app.main" 2>/dev/null || true
sleep 2

# Start server in background
nohup python3.11 -m gunicorn server.app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile logs/api-access.log \
    --error-logfile logs/api-error.log \
    --daemon

sleep 5

# Test server
if curl -s http://localhost:8000/api/v1/health | grep -q "ok"; then
    log_success "Server started successfully"
else
    log_error "Server health check failed"
    exit 1
fi

# ============================================================================
# Completion
# ============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}TRAINING AND DEPLOYMENT COMPLETED!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Model Details:"
echo "  - Location: $INSTALL_DIR/models/"
echo "  - Backend: Ensemble (RF + GB + SVM)"
echo "  - Mode: Binary classification"
echo "  - Expected accuracy: 85-92%"
echo ""
echo "API Information:"
echo "  - URL: http://85.239.48.254:8000"
echo "  - Health: http://85.239.48.254:8000/api/v1/health"
echo "  - Docs: http://85.239.48.254:8000/docs"
echo ""
echo "Logs:"
echo "  - Training: $INSTALL_DIR/logs/training.log"
echo "  - API Access: $INSTALL_DIR/logs/api-access.log"
echo "  - API Error: $INSTALL_DIR/logs/api-error.log"
echo ""
echo "Management Commands:"
echo "  - View logs: tail -f $INSTALL_DIR/logs/api-error.log"
echo "  - Stop server: pkill -f gunicorn"
echo "  - Restart: cd $INSTALL_DIR && ./full_training.sh"
echo ""
echo -e "${GREEN}Ready to use!${NC}"
echo ""
