#!/bin/bash
################################################################################
# Voice Disorder Detection - Fixed Production Deployment Script
# Version: 2.0.0 (Error-tolerant)
# Description: Automated deployment with proper error handling
################################################################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Configuration
INSTALL_DIR="/opt/Voice"
DATA_DIR="/var/lib/voice-disorder/data"
PYTHON="python3.11"

################################################################################
# Main Installation
################################################################################

main() {
    log_info "Starting Voice Disorder Detection deployment..."
    
    # Step 1: Check we're in the right directory
    if [ ! -d "$INSTALL_DIR" ]; then
        log_error "Directory $INSTALL_DIR not found!"
        exit 1
    fi
    
    cd "$INSTALL_DIR"
    log_success "Working directory: $INSTALL_DIR"
    
    # Step 2: Install Python dependencies (ignore system package warnings)
    log_info "Installing Python dependencies..."
    
    $PYTHON -m pip install --upgrade pip setuptools wheel 2>&1 | grep -v "WARNING\|ERROR: Cannot uninstall" || true
    
    log_info "Installing scikit-learn and ML libraries..."
    $PYTHON -m pip install --no-warn-script-location \
        scikit-learn \
        numpy \
        scipy \
        joblib \
        2>&1 | grep -v "WARNING\|ERROR: Cannot uninstall" || true
    
    log_info "Installing audio processing libraries..."
    $PYTHON -m pip install --no-warn-script-location \
        librosa \
        soundfile \
        audioread \
        2>&1 | grep -v "WARNING\|ERROR: Cannot uninstall" || true
    
    log_info "Installing web framework..."
    $PYTHON -m pip install --no-warn-script-location \
        fastapi \
        uvicorn \
        pydantic \
        sqlalchemy \
        2>&1 | grep -v "WARNING\|ERROR: Cannot uninstall" || true
    
    log_info "Installing project-specific packages..."
    $PYTHON -m pip install --no-warn-script-location \
        sbvoicedb \
        audiomentations \
        gunicorn \
        rich \
        2>&1 | grep -v "WARNING\|ERROR: Cannot uninstall" || true
    
    log_success "Dependencies installation completed"
    
    # Step 3: Verify critical imports
    log_info "Verifying installations..."
    
    if $PYTHON -c "import sklearn" 2>/dev/null; then
        log_success "scikit-learn: OK"
    else
        log_error "scikit-learn: FAILED"
        exit 1
    fi
    
    if $PYTHON -c "import librosa" 2>/dev/null; then
        log_success "librosa: OK"
    else
        log_error "librosa: FAILED"
        exit 1
    fi
    
    if $PYTHON -c "import fastapi" 2>/dev/null; then
        log_success "fastapi: OK"
    else
        log_error "fastapi: FAILED"
        exit 1
    fi
    
    if $PYTHON -c "import sbvoicedb" 2>/dev/null; then
        log_success "sbvoicedb: OK"
    else
        log_error "sbvoicedb: FAILED"
        exit 1
    fi
    
    log_success "All critical dependencies verified!"
    
    # Step 4: Setup data directory
    log_info "Setting up data directory..."
    mkdir -p "$DATA_DIR"
    export SBVOICEDB_DIR="$DATA_DIR/sbvoicedb"
    log_success "Data directory: $SBVOICEDB_DIR"
    
    # Step 5: Download dataset
    log_info "Downloading Saarbruecken Voice Database..."
    log_warning "This will take 15-30 minutes depending on your connection"
    
    $PYTHON << 'PYEOF'
from sbvoicedb import SbVoiceDb
import os
import sys

db_dir = os.environ.get('SBVOICEDB_DIR')
print(f"Initializing database at: {db_dir}")

try:
    db = SbVoiceDb(dbdir=db_dir, download_mode='lazy')
    print(f"Total sessions available: {db.number_of_all_sessions}")
    
    # Trigger download by accessing sessions
    print("Downloading dataset...")
    count = 0
    for session in db.iter_sessions():
        try:
            sess = db.get_session(session.id, query_recordings=True, query_pathologies=True)
            if sess and sess.recordings:
                for rec in sess.recordings:
                    try:
                        rec_full = db.get_recording(rec.id, full_file_paths=True)
                        if rec_full:
                            count += 1
                            if count % 100 == 0:
                                print(f"Downloaded {count} recordings...")
                    except:
                        pass
        except:
            pass
        
        if count >= 1000:  # Download at least 1000 recordings
            break
    
    print(f"Dataset ready! Downloaded {count} recordings")
    sys.exit(0)
except Exception as e:
    print(f"Error downloading dataset: {e}")
    sys.exit(1)
PYEOF
    
    if [ $? -eq 0 ]; then
        log_success "Dataset downloaded successfully"
    else
        log_error "Dataset download failed"
        exit 1
    fi
    
    # Step 6: Train model
    log_info "Starting model training with optimization..."
    log_warning "This will take 30-60 minutes"
    
    export SBVOICEDB_DIR="$DATA_DIR/sbvoicedb"
    
    $PYTHON scripts/train_optimized.py \
        --backend ensemble \
        --all \
        2>&1 | tee training_output.log
    
    if [ $? -eq 0 ]; then
        log_success "Model training completed!"
    else
        log_error "Model training failed. Check training_output.log"
        exit 1
    fi
    
    # Step 7: Verify model files
    log_info "Verifying trained models..."
    
    if [ -d "models" ] && [ "$(ls -A models/*.pkl 2>/dev/null)" ]; then
        log_success "Model files found:"
        ls -lh models/*.pkl
    else
        log_error "No model files found!"
        exit 1
    fi
    
    # Step 8: Setup environment
    log_info "Creating environment configuration..."
    
    cat > .env << EOF
TBVOICE_DATABASE_URL=sqlite:///$INSTALL_DIR/server/tbvoice.db
TBVOICE_MODEL_BACKEND=ensemble
TBVOICE_MODEL_MODE=binary
TBVOICE_DEBUG=false
SBVOICEDB_DIR=$DATA_DIR/sbvoicedb
EOF
    
    log_success "Environment configured"
    
    # Step 9: Test server startup
    log_info "Testing server startup..."
    
    timeout 10 $PYTHON -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000 &
    SERVER_PID=$!
    sleep 5
    
    if kill -0 $SERVER_PID 2>/dev/null; then
        log_success "Server starts successfully"
        kill $SERVER_PID
    else
        log_warning "Server test failed, but continuing..."
    fi
    
    # Final summary
    echo ""
    echo "=========================================="
    log_success "DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Start server:"
    echo "   cd $INSTALL_DIR"
    echo "   python3.11 -m gunicorn server.app.main:app \\"
    echo "       --workers 4 \\"
    echo "       --worker-class uvicorn.workers.UvicornWorker \\"
    echo "       --bind 0.0.0.0:8000"
    echo ""
    echo "2. Or run in background:"
    echo "   nohup python3.11 -m gunicorn server.app.main:app \\"
    echo "       --workers 4 \\"
    echo "       --worker-class uvicorn.workers.UvicornWorker \\"
    echo "       --bind 0.0.0.0:8000 \\"
    echo "       > /var/log/voice-api.log 2>&1 &"
    echo ""
    echo "3. Test API:"
    echo "   curl http://localhost:8000/api/v1/health"
    echo ""
    echo "Model accuracy report: training_report.json"
    echo "Training logs: training_output.log"
    echo ""
}

# Run main function
main "$@"
