#!/bin/bash
################################################################################
# Voice Disorder Detection - Production Deployment Script
# Version: 1.0.0
# Description: Automated deployment, training, and production setup
################################################################################

set -e  # Exit on error
set -o pipefail  # Catch errors in pipes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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
PROJECT_NAME="Voice"
INSTALL_DIR="/opt/voice-disorder-detection"
LOG_DIR="/var/log/voice-disorder"
DATA_DIR="/var/lib/voice-disorder/data"
MODELS_DIR="/var/lib/voice-disorder/models"
PYTHON_VERSION="3.11"

################################################################################
# System Check and Cleanup
################################################################################

check_system() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        log_info "Operating System: $NAME $VERSION"
    else
        log_error "Cannot determine OS version"
        exit 1
    fi
    
    # Check available disk space (need at least 15GB)
    available_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 15 ]; then
        log_error "Insufficient disk space. Need at least 15GB, available: ${available_space}GB"
        exit 1
    fi
    log_success "Disk space check passed: ${available_space}GB available"
    
    # Check RAM (recommend at least 4GB)
    total_ram=$(free -g | awk 'NR==2 {print $2}')
    if [ "$total_ram" -lt 4 ]; then
        log_warning "Low RAM detected: ${total_ram}GB. Recommended: 4GB+"
    else
        log_success "RAM check passed: ${total_ram}GB available"
    fi
    
    # Check if running as root
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

cleanup_old_installations() {
    log_info "Cleaning up old installations and temporary files..."
    
    # Remove old Python cache
    find /root -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find /tmp -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Clean apt cache
    apt-get clean
    apt-get autoclean
    
    # Remove old logs (keep last 7 days)
    if [ -d "$LOG_DIR" ]; then
        find "$LOG_DIR" -type f -mtime +7 -delete 2>/dev/null || true
    fi
    
    log_success "Cleanup completed"
}

################################################################################
# Dependencies Installation
################################################################################

install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    export DEBIAN_FRONTEND=noninteractive
    
    # Update package list
    apt-get update -qq
    
    # Install essential packages
    apt-get install -y -qq \
        software-properties-common \
        build-essential \
        git \
        curl \
        wget \
        libsndfile1 \
        libffi-dev \
        libssl-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        liblzma-dev \
        python3-openssl \
        htop \
        tmux \
        vim \
        supervisor \
        nginx
    
    log_success "System dependencies installed"
}

install_python() {
    log_info "Checking Python ${PYTHON_VERSION} installation..."
    
    if command -v python${PYTHON_VERSION} &> /dev/null; then
        log_success "Python ${PYTHON_VERSION} already installed"
        python${PYTHON_VERSION} --version
    else
        log_info "Installing Python ${PYTHON_VERSION}..."
        add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update -qq
        apt-get install -y -qq \
            python${PYTHON_VERSION} \
            python${PYTHON_VERSION}-dev \
            python${PYTHON_VERSION}-venv \
            python3-pip
        
        log_success "Python ${PYTHON_VERSION} installed"
    fi
    
    # Upgrade pip
    python${PYTHON_VERSION} -m pip install --upgrade pip setuptools wheel -q
}

################################################################################
# Project Setup
################################################################################

setup_directories() {
    log_info "Setting up project directories..."
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$DATA_DIR"
    mkdir -p "$MODELS_DIR"
    
    log_success "Directories created"
}

clone_or_update_repository() {
    log_info "Setting up project repository..."
    
    if [ -d "$INSTALL_DIR/.git" ]; then
        log_info "Repository exists, updating..."
        cd "$INSTALL_DIR"
        git fetch origin
        git reset --hard origin/main
        git pull origin main
    else
        log_info "Cloning repository..."
        git clone https://github.com/izoomlentoboy-creator/Voice.git "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
    
    log_success "Repository ready at $INSTALL_DIR"
}

install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    cd "$INSTALL_DIR"
    
    # Install main requirements
    python${PYTHON_VERSION} -m pip install -r requirements.txt -q
    
    # Install server requirements
    if [ -f "server/requirements.txt" ]; then
        python${PYTHON_VERSION} -m pip install -r server/requirements.txt -q
    fi
    
    # Install additional production dependencies
    python${PYTHON_VERSION} -m pip install -q \
        gunicorn \
        prometheus-client \
        python-dotenv
    
    log_success "Python dependencies installed"
}

################################################################################
# Data Preparation
################################################################################

download_dataset() {
    log_info "Preparing Saarbruecken Voice Database..."
    
    export SBVOICEDB_DIR="$DATA_DIR/sbvoicedb"
    
    cd "$INSTALL_DIR"
    
    # Create a script to download data
    cat > /tmp/download_svd.py << 'EOF'
from sbvoicedb import SbVoiceDb
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

db_dir = os.environ.get('SBVOICEDB_DIR')
print(f"Initializing database at: {db_dir}")

db = SbVoiceDb(dbdir=db_dir, download_mode='all')
print(f"Total sessions: {db.number_of_all_sessions}")
print(f"Downloaded sessions: {db.number_of_sessions_downloaded}")

# Trigger download by accessing first few sessions
print("Triggering data download...")
count = 0
for session in db.iter_sessions():
    sess_full = db.get_session(session.id, query_recordings=True, query_pathologies=True, query_speaker=True)
    if sess_full and sess_full.recordings:
        for rec in sess_full.recordings:
            rec_full = db.get_recording(rec.id, full_file_paths=True)
            if rec_full and rec_full.file_path:
                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} recordings...")
    if count >= 500:  # Process first 500 to trigger download
        break

print(f"Database ready! Downloaded sessions: {db.number_of_sessions_downloaded}")
EOF
    
    python${PYTHON_VERSION} /tmp/download_svd.py
    
    log_success "Dataset prepared"
}

################################################################################
# Model Training
################################################################################

train_model() {
    log_info "Starting optimized model training..."
    
    cd "$INSTALL_DIR"
    export SBVOICEDB_DIR="$DATA_DIR/sbvoicedb"
    
    # Run training with optimization
    log_info "Phase 1: Baseline training..."
    python${PYTHON_VERSION} scripts/train.py \
        --backend ensemble \
        2>&1 | tee "$LOG_DIR/training_baseline.log"
    
    log_info "Phase 2: Training with augmentation..."
    python${PYTHON_VERSION} scripts/train.py \
        --backend ensemble \
        --augment \
        2>&1 | tee "$LOG_DIR/training_augmented.log"
    
    log_info "Phase 3: Hyperparameter optimization..."
    python${PYTHON_VERSION} main.py optimize \
        --iterations 30 \
        2>&1 | tee "$LOG_DIR/training_optimization.log"
    
    log_info "Phase 4: Model calibration..."
    python${PYTHON_VERSION} main.py calibrate \
        --method isotonic \
        2>&1 | tee "$LOG_DIR/training_calibration.log"
    
    log_info "Phase 5: Cross-validation..."
    python${PYTHON_VERSION} main.py self-test --type cv \
        2>&1 | tee "$LOG_DIR/training_cv.log"
    
    log_info "Phase 6: Final evaluation..."
    python${PYTHON_VERSION} main.py self-test --type full \
        2>&1 | tee "$LOG_DIR/training_evaluation.log"
    
    log_info "Phase 7: Generating report..."
    python${PYTHON_VERSION} main.py report \
        --output-dir "$LOG_DIR/reports" \
        2>&1 | tee "$LOG_DIR/training_report.log"
    
    # Copy trained models to production directory
    cp -r models/* "$MODELS_DIR/"
    
    log_success "Model training completed!"
}

################################################################################
# Production Server Setup
################################################################################

setup_production_server() {
    log_info "Setting up production server..."
    
    cd "$INSTALL_DIR"
    
    # Create environment file
    cat > "$INSTALL_DIR/.env" << EOF
TBVOICE_DATABASE_URL=sqlite:///$INSTALL_DIR/server/tbvoice.db
TBVOICE_MODEL_BACKEND=ensemble
TBVOICE_MODEL_MODE=binary
TBVOICE_DEBUG=false
TBVOICE_WORKERS=4
SBVOICEDB_DIR=$DATA_DIR/sbvoicedb
EOF
    
    # Create systemd service
    cat > /etc/systemd/system/voice-disorder-api.service << EOF
[Unit]
Description=Voice Disorder Detection API Server
After=network.target

[Service]
Type=notify
User=root
WorkingDirectory=$INSTALL_DIR
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONPATH=$INSTALL_DIR"
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=/usr/local/bin/gunicorn server.app.main:app \\
    --workers 4 \\
    --worker-class uvicorn.workers.UvicornWorker \\
    --bind 0.0.0.0:8000 \\
    --access-logfile $LOG_DIR/access.log \\
    --error-logfile $LOG_DIR/error.log \\
    --log-level info
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Configure nginx
    cat > /etc/nginx/sites-available/voice-disorder << 'EOF'
server {
    listen 80;
    server_name _;
    
    client_max_body_size 50M;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    location /health {
        proxy_pass http://127.0.0.1:8000/api/v1/health;
        access_log off;
    }
}
EOF
    
    ln -sf /etc/nginx/sites-available/voice-disorder /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    # Test nginx configuration
    nginx -t
    
    # Enable and start services
    systemctl daemon-reload
    systemctl enable voice-disorder-api
    systemctl restart nginx
    systemctl start voice-disorder-api
    
    log_success "Production server configured"
}

################################################################################
# Health Check
################################################################################

health_check() {
    log_info "Running health checks..."
    
    sleep 5  # Wait for service to start
    
    # Check if service is running
    if systemctl is-active --quiet voice-disorder-api; then
        log_success "API service is running"
    else
        log_error "API service failed to start"
        systemctl status voice-disorder-api
        exit 1
    fi
    
    # Check API endpoint
    if curl -f -s http://localhost/api/v1/health > /dev/null; then
        log_success "API health check passed"
    else
        log_warning "API health check failed (may need time to initialize)"
    fi
    
    # Display service status
    systemctl status voice-disorder-api --no-pager
}

################################################################################
# Main Execution
################################################################################

main() {
    echo "=================================="
    echo "Voice Disorder Detection Deployment"
    echo "=================================="
    echo ""
    
    check_system
    cleanup_old_installations
    install_system_dependencies
    install_python
    setup_directories
    clone_or_update_repository
    install_python_dependencies
    download_dataset
    train_model
    setup_production_server
    health_check
    
    echo ""
    echo "=================================="
    log_success "Deployment completed successfully!"
    echo "=================================="
    echo ""
    echo "Service Information:"
    echo "  - API Endpoint: http://$(hostname -I | awk '{print $1}')"
    echo "  - Health Check: http://$(hostname -I | awk '{print $1}')/api/v1/health"
    echo "  - Logs: $LOG_DIR"
    echo "  - Models: $MODELS_DIR"
    echo ""
    echo "Useful Commands:"
    echo "  - View logs: tail -f $LOG_DIR/error.log"
    echo "  - Restart service: systemctl restart voice-disorder-api"
    echo "  - Check status: systemctl status voice-disorder-api"
    echo ""
}

# Run main function
main "$@"
