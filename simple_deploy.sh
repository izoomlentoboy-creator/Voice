#!/bin/bash
################################################################################
# Voice Disorder Detection - Simple Deployment
# Использует существующие скрипты проекта
################################################################################

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "========================================"
echo "Voice Disorder Detection - Simple Deploy"
echo "========================================"
echo ""

INSTALL_DIR="/opt/Voice"
DATA_DIR="/var/lib/voice-disorder/data/sbvoicedb"

cd "$INSTALL_DIR" || exit 1

# Step 1: Install dependencies
echo -e "${GREEN}[1/5]${NC} Installing dependencies..."
python3.11 -m pip install --quiet \
    scikit-learn numpy scipy joblib \
    librosa soundfile audioread \
    fastapi uvicorn pydantic sqlalchemy \
    sbvoicedb audiomentations \
    gunicorn rich 2>&1 | grep -v "WARNING\|ERROR: Cannot uninstall" || true

# Step 2: Setup environment
echo -e "${GREEN}[2/5]${NC} Setting up environment..."
mkdir -p "$DATA_DIR"
export SBVOICEDB_DIR="$DATA_DIR"

# Step 3: Train model using existing script
echo -e "${GREEN}[3/5]${NC} Training model (this will take 30-60 minutes)..."
echo "Using existing training script with ensemble backend..."

python3.11 scripts/train.py \
    --backend ensemble \
    --augment \
    --dbdir "$DATA_DIR" 2>&1 | tee training.log

if [ $? -ne 0 ]; then
    echo -e "${RED}Training failed. Check training.log${NC}"
    exit 1
fi

# Step 4: Verify model
echo -e "${GREEN}[4/5]${NC} Verifying model files..."
if [ -d "models" ] && [ -n "$(ls -A models/*.pkl 2>/dev/null)" ]; then
    echo "Model files found:"
    ls -lh models/*.pkl
else
    echo -e "${RED}No model files found!${NC}"
    exit 1
fi

# Step 5: Start server
echo -e "${GREEN}[5/5]${NC} Starting server..."

cat > .env << EOF
TBVOICE_DATABASE_URL=sqlite:///$INSTALL_DIR/server/tbvoice.db
TBVOICE_MODEL_BACKEND=ensemble
TBVOICE_MODEL_MODE=binary
TBVOICE_DEBUG=false
SBVOICEDB_DIR=$DATA_DIR
EOF

nohup python3.11 -m gunicorn server.app.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile /var/log/voice-api-access.log \
    --error-logfile /var/log/voice-api-error.log \
    > /dev/null 2>&1 &

sleep 5

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}DEPLOYMENT COMPLETED!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "API: http://85.239.48.254:8000"
echo "Health check: curl http://localhost:8000/api/v1/health"
echo "Training log: training.log"
echo ""
echo "To view server logs:"
echo "  tail -f /var/log/voice-api-error.log"
echo ""
echo "To stop server:"
echo "  pkill -f gunicorn"
echo ""
