#!/bin/bash
################################################################################
# Voice Disorder Detection - Simplified Training & Deployment
# Version: 3.0.0 (Final Working Version)
################################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Voice Disorder Detection${NC}"
echo -e "${BLUE}Training & Deployment Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration
INSTALL_DIR="/opt/Voice"
DATA_DIR="/var/lib/voice-disorder/data"
PYTHON="python3.11"

cd "$INSTALL_DIR" || exit 1

echo -e "${GREEN}[1/7]${NC} Installing dependencies..."
$PYTHON -m pip install --quiet --no-warn-script-location \
    scikit-learn numpy scipy joblib \
    librosa soundfile audioread \
    fastapi uvicorn pydantic sqlalchemy \
    sbvoicedb audiomentations \
    gunicorn rich 2>&1 | grep -v "WARNING\|ERROR: Cannot uninstall\|Attempting uninstall" || true

echo -e "${GREEN}[2/7]${NC} Setting up data directory..."
mkdir -p "$DATA_DIR"
export SBVOICEDB_DIR="$DATA_DIR/sbvoicedb"

echo -e "${GREEN}[3/7]${NC} Downloading dataset (15-30 min)..."
$PYTHON << 'PYEOF'
from sbvoicedb import SbVoiceDb
import os
db_dir = os.environ.get('SBVOICEDB_DIR')
print(f"Database location: {db_dir}")
db = SbVoiceDb(dbdir=db_dir, download_mode='lazy')
print(f"Total sessions: {db.number_of_all_sessions}")
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
                            print(f"Progress: {count} recordings")
                except:
                    pass
    except:
        pass
    if count >= 1000:
        break
print(f"Dataset ready: {count} recordings downloaded")
PYEOF

if [ $? -ne 0 ]; then
    echo -e "${RED}Dataset download failed${NC}"
    exit 1
fi

echo -e "${GREEN}[4/7]${NC} Training model (30-60 min)..."
export SBVOICEDB_DIR="$DATA_DIR/sbvoicedb"

$PYTHON scripts/train_optimized.py --backend ensemble --all 2>&1 | tee training.log

if [ $? -ne 0 ]; then
    echo -e "${RED}Training failed${NC}"
    exit 1
fi

echo -e "${GREEN}[5/7]${NC} Verifying model..."
if [ ! -d "models" ] || [ -z "$(ls -A models/*.pkl 2>/dev/null)" ]; then
    echo -e "${RED}Model files not found${NC}"
    exit 1
fi

echo -e "${GREEN}[6/7]${NC} Creating configuration..."
cat > .env << EOF
TBVOICE_DATABASE_URL=sqlite:///$INSTALL_DIR/server/tbvoice.db
TBVOICE_MODEL_BACKEND=ensemble
TBVOICE_MODEL_MODE=binary
TBVOICE_DEBUG=false
SBVOICEDB_DIR=$DATA_DIR/sbvoicedb
EOF

echo -e "${GREEN}[7/7]${NC} Starting server..."
nohup $PYTHON -m gunicorn server.app.main:app \
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
echo "API endpoint: http://85.239.48.254:8000"
echo "Health check: curl http://localhost:8000/api/v1/health"
echo "Training log: training.log"
echo "Model files: models/"
echo ""
echo "To view logs:"
echo "  tail -f /var/log/voice-api-error.log"
echo ""
echo "To stop server:"
echo "  pkill -f gunicorn"
echo ""
