#!/bin/bash
# Train the voice disorder detection model inside the Docker container
# Memory-optimized for servers with 4 GB RAM
# Usage: docker exec <container> bash /app/server/train_model.sh
#
# Prerequisites: sbvoicedb must be downloaded first. If not yet downloaded, run:
#   docker exec <container> python -c "
#     from sbvoicedb import SbVoiceDb
#     db = SbVoiceDb(download_mode='immediate')
#     print(f'Downloaded: {db.number_of_sessions_downloaded}')
#   "

set -e

cd /app

# --- Pre-flight check: verify sbvoicedb data is available ---
echo "=== TBVoice Model Training ==="
echo "Started at $(date)"
echo ""

echo "[0/3] Checking sbvoicedb data..."
DOWNLOADED=$(python -c "
from sbvoicedb import SbVoiceDb
db = SbVoiceDb()
print(db.number_of_sessions_downloaded)
" 2>/dev/null || echo "0")

if [ "$DOWNLOADED" = "0" ]; then
    echo "ERROR: sbvoicedb audio data not downloaded."
    echo "Run this first:"
    echo "  docker exec <container> python -c \\"
    echo "    from sbvoicedb import SbVoiceDb; SbVoiceDb(download_mode='immediate')\\""
    exit 1
fi
echo "  Found $DOWNLOADED downloaded sessions. OK."

# Step 1: Train the ensemble model
echo ""
echo "[1/3] Training ensemble model..."
python main.py --backend ensemble train

# Step 2: Train logistic regression baseline
echo ""
echo "[2/3] Training logistic regression baseline..."
python main.py --backend logreg train

# Step 3: Generate healthy reference stats for interpreter
echo ""
echo "[3/3] Generating healthy reference stats..."
python /app/server/generate_ref_stats.py

echo ""
echo "=== Training complete at $(date) ==="
echo "Models saved to /app/models/"
ls -la /app/models/
