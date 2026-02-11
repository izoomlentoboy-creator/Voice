#!/bin/bash
# Train the voice disorder detection model inside the Docker container
# Usage: docker exec tbvoice bash /app/server/train_model.sh

set -e

echo "=== TBVoice Model Training ==="
echo "Starting at $(date)"

cd /app

# Step 0: Download the sbvoicedb audio data
echo ""
echo "[0/4] Downloading voice database audio files..."
python -c "
from sbvoicedb import SbVoiceDb
db = SbVoiceDb(download_mode='immediate')
print(f'Total sessions: {db.number_of_all_sessions}')
print(f'Downloaded: {db.number_of_sessions_downloaded}')
print('Database ready.')
"

# Step 1: Train the ensemble model
echo ""
echo "[1/4] Training ensemble model..."
python main.py --backend ensemble train

# Step 2: Train logistic regression baseline
echo ""
echo "[2/4] Training logistic regression baseline..."
python main.py --backend logreg train

# Step 3: Fit domain monitor for OOD detection
echo ""
echo "[3/4] Fitting domain monitor..."
python main.py fit-monitor

# Step 4: Generate healthy reference stats for interpreter
echo ""
echo "[4/4] Generating healthy reference stats..."
python /app/server/generate_ref_stats.py

echo ""
echo "=== Training complete at $(date) ==="
echo "Models saved to /app/models/"
ls -la /app/models/
