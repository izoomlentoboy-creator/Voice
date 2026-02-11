#!/bin/bash
# Train the voice disorder detection model inside the Docker container
# Usage: docker exec tbvoice bash /app/server/train_model.sh

set -e

echo "=== TBVoice Model Training ==="
echo "Starting at $(date)"

cd /app

# Step 1: Train the ensemble model
echo ""
echo "[1/4] Training ensemble model..."
python -m voice_disorder_detection.main train --backend ensemble --mode binary

# Step 2: Train logistic regression baseline
echo ""
echo "[2/4] Training logistic regression baseline..."
python -m voice_disorder_detection.main train --backend logreg --mode binary

# Step 3: Fit domain monitor for OOD detection
echo ""
echo "[3/4] Fitting domain monitor..."
python -m voice_disorder_detection.main fit-monitor

# Step 4: Generate healthy reference stats for interpreter
echo ""
echo "[4/4] Generating healthy reference stats..."
python /app/server/generate_ref_stats.py

echo ""
echo "=== Training complete at $(date) ==="
echo "Models saved to /app/models/"
ls -la /app/models/
