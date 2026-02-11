#!/bin/bash
# Train the voice disorder detection model inside the Docker container
# Usage: docker exec tbvoice_tbvoice_1 bash /app/server/train_model.sh

set -e

echo "=== TBVoice Model Training ==="
echo "Starting at $(date)"

cd /app

# Step 1: Train the ensemble model
echo ""
echo "[1/4] Training ensemble model..."
python main.py train --backend ensemble

# Step 2: Train logistic regression baseline
echo ""
echo "[2/4] Training logistic regression baseline..."
python main.py train --backend logreg

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
