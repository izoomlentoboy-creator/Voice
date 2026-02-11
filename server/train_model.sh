#!/bin/bash
# Train the voice disorder detection model inside the Docker container
# Memory-optimized for servers with 4 GB RAM
# Usage: docker exec tbvoice bash /app/server/train_model.sh

set -e

echo "=== TBVoice Model Training (memory-optimized) ==="
echo "Starting at $(date)"

cd /app

# Step 1: Train the ensemble model (normal-pitch vowels only, gc every 50 sessions)
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
