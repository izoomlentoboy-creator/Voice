#!/bin/bash
################################################################################
# Quick Training Script - Optimized for 92%+ Accuracy
# This script runs the complete training pipeline with all optimizations
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Voice Disorder Detection - Quick Train${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration
PYTHON="python3.11"
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create log directory
mkdir -p "$LOG_DIR"

echo -e "${GREEN}[1/5]${NC} Training baseline ensemble model..."
$PYTHON scripts/train.py \
    --backend ensemble \
    2>&1 | tee "$LOG_DIR/train_baseline_$TIMESTAMP.log"

echo ""
echo -e "${GREEN}[2/5]${NC} Training with data augmentation..."
$PYTHON scripts/train_optimized.py \
    --backend ensemble \
    --augment \
    --calibrate \
    2>&1 | tee "$LOG_DIR/train_augmented_$TIMESTAMP.log"

echo ""
echo -e "${GREEN}[3/5]${NC} Running hyperparameter optimization..."
$PYTHON main.py optimize \
    --iterations 30 \
    2>&1 | tee "$LOG_DIR/train_optimize_$TIMESTAMP.log"

echo ""
echo -e "${GREEN}[4/5]${NC} Performing cross-validation..."
$PYTHON main.py self-test --type cv \
    2>&1 | tee "$LOG_DIR/train_cv_$TIMESTAMP.log"

echo ""
echo -e "${GREEN}[5/5]${NC} Generating comprehensive report..."
$PYTHON main.py report \
    --output-dir "$LOG_DIR/reports_$TIMESTAMP" \
    2>&1 | tee "$LOG_DIR/train_report_$TIMESTAMP.log"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Training completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Logs saved to: $LOG_DIR"
echo "Models saved to: ./models/"
echo ""
echo "Next steps:"
echo "  1. Review training report: cat $LOG_DIR/train_report_$TIMESTAMP.log"
echo "  2. Test model: python3.11 main.py predict --audio test.wav"
echo "  3. Start server: python3.11 server/app/main.py"
echo ""
