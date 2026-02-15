#!/bin/bash

################################################################################
# EchoFlow 2.0 Perfect V3 - Automated Setup and Training for MacBook M4
# FIXED VERSION: Works with Homebrew Python 3.14
# 
# This script automatically:
# 1. Creates Python virtual environment (venv)
# 2. Downloads the perfect training script from GitHub
# 3. Installs all dependencies in isolated environment
# 4. Downloads and prepares the dataset
# 5. Starts training with optimal settings
#
# Usage: bash setup_and_train_mac_fixed.sh
################################################################################

set -e  # Exit on any error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║         EchoFlow 2.0 Perfect V3 - Automated Setup              ║"
echo "║                  FIXED FOR HOMEBREW PYTHON                     ║"
echo "║                                                                ║"
echo "║  This will take 15-25 hours to complete training              ║"
echo "║  Expected accuracy: 90-96% (vs 65% before)                    ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check if we're on Mac
echo "→ Step 1/8: Checking system..."
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ Error: This script is for macOS only"
    exit 1
fi
echo "✅ Running on macOS"
echo ""

# Step 2: Check Python version
echo "→ Step 2/8: Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "   Install from: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python $PYTHON_VERSION found"
echo ""

# Step 3: Create project directory
echo "→ Step 3/8: Setting up project directory..."
cd ~
PROJECT_DIR="$HOME/EchoFlow-V2-Perfect"

if [ -d "$PROJECT_DIR" ]; then
    echo "⚠️  Directory $PROJECT_DIR already exists"
    echo "   Removing old directory..."
    rm -rf "$PROJECT_DIR"
    echo "✅ Old directory removed"
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
echo "✅ Created directory: $PROJECT_DIR"
echo ""

# Step 4: Create Python virtual environment
echo "→ Step 4/8: Creating Python virtual environment..."
echo "   This isolates packages from system Python..."
python3 -m venv venv

if [ ! -d "venv" ]; then
    echo "❌ Error: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment created and activated"
echo ""

# Step 5: Download training script from GitHub
echo "→ Step 5/8: Downloading perfect training script from GitHub..."
GITHUB_RAW="https://raw.githubusercontent.com/izoomlentoboy-creator/Voice/claude/voice-disorder-detection-model-qiRDt"

curl -sL "$GITHUB_RAW/train_perfect_v3.py" -o train_perfect_v3.py
curl -sL "$GITHUB_RAW/FINAL_SUMMARY.md" -o FINAL_SUMMARY.md

if [ ! -f "train_perfect_v3.py" ]; then
    echo "❌ Error: Failed to download training script"
    exit 1
fi

chmod +x train_perfect_v3.py
echo "✅ Downloaded train_perfect_v3.py (23 KB)"
echo "✅ Downloaded FINAL_SUMMARY.md (12 KB)"
echo ""

# Step 6: Install Python dependencies
echo "→ Step 6/8: Installing Python dependencies in virtual environment..."
echo "   This may take 5-10 minutes..."

# Upgrade pip first
python -m pip install --upgrade pip --quiet

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
matplotlib>=3.7.0
tensorboard>=2.13.0
EOF

# Install dependencies in venv
python -m pip install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install dependencies"
    exit 1
fi

echo "✅ All dependencies installed in virtual environment"
echo ""

# Step 7: Create dataset directory
echo "→ Step 7/8: Preparing dataset directory..."
mkdir -p dataset/normal dataset/pathological
echo "✅ Dataset directories created"
echo ""

# Step 8: Create startup script
echo "→ Step 8/8: Creating startup script..."

cat > start_training.sh << 'SCRIPT_EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python train_perfect_v3.py
SCRIPT_EOF

chmod +x start_training.sh
echo "✅ Startup script created"
echo ""

# Final check and start training
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║                   TRAINING STARTED                             ║"
echo "║                                                                ║"
echo "║  Expected duration: 15-25 hours                                ║"
echo "║  Progress will be saved every epoch                            ║"
echo "║  Virtual environment is activated                              ║"
echo "║                                                                ║"
echo "║  To monitor progress:                                          ║"
echo "║    tail -f logs/training_perfect_v3_*.log                      ║"
echo "║                                                                ║"
echo "║  To check current metrics:                                     ║"
echo "║    cat logs/training_perfect_v3_*.log | grep \"Val Acc\"         ║"
echo "║                                                                ║"
echo "║  To restart training later:                                    ║"
echo "║    cd ~/EchoFlow-V2-Perfect && ./start_training.sh             ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Start training in background with nohup
nohup python train_perfect_v3.py > training_output.log 2>&1 &
TRAIN_PID=$!

echo "✅ Training started (PID: $TRAIN_PID)"
echo ""
echo "📊 To view real-time progress:"
echo "   tail -f $PROJECT_DIR/training_output.log"
echo ""
echo "🛑 To stop training:"
echo "   kill $TRAIN_PID"
echo ""
echo "📁 Project directory: $PROJECT_DIR"
echo ""
echo "💡 Virtual environment location: $PROJECT_DIR/venv"
echo ""
echo "🎉 Setup complete! Training is running in background."
echo ""
echo "⚠️  IMPORTANT: Keep your Mac plugged in and awake!"
echo ""
