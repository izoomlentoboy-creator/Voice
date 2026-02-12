#!/bin/bash
# Server diagnostic script for Voice Disorder Detection
# Run on the server: bash scripts/check_server.sh

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }

echo "============================================"
echo " Voice Disorder Detection — Server Check"
echo "============================================"
echo ""

# --- System ---
echo "=== System ==="
echo "OS:       $(uname -sr)"
echo "RAM:      $(free -h | awk '/Mem:/{print $2}') total, $(free -h | awk '/Mem:/{print $7}') available"
echo "Disk:     $(df -h / | awk 'NR==2{print $4}') free on /"
echo "CPU:      $(nproc) cores"
echo ""

# --- Docker ---
echo "=== Docker ==="
if command -v docker &>/dev/null; then
    ok "Docker installed: $(docker --version 2>/dev/null | head -1)"
    if command -v docker &>/dev/null && docker compose version &>/dev/null 2>&1; then
        ok "Docker Compose: $(docker compose version 2>/dev/null | head -1)"
    else
        warn "Docker Compose not found (needed for docker compose up)"
    fi
    echo ""
    echo "Running containers:"
    docker ps --format "  {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "  (none or no access)"
    echo ""
    echo "All containers:"
    docker ps -a --format "  {{.Names}}\t{{.Status}}" 2>/dev/null || echo "  (none)"
else
    fail "Docker not installed"
fi
echo ""

# --- Project ---
echo "=== Project ==="
PROJECT_DIR=""
for d in /root/Voice /home/*/Voice /app; do
    if [ -f "$d/main.py" ] && [ -d "$d/voice_disorder_detection" ]; then
        PROJECT_DIR="$d"
        break
    fi
done

if [ -z "$PROJECT_DIR" ]; then
    fail "Project not found in /root/Voice, /home/*/Voice, or /app"
    echo "  Clone it: git clone https://github.com/izoomlentoboy-creator/Voice.git"
else
    ok "Project found: $PROJECT_DIR"
    cd "$PROJECT_DIR"

    # Git
    if [ -d .git ]; then
        BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
        echo "  Branch:  $BRANCH"
        echo "  Last commit: $(git log --oneline -1 2>/dev/null || echo 'unknown')"

        if [ "$BRANCH" != "master" ] && [ "$BRANCH" != "main" ]; then
            warn "Not on master branch (currently: $BRANCH)"
        fi
    fi
    echo ""

    # Python
    echo "=== Python ==="
    if command -v python3 &>/dev/null; then
        ok "Python: $(python3 --version)"
    else
        fail "python3 not found"
    fi

    if python3 -c "import sbvoicedb" 2>/dev/null; then
        ok "sbvoicedb installed"
    else
        warn "sbvoicedb not installed (pip install -r requirements.txt)"
    fi

    if python3 -c "import librosa" 2>/dev/null; then
        ok "librosa installed"
    else
        warn "librosa not installed"
    fi

    if python3 -c "import sklearn" 2>/dev/null; then
        ok "scikit-learn installed"
    else
        warn "scikit-learn not installed"
    fi
    echo ""

    # Models
    echo "=== Models ==="
    if [ -d "$PROJECT_DIR/models" ]; then
        MODEL_COUNT=$(find "$PROJECT_DIR/models" -name "*.joblib" 2>/dev/null | wc -l)
        if [ "$MODEL_COUNT" -gt 0 ]; then
            ok "Found $MODEL_COUNT trained model(s):"
            ls -lh "$PROJECT_DIR/models"/*.joblib 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
        else
            warn "No trained models found in $PROJECT_DIR/models/"
            echo "  Run: python main.py train"
        fi
    else
        fail "models/ directory missing"
    fi
    echo ""

    # Cache
    echo "=== Cache ==="
    CACHE_FILE="$PROJECT_DIR/cache/features_cache.npz"
    if [ -f "$CACHE_FILE" ]; then
        ok "Feature cache exists: $(ls -lh "$CACHE_FILE" | awk '{print $5}')"
    else
        echo "  No feature cache (will be created on first training)"
    fi
    echo ""

    # sbvoicedb data
    echo "=== sbvoicedb Data ==="
    for DB_DIR in /root/.local/share/sbvoicedb "$HOME/.local/share/sbvoicedb"; do
        if [ -d "$DB_DIR" ]; then
            ok "Data dir: $DB_DIR"
            if [ -f "$DB_DIR/sbvoice.db" ]; then
                ok "SQLite DB exists: $(ls -lh "$DB_DIR/sbvoice.db" | awk '{print $5}')"
            fi
            if [ -d "$DB_DIR/data" ]; then
                SESSION_COUNT=$(find "$DB_DIR/data" -maxdepth 1 -type d | wc -l)
                SESSION_COUNT=$((SESSION_COUNT - 1))
                echo "  Downloaded sessions: $SESSION_COUNT"
                if [ "$SESSION_COUNT" -lt 100 ]; then
                    warn "Very few sessions downloaded ($SESSION_COUNT). Training needs ~2000."
                elif [ "$SESSION_COUNT" -gt 1000 ]; then
                    ok "Sufficient data for training ($SESSION_COUNT sessions)"
                fi
            else
                warn "No data/ subdirectory — database not yet downloaded"
            fi
            break
        fi
    done
    echo ""

    # Duplicates check
    echo "=== Duplicate / Unnecessary Files ==="
    # Check for common artifacts
    for f in "$PROJECT_DIR"/*.pyc "$PROJECT_DIR"/__pycache__ "$PROJECT_DIR"/.DS_Store; do
        if [ -e "$f" ]; then
            warn "Artifact found: $f (can be removed)"
        fi
    done

    # Check for stale cache
    if [ -f "$CACHE_FILE" ]; then
        CACHE_AGE_DAYS=$(( ($(date +%s) - $(stat -c %Y "$CACHE_FILE" 2>/dev/null || echo 0)) / 86400 ))
        if [ "$CACHE_AGE_DAYS" -gt 30 ]; then
            warn "Feature cache is $CACHE_AGE_DAYS days old — consider rebuilding with --no-cache"
        fi
    fi

    # Check for docker image bloat
    if command -v docker &>/dev/null; then
        DANGLING=$(docker images -f "dangling=true" -q 2>/dev/null | wc -l)
        if [ "$DANGLING" -gt 0 ]; then
            warn "$DANGLING dangling Docker image(s) — clean with: docker image prune"
        fi
    fi
    echo ""

    # Screen
    echo "=== Screen ==="
    if command -v screen &>/dev/null; then
        ok "screen installed"
        SCREENS=$(screen -ls 2>/dev/null | grep -c "Detached\|Attached" || true)
        if [ "$SCREENS" -gt 0 ]; then
            echo "  Active sessions:"
            screen -ls 2>/dev/null | grep -E "Detached|Attached" | sed 's/^/  /'
        fi
    else
        warn "screen not installed — install: apt install screen"
    fi
fi

echo ""
echo "============================================"
echo " Check complete"
echo "============================================"
