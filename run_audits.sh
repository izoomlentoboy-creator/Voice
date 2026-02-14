#!/bin/bash

# Independent Audit Runner
# Runs multiple audit rounds independently

echo "======================================================================"
echo "ECHOFLOW 2.0 - INDEPENDENT AUDIT RUNNER"
echo "======================================================================"

AUDIT_LOG="audit_results.log"
> "$AUDIT_LOG"  # Clear log

CLEAN_RUNS=0
REQUIRED_CLEAN_RUNS=2
ROUND=1
MAX_ROUNDS=10

while [ $CLEAN_RUNS -lt $REQUIRED_CLEAN_RUNS ] && [ $ROUND -le $MAX_ROUNDS ]; do
    echo ""
    echo "======================================================================"
    echo "AUDIT ROUND $ROUND"
    echo "======================================================================"
    echo "Round $ROUND" >> "$AUDIT_LOG"
    
    ERRORS_FOUND=0
    
    # Test 1: Integration tests
    echo ""
    echo "[1/5] Running integration tests..."
    if python3 test_integration.py 2>&1 | tee -a "$AUDIT_LOG" | grep -q "ALL INTEGRATION TESTS PASSED"; then
        echo "✅ Integration tests passed"
    else
        echo "❌ Integration tests failed"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    fi
    
    # Test 2: Deep audit
    echo ""
    echo "[2/5] Running deep audit..."
    if python3 deep_audit.py 2>&1 | tee -a "$AUDIT_LOG" | grep -q "STATUS: ✅ PERFECT"; then
        echo "✅ Deep audit passed"
    else
        echo "❌ Deep audit found issues"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    fi
    
    # Test 3: Cross-check
    echo ""
    echo "[3/5] Running cross-check..."
    if python3 cross_check.py 2>&1 | tee -a "$AUDIT_LOG" | grep -q "STATUS: ✅ PERFECT MATCH"; then
        echo "✅ Cross-check passed"
    else
        echo "❌ Cross-check found mismatches"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    fi
    
    # Test 4: Python syntax
    echo ""
    echo "[4/5] Checking Python syntax..."
    if python3 -m py_compile models/*.py utils/*.py train.py 2>&1 | tee -a "$AUDIT_LOG"; then
        echo "✅ Python syntax OK"
    else
        echo "❌ Python syntax errors"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    fi
    
    # Test 5: Bash syntax
    echo ""
    echo "[5/5] Checking bash syntax..."
    if bash -n train_ultimate.sh 2>&1 | tee -a "$AUDIT_LOG"; then
        echo "✅ Bash syntax OK"
    else
        echo "❌ Bash syntax errors"
        ERRORS_FOUND=$((ERRORS_FOUND + 1))
    fi
    
    # Summary
    echo ""
    echo "======================================================================"
    echo "ROUND $ROUND SUMMARY"
    echo "======================================================================"
    
    if [ $ERRORS_FOUND -eq 0 ]; then
        echo "✅ NO ERRORS FOUND"
        CLEAN_RUNS=$((CLEAN_RUNS + 1))
        echo "Clean runs: $CLEAN_RUNS/$REQUIRED_CLEAN_RUNS"
    else
        echo "❌ ERRORS FOUND: $ERRORS_FOUND"
        CLEAN_RUNS=0
        echo "Clean runs reset to 0"
    fi
    
    echo "Round $ROUND: $ERRORS_FOUND errors" >> "$AUDIT_LOG"
    echo "" >> "$AUDIT_LOG"
    
    ROUND=$((ROUND + 1))
done

echo ""
echo "======================================================================"
echo "FINAL RESULT"
echo "======================================================================"

if [ $CLEAN_RUNS -ge $REQUIRED_CLEAN_RUNS ]; then
    echo "✅ SUCCESS: $REQUIRED_CLEAN_RUNS consecutive clean runs achieved!"
    echo "Total rounds: $((ROUND - 1))"
    exit 0
else
    echo "❌ FAILED: Could not achieve $REQUIRED_CLEAN_RUNS consecutive clean runs in $MAX_ROUNDS rounds"
    echo "Total rounds: $((ROUND - 1))"
    exit 1
fi
