#!/bin/bash
# Monitor Stage 1 pipeline progress

LOG="results/v15_production/stage1.log"

clear
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  QFD V15 Stage 1 Pipeline Monitor"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if process is running
if ps aux | grep -v grep | grep "stage1_optimize.py" > /dev/null; then
    PID=$(ps aux | grep -v grep | grep "stage1_optimize.py" | awk '{print $2}')
    RUNTIME=$(ps -p $PID -o etime= | tr -d ' ')
    CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
    MEM=$(ps -p $PID -o rss= | awk '{print int($1/1024)}')
    echo "✓ Stage 1 is RUNNING"
    echo "  PID: $PID"
    echo "  Runtime: $RUNTIME"
    echo "  CPU: ${CPU}%"
    echo "  Memory: ${MEM}MB"
else
    echo "✗ Stage 1 is NOT running"
    echo ""
    echo "To start: nohup python src/stage1_optimize.py \\"
    echo "    --lightcurves data/lightcurves_unified_v2_min3.csv \\"
    echo "    --out results/v15_production/stage1 \\"
    echo "    --global \"70.0,0.01,30.0\" \\"
    echo "    --tol 1e-4 --max-iters 500 --verbose \\"
    echo "    > results/v15_production/stage1.log 2>&1 &"
    exit 1
fi

echo ""
echo "────────────────────────────────────────────────────────"
echo ""

if [ ! -f "$LOG" ]; then
    echo "⚠ Log file not found: $LOG"
    exit 1
fi

# Progress
TOTAL=5468
CURRENT=$(grep -c "Optimizing SNID" "$LOG" 2>/dev/null || echo "0")
PCT=$((CURRENT * 100 / TOTAL))
echo "Progress: $CURRENT / $TOTAL SNe ($PCT%)"

# Success rate
OK_COUNT=$(grep -c "✓ OK" "$LOG" 2>/dev/null || echo "0")
FAIL_COUNT=$(grep -c "✗ FAIL" "$LOG" 2>/dev/null || echo "0")
TOTAL_PROCESSED=$((OK_COUNT + FAIL_COUNT))

if [ $TOTAL_PROCESSED -gt 0 ]; then
    SUCCESS_RATE=$((OK_COUNT * 100 / TOTAL_PROCESSED))
    echo "Success rate: $OK_COUNT OK / $FAIL_COUNT FAIL = $SUCCESS_RATE%"
else
    echo "Success rate: (starting...)"
fi

echo ""
echo "────────────────────────────────────────────────────────"
echo ""
echo "Recent activity (last 10):"
echo ""
tail -n 50 "$LOG" | grep -E "✓ OK|✗ FAIL" | tail -10 | while read line; do
    if echo "$line" | grep -q "✓ OK"; then
        echo "  $line"
    else
        echo "  $line"
    fi
done

echo ""
echo "────────────────────────────────────────────────────────"
echo ""
echo "Commands:"
echo "  watch -n 5 ./monitor_pipeline.sh   # Auto-refresh every 5s"
echo "  tail -f $LOG                        # Follow log in real-time"
echo "  kill $PID                           # Stop pipeline"
echo ""
