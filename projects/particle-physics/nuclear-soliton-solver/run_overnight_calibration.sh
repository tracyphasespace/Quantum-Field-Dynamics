#!/bin/bash
#
# Overnight Calibration - 4 Workers, 8 Hours
# ==========================================
#
# Runs parallel batch optimization for heavy isotope regional calibration.
#
# Usage:
#   ./run_overnight_calibration.sh
#
# Monitoring:
#   tail -f results/batch_optimization/checkpoint.json
#

set -e

# Configuration
WORKERS=3
HOURS=8.0
CONFIG="experiments/nuclear_heavy_region.runspec.json"
OUTPUT_DIR="results/batch_optimization_$(date +%Y%m%d_%H%M%S)"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Overnight Calibration - Heavy Region Optimization            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Configuration:"
echo "  Workers: $WORKERS"
echo "  Runtime: $HOURS hours"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo ""

# Check that config exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Configuration file not found: $CONFIG"
    exit 1
fi

# Check that we have enough memory
TOTAL_MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
REQUIRED_MEM_GB=$((WORKERS * 4))

echo "Memory check:"
echo "  Available: ${TOTAL_MEM_GB} GB"
echo "  Required: ${REQUIRED_MEM_GB} GB (${WORKERS} workers × 4 GB)"
echo ""

if [ "$TOTAL_MEM_GB" -lt "$REQUIRED_MEM_GB" ]; then
    echo "⚠️  Warning: May not have enough memory for all workers"
    echo "   Consider reducing --workers to $((TOTAL_MEM_GB / 4))"
    echo ""
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start optimization with nice priority (lower CPU priority to avoid system overload)
echo "🚀 Starting optimization..."
echo "   Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "   Expected completion: $(date -d "+$HOURS hours" '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "📊 Monitor progress:"
echo "   Checkpoint: $OUTPUT_DIR/checkpoint.json"
echo "   Log: $OUTPUT_DIR/optimization.log"
echo ""
echo "⏸️  To stop gracefully: Ctrl+C (will save checkpoint)"
echo ""

# Run optimization, logging to file
nice -n 10 python3 src/batch_optimize.py \
    --config "$CONFIG" \
    --workers "$WORKERS" \
    --hours "$HOURS" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/optimization.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Optimization Finished                                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Exit code: $EXIT_CODE"
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results:"
echo "  Directory: $OUTPUT_DIR"
echo "  Best parameters: $OUTPUT_DIR/best_parameters.json"
echo "  Checkpoint: $OUTPUT_DIR/checkpoint.json"
echo "  Log: $OUTPUT_DIR/optimization.log"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Optimization completed successfully!"

    # Show best result if available
    if [ -f "$OUTPUT_DIR/best_parameters.json" ]; then
        echo ""
        echo "🏆 Best Result:"
        python3 -c "
import json
with open('$OUTPUT_DIR/best_parameters.json') as f:
    data = json.load(f)
metrics = data['metrics']
print(f\"   Score: {data['score']:.2f}\")
print(f\"   Mean Error: {metrics['mean_error_pct']:.2f}%\")
print(f\"   Std Error: {metrics['std_error_pct']:.2f}%\")
print(f\"   Max Error: {metrics['max_abs_error_pct']:.2f}%\")
print(f\"   Virial: {metrics['mean_virial']:.3f}\")
"
    fi
else
    echo "⚠️  Optimization exited with errors (see log for details)"
fi

echo ""
