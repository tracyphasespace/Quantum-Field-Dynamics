#!/bin/bash
# V14 Stage-1 Parallel Runner
# Splits 5468 SNe across 8 parallel workers
# Each worker processes ~683 SNe

set -e

LIGHTCURVES="${1:-../../data/unified/lightcurves_unified_v2_min3.csv}"
OUTDIR="${2:-../../results/v14_production/stage1}"
GLOBAL_PARAMS="${3:-70,0.01,30}"
WORKERS="${4:-3}"  # Conservative: 3 workers for 7.6GB RAM (2.5GB per worker)

echo "=== V14 Stage-1 Parallel Runner ==="
echo "Lightcurves: $LIGHTCURVES"
echo "Output dir: $OUTDIR"
echo "Global params: $GLOBAL_PARAMS"
echo "Workers: $WORKERS"
echo ""

# Calculate total SNe
TOTAL_SNE=$(python3 -c "
import pandas as pd
df = pd.read_csv('$LIGHTCURVES')
print(len(df['snid'].unique()))
")

echo "Total SNe: $TOTAL_SNE"
CHUNK_SIZE=$((TOTAL_SNE / WORKERS))
echo "Chunk size: ~$CHUNK_SIZE SNe per worker"
echo ""

# Create output directory
mkdir -p "$OUTDIR"

# Set environment variables
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=1

# Launch parallel workers
PIDS=()
for i in $(seq 0 $((WORKERS - 1))); do
    START=$((i * CHUNK_SIZE))

    # Last worker gets any remainder
    if [ $i -eq $((WORKERS - 1)) ]; then
        END=$TOTAL_SNE
    else
        END=$(((i + 1) * CHUNK_SIZE))
    fi

    RANGE="$START:$END"
    LOGFILE="$OUTDIR/_worker_${i}.log"

    echo "[Worker $i] Processing SNe $RANGE → $LOGFILE"

    python3 stage1_optimize.py \
        --lightcurves "$LIGHTCURVES" \
        --sn-list "$RANGE" \
        --out "$OUTDIR" \
        --global "$GLOBAL_PARAMS" \
        --tol 1e-6 \
        --max-iters 200 \
        > "$LOGFILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "Launched $WORKERS workers (PIDs: ${PIDS[@]})"
echo ""
echo "Monitor progress:"
echo "  tail -f $OUTDIR/_worker_*.log"
echo "  watch -n 10 'ls $OUTDIR | grep -v _worker | wc -l'"
echo ""
echo "Waiting for all workers to complete..."

# Wait for all workers
START_TIME=$(date +%s)
for PID in "${PIDS[@]}"; do
    wait $PID
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: Worker PID $PID exited with code $EXIT_CODE"
    fi
done
END_TIME=$(date +%s)

ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "=== All workers completed in ${ELAPSED}s (~$((ELAPSED / 60)) minutes) ==="
echo ""

# Summary
SUCCESS=$(find "$OUTDIR" -name "status.txt" -exec grep -l "ok" {} \; | wc -l)
TOTAL_DIRS=$(find "$OUTDIR" -mindepth 1 -maxdepth 1 -type d ! -name "_*" | wc -l)

echo "Results:"
echo "  Success: $SUCCESS / $TOTAL_DIRS SNe"
echo "  Output: $OUTDIR"
echo ""
echo "Next: Run Stage-2 MCMC"
echo "  python3 stage2_fit.py \\"
echo "    --stage1 $OUTDIR \\"
echo "    --lightcurves $LIGHTCURVES \\"
echo "    --mode freeze --walkers 32 --steps 3000 --burn 1000 \\"
echo "    --init $GLOBAL_PARAMS --out ${OUTDIR}/../stage2"
