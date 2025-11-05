#!/bin/bash
# Automated V15 3-Stage Pipeline Runner
#
# This script monitors Stage 1 completion and automatically launches Stages 2 and 3

set -e  # Exit on error

LIGHTCURVES="../../data/unified/lightcurves_unified_v2_min3.csv"
STAGE1_DIR="results/v15_stage1_production"
STAGE2_DIR="results/v15_stage2_mcmc"
STAGE3_DIR="results/v15_stage3_hubble"

echo "================================================================================"
echo "V15 AUTOMATED PIPELINE MONITOR"
echo "================================================================================"
echo ""
echo "Monitoring Stage 1: $STAGE1_DIR"
echo ""

# Wait for Stage 1 to complete
while true; do
    completed=$(ls -d $STAGE1_DIR/*/ 2>/dev/null | wc -l)
    total=5468
    pct=$((completed * 100 / total))

    echo -ne "\rStage 1 Progress: $completed/$total ($pct%)  "

    # Check if process is still running
    if ! pgrep -f "stage1_optimize.py" > /dev/null; then
        echo ""
        echo "Stage 1 process completed!"
        break
    fi

    sleep 30
done

echo ""
echo "================================================================================"
echo "STAGE 1 COMPLETE - Analyzing Results"
echo "================================================================================"
echo ""

# Create summary script
python3 << 'EOF'
import json
from pathlib import Path
import numpy as np

stage1_dir = Path("results/v15_stage1_production")
results = []

for result_dir in stage1_dir.iterdir():
    if not result_dir.is_dir():
        continue

    metrics_file = result_dir / "metrics.json"
    if not metrics_file.exists():
        continue

    try:
        with open(metrics_file) as f:
            metrics = json.load(f)
        results.append(metrics)
    except:
        continue

print(f"Total SNe: {len(results)}")

# Filter quality
quality = [r for r in results if r.get('chi2_per_obs', 1e10) < 100]
print(f"Quality SNe (χ²/obs < 100): {len(quality)} ({len(quality)/len(results)*100:.1f}%)")

# Chi2 distribution
chi2_vals = [r['chi2_per_obs'] for r in results if 'chi2_per_obs' in r and r['chi2_per_obs'] < 1e6]
print(f"Median χ²/obs: {np.median(chi2_vals):.2f}")
print(f"Mean χ²/obs: {np.mean(chi2_vals):.2f}")

# Alpha variation
alphas = [r['persn_best'][3] for r in quality if 'persn_best' in r]
redshifts = [r['z'] for r in quality if 'z' in r]
corr = np.corrcoef(redshifts[:len(alphas)], alphas)[0, 1]
print(f"Alpha-z correlation: {corr:.3f}")

print("")
if len(quality) >= 50:
    print("✅ READY FOR STAGE 2!")
else:
    print("⚠️  WARNING: Only {len(quality)} quality SNe, need at least 50")
    exit(1)

EOF

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 1 results insufficient for Stage 2"
    exit 1
fi

echo ""
echo "================================================================================"
echo "LAUNCHING STAGE 2: MCMC (OPTIMIZED)"
echo "================================================================================"
echo ""

python stage2_mcmc_optimized.py \
    --stage1-results $STAGE1_DIR \
    --lightcurves $LIGHTCURVES \
    --out $STAGE2_DIR \
    --nwalkers 32 \
    --nsteps 5000 \
    --nburn 1000 2>&1 | tee stage2_mcmc.log

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 2 failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "LAUNCHING STAGE 3: HUBBLE DIAGRAM (OPTIMIZED)"
echo "================================================================================"
echo ""

python stage3_hubble_optimized.py \
    --stage1-results $STAGE1_DIR \
    --stage2-results $STAGE2_DIR \
    --out $STAGE3_DIR \
    --quality-cut 50 \
    --ncores 16 2>&1 | tee stage3_hubble.log

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 3 failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  Stage 1: $STAGE1_DIR"
echo "  Stage 2: $STAGE2_DIR"
echo "  Stage 3: $STAGE3_DIR"
echo ""
echo "Hubble diagram: $STAGE3_DIR/hubble_diagram.png"
echo ""
