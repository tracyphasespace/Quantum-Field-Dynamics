#!/bin/bash
#
# Staged Validation Pipeline: 50→500→5000 SNe Progressive Refinement
#
# This script orchestrates the complete validation workflow:
# - Stage 0: Select cleanest 50-100 SNe, train initial physics
# - Stage 1: Measure deviations on 500 new SNe, identify outliers
# - Stage 2: Refine physics on clean subset, apply to all 5000 SNe
#

set -e  # Exit on error

# Configuration
STAGE1_RESULTS="../../results/v15_clean/stage1_fullscale"
LIGHTCURVES="../data/lightcurves_unified_v2_min3.csv"
OUTPUT_BASE="../../results/v15_clean/staged_validation"

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "================================================================================"
echo "STAGED VALIDATION PIPELINE"
echo "================================================================================"
echo ""
echo "Stage 1 results: $STAGE1_RESULTS"
echo "Lightcurves: $LIGHTCURVES"
echo "Output base: $OUTPUT_BASE"
echo ""

# ================================================================================
# STAGE 0: Select cleanest SNe and train initial physics
# ================================================================================

echo "================================================================================"
echo "STAGE 0: Select Cleanest SNe for Initial Physics Training"
echo "================================================================================"
echo ""

STAGE0_OUT="$OUTPUT_BASE/stage0"
mkdir -p "$STAGE0_OUT"

# Select 100 cleanest SNe
echo "Step 1: Selecting 100 cleanest SNe..."
python3 stage0_select_clean.py \
  --stage1-results "$STAGE1_RESULTS" \
  --lightcurves "$LIGHTCURVES" \
  --n-select 100 \
  --out "$STAGE0_OUT/clean_snids.json" \
  2>&1 | tee "$STAGE0_OUT/selection.log"

echo ""
echo "Step 2: Running Stage 2 MCMC on clean subset..."
# Run Stage 2 on clean subset only
python3 stage2_simple.py \
  --stage1-results "$STAGE1_RESULTS" \
  --lightcurves "$LIGHTCURVES" \
  --out "$STAGE0_OUT/stage2_clean" \
  --snid-list "$STAGE0_OUT/clean_snids.json" \
  --nchains 2 \
  --nsamples 2000 \
  --nwarmup 1000 \
  --quality-cut 2000 \
  2>&1 | tee "$STAGE0_OUT/stage2_clean.log"

# Extract physics parameters for Stage 1
echo ""
echo "Step 3: Extracting physics parameters..."
python3 -c "
import json
import numpy as np
from pathlib import Path

# Load Stage 2 results
best_fit = np.load('$STAGE0_OUT/stage2_clean/best_fit.npy')
k_J, eta_prime, xi = best_fit

# Save physics parameters
physics = {
    'k_J': float(k_J),
    'eta_prime': float(eta_prime),
    'xi': float(xi),
    'source': 'stage0_clean_100sne'
}

with open('$STAGE0_OUT/physics_params.json', 'w') as f:
    json.dump(physics, f, indent=2)

print(f'Physics parameters from Stage 0:')
print(f'  k_J = {k_J:.6f}')
print(f'  η′  = {eta_prime:.6f}')
print(f'  ξ   = {xi:.6f}')
"

echo ""
echo "Stage 0 complete! Physics parameters saved to $STAGE0_OUT/physics_params.json"
echo ""

# ================================================================================
# STAGE 1: Measure deviations on new SNe
# ================================================================================

echo "================================================================================"
echo "STAGE 1: Measure Deviations on New SNe"
echo "================================================================================"
echo ""

STAGE1_OUT="$OUTPUT_BASE/stage1"
mkdir -p "$STAGE1_OUT"

echo "Measuring deviations on 500 new SNe with fixed physics..."
python3 stage1_measure_deviations.py \
  --stage1-results "$STAGE1_RESULTS" \
  --lightcurves "$LIGHTCURVES" \
  --physics-params "$STAGE0_OUT/physics_params.json" \
  --exclude-snids "$STAGE0_OUT/clean_snids.json" \
  --n-test 500 \
  --out "$STAGE1_OUT" \
  2>&1 | tee "$STAGE1_OUT/deviations.log"

echo ""
echo "Stage 1 complete! Deviation results saved to $STAGE1_OUT/deviations.json"
echo "Outlier SNIDs saved to $STAGE1_OUT/outlier_snids.json"
echo ""

# ================================================================================
# STAGE 2: Refine physics and apply to full dataset
# ================================================================================

echo "================================================================================"
echo "STAGE 2: Refine Physics on Clean Subset"
echo "================================================================================"
echo ""

STAGE2_OUT="$OUTPUT_BASE/stage2"
mkdir -p "$STAGE2_OUT"

# Combine Stage 0 clean SNe + Stage 1 good SNe (non-outliers)
echo "Step 1: Creating combined clean SNID list..."
python3 -c "
import json

# Load Stage 0 clean SNe
with open('$STAGE0_OUT/clean_snids.json') as f:
    stage0 = json.load(f)
stage0_snids = set(stage0['snids'])

# Load Stage 1 results
with open('$STAGE1_OUT/deviations.json') as f:
    stage1 = json.load(f)

# Load outliers
with open('$STAGE1_OUT/outlier_snids.json') as f:
    outliers = json.load(f)
outlier_snids = set(outliers['snids'])

# Get Stage 1 SNe that are NOT outliers
stage1_snids = set()
for r in stage1['results']:
    if r['snid'] not in outlier_snids:
        stage1_snids.add(r['snid'])

# Combine
combined_snids = sorted(stage0_snids | stage1_snids)

print(f'Stage 0 clean SNe: {len(stage0_snids)}')
print(f'Stage 1 good SNe: {len(stage1_snids)}')
print(f'Stage 1 outliers: {len(outlier_snids)}')
print(f'Combined clean SNe: {len(combined_snids)}')

# Save
with open('$STAGE2_OUT/combined_clean_snids.json', 'w') as f:
    json.dump({
        'snids': combined_snids,
        'n_total': len(combined_snids),
        'n_stage0': len(stage0_snids),
        'n_stage1_good': len(stage1_snids)
    }, f, indent=2)
"

echo ""
echo "Step 2: Running Stage 2 MCMC on combined clean subset..."
python3 stage2_simple.py \
  --stage1-results "$STAGE1_RESULTS" \
  --lightcurves "$LIGHTCURVES" \
  --out "$STAGE2_OUT/stage2_refined" \
  --snid-list "$STAGE2_OUT/combined_clean_snids.json" \
  --nchains 2 \
  --nsamples 4000 \
  --nwarmup 2000 \
  --quality-cut 2000 \
  2>&1 | tee "$STAGE2_OUT/stage2_refined.log"

echo ""
echo "Step 3: Extracting refined physics parameters..."
python3 -c "
import json
import numpy as np

# Load refined Stage 2 results
best_fit = np.load('$STAGE2_OUT/stage2_refined/best_fit.npy')
k_J, eta_prime, xi = best_fit

# Save refined physics parameters
physics = {
    'k_J': float(k_J),
    'eta_prime': float(eta_prime),
    'xi': float(xi),
    'source': 'stage2_refined_combined_clean'
}

with open('$STAGE2_OUT/physics_params_refined.json', 'w') as f:
    json.dump(physics, f, indent=2)

print(f'Refined physics parameters:')
print(f'  k_J = {k_J:.6f}')
print(f'  η′  = {eta_prime:.6f}')
print(f'  ξ   = {xi:.6f}')
"

echo ""
echo "================================================================================"
echo "VALIDATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  Stage 0 clean SNe: $STAGE0_OUT/clean_snids.json"
echo "  Stage 0 physics: $STAGE0_OUT/physics_params.json"
echo "  Stage 1 deviations: $STAGE1_OUT/deviations.json"
echo "  Stage 1 outliers: $STAGE1_OUT/outlier_snids.json"
echo "  Stage 2 combined clean: $STAGE2_OUT/combined_clean_snids.json"
echo "  Stage 2 refined physics: $STAGE2_OUT/physics_params_refined.json"
echo ""
echo "Next steps:"
echo "  1. Review Stage 1 deviation statistics and outlier fraction"
echo "  2. Apply refined physics to full 5000 SNe dataset with gating"
echo "  3. Compare refined parameters with Stage 0 to assess convergence"
echo ""
