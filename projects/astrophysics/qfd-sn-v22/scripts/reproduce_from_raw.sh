#!/bin/bash
#
# Complete Replication from Raw Lightcurve Data
# ==============================================
#
# This script runs the FULL pipeline from raw DES-SN5YR lightcurves
# through all stages to produce the final QFD vs ΛCDM comparison.
#
# For researchers who want 100% transparency and don't trust pre-computed Stage 1.
#
# Runtime: ~3-4 hours (Stage 1: 2-3 hours, Stage 2: 30 min, Stage 3: 1 min)
#
# Usage:
#   bash scripts/reproduce_from_raw.sh
#

set -e  # Exit on error

echo "============================================================"
echo "QFD SUPERNOVA ANALYSIS - COMPLETE REPLICATION"
echo "============================================================"
echo ""
echo "This will run the FULL pipeline from raw lightcurve data:"
echo "  Stage 1: Fit individual SN light curves (2-3 hours)"
echo "  Stage 2: MCMC global parameter fitting (30 minutes)"
echo "  Stage 3: Hubble diagram and model comparison (1 minute)"
echo ""
echo "Total runtime: ~3-4 hours on 8-core CPU"
echo ""

# Check if lightcurve data exists
LIGHTCURVE_FILE="${LIGHTCURVE_FILE:-data/raw/des_sn5yr_lightcurves.csv}"

if [ ! -f "$LIGHTCURVE_FILE" ]; then
    echo "ERROR: Lightcurve file not found: $LIGHTCURVE_FILE"
    echo ""
    echo "Please download DES-SN5YR data first:"
    echo "  bash scripts/download_des5yr.sh"
    echo ""
    echo "Or specify custom lightcurve file:"
    echo "  LIGHTCURVE_FILE=path/to/lightcurves.csv bash scripts/reproduce_from_raw.sh"
    exit 1
fi

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/full_replication_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Determine number of cores
NCORES=${NCORES:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}
echo "Using $NCORES CPU cores"
echo ""

# STAGE 1: Per-SN Light Curve Fitting
# ====================================
echo "------------------------------------------------------------"
echo "STAGE 1: Per-Supernova Light Curve Fitting"
echo "------------------------------------------------------------"
echo "Input: $LIGHTCURVE_FILE"
echo "Processing all supernovae..."
echo ""

python -m qfd_sn.stage1_fit \
    --lightcurves "$LIGHTCURVE_FILE" \
    --output "$OUTPUT_DIR/stage1" \
    --ncores "$NCORES"

if [ ! -f "$OUTPUT_DIR/stage1/stage1_results.csv" ]; then
    echo "ERROR: Stage 1 failed - output file not found"
    exit 1
fi

echo ""
echo "Stage 1 complete!"
echo "Results: $OUTPUT_DIR/stage1/stage1_results.csv"
echo ""

# Apply Quality Control Gates
# ============================
echo "------------------------------------------------------------"
echo "Applying Quality Control Gates"
echo "------------------------------------------------------------"
echo "Filtering Stage 1 results..."
echo "  chi²/dof < 2000"
echo "  -20 < ln_A < 20"
echo ""

python -c "
import pandas as pd
from qfd_sn.qc import QualityGates, apply_quality_gates

# Load Stage 1 results
data = pd.read_csv('$OUTPUT_DIR/stage1/stage1_results.csv')
print(f'Stage 1 total: {len(data)} SNe')

# Apply quality gates
gates = QualityGates(chi2_max=2000.0, ln_A_min=-20.0, ln_A_max=20.0)
qc_results = apply_quality_gates(data, gates, verbose=True)

# Save filtered results
filtered = data[qc_results.pass_all]
filtered.to_csv('$OUTPUT_DIR/stage1/stage1_results_filtered.csv', index=False)

print(f'After filtering: {len(filtered)} SNe')
print(f'Pass rate: {100*len(filtered)/len(data):.1f}%')
"

echo ""
echo "Quality control complete!"
echo "Filtered results: $OUTPUT_DIR/stage1/stage1_results_filtered.csv"
echo ""

# STAGE 2: MCMC Global Parameter Fitting
# =======================================
echo "------------------------------------------------------------"
echo "STAGE 2: MCMC Global Parameter Fitting"
echo "------------------------------------------------------------"
echo "Fitting QFD parameters to all SNe..."
echo "  nwalkers = 32"
echo "  nsteps = 4000"
echo "  nburn = 1000"
echo ""

python -m qfd_sn.stage2_mcmc \
    --input "$OUTPUT_DIR/stage1/stage1_results_filtered.csv" \
    --output "$OUTPUT_DIR/stage2" \
    --nwalkers 32 \
    --nsteps 4000 \
    --nburn 1000

if [ ! -f "$OUTPUT_DIR/stage2/summary.json" ]; then
    echo "ERROR: Stage 2 failed - summary file not found"
    exit 1
fi

echo ""
echo "Stage 2 complete!"
echo "Results: $OUTPUT_DIR/stage2/summary.json"
echo ""

# STAGE 3: Hubble Diagram and Model Comparison
# =============================================
echo "------------------------------------------------------------"
echo "STAGE 3: Hubble Diagram and Model Comparison"
echo "------------------------------------------------------------"
echo "Creating Hubble diagram..."
echo "Comparing QFD vs ΛCDM..."
echo ""

python -m qfd_sn.stage3_hubble \
    --stage1 "$OUTPUT_DIR/stage1/stage1_results_filtered.csv" \
    --stage2 "$OUTPUT_DIR/stage2" \
    --output "$OUTPUT_DIR/stage3"

if [ ! -f "$OUTPUT_DIR/stage3/summary.json" ]; then
    echo "ERROR: Stage 3 failed - summary file not found"
    exit 1
fi

echo ""
echo "Stage 3 complete!"
echo "Results: $OUTPUT_DIR/stage3/summary.json"
echo ""

# Display Final Results
# =====================
echo "============================================================"
echo "REPLICATION COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Summary:"
cat "$OUTPUT_DIR/stage3/summary.json"
echo ""
echo "============================================================"
echo ""
echo "Compare these results to published values:"
echo "  - k_J ≈ 120-130 km/s/Mpc"
echo "  - QFD RMS ≈ 1.7-1.9 mag"
echo "  - ΛCDM RMS ≈ 2.2-2.4 mag"
echo "  - Improvement: ~20-25%"
echo ""
echo "Full results:"
echo "  - Stage 1: $OUTPUT_DIR/stage1/"
echo "  - Stage 2: $OUTPUT_DIR/stage2/"
echo "  - Stage 3: $OUTPUT_DIR/stage3/"
echo "  - Hubble data: $OUTPUT_DIR/stage3/hubble_data.csv"
echo ""
