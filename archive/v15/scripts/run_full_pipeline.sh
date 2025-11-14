#!/bin/bash
#
# V15_CLEAN: FULL PRODUCTION PIPELINE
#
# This is the CANONICAL pipeline script. Use ONLY v15_clean/ code.
#
# Runs Stage 2 → Stage 3 with all corrections:
#   - Stage 2: Informed priors + sign fix
#   - Stage 3: Zero-point calibration
#

set -e

STAGE1_DIR="results/v15_clean/stage1_fullscale"
STAGE2_DIR="results/v15_clean/stage2_production"
STAGE3_DIR="results/v15_clean/stage3_hubble"
LIGHTCURVES="data/lightcurves_unified_v2_min3.csv"
LOG_DIR="logs/v15_clean"

mkdir -p "$STAGE2_DIR"
mkdir -p "$STAGE3_DIR"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "V15_CLEAN: FULL PRODUCTION PIPELINE"
echo "================================================================================"
echo "Start time: $(date)"
echo ""
echo "Pipeline stages:"
echo "  1. Stage 2 MCMC (v15_clean/stages/stage2_mcmc_numpyro.py)"
echo "  2. Stage 3 Hubble diagram (v15_clean/stages/stage3_hubble_optimized.py)"
echo ""
echo "Output directories:"
echo "  Stage 2: $STAGE2_DIR"
echo "  Stage 3: $STAGE3_DIR"
echo "  Logs:    $LOG_DIR"
echo ""
echo "================================================================================"
echo ""

# ============================================================================
# STAGE 2: MCMC WITH INFORMED PRIORS + SIGN FIX
# ============================================================================

echo ""
echo "================================================================================"
echo "STAGE 2: GLOBAL PARAMETER MCMC"
echo "================================================================================"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  - Code: v15_clean/stages/stage2_mcmc_numpyro.py (36KB, 2025-11-12 01:11)"
echo "  - Model: ln_A-space (fast)"
echo "  - Priors: Informed (k_J~10.7, η'~-8, ξ~-7)"
echo "  - Sign fix: APPLIED (lines 414, 424, 439, 448)"
echo "  - SNe: 4,727 (chi2 < 2000)"
echo "  - Chains: 4"
echo "  - Samples: 2000 per chain"
echo "  - Warmup: 1000"
echo ""
echo "Expected runtime: ~3-4 hours"
echo "================================================================================"
echo ""

python3 v15_clean/stages/stage2_mcmc_numpyro.py \
    --stage1-results "$STAGE1_DIR" \
    --out "$STAGE2_DIR" \
    --lightcurves "$LIGHTCURVES" \
    --nchains 4 \
    --nwarmup 1000 \
    --nsamples 2000 \
    --quality-cut 2000 \
    --use-ln-a-space \
    --constrain-signs informed \
    2>&1 | tee "$LOG_DIR/stage2.log"

STAGE2_EXIT=$?

if [ $STAGE2_EXIT -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: Stage 2 failed with exit code $STAGE2_EXIT"
    echo "================================================================================"
    exit $STAGE2_EXIT
fi

echo ""
echo "================================================================================"
echo "STAGE 2 COMPLETE!"
echo "================================================================================"
echo "Completion time: $(date)"
echo ""
echo "Results:"
cat "$STAGE2_DIR/best_fit.json"
echo ""
echo "================================================================================"
echo ""

sleep 5

# ============================================================================
# STAGE 3: HUBBLE DIAGRAM WITH ZERO-POINT CALIBRATION
# ============================================================================

echo ""
echo "================================================================================"
echo "STAGE 3: HUBBLE DIAGRAM ANALYSIS"
echo "================================================================================"
echo "Start time: $(date)"
echo ""
echo "Configuration:"
echo "  - Code: v15_clean/stages/stage3_hubble_optimized.py (15KB, 2025-11-12 07:46)"
echo "  - Zero-point calibration: APPLIED (lines 258-275)"
echo "  - Quality cut: chi2 < 2000"
echo "  - Parallel cores: 7"
echo ""
echo "Expected runtime: ~3 minutes"
echo "================================================================================"
echo ""

python3 v15_clean/stages/stage3_hubble_optimized.py \
    --stage1-results "$STAGE1_DIR" \
    --stage2-results "$STAGE2_DIR" \
    --lightcurves "$LIGHTCURVES" \
    --out "$STAGE3_DIR" \
    --quality-cut 2000 \
    --ncores 7 \
    2>&1 | tee "$LOG_DIR/stage3.log"

STAGE3_EXIT=$?

if [ $STAGE3_EXIT -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: Stage 3 failed with exit code $STAGE3_EXIT"
    echo "================================================================================"
    exit $STAGE3_EXIT
fi

echo ""
echo "================================================================================"
echo "STAGE 3 COMPLETE!"
echo "================================================================================"
echo "Completion time: $(date)"
echo ""

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "================================================================================"
echo "FULL PIPELINE COMPLETE - FINAL RESULTS"
echo "================================================================================"
echo "End time: $(date)"
echo ""

echo "--------------------------------------------------------------------------------"
echo "STAGE 2: Global Parameters"
echo "--------------------------------------------------------------------------------"
cat "$STAGE2_DIR/best_fit.json"
echo ""

echo "--------------------------------------------------------------------------------"
echo "STAGE 3: Hubble Diagram Statistics"
echo "--------------------------------------------------------------------------------"
cat "$STAGE3_DIR/summary.json"
echo ""

echo "================================================================================"
echo "OUTPUT FILES"
echo "================================================================================"
echo ""
echo "Stage 2 Results:"
echo "  - Parameters:     $STAGE2_DIR/best_fit.json"
echo "  - MCMC samples:   $STAGE2_DIR/samples.json"
echo "  - Diagnostics:    $STAGE2_DIR/diagnostics.json"
echo "  - Corner plot:    $STAGE2_DIR/corner_plot.png"
echo ""
echo "Stage 3 Results:"
echo "  - Summary:        $STAGE3_DIR/summary.json"
echo "  - Data CSV:       $STAGE3_DIR/hubble_data.csv"
echo "  - Hubble diagram: $STAGE3_DIR/hubble_diagram.png"
echo "  - Residuals plot: $STAGE3_DIR/residuals_analysis.png"
echo ""
echo "Logs:"
echo "  - Stage 2 log:    $LOG_DIR/stage2.log"
echo "  - Stage 3 log:    $LOG_DIR/stage3.log"
echo ""

echo "================================================================================"
echo "SUCCESS!"
echo "================================================================================"
