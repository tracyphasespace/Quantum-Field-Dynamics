#!/bin/bash
#
# V18 QFD Supernova - STAGE 2 PRODUCTION RUN
#
# This script executes the final, full-scale MCMC fit on the entire
# "clean" supernova dataset (~4,800 SNe from Stage 1) using the stable
# and validated v18 emcee-based model.
#

set -e
echo "========================================================================"
echo "RUNNING QFD SUPERNOVA PIPELINE V18 - STAGE 2 PRODUCTION"
echo "========================================================================"
echo

# --- Configuration ---
# Use all available CPU cores for maximum parallelization with emcee
NUM_CORES=$(nproc) 
# Use a high number of walkers for a robust exploration of the posterior
NUM_WALKERS=64 
# Production-level steps for high-quality posteriors
NUM_STEPS=10000
NUM_BURN=2000

# --- Paths ---
PIPELINE_DIR="./pipeline" # Assuming the script is run from the root of the repo
STAGES_DIR="$PIPELINE_DIR/stages"
DATA_FILE="$PIPELINE_DIR/data/lightcurves_unified_v2_min3.csv"
STAGE1_RESULTS_DIR="$PIPELINE_DIR/results/stage1_v17_fullscale" # Use the full Stage 1 results
RESULTS_DIR="$PIPELINE_DIR/results/v18_production_final"

echo "Configuration:"
echo "  - Script:          stage2_mcmc_v18_emcee.py"
echo "  - Model Basis:     ln_A (V18)"
echo "  - Sampler:         emcee (multi-core CPU)"
echo "  - CPU Cores:       $NUM_CORES"
echo "  - Walkers:         $NUM_WALKERS"
echo "  - Steps:           $NUM_STEPS (Burn-in: $NUM_BURN)"
echo "  - Input Data:      Full clean set from $STAGE1_RESULTS_DIR"
echo "  - Output Directory: $RESULTS_DIR"
echo "Expected Runtime: Several hours, depending on CPU performance."
echo "========================================================================"

# --- Pre-flight Check ---
if [ ! -d "$STAGE1_RESULTS_DIR" ]; then
    echo "ERROR: Full-scale Stage 1 results not found at $STAGE1_RESULTS_DIR"
    echo "Please run the v17 Stage 1 pipeline on the full dataset first."
    exit 1
fi

# Ensure output directory exists
mkdir -p "$RESULTS_DIR"

# --- Run the MCMC ---
echo "Starting Stage 2 Production MCMC..."

python "$STAGES_DIR/stage2_mcmc_v18_emcee.py" \
    --lightcurves "$DATA_FILE" \
    --stage1-results "$STAGE1_RESULTS_DIR" \
    --out "$RESULTS_DIR" \
    --max-sne 9999 \
    --nwalkers "$NUM_WALKERS" \
    --nsteps "$NUM_STEPS" \
    --nburn "$NUM_BURN" \
    --ncores "$NUM_CORES"

echo
echo "========================================================================"
echo "V18 STAGE 2 PRODUCTION RUN FINISHED"
echo "========================================================================"
echo "Final results, samples, and summary are in: $RESULTS_DIR"
echo "You are now ready for Stage 3 analysis and figure generation."
