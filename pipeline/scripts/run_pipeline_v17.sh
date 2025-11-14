#!/bin/bash
#
# V17 QFD Supernova Pipeline Runner - 50 SNe Test
#
# This script runs the new V17 Stage 2 MCMC on a small test set of 50
# supernovae to validate the implementation and memory usage.
#

set -e
echo "========================================================================"
echo "RUNNING QFD SUPERNOVA PIPELINE V17 - 50 SNE TEST"
echo "========================================================================"
echo

# --- Configuration ---
# Define paths relative to the v15_clean directory, where this script is run from
V17_DIR="./pipeline"
STAGES_DIR="$V17_DIR/stages"
DATA_FILE="$V17_DIR/data/lightcurves_unified_v2_min3.csv" # This symlink points to the full dataset
STAGE1_RESULTS_DIR="../results/v15_clean/stage1_fullscale" # Using pre-computed V15 results
RESULTS_DIR="$V17_DIR/results/stage2_50sne_test"

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# --- Stage 1 ---
echo "--- STAGE 1: Using Pre-computed V15 Stage 1 Results ---"
echo "Location: $STAGE1_RESULTS_DIR"
if [ ! -d "$STAGE1_RESULTS_DIR" ]; then
    echo "Error: Stage 1 results not found at $STAGE1_RESULTS_DIR"
    echo "Please ensure the v15 Stage 1 pipeline has been run successfully."
    exit 1
fi
echo


# --- Stage 2: Global MCMC Fit (V17 Model) ---
# This is the core of the new v17 implementation.
echo "--- STAGE 2: Global MCMC Fitting (V17 Model) on 50 SNe ---"
python "$STAGES_DIR/stage2_mcmc_v17.py" \
    --lightcurves "$DATA_FILE" \
    --stage1-results "$STAGE1_RESULTS_DIR" \
    --out "$RESULTS_DIR" \
    --max-sne 50 \
    --nchains 2 \
    --nsamples 1000 \
    --nwarmup 500

echo
echo "========================================================================"
echo "V17 50 SNE TEST FINISHED"
echo "========================================================================"
echo "Results are in: $RESULTS_DIR"
echo "Check the output for MCMC summary and saved sample files."
