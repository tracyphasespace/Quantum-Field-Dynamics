#!/bin/bash
#
# V17 QFD Supernova - Stage 1 Runner
#
# This script executes the parallel Stage 1 optimization processor, which fits
# the per-supernova nuisance parameters for the v17 model.
#

set -e
echo "========================================================================"
echo "RUNNING QFD SUPERNOVA PIPELINE V17 - STAGE 1"
echo "========================================================================"
echo

# --- Configuration ---
PIPELINE_DIR="./pipeline"
SCRIPTS_DIR="$PIPELINE_DIR/scripts"
DATA_FILE="pipeline/data/lightcurves_unified_v2_min3.csv"

# Use all available cores by default
NUM_CORES=$(nproc)

# For this initial test, we'll process 1 SN to diagnose performance.
LIMIT=1

# --- Run the Parallel Python Script ---
python "$SCRIPTS_DIR/run_stage1_parallel.py" \
    --lightcurves "$DATA_FILE" \
    --out "$RESULTS_DIR" \
    --n-cores "$NUM_CORES" \
    --limit "$LIMIT"

echo
echo "========================================================================"
echo "V17 STAGE 1 PROCESSING FINISHED"
echo "========================================================================"
echo "Results are in: $RESULTS_DIR"
