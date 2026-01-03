#!/bin/bash
# Complete pipeline for Harmonic Nuclear Model analysis

set -e  # Exit on error

echo "========================================================================"
echo "HARMONIC NUCLEAR MODEL: Complete Pipeline"
echo "========================================================================"
echo ""

# Step 1: Parse NUBASE data
echo "STEP 1: Parsing NUBASE2020 data..."
bash scripts/01_parse_nubase.sh

# Step 2: Fit harmonic families
echo ""
echo "STEP 2: Fitting harmonic families to stable nuclides..."
bash scripts/02_fit_families.sh

# Step 3: Score all nuclides
echo ""
echo "STEP 3: Scoring all nuclides with harmonic model..."
bash scripts/03_score_nuclides.sh

# Step 4: Run experiments
echo ""
echo "STEP 4: Running statistical experiments..."
bash scripts/04_run_experiments.sh

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE!"
echo "========================================================================"
echo ""
echo "Results available in:"
echo "  - reports/fits/family_params_stable.json"
echo "  - reports/exp1/exp1_results.json"
echo "  - reports/tacoma_narrows/tacoma_narrows_results.json"
echo "  - figures/"
echo ""
