#!/bin/bash
#
# Quick Stage 2 test with clean pipeline (corrected parameter ordering)
#
# This script runs a reduced MCMC test to verify:
# 1. Parameter ordering is correct
# 2. Physics model produces finite outputs
# 3. MCMC converges to reasonable values
#

set -e  # Exit on error

# Resource configuration (adjust for your system)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.63
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=6"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

# Test configuration (reduced for speed)
NUM_CHAINS=2
NUM_WARMUP=50
NUM_SAMPLES=100
CHI2_CUT=2000
MAX_SNE=20

# Input/output paths (relative to project root)
STAGE1_DIR="../results/v15_production/stage1_relaxed"
OUTPUT_DIR="../results/v15_production/stage2_clean_test"
LIGHTCURVES="../data/lightcurves_unified_v2_min3.csv"

echo "================================================================================"
echo "STAGE 2: QUICK TEST - CLEAN PIPELINE (CORRECTED PARAMETER ORDERING)"
echo "================================================================================"
echo "Resource configuration:"
echo "  GPU memory: ~2GB (0.63 fraction)"
echo "  CPU threads: 6"
echo ""
echo "MCMC Configuration (REDUCED FOR TESTING):"
echo "  - $NUM_CHAINS chains"
echo "  - $NUM_WARMUP warmup steps"
echo "  - $NUM_SAMPLES sampling steps"
echo "  - Quality cut: chi2 < $CHI2_CUT"
echo "  - Max SNe: $MAX_SNE (testing only!)"
echo ""
echo "CRITICAL FIX:"
echo "  ✅ Using PerSNParams.to_model_order() for correct parameter ordering"
echo "  ✅ Model receives: (t0, ln_A, A_plasma, beta)"
echo "  ✅ Should produce physically meaningful results"
echo ""
echo "Input: Stage 1 relaxed results (limited to $MAX_SNE SNe)"
echo "Output: $OUTPUT_DIR"
echo ""
echo "================================================================================"
echo "Starting Stage 2 MCMC test with clean pipeline..."
echo "================================================================================"

# Run from project root
cd "$(dirname "$0")/../.."

# Run Stage 2 with clean pipeline code
python3 v15_clean/stages/stage2_mcmc_numpyro.py \
    --stage1-results "$STAGE1_DIR" \
    --out "$OUTPUT_DIR" \
    --lightcurves "$LIGHTCURVES" \
    --nchains "$NUM_CHAINS" \
    --nwarmup "$NUM_WARMUP" \
    --nsamples "$NUM_SAMPLES" \
    --quality-cut "$CHI2_CUT" \
    --max-sne "$MAX_SNE"

echo ""
echo "================================================================================"
echo "Test complete! Check results in: $OUTPUT_DIR"
echo "================================================================================"
