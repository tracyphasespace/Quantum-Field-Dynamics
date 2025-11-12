#!/bin/bash
#
# Quick Stage 2 Test - Verify Parallel Execution
#
# This runs a minimal test to verify:
# 1. Parallel chains work (no sequential warnings)
# 2. Parameter ordering is correct
# 3. MCMC converges to reasonable values
#

set -e

# Resource configuration (same as production)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.63
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=6"
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6

# Quick test configuration
NUM_CHAINS=2
NUM_WARMUP=10
NUM_SAMPLES=20
CHI2_CUT=2000
MAX_SNE=50

# Input/output paths
STAGE1_DIR="results/v15_clean/stage1_fullscale"
OUTPUT_DIR="results/v15_clean/stage2_quicktest"
LIGHTCURVES="data/lightcurves_unified_v2_min3.csv"

echo "================================================================================"
echo "STAGE 2: QUICK TEST - VERIFY PARALLEL EXECUTION"
echo "================================================================================"
echo "Resource configuration:"
echo "  GPU memory: ~2.5GB (0.63 fraction)"
echo "  CPU threads: 6"
echo ""
echo "Test Configuration (MINIMAL):"
echo "  - $NUM_CHAINS chains (testing parallel execution)"
echo "  - $NUM_WARMUP warmup steps (minimal)"
echo "  - $NUM_SAMPLES sampling steps (minimal)"
echo "  - Max SNe: $MAX_SNE (quick test)"
echo ""
echo "Verification checks:"
echo "  1. No 'sequential' warnings (chains run in parallel)"
echo "  2. Both chains complete successfully"
echo "  3. Parameters converge to reasonable values"
echo ""
echo "Output: $OUTPUT_DIR"
echo "Expected runtime: ~2-3 minutes"
echo "================================================================================"

# Run from project root
cd "$(dirname "$0")/../.."

# Run quick test
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
echo "Quick Test Complete!"
echo "================================================================================"
echo ""
echo "Check for these success indicators:"
echo "  ✓ No 'Chains will be drawn sequentially' warning"
echo "  ✓ Two progress bars (or one updating 2x as fast)"
echo "  ✓ Parameters close to: k_J~10.7, η'~-8.0, ξ~-7.0"
echo ""
echo "If all checks pass, you're ready for the full production run:"
echo "  ./v15_clean/scripts/run_stage2_fullscale.sh"
echo ""
