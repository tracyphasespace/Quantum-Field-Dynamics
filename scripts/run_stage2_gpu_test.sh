#!/bin/bash
#
# Stage 2: GPU Test - Verify GPU configuration works
#
# Quick test with 50 SNe, 2 chains, 10 warmup + 20 samples
#

set -e

# --- GPU Memory Configuration ---
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# --- CPU Threading Configuration ---
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo "================================================================================"
echo "STAGE 2: GPU CONFIGURATION TEST"
echo "================================================================================"
echo "Testing with 50 SNe, 2 chains, 10 warmup + 20 samples"
echo "This should complete in ~2-3 minutes"
echo "================================================================================"

cd "$(dirname "$0")/../.."

python3 v15_clean/stages/stage2_mcmc_numpyro.py \
    --stage1-results results/v15_clean/stage1_fullscale \
    --out results/v15_clean/stage2_gpu_test \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --nchains 2 \
    --nwarmup 10 \
    --nsamples 20 \
    --quality-cut 2000 \
    --max-sne 50 \
    --use-ln-a-space

echo ""
echo "================================================================================"
echo "GPU Test Complete! If you see this message, GPU configuration is working."
echo "================================================================================"
