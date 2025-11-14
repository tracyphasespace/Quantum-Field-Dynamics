#!/bin/bash
#
# Stage 2: Test with Informed Priors
#
# Quick test with 50 SNe, 2 chains, 100 warmup + 200 samples
# Uses informed priors centered at paper values to constrain parameter space
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
echo "STAGE 2: INFORMED PRIORS TEST"
echo "================================================================================"
echo "Testing with 50 SNe, 2 chains, 100 warmup + 200 samples"
echo "Using informed priors: k_J ~ N(10.7, 3), η' ~ N(-8, 3), ξ ~ N(-7, 3)"
echo "Expected time: ~30 seconds"
echo "================================================================================"

cd "$(dirname "$0")/../.."

python3 v15_clean/stages/stage2_mcmc_numpyro.py \
    --stage1-results results/v15_clean/stage1_fullscale \
    --out results/v15_clean/stage2_informed_test \
    --lightcurves data/lightcurves_unified_v2_min3.csv \
    --nchains 2 \
    --nwarmup 100 \
    --nsamples 200 \
    --quality-cut 2000 \
    --max-sne 50 \
    --use-ln-a-space \
    --constrain-signs informed

echo ""
echo "================================================================================"
echo "INFORMED PRIORS TEST COMPLETE"
echo "================================================================================"
echo "Check results in: results/v15_clean/stage2_informed_test/"
echo ""
echo "Expected parameter ranges:"
echo "  k_J:      ~8-13 km/s/Mpc (paper: 10.7)"
echo "  eta':     ~-11 to -5     (paper: -8.0)"
echo "  xi:       ~-10 to -4     (paper: -7.0)"
echo "================================================================================"
