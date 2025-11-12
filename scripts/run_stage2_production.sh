#!/bin/bash
#
# Stage 2: Full-Scale MCMC with Clean Pipeline (Production Configuration)
#
# Hardware Target: 1x GPU (4GB VRAM), 1x 8-core/16-thread CPU with 16GB RAM
# Strategy: Run chains sequentially on the GPU for maximum speed per step.
#

set -e  # Exit on error

# --- GPU Memory Configuration ---
# Dedicate most of the VRAM to JAX.
# Pre-allocating everything can be faster if it fits.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# --- CPU Threading Configuration ---
# Even though the GPU does the heavy lifting, JAX uses the CPU for orchestration.
# Give it a healthy number of threads.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# --- MCMC Production Configuration ---
NUM_CHAINS=4
NUM_WARMUP=1000
NUM_SAMPLES=2000
CHI2_CUT=2000

# --- Input/Output Paths ---
STAGE1_DIR="results/v15_clean/stage1_fullscale"
OUTPUT_DIR="results/v15_clean/stage2_production"
LIGHTCURVES="data/lightcurves_unified_v2_min3.csv"

echo "================================================================================"
echo "STAGE 2: PRODUCTION RUN - CLEAN PIPELINE (GPU-CENTRIC)"
echo "================================================================================"
echo "Configuration:"
echo "  - GPU Memory Limit: ~3.4 GB (0.85 of 4GB)"
echo "  - CPU Threads: 8"
echo "  - MCMC Chains: $NUM_CHAINS (running SEQUENTIALLY on GPU)"
echo "  - MCMC Steps: $NUM_WARMUP warmup + $NUM_SAMPLES samples"
echo "  - Input SNe: All good fits from Stage 1 (quality cut: chi2 < $CHI2_CUT)"
echo "  - Output: $OUTPUT_DIR"
echo ""
echo "Pipeline Features:"
echo "  ✅ Correct parameter ordering via PerSNParams"
echo "  ✅ Fast ln_A-space model (100-1000x faster than full physics)"
echo "  ✅ GPU-accelerated parallel computation"
echo "  ✅ Informed priors centered at paper values (k_J~10.7, η'~-8, ξ~-7)"
echo ""
echo "Expected behavior:"
echo "  - A warning about 'Chains will be drawn sequentially' IS EXPECTED."
echo "  - One long progress bar will run 4 times (once per chain)."
echo "  - Expected runtime: ~30-60 minutes (with ln_A-space model)"
echo "================================================================================"
echo "Starting Stage 2 full-scale MCMC..."
echo "================================================================================"

# Run from project root
cd "$(dirname "$0")/../.."

# Launch the MCMC
python3 v15_clean/stages/stage2_mcmc_numpyro.py \
    --stage1-results "$STAGE1_DIR" \
    --out "$OUTPUT_DIR" \
    --lightcurves "$LIGHTCURVES" \
    --nchains "$NUM_CHAINS" \
    --nwarmup "$NUM_WARMUP" \
    --nsamples "$NUM_SAMPLES" \
    --quality-cut "$CHI2_CUT" \
    --use-ln-a-space \
    --constrain-signs informed

echo ""
echo "================================================================================"
echo "Stage 2 Production Run Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "================================================================================"
