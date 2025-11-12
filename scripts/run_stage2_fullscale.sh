#!/bin/bash
#
# Stage 2: Full-Scale MCMC with Clean Pipeline (Production Configuration)
#
# This script runs production-scale MCMC with:
# 1. All good SNe from Stage 1 fullscale results (~4,700 SNe)
# 2. 4 parallel chains (CPU-based) for robust convergence diagnostics
# 3. 1000 warmup + 2000 sampling steps for well-converged posteriors
# 4. Corrected parameter ordering via PerSNParams.to_model_order()
# 5. Single-threaded operations to minimize memory usage
#

set -e  # Exit on error

# Resource configuration (OPTIMIZED FOR LOW MEMORY - 2GB limit)
# Using single-threaded operations with minimal chains
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Low-memory configuration
NUM_CHAINS=2
NUM_WARMUP=1000
NUM_SAMPLES=2000
CHI2_CUT=2000

# Input/output paths (relative to project root)
STAGE1_DIR="results/v15_clean/stage1_fullscale"
OUTPUT_DIR="results/v15_clean/stage2_fullscale"
LIGHTCURVES="data/lightcurves_unified_v2_min3.csv"

echo "================================================================================"
echo "STAGE 2: FULL-SCALE MCMC - CLEAN PIPELINE (LOW-MEMORY MODE)"
echo "================================================================================"
echo "Resource configuration:"
echo "  CPU-based execution (JAX on CPU platform)"
echo "  Single-threaded operations for memory safety"
echo "  2 parallel chains (LOW MEMORY - optimized for 2GB limit)"
echo ""
echo "MCMC Configuration (PRODUCTION):"
echo "  - $NUM_CHAINS chains (parallel execution enabled)"
echo "  - $NUM_WARMUP warmup steps"
echo "  - $NUM_SAMPLES sampling steps"
echo "  - Quality cut: chi2 < $CHI2_CUT"
echo "  - Using ALL good SNe (~4,700)"
echo ""
echo "Pipeline Features:"
echo "  ✅ CPU-based parallel chain execution (numpyro.set_host_device_count)"
echo "  ✅ Corrected parameter ordering (PerSNParams.to_model_order)"
echo "  ✅ Full physics model with all DES-SN5YR lightcurve data"
echo "  ✅ Student-t robust likelihood from Stage 1"
echo ""
echo "Input: Stage 1 fullscale results (all good SNe with chi2 < $CHI2_CUT)"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Expected runtime: ~6-8 hours"
echo "================================================================================"
echo "Starting Stage 2 full-scale MCMC with clean pipeline..."
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
    --quality-cut "$CHI2_CUT"

echo ""
echo "================================================================================"
echo "Stage 2 Complete!"
echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
echo "  - mcmc_samples.nc: Full MCMC chain samples (NetCDF format)"
echo "  - global_params_summary.json: Posterior statistics for k_J, η′, ξ"
echo "  - convergence_diagnostics.txt: R-hat and ESS metrics"
echo ""
echo "Next step: Generate Hubble diagram with Stage 3"
echo "  ./v15_clean/scripts/run_stage3_hubble.sh"
echo ""
