#!/bin/bash
#
# Stage 2: Low-Memory Test (OPTIMIZED FOR 2GB LIMIT)
#
# This script tests Stage 2 with minimal memory usage:
# 1. Only 50 SNe (vs ~4,700 full)
# 2. 2 parallel chains (vs 4-7 typical)
# 3. Reduced warmup/sampling (10 warmup + 20 samples for quick test)
# 4. XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 (only 25% of available memory)
# 5. Single-threaded operations throughout
#

set -e  # Exit on error

# MINIMAL MEMORY CONFIGURATION (for 2GB limit in WSL)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Test configuration - minimal for memory testing
NUM_CHAINS=2
NUM_WARMUP=10
NUM_SAMPLES=20
CHI2_CUT=2000
MAX_SNE=50

# Input/output paths
STAGE1_DIR="results/v15_clean/stage1_fullscale"
OUTPUT_DIR="results/v15_clean/stage2_lowmem_test"
LIGHTCURVES="data/lightcurves_unified_v2_min3.csv"

echo "================================================================================"
echo "STAGE 2: LOW-MEMORY TEST (2GB LIMIT)"
echo "================================================================================"
echo "Resource configuration:"
echo "  XLA memory fraction: 25% (vs 63% default)"
echo "  CPU-based execution with 2 parallel chains"
echo "  Single-threaded operations"
echo ""
echo "Test Configuration:"
echo "  - $NUM_CHAINS chains"
echo "  - $NUM_WARMUP warmup steps (minimal)"
echo "  - $NUM_SAMPLES sampling steps (minimal)"
echo "  - Quality cut: chi2 < $CHI2_CUT"
echo "  - Max SNe: $MAX_SNE (vs ~4,700 full)"
echo ""
echo "Purpose: Verify memory usage stays under 2GB before running full production"
echo ""
echo "Input: Stage 1 fullscale results (limited to $MAX_SNE SNe)"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Expected runtime: ~2-3 minutes"
echo "================================================================================"
echo "Starting low-memory test..."
echo "================================================================================"

# Run from project root
cd "$(dirname "$0")/../.."

# Run Stage 2 with minimal memory configuration
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
echo "Low-Memory Test Complete!"
echo "================================================================================"
echo "If this completed without OOM errors, you can try increasing:"
echo "  1. Number of SNe: --max-sne 100, 200, 500, etc."
echo "  2. Number of chains: NUM_CHAINS=3 (if memory allows)"
echo "  3. Warmup/samples: For production use 1000/2000"
echo ""
echo "Monitor memory usage with: watch -n 1 free -h"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
