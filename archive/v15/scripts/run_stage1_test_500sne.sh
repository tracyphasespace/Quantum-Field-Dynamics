#!/bin/bash
#
# Stage 1: Test with 500 SNe (2-Parameter Model)
#
# CRITICAL: This uses the NEW 2-parameter model where k_J = 70.0 is FIXED
# Only fitting: eta_prime, xi (NOT k_J anymore)
#
# Memory configuration: Moderate for ~500 SNe
# Target: Operation within 4-6GB memory limit
#

set -e  # Exit on error

# Memory configuration (MODERATE for 500 SNe test)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# CPU thread limits (still conservative)
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Test configuration
NUM_SNE=500
WORKERS=4  # 4 parallel workers

echo "================================================================================"
echo "STAGE 1 TEST: 500 SNe with 2-PARAMETER MODEL"
echo "================================================================================"
echo "CRITICAL MODEL CHANGE (Nov 13, 2024):"
echo "  - k_J = 70.0 km/s/Mpc (FIXED, from QVD redshift model)"
echo "  - Fitting ONLY: (eta_prime, xi)"
echo "  - Command format: --global eta_prime,xi  (NOT k_J,eta_prime,xi)"
echo ""
echo "Resource configuration:"
echo "  Memory: Moderate (25% GPU fraction)"
echo "  Workers: $WORKERS parallel"
echo "  CPU threads: 1 per worker"
echo ""
echo "Test parameters:"
echo "  SNe: First $NUM_SNE from dataset"
echo "  Global params: eta_prime=0.01, xi=30.0"
echo "  Output: results/v15_clean/stage1_test_500sne"
echo ""

# Global parameters (2-PARAM MODEL - NO k_J!)
GLOBAL_PARAMS="0.01,30"

echo "Global parameters (eta_prime, xi): $GLOBAL_PARAMS"
echo "  eta_prime ≈ 0.01 (plasma veil strength)"
echo "  xi        ≈ 30.0 (FDR/sear strength)"
echo "  k_J = 70.0 (FIXED in v15_model.py, NOT fitted)"
echo ""
echo "================================================================================"
echo "Starting Stage 1 test with 500 SNe..."
echo "================================================================================"

# Run from project root
cd "$(dirname "$0")/../.."

python3 v15_clean/stages/stage1_optimize.py \
  --lightcurves data/lightcurves_unified_v2_min3.csv \
  --out results/v15_clean/stage1_test_500sne \
  --global "$GLOBAL_PARAMS" \
  --sn-list "0:$NUM_SNE" \
  --grad-tol 10.0 \
  --nu 5.0 \
  --use-studentt \
  --workers $WORKERS \
  --max-iters 200

echo ""
echo "================================================================================"
echo "Stage 1 Test Complete!"
echo "================================================================================"
echo "Results saved to: results/v15_clean/stage1_test_500sne/"
echo ""
echo "To check status:"
echo "  for status in results/v15_clean/stage1_test_500sne/*/status.txt; do cat \"\$status\"; done | sort | uniq -c"
echo ""
echo "Next step: If successful, consider full-scale or Stage 2 test"
echo "  ./v15_clean/scripts/run_stage2_test.sh"
echo ""
