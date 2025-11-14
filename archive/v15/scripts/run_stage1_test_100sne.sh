#!/bin/bash
#
# Stage 1: Test with 100 SNe (2-Parameter Model)
#
# CRITICAL: This uses the NEW 2-parameter model where k_J = 70.0 is FIXED
# Only fitting: eta_prime, xi (NOT k_J anymore)
#
# Memory configuration: Conservative for ~100 SNe
# Target: Safe operation within 2GB memory limit
#

set -e  # Exit on error

# Memory configuration (VERY CONSERVATIVE for 100 SNe test)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# CPU thread limits (single-threaded for safety)
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Test configuration
NUM_SNE=100
WORKERS=3  # Conservative: 3 parallel workers at ~500MB each

echo "================================================================================"
echo "STAGE 1 TEST: 100 SNe with 2-PARAMETER MODEL"
echo "================================================================================"
echo "CRITICAL MODEL CHANGE (Nov 13, 2024):"
echo "  - k_J = 70.0 km/s/Mpc (FIXED, from QVD redshift model)"
echo "  - Fitting ONLY: (eta_prime, xi)"
echo "  - Command format: --global eta_prime,xi  (NOT k_J,eta_prime,xi)"
echo ""
echo "Resource configuration:"
echo "  Memory: Very conservative (15% GPU fraction)"
echo "  Workers: $WORKERS parallel"
echo "  CPU threads: 1 per worker"
echo ""
echo "Test parameters:"
echo "  SNe: First $NUM_SNE from dataset"
echo "  Global params: eta_prime=0.01, xi=30.0"
echo "  Output: results/v15_clean/stage1_test_100sne"
echo ""

# Global parameters (2-PARAM MODEL - NO k_J!)
GLOBAL_PARAMS="0.01,30"

echo "Global parameters (eta_prime, xi): $GLOBAL_PARAMS"
echo "  eta_prime ≈ 0.01 (plasma veil strength)"
echo "  xi        ≈ 30.0 (FDR/sear strength)"
echo "  k_J = 70.0 (FIXED in v15_model.py, NOT fitted)"
echo ""
echo "================================================================================"
echo "Starting Stage 1 test with 100 SNe..."
echo "================================================================================"

# Run from project root
cd "$(dirname "$0")/../.."

python3 v15_clean/stages/stage1_optimize.py \
  --lightcurves data/lightcurves_unified_v2_min3.csv \
  --out results/v15_clean/stage1_test_100sne \
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
echo "Results saved to: results/v15_clean/stage1_test_100sne/"
echo ""
echo "To check status:"
echo "  for status in results/v15_clean/stage1_test_100sne/*/status.txt; do cat \"\$status\"; done | sort | uniq -c"
echo ""
echo "Next step: If successful, scale up to 500 SNe"
echo "  ./v15_clean/scripts/run_stage1_test_500sne.sh"
echo ""
