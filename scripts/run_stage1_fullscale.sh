#!/bin/bash
#
# Stage 1: Full-Scale Test with Clean Pipeline
# This tests the refactored pipeline with progress monitoring on the full dataset
#

set -e

# Low memory configuration for 6 workers
# Target: 2GB / 6 workers = ~333MB per worker
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.08
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# CPU thread limits (6 workers)
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "================================================================================"
echo "STAGE 1: FULL-SCALE TEST - CLEAN PIPELINE WITH PROGRESS MONITORING"
echo "================================================================================"
echo "Testing: Refactored pipeline with type-safe parameter handling"
echo ""
echo "Resource configuration:"
echo "  GPU memory: 2.0GB total (0.08 fraction × 6 workers = ~333MB/worker)"
echo "  Workers: 6"
echo "  CPU threads per worker: 1"
echo "  Memory mode: Dynamic allocation with cleanup"
echo ""
echo "Optimization settings:"
echo "  grad_tol: 10.0 (relaxed for better recovery)"
echo "  nu: 5.0 (Student-t likelihood for outlier robustness)"
echo "  use-studentt: enabled"
echo ""
echo "Features:"
echo "  ✅ Progress monitoring every 500 SNe"
echo "  ✅ Type-safe parameter structures (PerSNParams)"
echo "  ✅ Student-t robust fitting"
echo ""
echo "Dataset: All 5,468 SNe from DES-SN5YR"
echo "Output: results/v15_clean/stage1_fullscale/"
echo ""

# Global parameters from v15_production (Model A from paper)
GLOBAL_PARAMS="10.69,-7.97,-6.88"

echo "Global parameters (k_J, η′, ξ): $GLOBAL_PARAMS"
echo "  k_J  ≈ 10.69 (baseline drag kernel)"
echo "  η′   ≈ -7.97 (flux-dependent scattering)"
echo "  ξ    ≈ -6.88 (near-source saturation)"
echo ""
echo "================================================================================"
echo "Starting full-scale Stage 1 optimization with clean pipeline..."
echo "================================================================================"

# Run from project root
cd "$(dirname "$0")/../.."

python3 v15_clean/stages/stage1_optimize.py \
  --lightcurves data/lightcurves_unified_v2_min3.csv \
  --out results/v15_clean/stage1_fullscale \
  --global "$GLOBAL_PARAMS" \
  --grad-tol 10.0 \
  --nu 5.0 \
  --use-studentt \
  --workers 6 \
  --max-iters 200

echo ""
echo "================================================================================"
echo "Stage 1 Complete!"
echo "================================================================================"
echo "Results saved to: results/v15_clean/stage1_fullscale/"
echo ""
echo "To analyze results:"
echo "  python3 -c \"from pathlib import Path; d = Path('results/v15_clean/stage1_fullscale'); \\"
echo "    statuses = {'ok': 0, 'did_not_converge': 0, 'nan': 0}; \\"
echo "    [statuses.update({f.read_text().strip(): statuses.get(f.read_text().strip(), 0) + 1}) \\"
echo "     for f in d.glob('*/status.txt')]; \\"
echo "    print(f'Success: {statuses.get(\"ok\", 0)}/{sum(statuses.values())}')\""
echo ""
echo "Next step: Run Stage 2 MCMC with clean pipeline"
echo "  ./v15_clean/scripts/run_stage2_fullscale.sh"
echo ""
