#!/bin/bash
#
# Production run of NumPyro Stage 2 MCMC (~10-15 minutes)
#
# This runs the full MCMC sampling for publication-quality posteriors.

set -e  # Exit on error

echo "================================================================================"
echo "PRODUCTION: NumPyro Stage 2 MCMC"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  - 4 chains (parallel on GPU)"
echo "  - 1,000 warmup steps"
echo "  - 2,000 sampling steps"
echo "  - Total: 8,000 effective samples (4 × 2,000)"
echo "  - All 5,124 quality SNe"
echo ""
echo "Expected runtime: ~10-15 minutes"
echo "Output: results/v15_stage2_mcmc_numpyro/"
echo ""

# Create log directory
mkdir -p logs

# Run with logging
python stage2_mcmc_numpyro.py \
    --stage1-results results/v15_stage1_production \
    --lightcurves ../../data/unified/lightcurves_unified_v2_min3.csv \
    --out results/v15_stage2_mcmc_numpyro \
    --nchains 4 \
    --nsamples 2000 \
    --nwarmup 1000 \
    --quality-cut 2000 \
    2>&1 | tee logs/stage2_numpyro_production.log

echo ""
echo "================================================================================"
echo "PRODUCTION RUN COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to: results/v15_stage2_mcmc_numpyro/"
echo "Log saved to: logs/stage2_numpyro_production.log"
echo ""
echo "Next step: Run Stage 3 with new posteriors"
echo "  python stage3_hubble_optimized.py \\"
echo "    --stage1-results results/v15_stage1_production \\"
echo "    --stage2-results results/v15_stage2_mcmc_numpyro \\"
echo "    --lightcurves ../../data/unified/lightcurves_unified_v2_min3.csv \\"
echo "    --out results/v15_stage3_hubble_numpyro"
echo ""
