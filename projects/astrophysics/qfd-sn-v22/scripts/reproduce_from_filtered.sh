#!/usr/bin/env bash
#
# Reproduce QFD Results from Pre-computed Filtered Data
#
# This script uses the pre-computed Stage 1 results (6,724 filtered SNe)
# and runs Stage 2 (MCMC) and Stage 3 (Hubble diagram).
#
# Trust Level: Medium
#   - You trust our Stage 1 processing
#   - You verify Stage 2+3 yourself
#
# Runtime: ~30 minutes on 8-core CPU
#

set -e  # Exit on error

echo "================================================================================"
echo "QFD SUPERNOVA ANALYSIS - REPRODUCE FROM FILTERED DATA"
echo "================================================================================"
echo ""
echo "This script reproduces QFD results using pre-computed Stage 1 data."
echo "For full replication from raw DES-SN5YR, use scripts/reproduce_from_raw.sh"
echo ""

# Check if pre-computed data exists
if [ ! -f "data/precomputed_filtered/stage1_results_filtered.csv" ]; then
    echo "ERROR: Pre-computed filtered data not found!"
    echo "Expected: data/precomputed_filtered/stage1_results_filtered.csv"
    echo ""
    echo "Please ensure the data directory is populated, or use:"
    echo "  bash scripts/reproduce_from_raw.sh"
    exit 1
fi

# Create timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/reproduction_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Stage 1: Already done (using pre-computed)
echo "Stage 1: Using pre-computed filtered results"
echo "  File: data/precomputed_filtered/stage1_results_filtered.csv"
N_SNE=$(tail -n +2 data/precomputed_filtered/stage1_results_filtered.csv | wc -l)
echo "  SNe: $N_SNE"
echo ""

# Stage 2: MCMC parameter fitting
echo "================================================================================"
echo "Stage 2: Running MCMC Parameter Fitting"
echo "================================================================================"
echo ""
echo "This will take ~15-20 minutes..."
echo "Parameters: k_J_correction, eta_prime, xi, sigma_ln_A"
echo "MCMC config: 32 walkers × 4,000 steps"
echo ""

python -m qfd_sn.stage2_mcmc \
    --input data/precomputed_filtered/stage1_results_filtered.csv \
    --output "$OUTPUT_DIR/stage2" \
    --nwalkers 32 \
    --nsteps 4000 \
    --nburn 1000

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Stage 2 failed!"
    exit 1
fi

echo ""
echo "✅ Stage 2 complete"
echo ""

# Stage 3: Hubble diagram
echo "================================================================================"
echo "Stage 3: Creating Hubble Diagram and Model Comparison"
echo "================================================================================"
echo ""
echo "Computing QFD and ΛCDM distance moduli..."
echo ""

python -m qfd_sn.stage3_hubble \
    --stage1 data/precomputed_filtered/stage1_results_filtered.csv \
    --stage2 "$OUTPUT_DIR/stage2" \
    --output "$OUTPUT_DIR/stage3"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Stage 3 failed!"
    exit 1
fi

echo ""
echo "✅ Stage 3 complete"
echo ""

# Summary
echo "================================================================================"
echo "REPRODUCTION COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - $OUTPUT_DIR/stage2/summary.json           (QFD parameters)"
echo "  - $OUTPUT_DIR/stage3/hubble_data.csv        (distance moduli)"
echo "  - $OUTPUT_DIR/stage3/summary.json           (fit statistics)"
echo ""

# Display results
if [ -f "$OUTPUT_DIR/stage3/summary.json" ]; then
    echo "Quick Results:"
    echo "-------------"
    python3 << EOF
import json
with open('$OUTPUT_DIR/stage3/summary.json') as f:
    data = json.load(f)

print(f"  N_SNe:       {data['n_sne']:,}")
print(f"  QFD RMS:     {data['statistics']['qfd_rms']:.3f} mag")
print(f"  ΛCDM RMS:    {data['statistics']['lcdm_rms']:.3f} mag")
print(f"  Improvement: {data['statistics']['improvement_percent']:.1f}%")
print(f"")
print(f"  QFD Parameters:")
print(f"    k_J = {data['qfd_parameters']['k_J_total']:.2f} km/s/Mpc")
print(f"    η'  = {data['qfd_parameters']['eta_prime']:.4f}")
print(f"    ξ   = {data['qfd_parameters']['xi']:.4f}")
EOF
fi

echo ""
echo "Next steps:"
echo "  - Review results: cat $OUTPUT_DIR/stage3/summary.json"
echo "  - Generate plots: python scripts/create_plots.py --input $OUTPUT_DIR"
echo "  - Compare to published: cat benchmarks/v21_benchmark/parameters.json"
echo ""
echo "For full transparency, run from raw data:"
echo "  bash scripts/reproduce_from_raw.sh"
echo ""
