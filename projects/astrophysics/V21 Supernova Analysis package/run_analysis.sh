#!/bin/bash
# Run V21 Canonical Analysis
# This script demonstrates how to run the QFD vs LCDM comparison

set -e  # Exit on error

echo "======================================================================"
echo "V21 QFD vs ΛCDM Canonical Comparison"
echo "======================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# 1. Generate canonical comparison plots
echo "----------------------------------------------------------------------"
echo "Step 1: Generating Canonical Comparison Plots"
echo "----------------------------------------------------------------------"
echo ""
echo "This will create:"
echo "  - canonical_comparison.png (Hubble diagram)"
echo "  - time_dilation_test.png (stretch vs redshift)"
echo ""

python3 plot_canonical_comparison.py

echo ""
echo "✓ Canonical plots generated successfully"
echo ""

# 2. Optional: Run BBH forensics analysis
echo "----------------------------------------------------------------------"
echo "Step 2: BBH Forensics Analysis (Optional)"
echo "----------------------------------------------------------------------"
echo ""
echo "Note: This requires full lightcurves data, not just the sample."
echo "To run with sample data:"
echo ""
echo "  python3 analyze_bbh_candidates.py \\"
echo "    --stage2-results stage2_full_results.csv \\"
echo "    --lightcurves lightcurves_sample.csv \\"
echo "    --out . \\"
echo "    --top-n 10"
echo ""
echo "Skipping BBH analysis (requires full dataset)"
echo ""

# Summary
echo "======================================================================"
echo "Analysis Complete!"
echo "======================================================================"
echo ""
echo "Generated files:"
ls -lh *.png 2>/dev/null || echo "  (no new plots generated)"
echo ""
echo "View the key result:"
echo "  time_dilation_test.png - Shows data DOES NOT follow s = 1+z"
echo ""
echo "Documentation:"
echo "  README.md - Quick start guide"
echo "  ANALYSIS_SUMMARY.md - Detailed results"
echo "  QFD_PHYSICS.md - Model physics"
echo ""
