#!/bin/bash
# Run all solvers and save output

echo "Running QFD Nuclear Mass Solvers"
echo "================================"
echo ""

echo "1. Single nucleus solver (He-4 calibration)..."
python3 src/qfd_metric_solver.py > results/single_nucleus_output.txt
echo "   Output saved to: results/single_nucleus_output.txt"
echo ""

echo "2. Alpha cluster predictions..."
python3 src/alpha_cluster_solver.py > results/alpha_ladder_output.txt
echo "   Output saved to: results/alpha_ladder_output.txt"
echo ""

echo "Done! Check results/ directory for output files."
