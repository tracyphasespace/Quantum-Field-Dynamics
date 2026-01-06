#!/bin/bash
#
# Master Test Runner for V22 Hill Vortex Hardening Tests
# =======================================================
#
# Runs all validation tests in sequence and generates summary report.
#
# USAGE: ./run_all_hardening_tests.sh
#
# EXPECTED RUNTIME: ~20-30 minutes total
#   - Test 1 (Grid Convergence):       ~5-10 min
#   - Test 2 (Multi-Start Robustness): ~10-15 min
#   - Test 3 (Profile Sensitivity):    ~5 min
#

set -e  # Exit on first error

echo "================================================================================"
echo "V22 HILL VORTEX HARDENING TEST SUITE"
echo "================================================================================"
echo ""
echo "This suite runs three critical validation tests:"
echo "  1. Grid Convergence      - Verify numerical stability"
echo "  2. Multi-Start Robustness - Test solution uniqueness"
echo "  3. Profile Sensitivity    - Test β robustness"
echo ""
echo "Total expected runtime: ~20-30 minutes"
echo ""
read -p "Press Enter to start..."

# Create results directory
mkdir -p results

# Test 1: Grid Convergence
echo ""
echo "================================================================================"
echo "TEST 1/3: GRID CONVERGENCE"
echo "================================================================================"
echo ""
python3 test_01_grid_convergence.py
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Test 1 completed successfully"
else
    echo ""
    echo "✗ Test 1 failed"
    exit 1
fi

# Test 2: Multi-Start Robustness
echo ""
echo "================================================================================"
echo "TEST 2/3: MULTI-START ROBUSTNESS"
echo "================================================================================"
echo ""
python3 test_02_multistart_robustness.py
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Test 2 completed successfully"
else
    echo ""
    echo "✗ Test 2 failed"
    exit 1
fi

# Test 3: Profile Sensitivity
echo ""
echo "================================================================================"
echo "TEST 3/3: PROFILE SENSITIVITY"
echo "================================================================================"
echo ""
python3 test_03_profile_sensitivity.py
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Test 3 completed successfully"
else
    echo ""
    echo "✗ Test 3 failed"
    exit 1
fi

# Generate summary report
echo ""
echo "================================================================================"
echo "GENERATING SUMMARY REPORT"
echo "================================================================================"
echo ""
python3 generate_summary_report.py

echo ""
echo "================================================================================"
echo "ALL TESTS COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved in: ./results/"
echo ""
echo "Key files:"
echo "  - grid_convergence_results.json"
echo "  - multistart_robustness_results.json"
echo "  - profile_sensitivity_results.json"
echo "  - hardening_tests_summary.txt"
echo ""
echo "Review the summary report to assess publication readiness."
