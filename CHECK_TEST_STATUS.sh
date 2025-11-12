#!/bin/bash
#
# Quick script to check the status of the recovery test
#
# Usage: ./CHECK_TEST_STATUS.sh
#

echo "================================================================================"
echo "RECOVERY TEST STATUS CHECK"
echo "================================================================================"
echo ""

# Check if test is still running
if ps aux | grep "[s]tage2_mcmc_numpyro.py" > /dev/null; then
    echo "Status: ⏳ TEST STILL RUNNING"
    echo ""
    echo "Process info:"
    ps aux | grep "[s]tage2_mcmc_numpyro.py"
    echo ""
    echo "To kill the test if needed:"
    echo "  pkill -f stage2_mcmc_numpyro.py"
    echo ""
else
    echo "Status: ✅ TEST COMPLETED (or not running)"
    echo ""
fi

# Check if results exist
RESULTS_FILE="../results/v15_clean/stage2_recovery_test/best_fit.json"
if [[ -f "$RESULTS_FILE" ]]; then
    echo "================================================================================"
    echo "RESULTS AVAILABLE!"
    echo "================================================================================"
    echo ""
    cat "$RESULTS_FILE"
    echo ""
    echo "================================================================================"
    echo "VERIFICATION"
    echo "================================================================================"
    echo ""

    # Parse and check results using Python
    python3 << 'EOF'
import json
import sys

try:
    with open("../results/v15_clean/stage2_recovery_test/best_fit.json") as f:
        results = json.load(f)

    k_J = results.get("k_J", 0)
    eta_prime = results.get("eta_prime", 0)
    xi = results.get("xi", 0)
    k_J_std = results.get("k_J_std", 0)
    eta_prime_std = results.get("eta_prime_std", 0)
    xi_std = results.get("xi_std", 0)

    print(f"Parameters:")
    print(f"  k_J:       {k_J:8.2f} ± {k_J_std:.2f}")
    print(f"  eta_prime: {eta_prime:8.2f} ± {eta_prime_std:.2f}")
    print(f"  xi:        {xi:8.2f} ± {xi_std:.2f}")
    print()

    # Expected ranges (from November 5 working results)
    # k_J: 10.77 ± 4.57
    # eta': -7.99 ± 1.44
    # xi: -6.91 ± 3.75

    all_good = True

    # Check k_J
    if 7.0 < k_J < 14.0:
        print("✅ k_J is in expected range (7-14, target ~10.7)")
    else:
        print(f"❌ k_J OUT OF RANGE: {k_J:.2f} (expected 7-14)")
        all_good = False

    # Check eta_prime
    if -10.0 < eta_prime < -5.0:
        print("✅ eta_prime is in expected range (-10 to -5, target ~-8.0)")
    else:
        print(f"❌ eta_prime OUT OF RANGE: {eta_prime:.2f} (expected -10 to -5)")
        all_good = False

    # Check xi
    if -9.0 < xi < -4.0:
        print("✅ xi is in expected range (-9 to -4, target ~-7.0)")
    else:
        print(f"❌ xi OUT OF RANGE: {xi:.2f} (expected -9 to -4)")
        all_good = False

    # Check uncertainties (should be realistic, not tiny)
    if k_J_std > 0.5:
        print(f"✅ Uncertainties are realistic: σ(k_J) = {k_J_std:.2f}")
    else:
        print(f"❌ Uncertainty too small (overfitting!): σ(k_J) = {k_J_std:.4f}")
        all_good = False

    print()
    if all_good:
        print("=" * 80)
        print("🎉 SUCCESS! All parameters in expected range!")
        print("=" * 80)
        print()
        print("The fix worked! Results match November 5 working version.")
        print()
        print("Next steps:")
        print("  1. Review the full output above")
        print("  2. Run full production: ./scripts/run_stage2_fullscale.sh")
        print("  3. Then run Stage 3: ./scripts/run_stage3.sh")
        print()
    else:
        print("=" * 80)
        print("⚠️  SOME PARAMETERS OUT OF RANGE")
        print("=" * 80)
        print()
        print("This might be due to:")
        print("  - Small sample size (200 samples might not be enough)")
        print("  - Need to run full production (4000+ samples)")
        print("  - Random sampling variation")
        print()
        print("Recommend: Run full production pipeline anyway")
        print("  ./scripts/run_stage2_fullscale.sh")
        print()

except FileNotFoundError:
    print("ERROR: Results file not found")
    print(f"  Expected at: {RESULTS_FILE}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR parsing results: {e}")
    sys.exit(1)
EOF

else
    echo "Results file not found yet: $RESULTS_FILE"
    echo ""
    echo "The test is either:"
    echo "  - Still running (check status above)"
    echo "  - Failed (check log files)"
    echo "  - Not started yet"
    echo ""
    echo "To check logs:"
    echo "  tail -100 ../stage2_recovery_test.log 2>/dev/null || echo 'No log file'"
    echo ""
fi

echo "================================================================================"
echo "COMPARISON TO NOVEMBER 5 (WORKING VERSION)"
echo "================================================================================"
echo ""
echo "November 5 results (from abc_comparison_20251105_165123):"
cat ../results/abc_comparison_20251105_165123/A_unconstrained/best_fit.json 2>/dev/null || echo "  (File not found)"
echo ""

echo "================================================================================"
echo "For more details, see:"
echo "  - PROBLEM_SOLVED.md"
echo "  - REGRESSION_ANALYSIS.md"
echo "  - RECOVERY_INSTRUCTIONS.md"
echo "================================================================================"
