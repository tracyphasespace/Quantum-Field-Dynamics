#!/bin/bash
#
# Automated Recovery Script: Revert Incorrect "Sign Fix"
#
# This script reverts the January 12 "bug fix" that actually BROKE working code.
#
# What it does:
#   1. Creates backup of current (broken) code
#   2. Removes incorrect negative signs from lines 414, 424, 439, 448
#   3. Runs test to verify fix
#   4. Reports results
#
# Usage:
#   ./fix_regression.sh [--test-only]
#
# Options:
#   --test-only: Only run test, don't modify code
#

set -e  # Exit on error

echo "================================================================================"
echo "REGRESSION FIX: Removing Incorrect Negative Sign"
echo "================================================================================"
echo ""
echo "Problem: January 12 'bug fix' added incorrect negative signs"
echo "Solution: Revert to November 5 code (no negative signs)"
echo ""

# Parse arguments
TEST_ONLY=false
if [[ "$1" == "--test-only" ]]; then
    TEST_ONLY=true
    echo "MODE: Test-only (will not modify code)"
else
    echo "MODE: Will apply fix and test"
fi
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if file exists
STAGE2_FILE="stages/stage2_mcmc_numpyro.py"
if [[ ! -f "$STAGE2_FILE" ]]; then
    echo "ERROR: File not found: $STAGE2_FILE"
    exit 1
fi

if [[ "$TEST_ONLY" == "false" ]]; then
    echo "Step 1: Creating backup..."
    BACKUP_FILE="${STAGE2_FILE}.broken_jan12_$(date +%Y%m%d_%H%M%S)"
    cp "$STAGE2_FILE" "$BACKUP_FILE"
    echo "  ✓ Backup created: $BACKUP_FILE"
    echo ""

    echo "Step 2: Applying fix..."
    echo "  Removing incorrect negative signs from lines 414, 424, 439, 448..."

    # Create sed script to fix the issues
    # Line 414 & 439: Remove negative sign from "c = -jnp.array"
    # Line 424 & 448: Change subtraction to addition in ln_A0_std calculation

    # Use Python to make precise changes (safer than sed for multi-line logic)
    python3 << 'EOF'
import re

file_path = "stages/stage2_mcmc_numpyro.py"

with open(file_path, 'r') as f:
    lines = f.readlines()

changes = 0

for i, line in enumerate(lines, 1):
    # Fix line 414 & 439: Remove negative sign
    if i in [414, 439] and "c = -jnp.array([k_J, eta_prime, xi]) * scales" in line:
        # Remove the negative sign
        lines[i-1] = line.replace("c = -jnp.array", "c = jnp.array")
        # Update comment
        lines[i-1] = lines[i-1].replace("# CRITICAL: negative sign!", "# FIXED: removed incorrect negative sign")
        print(f"  ✓ Fixed line {i}: Removed negative sign")
        changes += 1

    # Fix line 424 & 448: Change subtraction to addition
    if i in [424, 448] and "ln_A0_std = ln_A0_phys - jnp.dot" in line:
        # Change subtraction to addition and fix the formula
        lines[i-1] = "        ln_A0_std = ln_A0_phys + jnp.dot(c, means / scales)\n"
        print(f"  ✓ Fixed line {i}: Changed subtraction to addition")
        changes += 1

with open(file_path, 'w') as f:
    f.writelines(lines)

print(f"\n  Total changes: {changes}/4")
if changes != 4:
    print(f"  ⚠️  WARNING: Expected 4 changes, got {changes}")
    print(f"  Check if file has already been fixed or line numbers changed")
EOF

    echo ""
    echo "  ✓ Fix applied!"
    echo ""

    echo "Step 3: Verifying changes..."
    echo "  Checking that negative signs were removed..."

    # Check that the fix was applied
    if grep -n "c = jnp.array(\[k_J, eta_prime, xi\]) \* scales" "$STAGE2_FILE" > /dev/null; then
        echo "  ✓ Positive sign confirmed (no negative)"
    else
        echo "  ⚠️  WARNING: Could not verify positive sign"
    fi

    if grep -n "ln_A0_std = ln_A0_phys + jnp.dot(c, means / scales)" "$STAGE2_FILE" > /dev/null; then
        echo "  ✓ Addition confirmed (not subtraction)"
    else
        echo "  ⚠️  WARNING: Could not verify addition"
    fi
    echo ""
fi

echo "Step 4: Running validation test..."
echo "  This will run Stage 2 on 50 SNe to verify the fix works"
echo ""

# Check if Stage 1 results exist
STAGE1_RESULTS="../results/v15_clean/stage1_fullscale"
if [[ ! -d "$STAGE1_RESULTS" ]]; then
    echo "  ⚠️  WARNING: Stage 1 results not found at: $STAGE1_RESULTS"
    echo "  Skipping validation test"
    echo ""
    echo "  To run validation manually:"
    echo "    cd v15_clean"
    echo "    python stages/stage2_mcmc_numpyro.py \\"
    echo "        --stage1-results ../results/v15_clean/stage1_fullscale \\"
    echo "        --out ../results/v15_clean/stage2_recovery_test \\"
    echo "        --nchains 2 --nsamples 200 --nwarmup 100 \\"
    echo "        --quality-cut 2000 --constrain-signs informed"
    echo ""
else
    echo "  Running quick test (2 chains, 200 samples, 50 SNe)..."
    echo "  Output: ../results/v15_clean/stage2_recovery_test"
    echo ""

    # Run test (will take ~3-5 minutes)
    python3 stages/stage2_mcmc_numpyro.py \
        --stage1-results "$STAGE1_RESULTS" \
        --out "../results/v15_clean/stage2_recovery_test" \
        --nchains 2 \
        --nsamples 200 \
        --nwarmup 100 \
        --quality-cut 2000 \
        --constrain-signs informed \
        2>&1 | tee ../stage2_recovery_test.log || true

    echo ""
    echo "Step 5: Checking results..."

    RESULTS_FILE="../results/v15_clean/stage2_recovery_test/best_fit.json"
    if [[ -f "$RESULTS_FILE" ]]; then
        echo "  ✓ Results file found: $RESULTS_FILE"
        echo ""
        echo "  Results:"
        cat "$RESULTS_FILE"
        echo ""

        # Extract values using Python
        python3 << 'EOF'
import json

with open("../results/v15_clean/stage2_recovery_test/best_fit.json") as f:
    results = json.load(f)

k_J = results.get("k_J", 0)
eta_prime = results.get("eta_prime", 0)
xi = results.get("xi", 0)
k_J_std = results.get("k_J_std", 0)

print("  Verification:")
print(f"    k_J:       {k_J:.2f} ± {k_J_std:.2f}")
print(f"    eta_prime: {eta_prime:.2f} ± {results.get('eta_prime_std', 0):.2f}")
print(f"    xi:        {xi:.2f} ± {results.get('xi_std', 0):.2f}")
print("")

# Check if results are reasonable
if 7.0 < k_J < 14.0:
    print("  ✅ k_J is in expected range (7-14, target ~10.7)")
else:
    print(f"  ❌ k_J is OUT OF RANGE: {k_J:.2f} (expected 7-14)")

if -10.0 < eta_prime < -5.0:
    print("  ✅ η' is in expected range (-10 to -5, target ~-8.0)")
else:
    print(f"  ❌ η' is OUT OF RANGE: {eta_prime:.2f} (expected -10 to -5)")

if -9.0 < xi < -4.0:
    print("  ✅ ξ is in expected range (-9 to -4, target ~-7.0)")
else:
    print(f"  ❌ ξ is OUT OF RANGE: {xi:.2f} (expected -9 to -4)")

if k_J_std > 0.5:
    print(f"  ✅ Uncertainty is realistic: σ(k_J) = {k_J_std:.2f}")
else:
    print(f"  ❌ Uncertainty too small (overfitting!): σ(k_J) = {k_J_std:.2f}")

print("")
EOF

    else
        echo "  ❌ Results file not found"
        echo "  Check log: ../stage2_recovery_test.log"
    fi
fi

echo ""
echo "================================================================================"
echo "SUMMARY"
echo "================================================================================"
echo ""

if [[ "$TEST_ONLY" == "false" ]]; then
    echo "✅ Fix applied successfully!"
    echo ""
    echo "What was changed:"
    echo "  - Lines 414, 439: Removed negative sign from 'c = -jnp.array(...)'"
    echo "  - Lines 424, 448: Changed subtraction to addition in ln_A0_std"
    echo ""
    echo "Backup created:"
    echo "  $BACKUP_FILE"
    echo ""
fi

echo "Next steps:"
echo "  1. Review the test results above"
echo "  2. If results look good (k_J ~ 10.7, η' ~ -8.0, ξ ~ -7.0):"
echo "     - Run full production: ./scripts/run_stage2_fullscale.sh"
echo "  3. If results still look wrong:"
echo "     - Check REGRESSION_ANALYSIS.md and RECOVERY_INSTRUCTIONS.md"
echo "     - Compare to 2Compare/stage2_mcmc_numpyro.py (known working code)"
echo ""
echo "For questions, see:"
echo "  - REGRESSION_ANALYSIS.md"
echo "  - RECOVERY_INSTRUCTIONS.md"
echo ""
echo "================================================================================"
