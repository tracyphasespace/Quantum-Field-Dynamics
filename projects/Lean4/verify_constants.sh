#!/bin/bash
# Constant Validation Script
# Checks for contaminated constant definitions in Lean files

echo "=== QFD Constant Validation Script ==="
echo "Checking for contaminated constant definitions..."
echo ""

# Track errors
ERRORS=0

# Check 1: Look for WRONG alpha_circ definition (1/(2π) instead of e/(2π))
echo "[1/4] Checking for contaminated alpha_circ definitions..."
WRONG_ALPHA=$(grep -r "def alpha_circ.*:=.*1 / (2 \* Real.pi)" QFD/ --include="*.lean" 2>/dev/null | grep -v "Real.exp 1")
if [ ! -z "$WRONG_ALPHA" ]; then
    echo "❌ ERROR: Found contaminated alpha_circ definition (1/(2π) instead of e/(2π)):"
    echo "$WRONG_ALPHA"
    echo ""
    ERRORS=$((ERRORS + 1))
else
    echo "✅ No contaminated alpha_circ definitions found"
fi

# Check 2: Look for hardcoded alpha_circ instead of import
echo ""
echo "[2/4] Checking for hardcoded alpha_circ (should import from VacuumParameters)..."
HARDCODED=$(grep -r "def alpha_circ.*:=.*Real.exp 1" QFD/Lepton --include="*.lean" 2>/dev/null | grep -v "VacuumParameters.lean")
if [ ! -z "$HARDCODED" ]; then
    echo "⚠️  WARNING: Found hardcoded alpha_circ (should import QFD.Vacuum.alpha_circ):"
    echo "$HARDCODED"
    echo ""
else
    echo "✅ All alpha_circ definitions properly import from VacuumParameters"
fi

# Check 3: Verify VacuumParameters.lean has correct definition
echo ""
echo "[3/4] Verifying VacuumParameters.lean has correct alpha_circ..."
CORRECT_DEF=$(grep "noncomputable def alpha_circ.*:=.*Real.exp 1 / (2 \* Real.pi)" QFD/Vacuum/VacuumParameters.lean 2>/dev/null)
if [ -z "$CORRECT_DEF" ]; then
    echo "❌ ERROR: VacuumParameters.lean missing correct alpha_circ definition!"
    echo "Expected: noncomputable def alpha_circ : ℝ := Real.exp 1 / (2 * Real.pi)"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ VacuumParameters.lean has correct definition"
fi

# Check 4: Verify files using alpha_circ import VacuumParameters
echo ""
echo "[4/4] Checking that files using alpha_circ import VacuumParameters..."
FILES_WITH_ALPHA=$(grep -l "alpha_circ" QFD/Lepton/*.lean 2>/dev/null | grep -v "VacuumParameters.lean")
MISSING_IMPORTS=""
for file in $FILES_WITH_ALPHA; do
    if ! grep -q "import QFD.Vacuum.VacuumParameters" "$file" 2>/dev/null; then
        MISSING_IMPORTS="$MISSING_IMPORTS\n  $file"
    fi
done

if [ ! -z "$MISSING_IMPORTS" ]; then
    echo "⚠️  WARNING: Files using alpha_circ without importing VacuumParameters:"
    echo -e "$MISSING_IMPORTS"
    echo ""
else
    echo "✅ All files using alpha_circ properly import VacuumParameters"
fi

# Summary
echo ""
echo "=== Validation Summary ==="
if [ $ERRORS -eq 0 ]; then
    echo "✅ PASSED: No critical errors found"
    echo ""
    echo "Next steps:"
    echo "  1. Review any warnings above"
    echo "  2. Run: lake build QFD.Vacuum.VacuumParameters QFD.Lepton.AnomalousMoment"
    echo "  3. If build fails, check CRITICAL_CONSTANTS.md for validation protocol"
    exit 0
else
    echo "❌ FAILED: $ERRORS critical error(s) found"
    echo ""
    echo "Action required:"
    echo "  1. Read CRITICAL_CONSTANTS.md for correct values"
    echo "  2. Fix contaminated definitions above"
    echo "  3. Run this script again to verify"
    echo "  4. Run: lake build to verify Lean compiles"
    exit 1
fi
