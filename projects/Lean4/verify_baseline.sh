#!/bin/bash
# Verify baseline modules still build successfully

echo "=== Baseline Verification: 9 Successfully Building Modules ==="
echo "Started: $(date)"
echo ""

modules=(
  "QFD.Lepton.KoideRelation"
  "QFD.Gravity.PerihelionShift"
  "QFD.Gravity.SnellLensing"
  "QFD.Electrodynamics.ProcaReal"
  "QFD.Cosmology.HubbleDrift"
  "QFD.GA.Cl33Instances"
  "QFD.Lepton.FineStructure"
  "QFD.Weak.ParityGeometry"
  "QFD.Cosmology.CosmicRestFrame"
)

pass=0
fail=0

for module in "${modules[@]}"; do
  echo "----------------------------------------"
  echo "Testing: $module"
  
  if lake build "$module" 2>&1 | tee "build_${module##*.}.log" | grep -q "error:"; then
    echo "❌ FAILED - Has errors"
    ((fail++))
  else
    # Check for sorry count
    sorry_count=$(grep -c "declaration uses 'sorry'" "build_${module##*.}.log" || echo "0")
    if [ "$sorry_count" -eq 0 ]; then
      echo "✅ SUCCESS - 0 sorries, 0 errors"
    else
      echo "⚠️  SUCCESS - $sorry_count sorries (may be documented)"
    fi
    ((pass++))
  fi
done

echo ""
echo "========================================"
echo "RESULTS: $pass passed, $fail failed"
echo "Completed: $(date)"
