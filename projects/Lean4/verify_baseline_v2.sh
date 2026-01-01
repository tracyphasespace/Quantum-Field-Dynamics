#!/bin/bash
echo "=== Baseline Verification (Corrected) ==="

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

for module in "${modules[@]}"; do
  echo "Testing: $module"
  
  result=$(lake build "$module" 2>&1)
  
  # Check for actual build failure (not linter warnings)
  if echo "$result" | grep -q "^error:.*build failed"; then
    echo "  ❌ BUILD FAILED"
  elif echo "$result" | grep -q "Build completed successfully"; then
    # Count sorries
    sorry_count=$(echo "$result" | grep -c "declaration uses 'sorry'" || echo "0")
    echo "  ✅ Builds - $sorry_count sorries"
  else
    echo "  ⚠️  UNKNOWN - check manually"
  fi
done
