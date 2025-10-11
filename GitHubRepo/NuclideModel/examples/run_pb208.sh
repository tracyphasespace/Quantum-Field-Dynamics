#!/bin/bash
# Example: Compute Pb-208 (heaviest stable doubly-magic nucleus)

python ../src/qfd_solver.py \
  --A 208 --Z 82 \
  --c-v2-base 2.201711 \
  --c-v2-iso 0.027035 \
  --c-v2-mass -0.000205 \
  --c-v4-base 5.282364 \
  --c-v4-size -0.085018 \
  --alpha-e-scale 1.007419 \
  --beta-e-scale 0.504312 \
  --c-sym 25.0 \
  --kappa-rho 0.029816 \
  --grid-points 48 \
  --iters-outer 360 \
  --emit-json \
  --out-json pb208_result.json

echo ""
echo "Results:"
jq '.E_model, .virial_abs, .physical_success' pb208_result.json

echo ""
echo "Expected: E_model ≈ -17900 MeV, virial < 0.10, error ≈ -8.4%"
echo "Note: Heavy isotope systematic underbinding present"
