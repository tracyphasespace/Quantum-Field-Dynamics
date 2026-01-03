#!/bin/bash
# Step 2: Fit harmonic families to stable nuclides

echo "Fitting harmonic families..."

python3 src/fit_families.py \
    --input data/derived/nubase_parsed.parquet \
    --output reports/fits/family_params_stable.json \
    --n-families 3 \
    --subset stable

echo "✓ Family parameters saved to reports/fits/family_params_stable.json"
