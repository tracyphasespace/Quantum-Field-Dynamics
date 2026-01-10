#!/bin/bash
# Step 3: Score all nuclides with harmonic model

echo "Scoring all nuclides..."

python3 src/score_harmonics.py \
    --nubase data/derived/nubase_parsed.parquet \
    --params reports/fits/family_params_stable.json \
    --output data/derived/harmonic_scores.parquet

echo "✓ Scored nuclides saved to data/derived/harmonic_scores.parquet"
