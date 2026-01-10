#!/bin/bash
# Step 4: Run statistical experiments

echo "Running experiments..."

# Generate null model candidates
echo "  Generating null candidates..."
python3 src/null_models.py \
    --nubase data/derived/nubase_parsed.parquet \
    --output data/derived/candidates_by_A.parquet \
    --valley-width 0.25

# Run Experiment 1: Existence clustering
echo "  Running Experiment 1 (existence clustering)..."
python3 src/experiments/exp1_existence.py \
    --candidates data/derived/candidates_by_A.parquet \
    --params reports/fits/family_params_stable.json \
    --out reports/exp1/

# Run Tacoma Narrows test: Half-life correlation
echo "  Running Tacoma Narrows test (half-life correlation)..."
python3 src/experiments/tacoma_narrows_test.py \
    --scores data/derived/harmonic_scores.parquet \
    --out reports/tacoma_narrows/

echo "✓ Experiments complete!"
echo "  Results in reports/exp1/ and reports/tacoma_narrows/"
