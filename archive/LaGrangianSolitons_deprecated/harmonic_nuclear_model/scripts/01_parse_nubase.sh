#!/bin/bash
# Step 1: Parse NUBASE2020 data

echo "Parsing NUBASE2020 data..."

python3 src/parse_nubase.py \
    --input data/raw/nubase_1.mas20.txt \
    --output data/derived/nubase_parsed.parquet

echo "✓ Parsed NUBASE data saved to data/derived/nubase_parsed.parquet"
