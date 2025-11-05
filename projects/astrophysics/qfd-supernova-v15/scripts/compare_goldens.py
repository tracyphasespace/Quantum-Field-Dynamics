#!/usr/bin/env python3
"""
Golden snapshot test: Compare pipeline outputs against frozen reference.

Usage:
    python scripts/compare_goldens.py results/smoke10_stage3.csv goldens/smoke10_stage3.csv
    python scripts/compare_goldens.py results/summary_overall.csv goldens/summary_overall.csv

Exits with code 1 if any numeric column drifts > 5e-3 from golden.
"""
import sys
import pathlib
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Usage: compare_goldens.py <new_csv> <golden_csv>")
        sys.exit(2)

    new_path = pathlib.Path(sys.argv[1])
    gold_path = pathlib.Path(sys.argv[2])

    if not new_path.exists():
        print(f"❌ New file not found: {new_path}")
        sys.exit(2)

    if not gold_path.exists():
        print(f"❌ Golden file not found: {gold_path}")
        print(f"   If this is the first run, copy {new_path} to {gold_path}")
        sys.exit(2)

    try:
        new = pd.read_csv(new_path)
        gold = pd.read_csv(gold_path)
    except Exception as e:
        print(f"❌ Error reading CSVs: {e}")
        sys.exit(2)

    # Find common columns
    cols = set(new.columns) & set(gold.columns)
    if not cols:
        print(f"❌ No common columns between {new_path.name} and {gold_path.name}")
        sys.exit(1)

    errs = []
    TOLERANCE = 5e-3

    for c in sorted(cols):
        # Only compare numeric columns
        if new[c].dtype.kind in "fc" and gold[c].dtype.kind in "fc":
            # Compute mean absolute deviation
            d = np.nanmean(np.abs(new[c] - gold[c]))
            if d > TOLERANCE:
                errs.append((c, d))

    if errs:
        print(f"❌ Goldens drift detected in {new_path.name}:")
        for col, drift in errs:
            print(f"   - {col}: mean |Δ| = {drift:.6f} (> {TOLERANCE})")
        print()
        print(f"If this is intentional (e.g., model improvement), update golden:")
        print(f"   cp {new_path} {gold_path}")
        sys.exit(1)

    print(f"✓ Goldens OK: {new_path.name} matches {gold_path.name} (tolerance={TOLERANCE})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
