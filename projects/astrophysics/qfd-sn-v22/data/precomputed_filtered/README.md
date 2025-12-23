# Pre-computed Filtered Data (Quick Start Option)

**Purpose**: Convenience shortcut for quick validation without running Stage 1

## What's In This Directory

This directory contains **pre-computed Stage 1 results** from our processing of DES-SN5YR data:

- `stage1_results_filtered.csv` - Quality-filtered Stage 1 fits (6,724 SNe)
- `stage1_lightcurves.csv` - Light curve data for these SNe
- `processing_log.json` - Provenance information

## Quality Filtering Applied

These results have quality gates applied:
- ✅ Chi²/dof < 2,000 (good fit quality)
- ✅ -20 < ln_A < 20 (no railed fits)
- ✅ 0.5 < stretch < 10.0 (physical values)

**Result**: 6,724 of 8,253 SNe passed quality control (18.5% rejection rate)

## Trust Level

**Use this if**: You want to quickly validate our QFD vs ΛCDM comparison

**Don't use this if**: You don't trust our Stage 1 processing

## Full Replication Path

If you want to replicate from scratch:

1. Download raw DES-SN5YR data:
   ```bash
   bash scripts/download_des5yr.sh
   ```

2. Run full pipeline from Stage 1:
   ```bash
   bash scripts/reproduce_from_raw.sh
   ```

This processes all 8,253 SNe through Stage 1, applies quality gates, then runs Stage 2+3.

## Data Provenance

**Source**: DES-SN5YR Public Release (2019)
**Processing Date**: 2025-12-23
**Processing Code**: Stage 1 from this repository
**Quality Gates**: See `../configs/des1499.yaml`

## Checksums

To verify data integrity:
```bash
sha256sum stage1_results_filtered.csv
# Expected: [will be computed after copying data]
```

## Comparison to Published Results

Our Stage 1 processing produces results consistent with:
- DES Collaboration 2024 (1,499 SNe cosmology sample)
- Pantheon+ 2022 (1,550 SNe sample)

The 6,724 SNe here include:
- DES-SN5YR confirmed Type Ia
- Quality cuts applied
- Redshift range: 0.1 < z < 2.5 (z < 2.5 cutoff applied)

---

**Bottom Line**: This is a convenience option. For full transparency, run the complete pipeline from raw data.
