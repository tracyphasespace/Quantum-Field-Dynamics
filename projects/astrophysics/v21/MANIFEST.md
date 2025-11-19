# V21 Package Manifest

**Created:** 2025-01-18
**Version:** v21
**Purpose:** Standalone analysis package for QFD vs ΛCDM comparison
**Status:** Ready for GitHub publication

## Package Contents

### Analysis Scripts (2 files, 29K)
- `plot_canonical_comparison.py` (16K) - Generate canonical cosmology plots
- `analyze_bbh_candidates.py` (13K) - BBH forensics with Lomb-Scargle periodogram
- `run_analysis.sh` (2.2K) - Simple wrapper script to run analysis

### Data Files (3 files, 1.6M)
- `stage2_full_results.csv` (1.2M, 8,254 rows) - Stage 1 fit results for 8,253 SNe
- `lightcurves_sample.csv` (415K, 5,001 rows) - Sample lightcurves for 100 SNe
- `bbh_forensics_results.csv` (1.5K) - Periodogram results for Top 10 BBH candidates

### Result Plots (4 files, 2.4M)
- `time_dilation_test.png` (499K) - **KEY RESULT**: Stretch vs redshift falsification
- `canonical_comparison.png` (325K) - Hubble diagram with residuals
- `population_overview.png` (515K) - Stretch and residual distributions
- `lcdm_comparison.png` (1.1M) - Population-level ΛCDM comparison

### Documentation (6 files, 54K)
- `README.md` (5.7K) - Quick start guide for AI assistants
- `ANALYSIS_SUMMARY.md` (5.6K) - Main results and interpretation
- `QFD_PHYSICS.md` (17K) - Detailed physics of QFD model
- `FORENSICS_RESULTS.md` (4.4K) - BBH forensics analysis details
- `DATA_PROVENANCE.md` (9.1K) - Data source and processing history
- `LCDM_VS_QFD_TESTS.md` (12K) - Detailed comparison methodology
- `MANIFEST.md` (this file)

## Package Statistics

**Total files:** 16
**Total size:** ~4.1 MB
**JSON files:** 0 (confirmed clean)
**Dependencies:** Python 3.12+, NumPy, Pandas, Matplotlib, SciPy

## Package Structure

```
v21/
├── README.md                          # Start here
├── MANIFEST.md                        # This file
├── ANALYSIS_SUMMARY.md                # Main results
├── run_analysis.sh                    # Quick run script
│
├── plot_canonical_comparison.py       # Main analysis script
├── analyze_bbh_candidates.py          # Forensics script
│
├── stage2_full_results.csv            # Main dataset (8,253 SNe)
├── lightcurves_sample.csv             # Sample lightcurves
├── bbh_forensics_results.csv          # Forensics results
│
├── time_dilation_test.png             # KEY RESULT
├── canonical_comparison.png           # Hubble diagram
├── population_overview.png            # Population stats
├── lcdm_comparison.png                # LCDM comparison
│
├── QFD_PHYSICS.md                     # Model physics
├── FORENSICS_RESULTS.md               # Forensics details
├── DATA_PROVENANCE.md                 # Data source
└── LCDM_VS_QFD_TESTS.md               # Methodology
```

## Key Features

✓ **Standalone** - All files in single directory
✓ **No nested directories** - AI-navigable flat structure
✓ **No JSON files** - Clean, minimal package
✓ **Self-contained** - Doesn't reference external directories
✓ **AI-readable** - Clear documentation and instructions
✓ **Reproducible** - Includes sample data for testing

## Quick Start

```bash
# Clone and run
cd v21
chmod +x run_analysis.sh
./run_analysis.sh
```

Or run Python directly:
```bash
python3 plot_canonical_comparison.py
```

## Main Result

The time dilation test (`time_dilation_test.png`) shows:
- ΛCDM predicts: stretch s = 1 + z (rising line)
- Data shows: s ≈ 1.0 (flat line)
- **Conclusion: Data falsifies ΛCDM time dilation prediction**

## For AI Assistants

This package is designed to be easily navigable by AI assistants. Key files:
1. `README.md` - Start here for instructions
2. `ANALYSIS_SUMMARY.md` - Main results and interpretation
3. `time_dilation_test.png` - The key falsification result
4. `plot_canonical_comparison.py` - The analysis code

## Data Provenance

- Source: Dark Energy Survey 5-Year Supernova Sample (DES-SN5YR)
- Processing: V20 pipeline with memory-safe optimizations
- Stage 1: 8,253 SNe successfully fit (99.7% success rate)
- Stage 2: 202 BBH candidates identified

## License

Please cite the DES-SN5YR dataset and this analysis if used in publications.

## Contact

For questions, open an issue on GitHub.
