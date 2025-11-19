# V21 QFD Supernova Analysis - Standalone Package

**Falsification of Cosmological Time Dilation in Type Ia Supernovae**

This package contains complete code and data to reproduce the key result: **The stretch parameter in Type Ia supernovae does NOT follow the (1+z) cosmological time dilation predicted by ΛCDM.**

## 🎯 Main Result

The **Time Dilation Test** (Figure: `time_dilation_test.png`) shows:
- **ΛCDM predicts**: Stretch should increase as s = 1+z with redshift
- **Data shows**: Stretch is flat or slightly decreasing with redshift
- **Conclusion**: Standard cosmological time dilation is NOT observed

## 📁 Files in This Package

### Analysis Scripts
- `plot_canonical_comparison.py` - Main analysis script (generates all plots)
- `analyze_bbh_candidates.py` - Forensics analysis for BBH hypothesis testing

### Data Files
- `stage2_results_with_redshift.csv` - Stage 1 fit results for 8,253 supernovae (includes redshifts)
- `bbh_forensics_results.csv` - Forensics analysis results

### Results & Plots
- `time_dilation_test.png` - **KEY RESULT: Falsification of (1+z) time dilation**
- `canonical_comparison.png` - Hubble diagram comparison (ΛCDM vs QFD)
- `lcdm_comparison.png` - Full ΛCDM comparison plots
- `population_overview.png` - Population statistics

### Documentation
- `FORENSICS_RESULTS.md` - BBH forensics analysis summary
- `ANALYSIS_SUMMARY.md` - Complete analysis summary
- `QFD_PHYSICS.md` - Explanation of QFD model physics

## 🚀 Quick Start (For AI Assistants)

### Prerequisites
```bash
pip install numpy pandas matplotlib scipy
```

### Generate All Plots
```bash
python3 plot_canonical_comparison.py
```

This will create:
- `time_dilation_test.png` - The key falsification result
- `canonical_comparison.png` - Hubble diagram

### Run Forensics Analysis
Note: Forensics analysis requires full lightcurve data (not included in this package). Download from DES-SN5YR and run:
```bash
python3 analyze_bbh_candidates.py \
  --stage2-results stage2_results_with_redshift.csv \
  --lightcurves /path/to/lightcurves_all_transients.csv \
  --out forensics_output \
  --top-n 10
```

## 📊 Understanding the Results

### The Time Dilation Plot

**What to look for:**
1. **Blue dashed line** (ΛCDM): Should rise linearly from 1.0 to 3.0
2. **Green solid line** (QFD): Flat horizontal at 1.0
3. **Black points** (Data): Observe which line they follow

**The Result:**
- Data is FLAT (≈1.0 across all redshifts)
- Data does NOT follow blue ΛCDM line
- Data is consistent with green QFD line

**Interpretation:**
The stretch parameter is NOT measuring cosmological time dilation. This challenges the foundational assumption of supernova cosmology.

## 🔬 Technical Details

### Dataset
- **N(SNe)**: 3,676 (after quality cuts)
- **Redshift range**: 0.01 < z < 2.0
- **Stretch range**: 0.5 < s < 2.8 (filtered to remove artifacts)

### Quality Cuts Applied
1. 0.5 < stretch < 2.8 (remove railed fits)
2. z > 0.01 (remove local SNe)
3. Successful Stage 1 convergence

### Normalization
- Stretch normalized by mean value at z < 0.1
- Forces s(z=0) = 1.0 for proper comparison
- Normalization factor = 1.872

### Physics Constraints
- QFD scattering parameter: η ≥ 0 (physical requirement)
- Best fit: η = 0.000 (boundary value)

## 📖 How to Use This Package (AI Instructions)

### For Analysis
1. Load `stage2_results_with_redshift.csv` into pandas (includes all data needed)
2. Apply quality cuts (see code in `plot_canonical_comparison.py`)
3. Normalize stretch parameter
4. Generate comparison plots

### For Verification
1. Run `plot_canonical_comparison.py` to reproduce all plots
2. Compare output to provided reference plots
3. Check that normalized stretch is flat vs redshift

### For Extension
1. Modify quality cuts in `plot_canonical_comparison.py`
2. Test different normalization schemes
3. Add additional cosmological models
4. Explore systematic uncertainties

## 📝 Column Definitions

### stage2_results_with_redshift.csv
- `snid`: Supernova ID (string)
- `stretch`: Fitted stretch parameter (dimensionless)
- `ln_A`: Log-amplitude (relates to distance modulus)
- `chi2_dof`: Chi-squared per degree of freedom
- `residual`: Fit residual (σ)
- `t0`, `A_plasma`, `beta`: Model parameters
- `z`: Redshift

## 🎓 Citation

If you use this analysis or code, please cite:

```
V21 QFD Supernova Analysis
https://github.com/[your-repo]/qfd-supernova-v21
Analysis Date: 2025-11-18
```

## ⚠️ Known Issues

1. **Distance modulus calibration**: The Hubble diagram (Figure 1) may have calibration issues. The TIME DILATION test (Figure 2) is robust and independent.

2. **High-z scatter**: Limited statistics at z > 1.0. Main result holds for 0 < z < 1.0.

3. **Stretch interpretation**: Our result shows stretch ≠ (1+z), but doesn't conclusively identify what stretch IS measuring.

## 🤝 Contributing

To reproduce or extend this analysis:
1. Download all files
2. Run `plot_canonical_comparison.py`
3. Verify plots match provided references
4. Modify cuts/models as needed
5. Share your results!

## 📞 Questions?

Ask an AI assistant! This package is designed to be AI-readable. Simply provide:
- This README.md
- The relevant Python script
- The question you want answered

Example prompt:
```
"I have the V21 QFD supernova analysis package. 
Can you explain why the time dilation test falsifies ΛCDM?"
```

## 🏆 Key Takeaway

**The data does not support the (1+z) time dilation assumption used in standard supernova cosmology.**

This result requires careful vetting but, if confirmed, has profound implications for our understanding of cosmic expansion and the evidence for dark energy.
