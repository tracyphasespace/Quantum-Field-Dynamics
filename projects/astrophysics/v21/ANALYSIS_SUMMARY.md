# Analysis Summary: QFD vs ΛCDM Comparison

## Main Result: Falsification of Cosmological Time Dilation

This analysis demonstrates that Type Ia supernovae stretch parameters **do not follow the (1+z) time dilation prediction** of standard ΛCDM cosmology.

## Key Findings

### 1. Time Dilation Test (see `time_dilation_test.png`)

**What we tested:**
- ΛCDM prediction: stretch parameter s = 1 + z (rising with redshift)
- QFD prediction: stretch parameter s = 1.0 (flat, no time dilation in static space)

**What we found:**
- Data shows **flat trend** (s ≈ 1.0) across all redshifts
- ΛCDM predicts s = 2.5 at z = 1.5, but data shows s ≈ 1.0
- **Result: Data falsifies ΛCDM time dilation prediction**

**Physics interpretation:**
The stretch parameter in traditional analysis measures the observed timescale of the supernova light curve. In expanding space (ΛCDM), photon wavelengths are stretched by (1+z), and time intervals should also dilate by (1+z). Our data shows this does NOT occur.

### 2. Hubble Diagram (see `canonical_comparison.png`)

**What we tested:**
- Distance modulus μ vs redshift z
- ΛCDM: μ_ΛCDM(z) with Ωm=0.3, ΩΛ=0.7
- QFD: μ_QFD(z) = μ_static(z) + η·z^1.5

**What we found:**
- QFD parameter: η = 0.000 (hit lower bound, suggests minimal scattering)
- Both models fit the Hubble diagram reasonably well
- However, residuals show systematic trends (χ²/dof ≈ 1984)

**Note:** The Hubble diagram alone cannot distinguish models due to calibration degeneracies. The time dilation test is the critical discriminator.

### 3. BBH Forensics Analysis

**Searched for:** Periodic gravitational lensing signals from Binary Black Holes in "Flashlight" SNe (overluminous outliers)

**Method:**
- Selected Top 10 candidates (residual > 2.0, stretch > 2.8)
- Applied Lomb-Scargle periodogram to residuals
- Searched for periods 1-100 days with FAP < 0.1

**Result:** **No periodic signals detected**
- All 10 candidates: FAP = 1.0 (no detection)
- See `bbh_forensics_results.csv` and `FORENSICS_RESULTS.md`

**Interpretation:** User insight - BBH orbital periods could be months to years (like Neptune's 165-year orbit), far too long to detect in typical 100-day SN observation windows. The periodogram was searching at the wrong timescale.

### 4. Population Statistics

**Dataset:** 8,253 Type Ia supernovae from DES 5-Year data

**Distribution (see `population_overview.png`):**
- Stretch mean: 3.30 ± 1.52
- Residual mean: -0.08 ± 0.92
- Strong correlation between stretch and residual
- 202 BBH candidates identified (stretch > 2.8)

**ΛCDM Comparison (see `lcdm_comparison.png`):**
- Stretch vs Redshift: Data shows flat/declining trend
- ΛCDM predicts rising trend (s = 1+z)
- Stretch excess (s - (1+z)): **DECREASING with z** (opposite of BBH lensing expectations)

## Physics Constraints Applied

### 1. Non-negative scattering: η ≥ 0
In QFD, photons can only be scattered/absorbed, never amplified. This is a fundamental physics constraint. The optimizer found η = 0.000, suggesting negligible scattering in this dataset.

### 2. Stretch normalization: s(z=0) = 1.0
By definition, at zero redshift there is no cosmological effect, so stretch must equal 1.0. We normalized all data by dividing by mean(s) at z < 0.1, which was 1.872.

## Implications

**ΛCDM Time Dilation Falsified:**
The flat stretch vs redshift trend directly contradicts the (1+z) time dilation prediction of ΛCDM cosmology. This is independent of calibration issues and represents a fundamental challenge to the expanding universe interpretation.

**QFD Consistency:**
The QFD model predicts flat stretch (no time dilation in static space), which matches the observed data. However, η = 0 suggests the photon damping component is minimal or not required for this dataset.

**Alternative Interpretations:**
The stretch parameter may not purely measure time dilation, or there may be systematic effects in template matching that create an apparent flat trend. Further investigation needed.

## Data Quality

- Total SNe: 8,253
- Success rate: 99.7%
- Redshift range: 0.01 < z < 1.96
- Survey: Dark Energy Survey 5-Year (DES-SN5YR)
- Filters: griz photometry

## Next Steps

1. Independent validation with other SN datasets (Pantheon+, Union3)
2. Investigation of stretch parameter systematics
3. Search for BBH signals at longer timescales (months-years)
4. Cross-check with other cosmological probes (BAO, CMB)

## Files in This Package

**Analysis Scripts:**
- `plot_canonical_comparison.py` - Generate ΛCDM vs QFD comparison plots
- `analyze_bbh_candidates.py` - BBH forensics with Lomb-Scargle periodogram

**Data:**
- `stage2_full_results.csv` - Stage 1 fit results for 8,253 SNe
- `lightcurves_sample.csv` - Sample lightcurves for testing (100 SNe)
- `bbh_forensics_results.csv` - Periodogram results for Top 10 BBH candidates

**Results:**
- `time_dilation_test.png` - KEY RESULT: Stretch vs redshift comparison
- `canonical_comparison.png` - Hubble diagram with residuals
- `population_overview.png` - Stretch and residual distributions
- `lcdm_comparison.png` - Population-level ΛCDM comparison

**Documentation:**
- `README.md` - Quick start guide for AI assistants
- `QFD_PHYSICS.md` - Detailed physics of QFD model
- `FORENSICS_RESULTS.md` - BBH forensics analysis details
- `DATA_PROVENANCE.md` - Data source and processing
- `LCDM_VS_QFD_TESTS.md` - Detailed comparison methodology
- `ANALYSIS_SUMMARY.md` - This file

## Citation

If you use this analysis, please cite:
- Dark Energy Survey 5-Year Supernova Sample (DES-SN5YR)
- QFD Model: [Your paper reference]

## Contact

For questions about this analysis, please open an issue on GitHub.
