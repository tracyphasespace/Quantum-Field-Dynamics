# Archive Manifest

**Archive:** harmonic_halflife_predictor_v1.0.0.zip  
**Date:** 2026-01-03  
**Author:** Tracy McSheery  
**License:** MIT  
**DOI:** [To be assigned by Zenodo]

---

## Contents

### Root Directory Files

| File | Size | Description |
|------|------|-------------|
| `README.md` | 11 KB | Main documentation and usage guide |
| `LICENSE` | 1 KB | MIT License text |
| `requirements.txt` | 59 B | Python dependencies (numpy, pandas, scipy, matplotlib) |
| `CITATION.cff` | 0.8 KB | Citation metadata in CFF format |
| `.zenodo.json` | 1.3 KB | Zenodo repository metadata |
| `MANIFEST.md` | This file | Complete archive contents listing |

### Documentation Files

| File | Size | Description |
|------|------|-------------|
| `SETUP.md` | 7.7 KB | Installation and setup instructions |
| `QUICK_START.txt` | 2.6 KB | Quick reference guide |
| `GITHUB_CHECKLIST.md` | 6.8 KB | Publication checklist |
| `PROCESS_AND_METHODS.md` | 31 KB | **Transparency document for scientific verification** |
| `NUCLEAR_SPECTROSCOPY_GUIDE.md` | 11 KB | Guide for creating yrast and spectral plots |
| `NEUTRON_DECAY_ANALYSIS.md` | 8.8 KB | Analysis of free neutron decay limitation |

### Python Scripts (`scripts/`)

| File | Lines | Description |
|------|-------|-------------|
| `nucleus_classifier.py` | 287 | 3-family harmonic classification algorithm |
| `predict_all_halflives.py` | 389 | Main prediction engine with regression models |
| `test_harmonic_vs_halflife.py` | 328 | Experimental validation against 47 isotopes |
| `analyze_all_decay_transitions.py` | 356 | Large-scale analysis of 4,878 transitions |
| `download_ame2020.py` | 45 | AME2020 data verification utility |

**Total:** 1,405 lines of Python code

### Data Files (`data/`)

| File | Rows | Size | Description |
|------|------|------|-------------|
| `ame2020_system_energies.csv` | 3,558 | 324 KB | AME2020 nuclear masses and binding energies |
| `harmonic_halflife_results.csv` | 47 | 6 KB | Experimental half-life calibration dataset |

**Source:** Atomic Mass Evaluation 2020 (Wang et al., Chinese Physics C 45, 030003, 2021)

### Results Files (`results/`)

| File | Rows | Size | Description |
|------|------|------|-------------|
| `predicted_halflives_all_isotopes.csv` | 3,530 | 400 KB | Complete predictions for all classified nuclei |
| `predicted_halflives_summary.md` | - | 8 KB | Statistical summary of predictions |
| `interesting_predictions.md` | - | 5 KB | Highlighted extreme and notable predictions |

### Technical Reports (`docs/`)

| File | Size | Description |
|------|------|-------------|
| `HALFLIFE_PREDICTION_REPORT.md` | 45 KB | Complete technical report |
| `BETA_PLUS_MODEL_FIX.md` | 12 KB | Beta+ regression model analysis |
| `harmonic_halflife_summary.md` | 15 KB | Experimental validation summary |

### Figures (`figures/`)

| File | Format | Description |
|------|--------|-------------|
| `halflife_prediction_validation.png` | PNG | 4-panel validation plot |
| `harmonic_halflife_analysis.png` | PNG | Experimental correlation analysis |
| `yrast_spectral_analysis.png` | PNG | 4-panel yrast and spectral diagrams |
| `nuclear_spectroscopy_complete.png` | PNG | 6-panel comprehensive spectroscopy |
| `yrast_comparison.png` | PNG | Traditional vs. harmonic yrast comparison |

**Total:** 5 publication-quality figures (~200 KB)

---

## Scientific Results Summary

### Coverage
- **Nuclei classified:** 3,530 / 3,557 (99.2%)
- **Decay transitions analyzed:** 4,878
- **Experimental validation isotopes:** 47 (Alpha: 24, Beta⁻: 15, Beta⁺: 8)

### Accuracy Metrics

| Decay Mode | Directional Accuracy | RMSE (log units) | Typical Error |
|------------|---------------------|------------------|---------------|
| **Alpha** | 56% mode preserved | 3.87 | ~10³× |
| **Beta⁻** | 99.7% (1494/1498) | 2.91 | ~10³× |
| **Beta⁺** | 83.6% (1331/1592) | 7.75 | ~10⁸× |

### Selection Rule Validation

| Transition Type | Count | Percentage | Relative Rate |
|----------------|-------|------------|---------------|
| **Allowed** (|ΔN| ≤ 1) | 2,396 | 75.3% | 1.00× |
| **Forbidden** (|ΔN| ≥ 2) | 785 | 24.7% | 0.18× (5.5× slower) |

### Key Parameters

**3-Family Model:**
- Family A: c2/c1 = 0.26 (volume-dominated), N ∈ {-3, +3}
- Family B: c2/c1 = 0.12 (surface-dominated), N ∈ {-3, +3}  
- Family C: c2/c1 = 0.20 (neutron-rich), N ∈ {4, +10}
- **Universal constant:** dc3 = -0.865 MeV

**Regression Models:**
- Alpha: log(t) = -24.14 + 67.05/√Q + 2.56·|ΔN|
- Beta⁻: log(t) = 9.35 - 0.63·log(Q) - 0.61·|ΔN|
- Beta⁺: log(t) = 11.39 - 23.12·log(Q) (simplified)

---

## Reproducibility

### System Requirements
- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0

### Installation
```bash
pip install -r requirements.txt
```

### Running Predictions
```bash
cd scripts
python predict_all_halflives.py
```

### Validation
```bash
python test_harmonic_vs_halflife.py
```

### Full Analysis
```bash
python analyze_all_decay_transitions.py
```

**Total runtime:** < 2 minutes on standard hardware

---

## Verification Checklist

For independent verification, see `PROCESS_AND_METHODS.md` which provides:

- [ ] AME2020 data matches official IAEA source
- [ ] Classification parameters not tuned for half-lives
- [ ] All 47 experimental isotopes used (no cherry-picking)
- [ ] No outlier removal in regression
- [ ] Q-values calculated using standard formulas
- [ ] All failures documented (Beta⁺, EC, neutron)
- [ ] Residuals reported for all isotopes
- [ ] Results reproducible with provided scripts

---

## Known Limitations

1. **Beta⁺ decay:** Only 8 calibration isotopes, all with |ΔN|=1 (zero variance)
2. **Electron capture:** Not modeled (isotopes like Fe-55 misclassified as stable)
3. **Free neutron decay:** Model fails (178,888× too slow) - designed for nuclear structure
4. **Long-lived actinides:** Underpredicted by 10⁵-10⁷ factors (U-235, Th-232)
5. **Proton emission:** Not included in current version

See `NEUTRON_DECAY_ANALYSIS.md` for detailed discussion of free particle limitation.

---

## Citation

### BibTeX
```bibtex
@software{mcsheery2026harmonic,
  author = {McSheery, Tracy},
  title = {Harmonic Half-Life Predictor: Nuclear Decay Prediction from Geometric Quantization},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/tracyphasespace/Quantum-Field-Dynamics},
  note = {Quantum Field Dynamics Project}
}
```

### APA
McSheery, T. (2026). *Harmonic Half-Life Predictor: Nuclear Decay Prediction from Geometric Quantization* (Version 1.0.0) [Computer software]. https://github.com/tracyphasespace/Quantum-Field-Dynamics

---

## Contact

**Author:** Tracy McSheery  
**Project:** Quantum Field Dynamics  
**Repository:** https://github.com/tracyphasespace/Quantum-Field-Dynamics  
**Issues:** https://github.com/tracyphasespace/Quantum-Field-Dynamics/issues

---

## License

MIT License - See `LICENSE` file for full text.

Copyright (c) 2026 Tracy McSheery (Quantum Field Dynamics Project)

---

**Archive created:** 2026-01-03  
**Total size:** ~2.1 MB (uncompressed)  
**Files:** 29 (8 documentation, 5 scripts, 2 data, 3 results, 3 technical reports, 5 figures, 3 metadata)  
**Lines of code:** 1,405 (Python)  
**Lines of documentation:** ~3,500 (Markdown)
