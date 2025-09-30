# QFD Supernova Analysis Package

**Comprehensive framework for testing Quantum Field Dynamics (QFD) theory using supernova observations**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This package provides the first comprehensive, multi-observable framework for testing QFD theory against supernova cosmology data. It implements a novel two-script synergistic approach that tests all three stages of QFD physics:

- **Stage 1**: Plasma veil effects (time-wavelength dependent)
- **Stage 2**: Field Depletion Region (FDR) dimming
- **Stage 3**: Cosmological drag modifications

## 🚀 Key Features

### Complete QFD Physics Testing
- **Cosmological Analysis**: μ-z Hubble diagram fitting with QFD vs ΛCDM comparison
- **Light Curve Analysis**: Direct plasma veil mechanism validation
- **Multi-Observable**: Cross-correlation between distance and time-flux measurements

### Production-Ready Pipeline
- **Vectorized Models**: High-performance numerical implementations
- **Realistic Uncertainties**: Proper error modeling for supernova datasets
- **Statistical Rigor**: AIC/BIC model comparison, cross-validation
- **Publication Quality**: Comprehensive diagnostics and visualization

### Scientific Validation Framework
- **Model-Neutral**: Fair comparison between QFD and standard models
- **Falsifiable Predictions**: Direct tests of QFD's unique predictions
- **Multi-Survey Support**: OSC, SDSS, DES data ingestion

## 📊 Scientific Results Summary

### μ-z Analysis (Union2.1 Dataset)
- **580 supernovae** analyzed with realistic uncertainties
- **ΛCDM strongly preferred** (ΔAIC = +303) for geometric expansion
- **Expected result**: Confirms QFD geometry degeneracy with ΛCDM

### Light Curve Analysis (Individual SNe)
- **Direct plasma veil testing** using time-wavelength structure
- **Template-based comparison** ensures fair baseline
- **Mixed results**: Some evidence for QFD effects, requires larger sample

### Framework Impact
- **First comprehensive QFD observational test** across multiple domains
- **Reusable methodology** for other modified gravity theories
- **Publication-ready** analysis pipeline

## 🛠️ Installation

### Requirements
```bash
python >= 3.8
numpy >= 1.20
scipy >= 1.7
pandas >= 1.3
matplotlib >= 3.4
emcee >= 3.0
corner >= 2.2
astropy >= 4.0
```

### Quick Start
```bash
git clone [your-repo-url]
cd QFD_Supernova_Analysis_Package
pip install -r requirements.txt
```

## 📚 Usage

### 1. Cosmological Parameter Fitting
Determine background QFD cosmological parameters using Union2.1 distance data:

```bash
python src/QFD_Cosmology_Fitter_v5.6.py \
    --walkers 32 --steps 3000 \
    --outdir results/cosmology
```

**Output**: Fixed cosmological parameters (η', ξ, H₀) for Stage 2+3 QFD effects

### 2. Plasma Veil Analysis
Test QFD Stage 1 mechanism using individual supernova light curves:

```bash
python src/qfd_plasma_veil_fitter.py \
    --data data/lightcurves_osc.csv \
    --snid SN2011fe \
    --cosmology results/cosmology/best_fit_params.json \
    --outdir results/plasma_analysis
```

**Output**: Plasma parameters (A_plasma, τ_decay, β) for each supernova

### 3. Comprehensive Model Comparison
Compare QFD against ΛCDM and template models:

```bash
python src/compare_qfd_lcdm_mu_z.py \
    --data data/union2.1_data_with_errors.txt \
    --out results/model_comparison.json

python src/qfd_lightcurve_comparison_v2.py \
    --data data/lightcurves_osc.csv \
    --snid SN2011fe \
    --outdir results/lightcurve_comparison
```

## 📁 Package Structure

```
QFD_Supernova_Analysis_Package/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                           # MIT License
├── src/                              # Core analysis scripts
│   ├── QFD_Cosmology_Fitter_v5.6.py     # Main cosmological analysis
│   ├── qfd_plasma_veil_fitter.py         # Plasma veil mechanism testing
│   ├── compare_qfd_lcdm_mu_z.py          # μ-z model comparison
│   ├── qfd_lightcurve_comparison_v2.py   # Enhanced light curve analysis
│   ├── qfd_ingest_lightcurves.py         # Multi-survey data ingestion
│   ├── qfd_predictions_framework.py     # Unique predictions testing
│   └── add_union21_errors.py             # Realistic uncertainty modeling
├── data/                             # Sample datasets
│   ├── union2.1_data_with_errors.txt     # Union2.1 with realistic errors
│   └── sample_lightcurves/               # Example light curve data
├── docs/                             # Documentation
│   ├── QFD_Evidence_Summary_v2.md        # Scientific results summary
│   ├── QFD_Complete_Analysis_Framework.md # Complete methodology
│   ├── methodology.md                    # Detailed methods
│   └── api_reference.md                  # Code documentation
├── examples/                         # Usage examples
│   ├── basic_cosmology_fit.py            # Simple cosmology example
│   ├── plasma_analysis_example.py        # Light curve analysis example
│   └── complete_workflow.py              # Full analysis pipeline
├── tests/                            # Unit tests
│   ├── test_cosmology_fitter.py          # Cosmology fitter tests
│   ├── test_plasma_fitter.py             # Plasma fitter tests
│   └── test_data_ingestion.py            # Data ingestion tests
└── results/                          # Example outputs
    ├── sample_cosmology_run/             # Example cosmology results
    ├── sample_plasma_analysis/           # Example plasma results
    └── publication_figures/              # High-quality plots
```

## 🔬 Scientific Methodology

### Two-Script Synergistic Approach

1. **Cosmic Canvas** (`QFD_Cosmology_Fitter_v5.6.py`)
   - Analyzes Union2.1 distance-redshift data
   - Determines fixed background cosmological parameters
   - Tests QFD Stages 2+3 (FDR + cosmological drag)

2. **Individual Portraits** (`qfd_plasma_veil_fitter.py`)
   - Analyzes raw supernova light curves
   - Fits plasma veil parameters with fixed cosmology
   - Tests QFD Stage 1 (time-wavelength dependent effects)

### Key Scientific Innovation

**Smoking Gun Test**: Correlation between plasma parameters from light curve fits and residuals in the Hubble diagram. This would be unique evidence for QFD theory.

### Model Comparison Framework

- **Information Criteria**: AIC/BIC for rigorous model selection
- **Cross-Validation**: Out-of-sample testing prevents overfitting
- **Realistic Baselines**: Strong template models for fair comparison
- **Statistical Significance**: Proper uncertainty propagation

## 📈 Results & Impact

### Publication-Ready Claims
1. "First comprehensive observational test of QFD theory across multiple cosmological observables"
2. "QFD provides complete theoretical framework but limited observational evidence relative to ΛCDM"
3. "Novel multi-observable methodology applicable to other modified gravity theories"

### Scientific Contributions
- **Methodology**: Reusable framework for alternative cosmology testing
- **Null Results**: Important constraints on QFD parameter space
- **Framework**: Template for honest, rigorous model comparison

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/contributing.md) for details.

### Areas for Enhancement
- Additional supernova surveys (Roman, LSST)
- Advanced light curve templates
- Bayesian model averaging
- Machine learning classification

## 📄 Citation

If you use this package in your research, please cite:

```bibtex
@software{qfd_supernova_analysis,
  title={QFD Supernova Analysis Package: Comprehensive Framework for Testing Quantum Field Dynamics},
  author={[Your Name]},
  year={2025},
  url={[Your GitHub URL]},
  note={Version 1.0}
}
```

## 🏆 Acknowledgments

- Union2.1 Supernova Cosmology Project
- Open Supernova Catalog (OSC)
- Dark Energy Survey (DES)
- Sloan Digital Sky Survey (SDSS)

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [Your Email]
- **GitHub**: [Your GitHub Profile]
- **Institution**: [Your Institution]

---

**Status**: ✅ Production-ready framework for comprehensive QFD cosmological analysis