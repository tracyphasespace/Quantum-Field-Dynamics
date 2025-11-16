# QFD Supernova Analysis Pipeline V18

**Quantum Field Dynamics (QFD) analysis of Type Ia supernovae using the DES 5-Year dataset**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12720778.svg)](https://doi.org/10.5281/zenodo.12720778)

## Overview

This pipeline implements a three-stage analysis of Type Ia supernova light curves to test the Quantum Field Dynamics (QFD) cosmological model against ΛCDM. The analysis uses publicly available data from the Dark Energy Survey (DES) 5-Year Supernova Program.

**Key Results:**
- QFD provides **8.1% better fit** than ΛCDM (RMS: 2.381 vs 2.592 mag)
- Identified **68 outlier candidates** for Binary Black Hole (BBH) lensing at 2.5σ threshold
- Asymmetric outlier distribution (5:1 dark/bright ratio) supports BBH lensing hypothesis

## Pipeline Stages

### Stage 1: Individual SN Optimization
Fits QFD light curve model to each supernova independently using L-BFGS-B optimization.

**Model parameters (per SN):**
- `t0`: Time of peak brightness
- `ln_A`: Log-amplitude (brightness)
- `A_plasma`: Local plasma density parameter
- `beta`: Thermal effect parameter

### Stage 2: Global MCMC Inference
Uses emcee to infer global QFD parameters via Bayesian MCMC.

**Global parameters:**
- `k_J_correction`: J-band magnitude correction
- `eta_prime`: QFD plasma veil parameter (η')
- `xi`: QFD frequency-dependent refraction parameter (ξ)
- `sigma_ln_A`: Intrinsic scatter in log-amplitude

### Stage 3: Hubble Diagram & Outlier Detection
Constructs Hubble diagram, compares QFD vs ΛCDM, and identifies outliers for BBH lensing analysis.

**Outputs:**
- Distance modulus vs redshift plots
- Residual analysis
- Outlier classification (BBH scatter vs magnify)

## Installation

### Requirements

- Python 3.12+
- 8+ CPU cores recommended
- 16+ GB RAM recommended

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/tracyphasespace/Quantum-Field-Dynamics.git
cd Quantum-Field-Dynamics/qfd-supernova-v15/v15_clean/v18
```

2. **Create virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

**Important:** This pipeline requires **NumPy 1.x** (not 2.x) due to JAX compatibility. The `requirements.txt` is configured correctly.

## Data Setup

### Download DES 5-Year Supernova Dataset

The pipeline uses publicly available DES data:

**Option 1: Download from Zenodo (Recommended)**
```bash
# Download 1.5 GB archive
wget https://zenodo.org/records/12720778/files/DES-SN5YR_PHOTOMETRY.zip
unzip DES-SN5YR_PHOTOMETRY.zip -d data/raw/
```

**Option 2: Use provided unified CSV**
```bash
# If you already have lightcurves_unified_v2_min3.csv
cp /path/to/lightcurves_unified_v2_min3.csv data/
```

**Dataset Information:**
- **Total light curves:** 31,636 (DIFFIMG) + 19,706 (SMP)
- **DOI:** [10.5281/zenodo.12720778](https://doi.org/10.5281/zenodo.12720778)
- **GitHub:** [des-science/DES-SN5YR](https://github.com/des-science/DES-SN5YR)

## Usage

### Quick Start

Run the full pipeline with default settings:

```bash
# Stage 1: Individual SN fits (most time-consuming)
python3 pipeline/stages/stage1_optimize_v17.py \
  --lightcurves data/lightcurves_unified_v2_min3.csv \
  --out results/stage1_fullscale \
  --ncores 8

# Stage 2: MCMC global parameter inference
python3 pipeline/stages/stage2_mcmc_v18_emcee.py \
  --lightcurves data/lightcurves_unified_v2_min3.csv \
  --stage1-results results/stage1_fullscale \
  --out results/stage2_emcee \
  --max-sne 1000 \
  --nwalkers 32 \
  --nsteps 2000 \
  --nburn 500 \
  --ncores 8

# Stage 3: Hubble diagram with 2.5σ outlier threshold
python3 pipeline/stages/stage3_v18.py \
  --lightcurves data/lightcurves_unified_v2_min3.csv \
  --stage1-results results/stage1_fullscale \
  --stage2-results results/stage2_emcee \
  --out results/stage3_hubble \
  --quality-cut 2000 \
  --outlier-sigma-threshold 2.5 \
  --ncores 8
```

### Command-Line Options

**Stage 1:**
```
--lightcurves     Path to unified lightcurve CSV
--out             Output directory for results
--ncores          Number of CPU cores (default: 8)
--max-sne         Maximum number of SNe to process (optional)
```

**Stage 2:**
```
--lightcurves     Path to unified lightcurve CSV
--stage1-results  Path to Stage 1 results
--out             Output directory for results
--max-sne         Number of SNe for MCMC (default: 1000)
--nwalkers        Number of MCMC walkers (default: 32)
--nsteps          Total MCMC steps (default: 2000)
--nburn           Burn-in steps to discard (default: 500)
--ncores          Number of CPU cores (default: 8)
```

**Stage 3:**
```
--lightcurves               Path to unified lightcurve CSV
--stage1-results            Path to Stage 1 results
--stage2-results            Path to Stage 2 results
--out                       Output directory for results
--quality-cut               Max chi² for quality SNe (default: 2000)
--outlier-sigma-threshold   Sigma threshold for outliers (default: 3.0)
--ncores                    Number of CPU cores (default: 8)
```

## Output Files

### Stage 1
```
results/stage1_fullscale/
├── <snid>_fit.json          # Fit results for each SN
└── summary_statistics.json   # Overall statistics
```

### Stage 2
```
results/stage2_emcee/
├── samples.npz              # MCMC chains (NumPy archive)
└── summary.json             # Parameter statistics (median, mean, std)
```

### Stage 3
```
results/stage3_hubble/
├── hubble_data.csv                # Full Hubble diagram dataset
├── hubble_diagram.png             # Visualization
├── residuals_analysis.png         # Residual plots
├── summary.json                   # Statistics and fit results
├── outliers_too_dark.csv          # BBH scatter candidates
└── outliers_too_bright.csv        # BBH magnify candidates
```

## Results

### QFD vs ΛCDM Performance

| Metric | QFD | ΛCDM | Improvement |
|--------|-----|------|-------------|
| RMS Residual | 2.381 mag | 2.592 mag | **8.1%** |
| χ² (total) | 27,689 | 295,965 | **90.6%** |
| Trend slope | +4.21 | -5.66 | - |
| Correlation | r=0.49 | r=-0.61 | - |

### Best-Fit Parameters (Stage 2 MCMC)

| Parameter | Median | Mean | Std |
|-----------|--------|------|-----|
| k_J_correction | 19.997 | 19.965 | 0.112 |
| η' (eta_prime) | -6.000 | -5.999 | 0.0035 |
| ξ (xi) | -6.000 | -5.998 | 0.0057 |
| σ_ln_A | 1.000 | 1.000 | 7.65×10⁻⁵ |

### Outlier Analysis (2.5σ Threshold)

**Total outliers:** 68 / 4,885 SNe (1.39%)

- **Too dark (BBH scatter):** 57 SNe (84%)
  - Off-axis BBH geometry
  - Dimmer than QFD prediction

- **Too bright (BBH magnify):** 11 SNe (16%)
  - On-axis BBH alignment (rare)
  - Brighter than QFD prediction

**Physical interpretation:** The 5:1 asymmetry supports BBH lensing hypothesis, as off-axis geometry is statistically more common than precise alignment.

## Three-Population Model

Based on outlier analysis, the dataset consists of:

1. **Normal Population** (98.6%): Standard QFD model
2. **BBH Scatter Population** (1.2%): Off-axis BBH lensing (dimming)
3. **BBH Magnify Population** (0.2%): On-axis BBH lensing (brightening)

## Code Structure

```
v18/
├── pipeline/
│   ├── core/
│   │   ├── v17_qfd_model.py          # QFD redshift/distance model
│   │   ├── v17_lightcurve_model.py   # Light curve fitting (JAX)
│   │   ├── v17_data.py               # Data loader classes
│   │   └── pipeline_io.py            # I/O utilities
│   ├── stages/
│   │   ├── stage1_optimize_v17.py    # Individual SN optimization
│   │   ├── stage2_mcmc_v18_emcee.py  # MCMC with emcee
│   │   └── stage3_v18.py             # Hubble diagram & outliers
│   └── scripts/
│       └── run_stage2_fullscale_v18.sh  # Example run script
├── data/
│   └── lightcurves_unified_v2_min3.csv  # Input data (user-provided)
├── results/                          # Pipeline outputs
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── Outcome.md                        # Detailed analysis results
```

## Troubleshooting

### NumPy Version Error
```
AttributeError: _ARRAY_API not found
```
**Solution:** JAX 0.4.23 requires NumPy 1.x. Downgrade:
```bash
pip uninstall -y numpy
pip install numpy==1.26.4
```

### SupernovaData Attribute Error
```
AttributeError: 'SupernovaData' object has no attribute 'wavelength'
```
**Solution:** Use correct field names: `wavelength_nm`, `flux_jy`, `flux_err_jy`

### MCMC Convergence Issues
- Increase `--nsteps` (e.g., 5000)
- Increase `--nburn` (e.g., 1000)
- Check walker initialization in Stage 2 code

### A_lens Fitting Failures
Currently, most A_lens diagnostic fits fail (ABNORMAL_TERMINATION_IN_LNSRCH). This is a known issue requiring improved optimization strategies for the BBH lensing model.

## Citation

If you use this pipeline or dataset, please cite:

**DES 5-Year Supernova Dataset:**
```bibtex
@software{des_sn5yr_2024,
  author       = {DES Collaboration},
  title        = {DES 5-Year Supernova Photometry Release},
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.12720778},
  url          = {https://doi.org/10.5281/zenodo.12720778}
}
```

**QFD Analysis Pipeline:**
```bibtex
@software{qfd_pipeline_v18,
  author       = {Tracy Phase Space},
  title        = {QFD Supernova Analysis Pipeline V18},
  year         = 2025,
  publisher    = {GitHub},
  url          = {https://github.com/tracyphasespace/Quantum-Field-Dynamics}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add appropriate license information]

## Contact

For questions or issues:
- GitHub Issues: https://github.com/tracyphasespace/Quantum-Field-Dynamics/issues
- Primary maintainer: Tracy Phase Space

## Acknowledgments

- Dark Energy Survey (DES) Collaboration for public data release
- DES-SN5YR team for photometry pipeline and documentation
- Zenodo for data hosting

## References

- **DES 5-Year Data:** https://doi.org/10.5281/zenodo.12720778
- **DES-SN5YR GitHub:** https://github.com/des-science/DES-SN5YR
- **Analysis Results:** See `Outcome.md` for detailed findings

---

**Version:** 1.0
**Last Updated:** November 15, 2025
