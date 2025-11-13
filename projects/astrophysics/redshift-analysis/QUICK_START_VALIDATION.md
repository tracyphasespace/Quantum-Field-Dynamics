# Quick Start: QFD Hubble Constant Validation

This guide helps you quickly replicate the QFD validation results showing that cosmological observations can be matched at H₀ ≈ 70 km/s/Mpc **without dark energy**.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (to clone the repository)

## Installation

```bash
# Clone the repository (if not already cloned)
git clone https://github.com/tracyphasespace/Quantum-Field-Dynamics.git
cd Quantum-Field-Dynamics

# Navigate to the RedShift analysis directory
cd projects/astrophysics/redshift-analysis/RedShift/

# Install the QFD CMB module
pip install -e .
```

## Run the Validation

### Option 1: Complete Hubble Constant Validation

This runs the full validation comparing QFD vs ΛCDM:

```bash
python hubble_constant_validation.py
```

**Output:**
- Statistical comparison (printed to console)
- Publication-quality figures (PNG and PDF)
- JSON results file with all data
- Saved to: `validation_output/`

**Expected Results:**
```
✓ PASSED: QFD matches observations within experimental uncertainty
  QFD achieves RMS = 0.143 mag (< 0.2 mag threshold)

Key finding: QFD reproduces H₀ = 70 km/s/Mpc WITHOUT dark energy!
```

### Option 2: CMB Module Demo

This runs the CMB power spectrum calculations:

```bash
python run_demo.py --outdir validation_output
```

**Output:**
- CMB TT, TE, EE power spectra (CSV and plots)
- Demonstrates QFD photon-photon scattering physics

## Validation Results

After running the validation, check:

### 1. Validation Report
```bash
cat ../HUBBLE_VALIDATION_REPORT.md
```

### 2. Visual Results
```bash
# View the validation figure
open validation_output/hubble_constant_validation.png
# or: xdg-open validation_output/hubble_constant_validation.png (Linux)
```

### 3. Numerical Results
```bash
cat validation_output/hubble_constant_validation_results.json
```

Key metrics to look for:
- `"qfd_rms_error_mag"`: Should be ~0.14 mag
- `"qfd_reduced_chi2"`: Should be ~0.94 (excellent fit)
- `"validation_passed"`: Should be `true`

## Understanding the Results

### What This Validation Shows

The validation demonstrates that:

1. **QFD uses H₀ = 70 km/s/Mpc** (standard value)
2. **No dark energy needed** (Ω_Λ = 0 vs. 0.7 in ΛCDM)
3. **Better fit than ΛCDM** (RMS 0.143 vs. 0.178 mag)
4. **Photon-ψ field interactions** replace cosmic acceleration

### How It Works

```
Traditional View (ΛCDM):
  Distant supernovae dim → Universe accelerating → Dark energy (68% of universe)

QFD View:
  Distant supernovae dim → Photon-ψ field interactions → No dark energy needed
```

### The Physics

- **SLAC E144 basis**: Experimentally validated photon-photon scattering
- **Scaling**: Laboratory → Cosmological distances
- **Mechanism**: High-energy photons → ψ field → CMB photons
- **Result**: Systematic dimming ∝ z^0.6 (mimics dark energy)

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'qfd_cmb'`:

```bash
# Make sure you're in the RedShift directory
cd projects/astrophysics/redshift-analysis/RedShift/

# Reinstall the module
pip install -e .
```

### Missing Dependencies

```bash
pip install numpy scipy matplotlib pandas
```

### Permission Issues

If you get permission errors:

```bash
# Use a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

## Next Steps

### Explore the Code

Key files to examine:

1. **`hubble_constant_validation.py`** - Main validation script
   - Line 31: Hubble constant H₀ = 70.0 km/s/Mpc
   - Line 39: QFD parameters (no dark energy)
   - Line 84: QVD dimming calculation
   - Line 398: Main validation function

2. **`qfd_cmb/`** - QFD physics modules
   - `ppsi_models.py` - ψ field power spectra
   - `visibility.py` - Cosmological visibility functions
   - `kernels.py` - Photon-photon scattering kernels
   - `projector.py` - CMB projection calculations

### Modify Parameters

Try changing parameters in `hubble_constant_validation.py`:

```python
# Line 25-26: Adjust QVD parameters
QVD_COUPLING = 0.85  # Try 0.7 - 1.0
REDSHIFT_POWER = 0.6  # Try 0.5 - 0.7

# Rerun to see impact on fit quality
```

### Generate Additional Plots

The validation script includes functions you can call individually:

```python
from hubble_constant_validation import *

# Generate specific analysis
results = validate_hubble_constant()
create_validation_plots(results)
save_validation_results(results)
```

## Publication-Ready Outputs

The validation generates publication-quality outputs:

- **Figure (PNG)**: 300 DPI, 16×10 inches
- **Figure (PDF)**: Vector graphics for LaTeX
- **Data (JSON)**: All numerical results
- **Report (MD)**: Complete analysis documentation

Use these directly in papers or presentations!

## Questions?

- **Repository**: https://github.com/tracyphasespace/Quantum-Field-Dynamics
- **Issues**: https://github.com/tracyphasespace/Quantum-Field-Dynamics/issues
- **Documentation**: See `HUBBLE_VALIDATION_REPORT.md`

## Citation

If you use this validation in your research:

```bibtex
@software{qfd_hubble_validation,
  title={QFD Hubble Constant Validation Without Dark Energy},
  author={QFD Project},
  year={2025},
  url={https://github.com/tracyphasespace/Quantum-Field-Dynamics}
}
```

---

**Happy Validating!** 🚀

If you successfully replicate the validation, you've demonstrated that:
- Dark energy may be unnecessary
- H₀ ≈ 70 km/s/Mpc works without acceleration
- QFD provides a viable alternative to ΛCDM cosmology
