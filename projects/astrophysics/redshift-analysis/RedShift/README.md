# QFD CMB Module — Photon–Photon Scattering Projection

[![CI Status](https://github.com/username/qfd-cmb/workflows/CI/badge.svg)](https://github.com/username/qfd-cmb/actions)
[![Coverage](https://codecov.io/gh/username/qfd-cmb/branch/main/graph/badge.svg)](https://codecov.io/gh/username/qfd-cmb)
[![PyPI version](https://badge.fury.io/py/qfd-cmb.svg)](https://badge.fury.io/py/qfd-cmb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Version:** 0.1.0

## Description

This package provides a minimal, production-ready scaffold to compute QFD-based CMB TT/TE/EE spectra using a
**photon–photon (γγ) sin² kernel** in place of Thomson scattering. It includes both **Limber** and **full line‑of‑sight (LOS)**
projectors, parametric ψ‑field power spectra, flexible visibility functions, and publication‑quality plotting utilities.

The QFD CMB Module implements quantum field theory corrections to cosmic microwave background (CMB) calculations,
specifically focusing on photon-photon scattering effects that can introduce oscillatory features in the power spectra.
This is particularly relevant for precision cosmology and tests of fundamental physics using CMB observations.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for development installation)

### Quick Installation

For most users, install directly from PyPI:

```bash
pip install qfd-cmb
```

### Development Installation

For development or to run the latest version:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/qfd-cmb.git
   cd qfd-cmb
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Using venv (recommended)
   python -m venv .venv
   
   # On Linux/macOS:
   source .venv/bin/activate
   
   # On Windows:
   .venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e .
   ```

4. **Install development dependencies (optional):**
   ```bash
   pip install -r requirements-dev.txt
   ```

### Verify Installation

Test your installation by running the demo:

```bash
python run_demo.py --outdir outputs
```

This should generate CSV files and plots in the `outputs/` directory using **Planck‑anchored** parameters (ℓ_A≈301, rψ≈147 Mpc, τ≈0.054).

## Modules

- `qfd_cmb/ppsi_models.py` — families of P_ψ(k), including oscillatory modulation with damping.
- `qfd_cmb/visibility.py` — parametric visibility g(η) and helpers to convert (z,η,χ).
- `qfd_cmb/kernels.py` — γγ sin² angular/polarization kernel as a Mueller‑matrix analogue.
- `qfd_cmb/projector.py` — Limber and full LOS projectors; accepts user source functions S_T,S_E.
- `qfd_cmb/figures.py` — clean, black‑and‑white publication plots (TT/TE/EE; signed TE).
- `fit_planck.py` — scaffold for parameter fitting with `emcee` given Planck/BAO CSVs.
- `run_demo.py` — reproduces the "cosmic‑anchored" toy and saves outputs.

## Data

Place Planck binned TT/TE/EE CSVs and any BAO compilation in `data/` (or pass paths on CLI).
Expected CSV columns for C_ℓ: `ell,C_TT,C_TE,C_EE,error_*` (errors optional).

## Usage Examples

### Basic Usage

```python
import numpy as np
from qfd_cmb import ppsi_models, visibility, projector

# Set up parameters
lA = 301.0
rpsi = 147.0
chi_star = 14065.0
sigma_chi = 250.0

# Generate power spectrum
ell = np.arange(2, 2501)
Psi_k = ppsi_models.oscillatory_psik(k_values, lA, rpsi)

# Compute CMB spectra
Ctt, Cee, Cte = projector.project_limber(ell, Psi_k, chi_star, sigma_chi)
```

### Advanced Parameter Fitting

See `examples/advanced_fitting.py` for a complete example using `emcee` to fit parameters to Planck data.

## Testing

Run the test suite to verify your installation:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=qfd_cmb

# Run specific test module
pytest tests/test_ppsi_models.py
```

## Troubleshooting

### Common Installation Issues

**Problem:** `ModuleNotFoundError: No module named 'qfd_cmb'`
- **Solution:** Ensure you've activated your virtual environment and installed the package:
  ```bash
  source .venv/bin/activate  # or .venv\Scripts\activate on Windows
  pip install -e .
  ```

**Problem:** `ImportError` related to NumPy/SciPy
- **Solution:** Update to compatible versions:
  ```bash
  pip install --upgrade numpy scipy matplotlib
  ```

**Problem:** Demo script fails with "Permission denied" or file access errors
- **Solution:** Ensure the output directory is writable:
  ```bash
  mkdir -p outputs
  chmod 755 outputs
  ```

### Platform-Specific Issues

**Windows:**
- Use `python -m pip` instead of `pip` if you encounter permission issues
- Ensure you're using Command Prompt or PowerShell as Administrator if needed
- Use forward slashes or raw strings for file paths in Python scripts

**macOS:**
- If using Homebrew Python, you may need to install additional dependencies:
  ```bash
  brew install python-tk  # for matplotlib GUI backend
  ```

**Linux:**
- Install system dependencies for matplotlib:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install python3-tk
  
  # CentOS/RHEL
  sudo yum install tkinter
  ```

### Performance Issues

**Problem:** Slow computation times
- **Solution:** Ensure you're using optimized NumPy/SciPy builds:
  ```bash
  pip install numpy[mkl] scipy  # Intel MKL acceleration
  ```

**Problem:** Memory errors with large parameter ranges
- **Solution:** Reduce grid resolution or process in chunks:
  ```python
  # Reduce chi_grid_points in projector calls
  # Process ell ranges in smaller batches
  ```

### Scientific Validation Issues

**Problem:** Results don't match expected values
- **Solution:** 
  1. Verify input parameters match the expected ranges
  2. Check that reference data files are in the correct format
  3. Run the validation tests: `pytest tests/test_scientific_validation.py`

**Problem:** Numerical instabilities or NaN values
- **Solution:**
  1. Check input parameter ranges for physical validity
  2. Increase numerical precision in integration routines
  3. Verify that visibility function parameters are reasonable

### Getting Help

If you encounter issues not covered here:

1. **Check the [Issues](https://github.com/username/qfd-cmb/issues)** page for similar problems
2. **Run the test suite** to identify specific failing components
3. **Create a minimal example** that reproduces the issue
4. **Open a new issue** with:
   - Your operating system and Python version
   - Complete error message and traceback
   - Minimal code example
   - Steps to reproduce

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Setting up a development environment
- Code style and testing requirements
- Submitting pull requests
- Reporting bugs and requesting features

## Notes

- The LOS projector here is **modular**: you can pass any S_T(k,η), S_E(k,η) callable (e.g., from a more detailed ψ‑dynamics model).
- The included γγ kernel follows the **same quadrupole geometry** as Thomson; only the optical‑depth history differs.
- B‑modes are zero at linear order for parity‑even scalar sources; lensing B can be added in a post‑processing step.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{qfd_cmb,
  title={QFD CMB Module: Photon-Photon Scattering Projection},
  author={QFD Project},
  version={0.1.0},
  year={2025},
  url={https://github.com/username/qfd-cmb}
}
```

© QFD Project. Apache‑2.0 license.
