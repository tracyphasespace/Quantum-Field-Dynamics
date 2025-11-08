# MNRAS Publication Figures

This directory contains scripts to generate publication-quality figures for the MNRAS letter on QFD Supernova Analysis V15.

## Quick Start

```bash
# Generate all figures
make all

# Or generate individually
make figure_hubble.pdf
make figure_basis_inference.pdf
make figure_residuals.pdf
make figure_corner.pdf
```

## Figure Specifications

All figures follow MNRAS two-column layout specifications:
- **Vector format:** PDF with editable text (fonttype 42)
- **Fonts:** TeX Gyre Termes / Times (matching newtx)
- **Sizes:** 7-8 pt for labels (readable at print)
- **Monochrome-friendly:** Distinguishable in grayscale
- **Provenance:** Automatic metadata tracking

### Figure 1: Hubble Diagram & Residuals
**File:** `figure_hubble.pdf`
**Size:** 244 pt × 216 pt (single column)

Two-panel stacked plot:
- (a) Hubble diagram: μ_obs vs z with QFD model and ΛCDM reference
- (b) Residuals: Δμ vs z with binned statistics

### Figure 2: Basis Functions & Correlations
**File:** `figure_basis_inference.pdf`
**Size:** 244 pt × 208 pt (single column)

2×2 grid showing:
- (a) Basis functions φ_k(z)
- (b) Finite-difference derivatives
- (c) Correlation matrix with annotations
- (d) Identifiability metrics (κ, max |ρ|)

### Figure 3: Residual Diagnostics
**File:** `figure_residuals.pdf`
**Size:** 244 pt × 216 pt (single column)

Three-panel diagnostics:
- (a) Residuals vs z with running median and 16-84% band
- (b) Residuals vs nuisance parameter
- (c) Q-Q plot with normality tests

### Figure 4: Parameter Posteriors
**File:** `figure_corner.pdf`
**Size:** 508 pt × 320 pt (double column)

Corner plot of QFD parameters:
- k_J (cosmological drag)
- η' (plasma veil evolution)
- ξ (FDR/saturation)
- σ_α (intrinsic scatter, if available)
- ν (Student-t DOF, if available)

## Requirements

```bash
# Python 3.9+
pip install numpy scipy matplotlib pandas

# Optional (for MCMC samples)
pip install jax numpyro
```

## Reproducibility

### Lock Files

Generate reproducibility locks:
```bash
make environment.lock  # Package versions
make data.lock         # Dataset hashes
```

These files record:
- Python version
- Package versions (numpy, scipy, matplotlib, etc.)
- Git SHA
- Dataset file hashes (SHA256)
- SNe count and redshift range

### Provenance Tracking

Each figure automatically generates a `*_provenance.json` file containing:
- Creation timestamp
- Software versions
- Git SHA
- Input data paths and hashes
- Model parameters
- Summary statistics

Example:
```json
{
  "figure": "figure_hubble.pdf",
  "created": "2025-11-06T12:34:56",
  "software": {
    "numpy": "1.24.0",
    "matplotlib": "3.7.1"
  },
  "git_sha": "abc123...",
  "best_fit": {
    "k_J": 70.0,
    "eta_prime": 0.01,
    "xi": 30.0
  },
  ...
}
```

## Usage with LaTeX

### Single Column
```latex
\begin{figure}
    \centering
    \includegraphics[width=0.48\textwidth]{figure_hubble.pdf}
    \caption{Hubble diagram and residuals ...}
    \label{fig:hubble}
\end{figure}
```

### Double Column
```latex
\begin{figure*}
    \centering
    \includegraphics[width=\textwidth]{figure_corner.pdf}
    \caption{Posterior constraints ...}
    \label{fig:corner}
\end{figure*}
```

## Makefile Targets

```bash
make all          # Build all figures
make clean        # Remove figures
make clean-locks  # Remove lock files
make distclean    # Remove everything
make status       # Show build status
make validate     # Validate PDFs
make help         # Show help
```

## File Structure

```
figures/
├── README.md                    # This file
├── Makefile                     # Build system
├── mnras_style.py               # MNRAS style configuration
├── make_hubble.py               # Figure 1 generator
├── make_basis_inference.py      # Figure 2 generator
├── make_residuals.py            # Figure 3 generator
├── make_corner.py               # Figure 4 generator
├── make_locks.py                # Lock file generator
├── environment.lock             # Package versions (generated)
├── data.lock                    # Dataset hashes (generated)
├── figure_*.pdf                 # Generated figures
└── figure_*_provenance.json     # Provenance metadata
```

## Style Configuration

All figures use consistent styling defined in `mnras_style.py`:

```python
# MNRAS column widths
SINGLE_COLUMN = 244 pt (~84 mm)
DOUBLE_COLUMN = 508 pt (~178 mm)

# Font sizes
axes.labelsize = 7.5 pt
tick.labelsize = 7.0 pt
legend.fontsize = 7.0 pt
title.fontsize = 8.0 pt

# Line widths
axes.linewidth = 0.6 pt
lines.linewidth = 0.8 pt
```

## Customization

### Changing Data Paths

Edit the `stage3_dir` and `stage2_dir` variables in each `make_*.py` script:

```python
# Configuration
stage3_dir = "../results/v15_production/stage3"
stage2_dir = "../results/v15_production/stage2"
```

### Adjusting Figure Style

Edit `mnras_style.py` to modify:
- Font sizes
- Line widths
- Colors (for draft versions)
- Aspect ratios

### Adding New Figures

1. Create `make_newfigure.py` using `mnras_style` module
2. Add target to `Makefile`:
   ```makefile
   figure_newfigure.pdf: make_newfigure.py mnras_style.py $(LOCKS)
       python make_newfigure.py
   ```
3. Add to `FIGS` list in Makefile

## Troubleshooting

### Missing Data Files

```
FileNotFoundError: No such file or directory: '../results/v15_production/stage3/hubble_data.csv'
```

**Solution:** Run the full pipeline first (Stages 1-3) or update paths in scripts.

### Missing Packages

```
ModuleNotFoundError: No module named 'scipy'
```

**Solution:** Install requirements:
```bash
pip install numpy scipy matplotlib pandas
```

### Invalid PDF Output

```
Error: figure_hubble.pdf is NOT a valid PDF
```

**Solution:** Check matplotlib installation and backend:
```python
import matplotlib
print(matplotlib.get_backend())  # Should support PDF
```

## Testing

```bash
# Generate lock files
make environment.lock data.lock

# Build all figures
make all

# Validate outputs
make validate

# Check status
make status
```

Expected output:
```
✓ figure_hubble.pdf (valid PDF)
✓ figure_basis_inference.pdf (valid PDF)
✓ figure_residuals.pdf (valid PDF)
✓ figure_corner.pdf (valid PDF)
```

## Citation

If you use these figure generation scripts, please cite:

```bibtex
@software{qfd_figures_2025,
  title={MNRAS Publication Figures for QFD Supernova Analysis V15},
  author={QFD Collaboration},
  year={2025},
  url={https://github.com/tracyphasespace/Quantum-Field-Dynamics}
}
```

## License

Part of the QFD Supernova V15 project. See main repository for license.

## Contact

For issues or questions:
- GitHub: https://github.com/tracyphasespace/Quantum-Field-Dynamics/issues
- Documentation: See `../docs/` directory

---

**Last Updated:** 2025-11-06
**Version:** V15-rc1+mnras
