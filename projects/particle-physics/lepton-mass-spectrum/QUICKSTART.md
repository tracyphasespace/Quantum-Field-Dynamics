# Quick Start Guide

## Installation (5 minutes)

### Prerequisites

- Python 3.8 or later
- pip package manager

### Setup

```bash
# Navigate to the project directory
cd lepton-mass-spectrum

# Install dependencies
pip install -r requirements.txt

# Verify installation
python scripts/verify_installation.py
```

Expected output:
```
✓ numpy               version 1.24.3
✓ scipy               version 1.10.1
✓ matplotlib          version 3.7.1
✓ emcee               version 3.1.4
✓ corner              version 2.2.2
✓ h5py                version 3.8.0
✓ Local modules       imported successfully
```

## Running the Analysis (20 minutes)

### Quick Test Run

```bash
cd scripts
python run_mcmc.py
```

This will:
1. Initialize model with Compton-scale radii
2. Run MCMC sampling (24 walkers × 1000 steps ≈ 15 minutes)
3. Generate results and plots in `../results/`

### Expected Output

Console output:
```
Model initialized with Compton-scale radii:
  Electron: R_flow = 386.2 fm, R_core = 245.8 fm
  Muon:     R_flow = 1.87 fm, R_core = 1.19 fm
  Tau:      R_flow = 0.11 fm, R_core = 0.07 fm

Running MCMC...
[Progress bar]

MCMC Complete
Acceptance fraction: 0.712

Posterior Summary:
β: 3.0627 ± 0.1491
   [2.9150, 3.2103] (68% CI)

ξ: 0.9655 ± 0.5494
   [0.5953, 1.5856] (68% CI)

τ: 1.0073 ± 0.6584
   [0.6245, 1.7400] (68% CI)

β-ξ correlation: 0.008
```

### Generated Files

```
results/
├── mcmc_chain.h5       # Full MCMC chain (HDF5 format)
├── results.json        # Parameter estimates (JSON)
└── corner_plot.png     # Posterior visualization
```

## Interpreting Results

### Parameter Estimates

**β (vacuum compression stiffness)**:
- Value: 3.06 ± 0.15
- Interpretation: Dimensionless vacuum bulk modulus
- Comparison: Theoretical prediction β = 3.058 (Golden Loop)
- Status: Consistent within uncertainty

**ξ (vacuum gradient stiffness)**:
- Value: 0.97 ± 0.55
- Interpretation: Surface tension parameter
- Comparison: Expected ξ ~ 1 from dimensional analysis
- Status: Physically reasonable

**τ (vacuum temporal stiffness)**:
- Value: 1.01 ± 0.66
- Interpretation: Inertial parameter
- Comparison: Expected τ ~ 1 from dimensional analysis
- Status: Validates temporal term inclusion

### Corner Plot

The corner plot (`results/corner_plot.png`) shows:

- **Diagonal panels**: Marginalized posterior distributions (histograms)
- **Off-diagonal panels**: 2D correlations between parameters
- **Contours**: 68% and 95% credible regions

Key observations:
- β is well-constrained (narrow distribution)
- ξ and τ have broader distributions (less constrained)
- β-ξ correlation ≈ 0 (parameters independent)

### Results File

`results/results.json` contains:
```json
{
  "β": {
    "median": 3.0627,
    "std": 0.1491,
    "q16": 2.9150,
    "q84": 3.2103
  },
  "ξ": { ... },
  "τ": { ... },
  "correlation_beta_xi": 0.008,
  "R_e_flow_fm": 386.16,
  "R_e_core_fm": 245.84
}
```

## Customization

### Modify MCMC Parameters

Edit `scripts/run_mcmc.py`:

```python
# Around line 280
sampler = run_mcmc(
    n_walkers=24,      # Number of walkers (multiple of 8)
    n_steps=1000,      # Production steps
    n_burn=200,        # Burn-in steps
    output_dir='../results'
)
```

For publication-quality results, increase to:
```python
n_walkers=48
n_steps=10000
n_burn=1000
```

(Runtime: ~2 hours)

### Change Priors

Edit the `log_prior()` function in `scripts/run_mcmc.py`:

```python
# Around line 150
def log_prior(params):
    beta, xi, tau = params

    # Modify β prior
    beta_mean = 3.058  # Center value
    beta_std = 0.15    # Standard deviation

    # Modify ξ prior
    xi_log_std = 0.5   # Log-space std dev

    # Modify τ prior
    tau_log_std = 0.5  # Log-space std dev
```

### Test Different Scales

To verify scale dependence, modify radius in `LeptonMassModel.__init__()`:

```python
# Test wrong scale (will fail)
self.R_e_flow = 2.82  # Classical radius

# Or
self.R_e_flow = 0.84  # Proton radius (catastrophic)

# Correct scale (should work)
self.R_e_flow = LAMBDA_COMPTON_E  # Default
```

## Troubleshooting

### Low Acceptance Fraction (< 0.2)

**Problem**: MCMC steps too large, walkers rejecting moves.

**Solution**: Reduce step size by tightening initial scatter:
```python
pos = θ_init + 0.05 × randn(n_walkers, 3)  # Was 0.1
```

### High Acceptance Fraction (> 0.8)

**Problem**: MCMC steps too small, walkers not exploring.

**Solution**: Increase initial scatter:
```python
pos = θ_init + 0.2 × randn(n_walkers, 3)  # Was 0.1
```

### Parameters Not Converging

**Problem**: Insufficient burn-in or production steps.

**Solution**: Increase both:
```python
n_burn = 500     # Was 200
n_steps = 5000   # Was 1000
```

### Import Errors

**Problem**: Missing dependencies.

**Solution**:
```bash
pip install -r requirements.txt

# Or install individually
pip install numpy scipy matplotlib emcee corner h5py
```

## Next Steps

### Reproduce Results

1. Run with same random seed for exact reproduction:
   ```python
   np.random.seed(42)  # Add to beginning of run_mcmc()
   ```

2. Compare your results to documented values in `docs/RESULTS.md`

3. Variation should be within ±0.01 for β, ±0.05 for ξ, τ

### Explore Model

1. **Scale test**: Run with different R values (see Customization)
2. **Prior sensitivity**: Run with different prior widths
3. **Convergence**: Increase n_steps to 10000, check stability

### Predictive Tests

The model is now calibrated. To test predictive power:

1. **Muon g-2**: Compute anomalous magnetic moment from same parameters
2. **Charge radii**: Predict R_core and compare to experiments
3. **Magnetic moments**: Calculate from angular momentum

(These require extending the model - see `docs/THEORY.md` for guidance)

## Documentation

- `README.md`: Project overview
- `docs/METHODOLOGY.md`: Detailed methods
- `docs/THEORY.md`: Theoretical background
- `docs/RESULTS.md`: Extended results and analysis
- `CITATION.cff`: How to cite this work

## Support

### Common Questions

**Q: Why Compton wavelength, not classical radius?**

A: Classical radius (2.82 fm) is from equating Coulomb energy to mass. Compton wavelength (386 fm) is the quantum length scale ℏ/mc where particle physics transitions to field physics. The model uses the latter.

**Q: Is this a prediction or a fit?**

A: It's a 3-parameter fit to 3 masses, so not predictive yet. True tests require computing other observables (g-2, charge radius, etc.) with the same parameters.

**Q: Why is ξ so uncertain?**

A: Gradient energy is only ~0.002% of electron energy at Compton scale. Data weakly constrains ξ. Better constraints require smaller particles (muon, tau) where gradient term is larger.

**Q: How does this compare to Standard Model?**

A: Standard Model uses QFT with renormalization. This model uses classical geometry + phenomenological functional. Both reproduce masses, but mechanisms differ. See `docs/THEORY.md` for comparison.

### Getting Help

1. Check `docs/` folder for detailed documentation
2. Review `scripts/verify_installation.py` output
3. Ensure Python 3.8+ and all dependencies installed
4. Open issue on GitHub with error messages and system info

## License

MIT License - See LICENSE file.

Free to use, modify, and distribute with attribution.
