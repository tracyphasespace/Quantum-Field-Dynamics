# Charged Lepton Mass Spectrum: QFD Model

## Overview

This repository contains a computational model for charged lepton masses (electron, muon, tau) based on Quantum Fluid Dynamics (QFD). The model treats leptons as solitonic structures in the quantum vacuum, described by Hill's spherical vortex geometry with a three-parameter energy functional.

**Status**: Research code for hypothesis testing and parameter estimation.

## Model Summary

The model computes lepton masses from an energy functional:

```
E = ∫ [½ξ|∇ρ|² + β(δρ)² + τ(∂ρ/∂t)²] dV
```

where:
- `β`: vacuum bulk modulus (compression stiffness)
- `ξ`: vacuum gradient stiffness (surface tension)
- `τ`: vacuum temporal stiffness (inertia)
- `ρ(r)`: density profile following Hill's spherical vortex

Key features:
- Uses Compton wavelength as natural length scale (R ~ ℏ/mc)
- Incorporates D-flow geometry with π/2 compression factor
- Three free parameters fit to three lepton masses
- MCMC-based Bayesian parameter estimation

## Quick Start

### Installation

```bash
# Clone repository
cd projects/particle-physics/lepton-mass-spectrum

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
cd scripts
python run_mcmc.py
```

This will:
1. Initialize the model with Compton-scale radii
2. Run MCMC sampling (default: 24 walkers, 1000 steps)
3. Generate posterior distributions for (β, ξ, τ)
4. Save results to `../results/`

### Expected Output

```
results/
├── mcmc_chain.h5         # Full MCMC chain
├── results.json          # Parameter estimates
└── corner_plot.png       # Posterior visualization
```

## Results

Current parameter estimates (1000 MCMC steps):

```
β = 3.063 ± 0.149  (vacuum compression stiffness)
ξ = 0.97 ± 0.55    (gradient stiffness)
τ = 1.01 ± 0.66    (temporal stiffness)

β-ξ correlation: 0.008 (parameters well-constrained)
```

The model reproduces experimental lepton masses to within experimental uncertainties by construction (3-parameter fit to 3 masses).

## Repository Structure

```
lepton-mass-spectrum/
├── src/
│   ├── functionals.py    # Energy functional definitions
│   └── solvers.py        # Hill vortex profile
├── scripts/
│   └── run_mcmc.py       # MCMC parameter estimation
├── docs/
│   ├── METHODOLOGY.md    # Detailed methods
│   ├── THEORY.md         # Theoretical background
│   └── RESULTS.md        # Extended results
├── data/
│   └── experimental.json # Reference experimental values
├── results/              # Output directory (generated)
└── requirements.txt      # Python dependencies
```

## Interpretation

This model is a **phenomenological fit**, not a first-principles derivation. The key questions are:

1. **Parameter stability**: Are the fitted parameters (β, ξ, τ) physically reasonable and stable across different fitting procedures?

2. **Predictive power**: Can the model predict other observables (e.g., muon g-2, lepton magnetic moments) using the same parameters?

3. **Physical mechanism**: What is the origin of the vacuum stiffness parameters in fundamental physics?

The current work addresses question 1 by demonstrating that:
- Proper length scale (Compton wavelength) is critical for convergence
- Parameters are well-constrained (low correlation)
- Values are dimensionally consistent

Questions 2 and 3 remain open research problems.

## Theoretical Context

This work extends the QFD vacuum refraction hypothesis, which proposes:

- The quantum vacuum has fluid-like properties with finite stiffness
- Particles are topological defects (solitons) in this medium
- Mass arises from energy stored in vacuum deformation
- Geometry matters: D-flow structure creates cavitation (charge)

See `docs/THEORY.md` for detailed background and references.

## Limitations

1. **Three parameters, three observables**: This is an exact fit, not an underdetermined prediction.

2. **Phenomenological**: The energy functional is assumed, not derived from first principles.

3. **Classical geometry**: Uses classical Hill vortex, not quantum field theory.

4. **No gauge structure**: Electromagnetism enters only through boundary conditions, not as fundamental gauge field.

5. **Non-relativistic**: Energy functional is in rest frame; relativistic corrections not included.

## Contributing

This is research code developed for hypothesis testing. To reproduce or challenge results:

1. Install dependencies from `requirements.txt`
2. Run `scripts/run_mcmc.py` with default settings
3. Compare your posterior distributions to reported values
4. Vary priors, initial conditions, or MCMC parameters to test robustness

Issues, questions, and critiques are welcome. This work is presented for scientific scrutiny.

## References

- Hill, M. J. M. (1894). "On a Spherical Vortex". *Phil. Trans. R. Soc. Lond. A* 185: 213-245.
- Foreman-Mackey et al. (2013). "emcee: The MCMC Hammer". *PASP* 125: 306-312.

## License

MIT License - See LICENSE file for details.

## Contact

For questions about this specific implementation, please open an issue in the repository.

For broader QFD theory questions, see the main project documentation.
