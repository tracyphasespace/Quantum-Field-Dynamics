"""
Complete QFD Energy Functional with Gradient Density and Emergent Time

This package implements hierarchical energy functionals for lepton mass prediction:

Stage 1: E = ∫ [½ξ|∇ρ|² + β(δρ)²] dV  (gradient density)
Stage 2: E = ∫ [½ξ|∇ρ|² + β(δρ)² + τ(∂ρ/∂t)²] dV  (emergent time)
Stage 3: E = ∫ [... + E_EM[ρ] + E_swirl] dV  (full EM response)

Goal: Resolve 3% offset between V22 fit (β≈3.15) and Golden Loop prediction (β=3.058)
"""

__version__ = "0.1.0"
__author__ = "QFD Spectral Gap Project"

from .functionals import (
    gradient_energy_functional,
    temporal_energy_functional,
    compute_mass_from_energy
)

from .solvers import (
    solve_euler_lagrange,
    hill_vortex_profile,
    integrate_energy
)

from .mcmc_stage1_gradient import (
    run_stage1_mcmc,
    analyze_stage1_results
)

__all__ = [
    'gradient_energy_functional',
    'temporal_energy_functional',
    'compute_mass_from_energy',
    'solve_euler_lagrange',
    'hill_vortex_profile',
    'integrate_energy',
    'run_stage1_mcmc',
    'analyze_stage1_results'
]
