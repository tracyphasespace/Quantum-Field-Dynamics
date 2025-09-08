"""
QFD Solvers Module
================

Phoenix core solvers for quantum field dynamics simulations.

Theoretical Framework Capabilities:
- Phoenix Core Hamiltonian field evolution
- QFD Zeeman experiments for g-2 predictions
- Isomer theory: electron/muon/tau as unified field states
- Excited state search and analysis
"""

from .phoenix_solver import solve_psi_field, load_particle_constants
from .zeeman_experiments import ZeemanExperiment, IsomerZeemanAnalysis, ExcitedStateZeemanSolver
from .backend import get_backend
from .hamiltonian import PhoenixHamiltonian

__all__ = [
    'solve_psi_field', 
    'load_particle_constants',
    'ZeemanExperiment',
    'IsomerZeemanAnalysis', 
    'ExcitedStateZeemanSolver',
    'get_backend',
    'PhoenixHamiltonian'
]