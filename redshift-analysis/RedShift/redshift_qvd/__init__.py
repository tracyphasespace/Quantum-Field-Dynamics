"""
RedShift QVD Cosmological Model
==============================

A physics-based alternative to dark energy using wavelength-independent 
QVD interactions between high-energy photons and the quantum vacuum field.

This package provides:
- Direct photon-ψ field interactions (no plasma required)
- Wavelength-independent cosmological dimming
- CMB energy transfer mechanism
- Alternative to dark energy acceleration
- Comprehensive numerical stability and validation

Key Physics Distinction:
- RedShift: Direct photon-ψ vacuum interactions (wavelength-independent)
- Supernova: Plasma-mediated scattering (wavelength-dependent)

Example usage:
    >>> from redshift_qvd import RedshiftAnalyzer
    >>> analyzer = RedshiftAnalyzer(qvd_coupling=0.85, redshift_power=0.6)
    >>> results = analyzer.run_complete_analysis()
"""

__version__ = "1.0.0"
__author__ = "PhaseSpace"
__email__ = "contact@phasespace.com"
__license__ = "Proprietary"

# Import main classes for easy access
from .redshift_analyzer import RedshiftAnalyzer
from .cosmology import QVDCosmology
from .physics import QVDVacuumPhysics
from .visualization import RedshiftPlotter
from .cmb_coupling import CMBEnergyTransfer

# Import numerical safety utilities
from .numerical_safety import (
    safe_power,
    safe_log10,
    safe_exp,
    safe_divide,
    safe_sqrt,
    validate_finite,
    clamp_to_range
)

from .physical_bounds import (
    CosmologicalBounds,
    BoundsEnforcer,
    create_safe_cosmological_state
)

from .error_handling import (
    setup_redshift_logging,
    ErrorReporter
)

# Define what gets imported with "from redshift_qvd import *"
__all__ = [
    # Main classes
    'RedshiftAnalyzer',
    'QVDCosmology',
    'QVDVacuumPhysics',
    'RedshiftPlotter',
    'CMBEnergyTransfer',
    
    # Numerical safety
    'safe_power',
    'safe_log10', 
    'safe_exp',
    'safe_divide',
    'safe_sqrt',
    'validate_finite',
    'clamp_to_range',
    
    # Physical bounds
    'CosmologicalBounds',
    'BoundsEnforcer',
    'create_safe_cosmological_state',
    
    # Error handling
    'setup_redshift_logging',
    'ErrorReporter',
    
    # Package metadata
    '__version__',
    '__author__',
    '__email__',
    '__license__'
]