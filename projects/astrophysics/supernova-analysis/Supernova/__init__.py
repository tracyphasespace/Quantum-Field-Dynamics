"""
Supernova QVD Scattering Model
=============================

A numerically stable, production-ready implementation of the E144-scaled 
Quantum Vacuum Dynamics (QVD) model for supernova light curve analysis.

This package provides:
- Robust numerical calculations with 100% finite results
- Physical bounds enforcement for all parameters
- Comprehensive error handling and logging
- High-performance supernova curve generation
- Multi-wavelength analysis capabilities
- Extensive validation and testing framework

Example usage:
    >>> from supernova_qvd_scattering import E144ExperimentalData, SupernovaParameters, E144ScaledQVDModel
    >>> e144_data = E144ExperimentalData()
    >>> sn_params = SupernovaParameters()
    >>> model = E144ScaledQVDModel(e144_data, sn_params)
    >>> curve = model.generate_luminance_curve(distance_Mpc=100.0, wavelength_nm=500.0)
"""

__version__ = "1.0.0"
__author__ = "PhaseSpace"
__email__ = "contact@phasespace.com"
__license__ = "Proprietary"

# Import main classes for easy access
from .supernova_qvd_scattering import (
    E144ExperimentalData,
    SupernovaParameters, 
    E144ScaledQVDModel
)

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
    PhysicalBounds,
    BoundsEnforcer,
    create_safe_plasma_state,
    create_safe_scattering_results
)

from .error_handling import (
    setup_qvd_logging,
    ErrorReporter
)

# Define what gets imported with "from supernova import *"
__all__ = [
    # Main classes
    'E144ExperimentalData',
    'SupernovaParameters',
    'E144ScaledQVDModel',
    
    # Numerical safety
    'safe_power',
    'safe_log10', 
    'safe_exp',
    'safe_divide',
    'safe_sqrt',
    'validate_finite',
    'clamp_to_range',
    
    # Physical bounds
    'PhysicalBounds',
    'BoundsEnforcer',
    'create_safe_plasma_state',
    'create_safe_scattering_results',
    
    # Error handling
    'setup_qvd_logging',
    'ErrorReporter',
    
    # Package metadata
    '__version__',
    '__author__',
    '__email__',
    '__license__'
]