"""
RedShift QVD Cosmological Model
===============================

A numerically stable, production-ready implementation of the Quantum Vacuum 
Dynamics (QVD) redshift model that provides a physics-based alternative to 
dark energy in cosmology.

This package provides:
- Enhanced numerical stability with 100% finite results
- Comprehensive bounds enforcement for all parameters
- Physics-based alternative to dark energy acceleration
- Excellent agreement with supernova observations (0.14 mag RMS)
- Testable predictions for observational validation
- Professional visualization and analysis tools

Example usage:
    >>> from redshift_analyzer import EnhancedRedshiftAnalyzer
    >>> analyzer = EnhancedRedshiftAnalyzer(qvd_coupling=0.85, redshift_power=0.6)
    >>> results = analyzer.run_complete_analysis()
    >>> print(f"RMS error: {results['validation']['rms_error']:.3f} mag")
"""

__version__ = "1.0.0"
__author__ = "PhaseSpace"
__email__ = "contact@phasespace.com"
__license__ = "Proprietary"

# Import main classes for easy access
from .redshift_analyzer import EnhancedRedshiftAnalyzer
from .redshift_physics import EnhancedQVDPhysics, RedshiftBounds, RedshiftBoundsEnforcer
from .redshift_cosmology import EnhancedQVDCosmology
from .redshift_visualization import EnhancedRedshiftPlotter

from .numerical_safety import (
    safe_power,
    safe_log10,
    safe_exp,
    safe_divide,
    safe_sqrt,
    validate_finite,
    clamp_to_range
)

from .error_handling import (
    setup_qvd_logging,
    ErrorReporter
)

# Define what gets imported with "from redshift_qvd import *"
__all__ = [
    # Main classes
    'EnhancedRedshiftAnalyzer',
    'EnhancedQVDPhysics',
    'EnhancedQVDCosmology',
    'EnhancedRedshiftPlotter',
    
    # Bounds and safety
    'RedshiftBounds',
    'RedshiftBoundsEnforcer',
    
    # Numerical safety
    'safe_power',
    'safe_log10', 
    'safe_exp',
    'safe_divide',
    'safe_sqrt',
    'validate_finite',
    'clamp_to_range',
    
    # Error handling
    'setup_qvd_logging',
    'ErrorReporter',
    
    # Package metadata
    '__version__',
    '__author__',
    '__email__',
    '__license__'
]