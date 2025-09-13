"""
QFD CMB Module - Photon-Photon Scattering Projection for CMB TT/TE/EE Spectra

This package provides tools for computing QFD-based CMB spectra using
photon-photon scattering kernels in place of Thomson scattering.
"""

__version__ = "0.1.0"
__author__ = "QFD Project"
__email__ = "contact@qfd-project.org"

from . import ppsi_models, visibility, kernels, projector, figures

# Make key functions available at package level
from .ppsi_models import oscillatory_psik
from .visibility import gaussian_window_chi
from .kernels import sin2_mueller_coeffs, te_correlation_phase
from .projector import project_limber
from .figures import plot_TT, plot_EE, plot_TE

__all__ = [
    "ppsi_models",
    "visibility", 
    "kernels",
    "projector",
    "figures",
    "oscillatory_psik",
    "gaussian_window_chi",
    "sin2_mueller_coeffs",
    "te_correlation_phase",
    "project_limber",
    "plot_TT",
    "plot_EE",
    "plot_TE",
]
