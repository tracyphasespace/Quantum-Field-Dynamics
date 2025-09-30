"""
Scattering kernels and correlation functions for photon-photon interactions.

This module provides the Mueller matrix coefficients and correlation functions
needed for computing CMB temperature and polarization spectra with photon-photon
scattering effects.
"""

import numpy as np

def sin2_mueller_coeffs(mu):
    """
    Compute Mueller matrix coefficients for sin²(θ) scattering kernel.
    
    Parameters
    ----------
    mu : array_like
        Cosine of scattering angle, mu = cos(θ)
        
    Returns
    -------
    w_T : ndarray
        Temperature (intensity) scattering weights
    w_E : ndarray
        Polarization (E-mode) scattering weights
        
    Notes
    -----
    For photon-photon scattering, the differential cross-section goes as sin²(θ),
    leading to scattering weights proportional to (1 - μ²).
    """
    w_T = 1.0 - mu**2          # intensity weight
    w_E = (1.0 - mu**2)        # polarization efficiency (schematic)
    return w_T, w_E

def te_correlation_phase(k, rpsi, ell, chi_star, sigma_phase=0.16, phi0=0.0):
    """
    Compute scale-dependent TE correlation coefficient with oscillatory features.
    
    Parameters
    ----------
    k : float
        Wavenumber in 1/Mpc (used as fallback if chi_star=0)
    rpsi : float
        Oscillation scale in Mpc
    ell : array_like
        Multipole values
    chi_star : float
        Comoving distance to last scattering in Mpc
    sigma_phase : float, optional
        Damping parameter for high-ell suppression (default: 0.16)
    phi0 : float, optional
        Phase offset in radians (default: 0.0)
        
    Returns
    -------
    ndarray
        TE correlation coefficient ρ_ℓ as function of multipole
        
    Notes
    -----
    The correlation coefficient includes oscillatory features that can lead to
    sign changes in the TE spectrum, mimicking observed CMB data features.
    """
    keff = (ell + 0.5)/chi_star if chi_star>0 else k
    rho = np.cos(keff * rpsi + phi0) * np.exp(- (sigma_phase * (ell/200.0))**2)
    return rho
