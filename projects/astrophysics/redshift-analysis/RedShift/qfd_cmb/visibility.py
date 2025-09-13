"""
Visibility and window functions for CMB analysis.

This module provides Gaussian visibility and window functions used in
photon-photon scattering calculations for CMB temperature and polarization.
"""

import numpy as np

def gaussian_visibility(eta, eta_star, sigma_eta):
    """
    Compute normalized Gaussian visibility function g(η).
    
    Parameters
    ----------
    eta : array_like
        Conformal time values
    eta_star : float
        Central conformal time for the Gaussian
    sigma_eta : float
        Width parameter for the Gaussian in conformal time
        
    Returns
    -------
    ndarray
        Normalized Gaussian visibility function values
        
    Notes
    -----
    The visibility function is L2-normalized: ∫ g²(η) dη = 1
    """
    x = (eta - eta_star)/sigma_eta
    g = np.exp(-0.5*x*x)
    norm = np.sqrt(np.trapz(g**2, eta))
    return g / (norm + 1e-30)

def gaussian_window_chi(chi, chi_star, sigma_chi):
    """
    Compute normalized Gaussian window function in comoving distance.
    
    Parameters
    ----------
    chi : array_like
        Comoving distance values in Mpc
    chi_star : float
        Central comoving distance for the Gaussian in Mpc
    sigma_chi : float
        Width parameter for the Gaussian in Mpc
        
    Returns
    -------
    ndarray
        Normalized Gaussian window function values
        
    Notes
    -----
    The window function is L2-normalized: ∫ W²(χ) dχ = 1
    """
    x = (chi - chi_star)/sigma_chi
    W = np.exp(-0.5*x*x)
    norm = np.sqrt(np.trapz(W**2, chi))
    return W / (norm + 1e-30)
