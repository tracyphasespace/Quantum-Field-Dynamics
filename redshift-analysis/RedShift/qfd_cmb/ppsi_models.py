"""
Photon-photon scattering power spectrum models for CMB analysis.

This module provides functions for computing oscillatory power spectra
that arise from photon-photon scattering effects in cosmology.
"""

import numpy as np

def oscillatory_psik(k, A=1.0, ns=0.96, rpsi=147.0, Aosc=0.55, sigma_osc=0.025):
    """
    Compute oscillatory power spectrum P_ψ(k) with tilt and Gaussian-damped cosine modulation.
    
    This function calculates the photon-photon scattering power spectrum with
    oscillatory features that can affect CMB temperature and polarization spectra.
    
    Parameters
    ----------
    k : array_like
        Wavenumber values in units of 1/Mpc
    A : float, optional
        Overall amplitude normalization (default: 1.0)
    ns : float, optional
        Spectral index for the power-law component (default: 0.96)
    rpsi : float, optional
        Oscillation scale in Mpc (default: 147.0)
    Aosc : float, optional
        Oscillation amplitude (default: 0.55)
    sigma_osc : float, optional
        Gaussian damping scale for oscillations (default: 0.025)
    
    Returns
    -------
    ndarray
        Power spectrum values P_ψ(k) at the input wavenumbers
        
    Notes
    -----
    The power spectrum is computed as:
    P_ψ(k) = A * k^(ns-1) * [1 + Aosc * cos(k * rpsi) * exp(-(k * sigma_osc)^2)]^2
    
    The squaring ensures positivity of the power spectrum.
    """
    k = np.asarray(k)
    base = (k + 1e-16)**(ns - 1.0)
    osc  = (1.0 + Aosc * np.cos(k * rpsi) * np.exp(-(k * sigma_osc)**2))
    return A * base * osc**2
