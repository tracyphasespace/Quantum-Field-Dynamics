qfd_cmb.kernels module
======================

.. automodule:: qfd_cmb.kernels
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: qfd_cmb.kernels.sin2_mueller_coeffs

.. autofunction:: qfd_cmb.kernels.te_correlation_phase

The kernels module provides scattering kernel functions and correlation coefficients 
for computing polarized CMB spectra with photon-photon scattering.

sin2_mueller_coeffs
~~~~~~~~~~~~~~~~~~~

Computes intensity and polarization weights for a sin²(θ) scattering kernel, 
which models the angular dependence of photon-photon scattering.

Parameters
^^^^^^^^^^

* **mu** : array_like
    Cosine of scattering angle, μ = cos(θ)

Returns
^^^^^^^

* **w_T** : ndarray
    Intensity weight factor (1 - μ²)
* **w_E** : ndarray  
    Polarization efficiency factor

Notes
^^^^^

The sin²(θ) kernel arises naturally in photon-photon scattering calculations. 
The intensity weight is:

.. math::

   w_T = \\sin^2(\\theta) = 1 - \\mu^2

The polarization efficiency follows a similar functional form but may include 
additional physics-dependent factors.

te_correlation_phase
~~~~~~~~~~~~~~~~~~~~

Computes the scale-dependent TE correlation coefficient that introduces sign flips 
and correlations between temperature and E-mode polarization.

Parameters
^^^^^^^^^^

* **k** : array_like
    Wavenumber values in 1/Mpc
* **rpsi** : float
    Characteristic oscillation scale in Mpc
* **ell** : int or array_like
    Multipole moment(s)
* **chi_star** : float
    Comoving distance to last scattering in Mpc
* **sigma_phase** : float, optional
    Damping parameter for large-scale suppression (default: 0.16)
* **phi0** : float, optional
    Phase offset in radians (default: 0.0)

Returns
^^^^^^^

* **rho** : ndarray
    TE correlation coefficient ρₗ

Notes
^^^^^

The correlation coefficient is computed as:

.. math::

   \\rho_\\ell = \\cos(k_{\\text{eff}} r_\\psi + \\phi_0) \\exp\\left[-\\left(\\sigma_{\\text{phase}} \\frac{\\ell}{200}\\right)^2\\right]

where :math:`k_{\\text{eff}} = (\\ell + 0.5)/\\chi_*` for the Limber approximation.

Examples
~~~~~~~~

.. code-block:: python

   import numpy as np
   from qfd_cmb.kernels import sin2_mueller_coeffs, te_correlation_phase
   
   # Compute scattering weights
   mu = np.linspace(-1, 1, 100)
   w_T, w_E = sin2_mueller_coeffs(mu)
   
   # Compute TE correlation for specific multipoles
   k = np.logspace(-4, 1, 50)
   ells = [10, 100, 1000]
   
   for ell in ells:
       rho = te_correlation_phase(k, rpsi=147.0, ell=ell, chi_star=14065.0)
       print(f"ell={ell}: rho_max={rho.max():.3f}, rho_min={rho.min():.3f}")