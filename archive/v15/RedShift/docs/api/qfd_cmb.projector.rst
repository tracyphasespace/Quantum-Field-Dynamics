qfd_cmb.projector module
========================

.. automodule:: qfd_cmb.projector
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: qfd_cmb.projector.project_limber

.. autofunction:: qfd_cmb.projector.los_transfer

.. autofunction:: qfd_cmb.projector.project_los

The projector module implements the mathematical machinery for converting 3D power 
spectra into 2D angular power spectra on the sky, using both Limber approximation 
and exact line-of-sight integration methods.

project_limber
~~~~~~~~~~~~~~

Performs Limber projection to compute angular power spectra from 3D power spectra. 
This is the primary function for high-ℓ calculations.

Parameters
^^^^^^^^^^

* **ells** : array_like
    Multipole moments ℓ
* **Pk_func** : callable
    Function that returns P(k) given wavenumber k
* **W_chi** : array_like
    Window function W(χ) evaluated on chi_grid
* **chi_grid** : array_like
    Comoving distance grid in Mpc

Returns
^^^^^^^

* **C** : ndarray
    Angular power spectrum Cₗ values

Notes
^^^^^

The Limber approximation computes:

.. math::

   C_\\ell \\approx \\int d\\chi \\frac{W(\\chi)^2}{\\chi^2} P\\left(\\frac{\\ell + 1/2}{\\chi}\\right)

This is accurate for ℓ ≳ 10 and provides significant computational speedup compared 
to exact line-of-sight integration.

los_transfer
~~~~~~~~~~~~

Computes line-of-sight transfer functions Δₗ(k) using exact spherical Bessel function 
integration. Used for accurate low-ℓ calculations.

Parameters
^^^^^^^^^^

* **ells** : array_like
    Multipole moments ℓ
* **k_grid** : array_like
    Wavenumber grid in 1/Mpc
* **eta_grid** : array_like
    Conformal time grid
* **S_func** : callable
    Source function S(k,η)

Returns
^^^^^^^

* **Delta** : ndarray
    Transfer functions Δₗ(k) with shape (len(ells), len(k_grid))

Notes
^^^^^

Computes the exact integral:

.. math::

   \\Delta_\\ell(k) = \\int d\\eta \\, S(k,\\eta) \\, j_\\ell[k(\\eta_0 - \\eta)]

where jₗ are spherical Bessel functions and η₀ is the present conformal time.

project_los
~~~~~~~~~~~

Projects transfer functions to angular power spectra using exact line-of-sight method.

Parameters
^^^^^^^^^^

* **ells** : array_like
    Multipole moments ℓ
* **k_grid** : array_like
    Wavenumber grid in 1/Mpc  
* **Pk_func** : callable
    Function that returns P(k) given wavenumber k
* **DeltaX** : array_like
    Transfer functions for field X with shape (len(ells), len(k_grid))
* **DeltaY** : array_like
    Transfer functions for field Y with shape (len(ells), len(k_grid))

Returns
^^^^^^^

* **C** : ndarray
    Cross-power spectrum Cₗˣʸ values

Notes
^^^^^

Computes the exact projection:

.. math::

   C_\\ell^{XY} = \\int \\frac{k^2 dk}{2\\pi^2} P(k) \\Delta_\\ell^X(k) \\Delta_\\ell^Y(k)

For auto-spectra, use DeltaX = DeltaY. For cross-spectra like TE, use different 
transfer functions for temperature and E-mode polarization.

Examples
~~~~~~~~

.. code-block:: python

   import numpy as np
   from qfd_cmb.projector import project_limber, los_transfer, project_los
   from qfd_cmb.ppsi_models import oscillatory_psik
   from qfd_cmb.visibility import gaussian_window_chi
   
   # Limber projection example
   ells = np.arange(2, 3000)
   chi = np.linspace(100, 15000, 200)
   W = gaussian_window_chi(chi, 14065.0, 250.0)
   
   Pk_func = lambda k: oscillatory_psik(k, rpsi=147.0)
   Cl_limber = project_limber(ells, Pk_func, W, chi)
   
   # Line-of-sight projection example (for low ells)
   ells_low = np.arange(2, 50)
   k_grid = np.logspace(-4, 0, 100)
   eta_grid = np.linspace(0, 14000, 200)
   
   # Define a simple source function
   def S_func(k, eta):
       return np.exp(-k * eta / 1000)  # Example source
   
   # Compute transfer functions
   Delta = los_transfer(ells_low, k_grid, eta_grid, S_func)
   
   # Project to get spectrum
   Cl_los = project_los(ells_low, k_grid, Pk_func, Delta, Delta)