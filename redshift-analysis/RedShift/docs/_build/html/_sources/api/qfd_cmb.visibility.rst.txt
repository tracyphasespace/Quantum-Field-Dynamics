qfd_cmb.visibility module
=========================

.. automodule:: qfd_cmb.visibility
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: qfd_cmb.visibility.gaussian_visibility

.. autofunction:: qfd_cmb.visibility.gaussian_window_chi

The visibility functions define the weighting of different epochs or distances in 
the CMB calculation. These functions are normalized to ensure proper integration.

gaussian_visibility
~~~~~~~~~~~~~~~~~~~

Implements a Gaussian visibility function in conformal time η, typically used for 
modeling the last scattering surface.

Parameters
^^^^^^^^^^

* **eta** : array_like
    Conformal time values
* **eta_star** : float
    Central conformal time (peak of visibility)
* **sigma_eta** : float
    Width of the visibility function

Returns
^^^^^^^

* **g** : ndarray
    Normalized Gaussian visibility function

gaussian_window_chi  
~~~~~~~~~~~~~~~~~~~

Implements a Gaussian window function in comoving distance χ, used for Limber 
projection calculations.

Parameters
^^^^^^^^^^

* **chi** : array_like
    Comoving distance values in Mpc
* **chi_star** : float
    Central comoving distance in Mpc (typically ~14065 Mpc for last scattering)
* **sigma_chi** : float
    Width of the window function in Mpc (typically ~250 Mpc)

Returns
^^^^^^^

* **W** : ndarray
    L2-normalized Gaussian window function

Notes
~~~~~

Both functions are L2-normalized such that:

.. math::

   \\int g(\\eta)^2 d\\eta = 1

   \\int W(\\chi)^2 d\\chi = 1

This normalization ensures proper power spectrum normalization in projection integrals.

Examples
~~~~~~~~

.. code-block:: python

   import numpy as np
   from qfd_cmb.visibility import gaussian_window_chi
   
   # Define comoving distance grid
   chi = np.linspace(100, 15000, 200)
   
   # Create visibility window for last scattering
   chi_star = 14065.0  # Comoving distance to last scattering
   sigma_chi = 250.0   # Width of last scattering surface
   W = gaussian_window_chi(chi, chi_star, sigma_chi)
   
   # Verify normalization
   norm = np.sqrt(np.trapz(W**2, chi))
   print(f"Normalization: {norm:.6f}")  # Should be ~1.0