qfd_cmb.ppsi_models module
===========================

.. automodule:: qfd_cmb.ppsi_models
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: qfd_cmb.ppsi_models.oscillatory_psik

The oscillatory_psik function implements the core QFD power spectrum model with 
oscillatory features. This function combines a power-law base spectrum with 
Gaussian-damped cosine modulations to model the effects of photon-photon scattering.

Parameters
~~~~~~~~~~

* **k** : array_like
    Wavenumber values in units of 1/Mpc
* **A** : float, optional
    Overall amplitude normalization (default: 1.0)
* **ns** : float, optional  
    Spectral index for the power-law component (default: 0.96)
* **rpsi** : float, optional
    Characteristic scale for oscillations in Mpc (default: 147.0)
* **Aosc** : float, optional
    Amplitude of oscillatory modulation (default: 0.55)
* **sigma_osc** : float, optional
    Damping scale for oscillations (default: 0.025)

Returns
~~~~~~~

* **Pk** : ndarray
    Power spectrum values P(k) at the input wavenumbers

Notes
~~~~~

The function implements the model:

.. math::

   P_\\psi(k) = A \\cdot k^{n_s - 1} \\cdot \\left[1 + A_{\\text{osc}} \\cos(k r_\\psi) e^{-(k \\sigma_{\\text{osc}})^2}\\right]^2

The squaring of the bracketed term ensures positivity of the power spectrum.

Examples
~~~~~~~~

.. code-block:: python

   import numpy as np
   from qfd_cmb.ppsi_models import oscillatory_psik
   
   # Basic usage with default parameters
   k = np.logspace(-4, 1, 100)
   Pk = oscillatory_psik(k)
   
   # Custom parameters
   Pk_custom = oscillatory_psik(k, A=1.2, ns=0.965, rpsi=150.0)