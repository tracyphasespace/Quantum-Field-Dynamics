Quick Start Guide
=================

This guide will get you up and running with the QFD CMB Module in just a few minutes.

Basic Usage
-----------

The QFD CMB Module provides a simple interface for computing CMB spectra using 
photon-photon scattering models. Here's a minimal example:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from qfd_cmb import oscillatory_psik, gaussian_window_chi, project_limber

   # Define wavenumber grid (in 1/Mpc)
   k = np.logspace(-4, 1, 100)
   
   # Create oscillatory power spectrum with default parameters
   Pk_values = oscillatory_psik(k)
   
   # Plot the power spectrum
   plt.figure(figsize=(8, 6))
   plt.loglog(k, Pk_values)
   plt.xlabel('k [1/Mpc]')
   plt.ylabel('P(k)')
   plt.title('Oscillatory Power Spectrum')
   plt.show()

Computing CMB Spectra
---------------------

To compute CMB angular power spectra, you need to define the visibility function 
and perform the projection:

.. code-block:: python

   import numpy as np
   from qfd_cmb import oscillatory_psik, gaussian_window_chi, project_limber

   # Define comoving distance grid (in Mpc)
   chi = np.linspace(100, 15000, 200)
   
   # Define visibility window centered at last scattering
   chi_star = 14065.0  # Comoving distance to last scattering
   sigma_chi = 250.0   # Width of last scattering surface
   W = gaussian_window_chi(chi, chi_star, sigma_chi)
   
   # Define multipole range
   ells = np.arange(2, 3000)
   
   # Project to get CMB TT spectrum
   def Pk_func(k):
       return oscillatory_psik(k, rpsi=147.0, Aosc=0.55)
   
   Ctt = project_limber(ells, Pk_func, W, chi)

Plotting Results
----------------

The module includes convenient plotting functions:

.. code-block:: python

   from qfd_cmb.figures import plot_TT, plot_EE, plot_TE
   
   # Plot TT spectrum
   plot_TT(ells, Ctt, 'tt_spectrum.png')
   
   # For EE and TE, you would compute them similarly
   # Cee = project_limber(ells, Pk_func_EE, W, chi)
   # Cte = project_limber(ells, Pk_func_TE, W, chi)
   # plot_EE(ells, Cee, 'ee_spectrum.png')
   # plot_TE(ells, Cte, 'te_spectrum.png')

Working with Parameters
-----------------------

The oscillatory power spectrum model has several adjustable parameters:

.. code-block:: python

   from qfd_cmb import oscillatory_psik
   
   # Default parameters
   k = np.logspace(-4, 1, 100)
   Pk_default = oscillatory_psik(k)
   
   # Custom parameters
   Pk_custom = oscillatory_psik(
       k,
       A=1.2,           # Amplitude
       ns=0.965,        # Spectral index
       rpsi=150.0,      # Oscillation scale (Mpc)
       Aosc=0.6,        # Oscillation amplitude
       sigma_osc=0.03   # Oscillation damping
   )

Advanced Features
-----------------

Mueller Matrix Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For polarization calculations, use the Mueller matrix coefficients:

.. code-block:: python

   from qfd_cmb.kernels import sin2_mueller_coeffs, te_correlation_phase
   
   # Compute intensity and polarization weights
   mu = np.linspace(-1, 1, 100)  # cos(theta)
   w_T, w_E = sin2_mueller_coeffs(mu)
   
   # Compute TE correlation coefficient
   rho = te_correlation_phase(k, rpsi=147.0, ell=100, chi_star=14065.0)

Line-of-Sight Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

For more accurate calculations at low multipoles, use line-of-sight integration:

.. code-block:: python

   from qfd_cmb.projector import los_transfer, project_los
   
   # Define conformal time grid
   eta = np.linspace(0, 14000, 200)
   
   # Define source function (example)
   def S_func(k, eta):
       return np.exp(-k * eta / 1000)  # Simple example
   
   # Compute transfer functions
   Delta = los_transfer(ells[:10], k, eta, S_func)
   
   # Project to get spectrum
   Cl_los = project_los(ells[:10], k, oscillatory_psik, Delta, Delta)

Next Steps
----------

* Check out the :doc:`examples` for more detailed tutorials
* Read the :doc:`api/modules` for complete function documentation
* See :doc:`contributing` if you want to contribute to the project

Common Patterns
---------------

Here are some common usage patterns:

**Parameter Sweeps**

.. code-block:: python

   rpsi_values = [140, 147, 154]
   spectra = {}
   
   for rpsi in rpsi_values:
       Pk_func = lambda k: oscillatory_psik(k, rpsi=rpsi)
       Cl = project_limber(ells, Pk_func, W, chi)
       spectra[rpsi] = Cl

**Comparing Models**

.. code-block:: python

   # Standard model (no oscillations)
   Pk_standard = lambda k: oscillatory_psik(k, Aosc=0.0)
   Cl_standard = project_limber(ells, Pk_standard, W, chi)
   
   # QFD model (with oscillations)
   Pk_qfd = lambda k: oscillatory_psik(k, Aosc=0.55)
   Cl_qfd = project_limber(ells, Pk_qfd, W, chi)
   
   # Plot comparison
   plt.figure()
   plt.loglog(ells, ells*(ells+1)*Cl_standard, label='Standard')
   plt.loglog(ells, ells*(ells+1)*Cl_qfd, label='QFD')
   plt.legend()
   plt.show()