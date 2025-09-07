QFD CMB Module Documentation
============================

Welcome to the QFD CMB Module documentation. This package provides tools for computing 
QFD-based CMB spectra using photon-photon scattering kernels in place of Thomson scattering.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/modules
   examples
   contributing

Overview
--------

The QFD CMB Module implements photon-photon scattering projection for CMB TT/TE/EE spectra.
It provides a complete framework for:

* Oscillatory power spectrum models with tilt and modulation
* Gaussian visibility functions for last scattering surface
* Mueller matrix coefficients for intensity and polarization
* Limber and line-of-sight projection methods
* Plotting utilities for CMB spectra

Key Features
------------

* **Scientific Accuracy**: Implements physically motivated QFD models
* **Numerical Stability**: Robust handling of edge cases and numerical precision
* **Flexible Interface**: Easy-to-use functions with sensible defaults
* **Comprehensive Testing**: Extensive validation against reference values
* **Well Documented**: Complete API documentation with examples

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from qfd_cmb import oscillatory_psik, gaussian_window_chi, project_limber

   # Define wavenumber and distance grids
   k = np.logspace(-4, 1, 100)
   chi = np.linspace(100, 15000, 200)
   
   # Create oscillatory power spectrum
   Pk = oscillatory_psik(k, rpsi=147.0)
   
   # Define visibility window
   W = gaussian_window_chi(chi, chi_star=14065.0, sigma_chi=250.0)
   
   # Project to get CMB spectrum
   ells = np.arange(2, 3000)
   Cl = project_limber(ells, lambda k: oscillatory_psik(k), W, chi)

Installation
------------

Install the package using pip:

.. code-block:: bash

   pip install qfd-cmb

Or install from source:

.. code-block:: bash

   git clone https://github.com/qfd-project/qfd-cmb.git
   cd qfd-cmb
   pip install -e .

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`