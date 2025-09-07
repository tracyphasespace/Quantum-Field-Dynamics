qfd_cmb.figures module
=====================

.. automodule:: qfd_cmb.figures
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: qfd_cmb.figures.plot_TT

.. autofunction:: qfd_cmb.figures.plot_EE

.. autofunction:: qfd_cmb.figures.plot_TE

The figures module provides convenient plotting functions for visualizing CMB angular 
power spectra with appropriate scaling and formatting.

plot_TT
~~~~~~~

Creates a log-log plot of the TT (temperature-temperature) angular power spectrum.

Parameters
^^^^^^^^^^

* **ells** : array_like
    Multipole moments ℓ
* **Ctt** : array_like
    TT angular power spectrum values Cₗᵀᵀ
* **path** : str
    Output file path for saving the plot

Notes
^^^^^

The function plots ℓ(ℓ+1)Cₗᵀᵀ vs ℓ on log-log axes, which is the standard 
presentation for CMB temperature spectra. This scaling removes the geometric 
factor and makes features more visible.

plot_EE
~~~~~~~

Creates a log-log plot of the EE (E-mode polarization) angular power spectrum.

Parameters
^^^^^^^^^^

* **ells** : array_like
    Multipole moments ℓ
* **Cee** : array_like
    EE angular power spectrum values Cₗᴱᴱ
* **path** : str
    Output file path for saving the plot

Notes
^^^^^

Similar to plot_TT, this function plots ℓ(ℓ+1)Cₗᴱᴱ vs ℓ on log-log axes. 
The EE spectrum is typically smaller in amplitude than the TT spectrum.

plot_TE
~~~~~~~

Creates a semi-log plot of the TE (temperature-E-mode cross-correlation) spectrum.

Parameters
^^^^^^^^^^

* **ells** : array_like
    Multipole moments ℓ
* **Cte** : array_like
    TE angular power spectrum values Cₗᵀᴱ
* **path** : str
    Output file path for saving the plot

Notes
^^^^^

The TE spectrum can be positive or negative, so this function plots 
sign(Cₗᵀᴱ) × ℓ(ℓ+1)|Cₗᵀᴱ| vs ℓ on semi-log axes. A horizontal line at zero 
is included for reference.

Common Features
~~~~~~~~~~~~~~~

All plotting functions:

* Save plots with 200 DPI for publication quality
* Use tight layout for optimal spacing
* Close the figure after saving to prevent memory leaks
* Include appropriate axis labels with LaTeX formatting
* Add descriptive titles

Examples
~~~~~~~~

.. code-block:: python

   import numpy as np
   from qfd_cmb.figures import plot_TT, plot_EE, plot_TE
   from qfd_cmb.projector import project_limber
   from qfd_cmb.ppsi_models import oscillatory_psik
   from qfd_cmb.visibility import gaussian_window_chi
   
   # Compute spectra (example)
   ells = np.arange(2, 3000)
   chi = np.linspace(100, 15000, 200)
   W = gaussian_window_chi(chi, 14065.0, 250.0)
   
   # TT spectrum
   Pk_func = lambda k: oscillatory_psik(k, rpsi=147.0)
   Ctt = project_limber(ells, Pk_func, W, chi)
   plot_TT(ells, Ctt, 'output/tt_spectrum.png')
   
   # EE spectrum (would need polarization-specific calculation)
   # Cee = project_limber(ells, Pk_func_EE, W, chi)
   # plot_EE(ells, Cee, 'output/ee_spectrum.png')
   
   # TE spectrum (would need cross-correlation calculation)  
   # Cte = project_limber(ells, Pk_func_TE, W, chi)
   # plot_TE(ells, Cte, 'output/te_spectrum.png')

Customization
~~~~~~~~~~~~~

For more control over plot appearance, you can use matplotlib directly:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Custom TT plot
   plt.figure(figsize=(10, 6))
   plt.loglog(ells, ells*(ells+1)*Ctt, 'b-', linewidth=2, label='QFD Model')
   plt.xlabel(r'Multipole $\\ell$', fontsize=14)
   plt.ylabel(r'$\\ell(\\ell+1)C_\\ell^{TT}$ [$\\mu$K$^2$]', fontsize=14)
   plt.title('Temperature Angular Power Spectrum', fontsize=16)
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('custom_tt_plot.png', dpi=300)
   plt.show()