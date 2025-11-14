Examples
========

This section provides detailed examples of using the QFD CMB Module for various 
scientific calculations.

.. toctree::
   :maxdepth: 2

   examples/basic_usage
   examples/parameter_studies
   examples/advanced_calculations

Basic Usage Examples
--------------------

Simple Power Spectrum Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from qfd_cmb import oscillatory_psik
   
   # Define wavenumber range
   k = np.logspace(-4, 1, 200)
   
   # Compute power spectrum with default parameters
   Pk = oscillatory_psik(k)
   
   # Plot the result
   plt.figure(figsize=(10, 6))
   plt.loglog(k, Pk, 'b-', linewidth=2)
   plt.xlabel('k [1/Mpc]')
   plt.ylabel('P(k)')
   plt.title('QFD Oscillatory Power Spectrum')
   plt.grid(True, alpha=0.3)
   plt.show()

Computing CMB Angular Power Spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from qfd_cmb import oscillatory_psik, gaussian_window_chi, project_limber
   from qfd_cmb.figures import plot_TT
   
   # Define grids
   ells = np.arange(2, 3000)
   chi = np.linspace(100, 15000, 300)
   
   # Define visibility window
   chi_star = 14065.0  # Comoving distance to last scattering
   sigma_chi = 250.0   # Width of last scattering surface
   W = gaussian_window_chi(chi, chi_star, sigma_chi)
   
   # Define power spectrum function
   def Pk_func(k):
       return oscillatory_psik(k, rpsi=147.0, Aosc=0.55)
   
   # Compute TT spectrum
   Ctt = project_limber(ells, Pk_func, W, chi)
   
   # Plot the result
   plot_TT(ells, Ctt, 'tt_spectrum.png')
   
   # Also create custom plot
   import matplotlib.pyplot as plt
   plt.figure(figsize=(12, 8))
   plt.loglog(ells, ells*(ells+1)*Ctt, 'r-', linewidth=2)
   plt.xlabel(r'Multipole $\\ell$')
   plt.ylabel(r'$\\ell(\\ell+1)C_\\ell^{TT}$')
   plt.title('QFD CMB Temperature Power Spectrum')
   plt.grid(True, alpha=0.3)
   plt.show()

Parameter Studies
-----------------

Varying Oscillation Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from qfd_cmb import oscillatory_psik, gaussian_window_chi, project_limber
   
   # Define grids
   ells = np.arange(2, 2000)
   chi = np.linspace(100, 15000, 200)
   W = gaussian_window_chi(chi, 14065.0, 250.0)
   
   # Parameter values to explore
   rpsi_values = [140, 147, 154]  # Different oscillation scales
   Aosc_values = [0.0, 0.3, 0.55, 0.8]  # Different oscillation amplitudes
   
   # Study rpsi dependence
   plt.figure(figsize=(12, 8))
   for rpsi in rpsi_values:
       Pk_func = lambda k: oscillatory_psik(k, rpsi=rpsi, Aosc=0.55)
       Ctt = project_limber(ells, Pk_func, W, chi)
       plt.loglog(ells, ells*(ells+1)*Ctt, linewidth=2, 
                  label=f'$r_\\psi = {rpsi}$ Mpc')
   
   plt.xlabel(r'Multipole $\\ell$')
   plt.ylabel(r'$\\ell(\\ell+1)C_\\ell^{TT}$')
   plt.title('Effect of Oscillation Scale')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()
   
   # Study Aosc dependence
   plt.figure(figsize=(12, 8))
   for Aosc in Aosc_values:
       Pk_func = lambda k: oscillatory_psik(k, rpsi=147.0, Aosc=Aosc)
       Ctt = project_limber(ells, Pk_func, W, chi)
       plt.loglog(ells, ells*(ells+1)*Ctt, linewidth=2, 
                  label=f'$A_{{\\text{{osc}}}} = {Aosc}$')
   
   plt.xlabel(r'Multipole $\\ell$')
   plt.ylabel(r'$\\ell(\\ell+1)C_\\ell^{TT}$')
   plt.title('Effect of Oscillation Amplitude')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Comparing with Standard Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from qfd_cmb import oscillatory_psik, gaussian_window_chi, project_limber
   
   # Define grids
   ells = np.arange(2, 3000)
   chi = np.linspace(100, 15000, 300)
   W = gaussian_window_chi(chi, 14065.0, 250.0)
   
   # Standard model (no oscillations)
   Pk_standard = lambda k: oscillatory_psik(k, Aosc=0.0)
   Ctt_standard = project_limber(ells, Pk_standard, W, chi)
   
   # QFD model (with oscillations)
   Pk_qfd = lambda k: oscillatory_psik(k, rpsi=147.0, Aosc=0.55)
   Ctt_qfd = project_limber(ells, Pk_qfd, W, chi)
   
   # Plot comparison
   plt.figure(figsize=(14, 10))
   
   # Main comparison plot
   plt.subplot(2, 1, 1)
   plt.loglog(ells, ells*(ells+1)*Ctt_standard, 'k-', linewidth=2, 
              label='Standard Model')
   plt.loglog(ells, ells*(ells+1)*Ctt_qfd, 'r-', linewidth=2, 
              label='QFD Model')
   plt.xlabel(r'Multipole $\\ell$')
   plt.ylabel(r'$\\ell(\\ell+1)C_\\ell^{TT}$')
   plt.title('CMB Temperature Spectra: Standard vs QFD')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   # Residual plot
   plt.subplot(2, 1, 2)
   residual = (Ctt_qfd - Ctt_standard) / Ctt_standard * 100
   plt.semilogx(ells, residual, 'g-', linewidth=2)
   plt.axhline(0, color='k', linestyle='--', alpha=0.5)
   plt.xlabel(r'Multipole $\\ell$')
   plt.ylabel('Relative Difference [%]')
   plt.title('QFD vs Standard Model Residuals')
   plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Advanced Calculations
---------------------

Polarization Spectra
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from qfd_cmb import oscillatory_psik, gaussian_window_chi, project_limber
   from qfd_cmb.kernels import sin2_mueller_coeffs, te_correlation_phase
   
   # Define grids
   ells = np.arange(2, 2000)
   chi = np.linspace(100, 15000, 200)
   W = gaussian_window_chi(chi, 14065.0, 250.0)
   
   # Base power spectrum
   def Pk_base(k):
       return oscillatory_psik(k, rpsi=147.0, Aosc=0.55)
   
   # TT spectrum (temperature)
   Ctt = project_limber(ells, Pk_base, W, chi)
   
   # EE spectrum (simplified - would need proper polarization calculation)
   # This is a schematic example
   def Pk_EE(k):
       return 0.1 * Pk_base(k)  # EE is typically ~10% of TT
   
   Cee = project_limber(ells, Pk_EE, W, chi)
   
   # TE spectrum with correlation
   def Pk_TE(k):
       # Simplified TE correlation
       rho = 0.5 * np.cos(k * 147.0)  # Oscillatory correlation
       return rho * np.sqrt(Pk_base(k) * Pk_EE(k))
   
   Cte = project_limber(ells, Pk_TE, W, chi)
   
   # Plot all spectra
   plt.figure(figsize=(15, 5))
   
   plt.subplot(1, 3, 1)
   plt.loglog(ells, ells*(ells+1)*Ctt, 'r-', linewidth=2)
   plt.xlabel(r'$\\ell$')
   plt.ylabel(r'$\\ell(\\ell+1)C_\\ell^{TT}$')
   plt.title('TT Spectrum')
   plt.grid(True, alpha=0.3)
   
   plt.subplot(1, 3, 2)
   plt.loglog(ells, ells*(ells+1)*Cee, 'b-', linewidth=2)
   plt.xlabel(r'$\\ell$')
   plt.ylabel(r'$\\ell(\\ell+1)C_\\ell^{EE}$')
   plt.title('EE Spectrum')
   plt.grid(True, alpha=0.3)
   
   plt.subplot(1, 3, 3)
   # Handle sign for TE spectrum
   sign = np.sign(Cte + 1e-30)
   plt.semilogx(ells, sign * ells*(ells+1)*np.abs(Cte), 'g-', linewidth=2)
   plt.axhline(0, color='k', linestyle='--', alpha=0.5)
   plt.xlabel(r'$\\ell$')
   plt.ylabel(r'sign×$\\ell(\\ell+1)|C_\\ell^{TE}|$')
   plt.title('TE Spectrum')
   plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Line-of-Sight vs Limber Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from qfd_cmb import oscillatory_psik, gaussian_window_chi, project_limber
   from qfd_cmb.projector import los_transfer, project_los
   
   # Define grids for comparison
   ells_low = np.arange(2, 100)  # Low ells where differences matter
   ells_high = np.arange(2, 1000)  # Full range for Limber
   
   chi = np.linspace(100, 15000, 200)
   W = gaussian_window_chi(chi, 14065.0, 250.0)
   
   k_grid = np.logspace(-4, 0, 100)
   eta_grid = np.linspace(0, 14000, 150)
   
   # Power spectrum function
   Pk_func = lambda k: oscillatory_psik(k, rpsi=147.0, Aosc=0.55)
   
   # Limber calculation
   Cl_limber = project_limber(ells_high, Pk_func, W, chi)
   
   # Line-of-sight calculation (simplified example)
   def S_func(k, eta):
       # Simplified source function
       eta_star = 13500  # Approximate conformal time at last scattering
       sigma_eta = 100
       return np.exp(-0.5 * ((eta - eta_star) / sigma_eta)**2)
   
   # Compute transfer functions
   Delta = los_transfer(ells_low, k_grid, eta_grid, S_func)
   
   # Project to get spectrum
   Cl_los = project_los(ells_low, k_grid, Pk_func, Delta, Delta)
   
   # Compare results
   plt.figure(figsize=(12, 8))
   
   # Full Limber result
   plt.loglog(ells_high, ells_high*(ells_high+1)*Cl_limber, 'b-', 
              linewidth=2, label='Limber Approximation', alpha=0.7)
   
   # Line-of-sight result for low ells
   plt.loglog(ells_low, ells_low*(ells_low+1)*Cl_los, 'ro', 
              markersize=4, label='Line-of-Sight (low $\\ell$)')
   
   plt.xlabel(r'Multipole $\\ell$')
   plt.ylabel(r'$\\ell(\\ell+1)C_\\ell^{TT}$')
   plt.title('Limber vs Line-of-Sight Projection Methods')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.xlim(2, 1000)
   
   # Inset showing low-ell region
   from mpl_toolkits.axes_grid1.inset_locator import inset_axes
   axins = inset_axes(plt.gca(), width="40%", height="40%", loc='upper right')
   axins.loglog(ells_high[:50], ells_high[:50]*(ells_high[:50]+1)*Cl_limber[:50], 
                'b-', linewidth=2, alpha=0.7)
   axins.loglog(ells_low[:30], ells_low[:30]*(ells_low[:30]+1)*Cl_los[:30], 
                'ro', markersize=3)
   axins.set_xlim(2, 50)
   axins.grid(True, alpha=0.3)
   
   plt.show()

Performance Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import time
   from qfd_cmb import oscillatory_psik, gaussian_window_chi, project_limber
   
   def benchmark_calculation(n_ells, n_chi, n_runs=5):
       """Benchmark the Limber projection calculation."""
       
       # Setup
       ells = np.arange(2, n_ells + 2)
       chi = np.linspace(100, 15000, n_chi)
       W = gaussian_window_chi(chi, 14065.0, 250.0)
       Pk_func = lambda k: oscillatory_psik(k, rpsi=147.0, Aosc=0.55)
       
       # Timing runs
       times = []
       for i in range(n_runs):
           start_time = time.time()
           Cl = project_limber(ells, Pk_func, W, chi)
           end_time = time.time()
           times.append(end_time - start_time)
       
       avg_time = np.mean(times)
       std_time = np.std(times)
       
       return avg_time, std_time, len(ells), len(chi)
   
   # Run benchmarks
   test_cases = [
       (100, 50),    # Small
       (500, 100),   # Medium  
       (2000, 200),  # Large
       (5000, 500),  # Very large
   ]
   
   print("Performance Benchmark Results:")
   print("=" * 50)
   print(f"{'N_ells':<8} {'N_chi':<8} {'Time (s)':<12} {'Std (s)':<10}")
   print("-" * 50)
   
   for n_ells, n_chi in test_cases:
       avg_time, std_time, actual_ells, actual_chi = benchmark_calculation(n_ells, n_chi)
       print(f"{actual_ells:<8} {actual_chi:<8} {avg_time:<12.4f} {std_time:<10.4f}")
   
   print("=" * 50)