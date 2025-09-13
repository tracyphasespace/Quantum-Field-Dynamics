#!/usr/bin/env python3
"""
Advanced Parameter Fitting Example for QFD CMB Module

This script demonstrates advanced parameter fitting techniques using the QFD CMB Module,
including MCMC sampling, likelihood analysis, and parameter constraint visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from scipy.stats import chi2
import emcee
import corner
from qfd_cmb import oscillatory_psik, gaussian_window_chi, project_limber
from qfd_cmb.kernels import te_correlation_phase


class QFDLikelihood:
    """Likelihood class for QFD parameter fitting."""
    
    def __init__(self, ell_data, Ctt_data, Ctt_errors, chi_star=14065.0, sigma_chi=250.0):
        """
        Initialize likelihood with observational data.
        
        Parameters:
        -----------
        ell_data : array
            Multipole values for data points
        Ctt_data : array
            Observed TT power spectrum values
        Ctt_errors : array
            Uncertainties on TT power spectrum
        chi_star : float
            Comoving distance to last scattering (Mpc)
        sigma_chi : float
            Width of last scattering surface (Mpc)
        """
        self.ell_data = ell_data
        self.Ctt_data = Ctt_data
        self.Ctt_errors = Ctt_errors
        self.chi_star = chi_star
        self.sigma_chi = sigma_chi
        
        # Setup visibility window
        chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 300)
        self.W_chi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        self.chi_grid = chi_grid
        
        print(f"Likelihood initialized with {len(ell_data)} data points")
        print(f"Multipole range: {ell_data.min()} to {ell_data.max()}")
    
    def compute_theory_spectrum(self, params):
        """Compute theoretical TT spectrum for given parameters."""
        rpsi, Aosc, sigma_osc, ns = params
        
        # Define power spectrum function
        def Pk_func(k):
            return oscillatory_psik(k, ns=ns, rpsi=rpsi, Aosc=Aosc, sigma_osc=sigma_osc)
        
        # Compute TT spectrum
        Ctt_theory = project_limber(self.ell_data, Pk_func, self.W_chi, self.chi_grid)
        
        return Ctt_theory
    
    def log_likelihood(self, params):
        """Compute log-likelihood for given parameters."""
        try:
            Ctt_theory = self.compute_theory_spectrum(params)
            
            # Chi-squared calculation
            chi_sq = np.sum(((self.Ctt_data - Ctt_theory) / self.Ctt_errors)**2)
            
            return -0.5 * chi_sq
        
        except Exception as e:
            # Return very low likelihood for invalid parameters
            return -1e10
    
    def log_prior(self, params):
        """Compute log-prior for parameters."""
        rpsi, Aosc, sigma_osc, ns = params
        
        # Define reasonable priors
        if not (100 < rpsi < 200):
            return -np.inf
        if not (0.0 <= Aosc <= 1.0):
            return -np.inf
        if not (0.01 <= sigma_osc <= 0.1):
            return -np.inf
        if not (0.9 <= ns <= 1.1):
            return -np.inf
        
        return 0.0  # Flat priors within bounds
    
    def log_posterior(self, params):
        """Compute log-posterior (likelihood + prior)."""
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)


def generate_mock_data():
    """Generate mock observational data for fitting demonstration."""
    print("=" * 60)
    print("Generating Mock Observational Data")
    print("=" * 60)
    
    # True parameters for mock data
    true_params = {
        'rpsi': 147.0,
        'Aosc': 0.55,
        'sigma_osc': 0.025,
        'ns': 0.96
    }
    
    print("True parameters:")
    for key, value in true_params.items():
        print(f"  {key}: {value}")
    
    # Setup calculation
    chi_star = 14065.0
    sigma_chi = 250.0
    chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 300)
    W_chi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
    
    # Define multipole range (sparse sampling to simulate real data)
    ell_data = np.logspace(np.log10(50), np.log10(2000), 30).astype(int)
    
    # Compute true spectrum
    def Pk_true(k):
        return oscillatory_psik(k, **true_params)
    
    Ctt_true = project_limber(ell_data, Pk_true, W_chi, chi_grid)
    
    # Add realistic noise (5% relative error)
    relative_error = 0.05
    Ctt_errors = relative_error * Ctt_true
    Ctt_data = Ctt_true + np.random.normal(0, Ctt_errors)
    
    print(f"Generated {len(ell_data)} mock data points")
    print(f"Relative error level: {relative_error*100}%")
    print()
    
    return ell_data, Ctt_data, Ctt_errors, true_params


def maximum_likelihood_fitting(likelihood):
    """Perform maximum likelihood parameter estimation."""
    print("=" * 60)
    print("Maximum Likelihood Fitting")
    print("=" * 60)
    
    # Initial guess
    initial_params = [147.0, 0.5, 0.03, 0.96]  # rpsi, Aosc, sigma_osc, ns
    
    print("Initial parameter guess:")
    param_names = ['rpsi', 'Aosc', 'sigma_osc', 'ns']
    for name, value in zip(param_names, initial_params):
        print(f"  {name}: {value}")
    
    # Define objective function (negative log-likelihood)
    def objective(params):
        return -likelihood.log_likelihood(params)
    
    # Perform optimization
    print("\nRunning optimization...")
    result = minimize(objective, initial_params, method='Nelder-Mead',
                     options={'maxiter': 1000, 'disp': True})
    
    if result.success:
        print("\nOptimization successful!")
        print("Best-fit parameters:")
        for name, value in zip(param_names, result.x):
            print(f"  {name}: {value:.4f}")
        print(f"Best-fit log-likelihood: {-result.fun:.2f}")
    else:
        print("\nOptimization failed!")
        print(f"Message: {result.message}")
    
    print()
    return result.x if result.success else initial_params


def mcmc_sampling(likelihood, best_fit_params):
    """Perform MCMC sampling for parameter uncertainties."""
    print("=" * 60)
    print("MCMC Parameter Sampling")
    print("=" * 60)
    
    # MCMC setup
    ndim = 4  # Number of parameters
    nwalkers = 32
    nsteps = 2000
    
    # Initialize walkers around best-fit
    pos = best_fit_params + 1e-2 * np.random.randn(nwalkers, ndim)
    
    # Ensure all walkers start in valid parameter space
    for i in range(nwalkers):
        while not np.isfinite(likelihood.log_prior(pos[i])):
            pos[i] = best_fit_params + 1e-2 * np.random.randn(ndim)
    
    print(f"MCMC setup:")
    print(f"  Dimensions: {ndim}")
    print(f"  Walkers: {nwalkers}")
    print(f"  Steps: {nsteps}")
    
    # Run MCMC
    print("\nRunning MCMC sampling...")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood.log_posterior)
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    # Analyze results
    print("\nMCMC completed!")
    print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    
    # Remove burn-in
    burnin = 500
    samples = sampler.get_chain(discard=burnin, flat=True)
    
    print(f"Using {len(samples)} samples after burn-in")
    
    # Compute parameter statistics
    param_names = ['rpsi', 'Aosc', 'sigma_osc', 'ns']
    print("\nParameter constraints (68% confidence):")
    for i, name in enumerate(param_names):
        mcmc_median = np.percentile(samples[:, i], 50)
        mcmc_lower = np.percentile(samples[:, i], 16)
        mcmc_upper = np.percentile(samples[:, i], 84)
        print(f"  {name}: {mcmc_median:.4f} +{mcmc_upper-mcmc_median:.4f} -{mcmc_median-mcmc_lower:.4f}")
    
    print()
    return samples, param_names


def create_corner_plot(samples, param_names, true_params, output_file):
    """Create corner plot showing parameter constraints."""
    print("Creating corner plot...")
    
    # Extract true parameter values in correct order
    true_values = [true_params[name] for name in param_names]
    
    # Create corner plot
    fig = corner.corner(samples, labels=param_names, truths=true_values,
                       truth_color='red', show_titles=True, title_kwargs={'fontsize': 12})
    
    # Save plot
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"Corner plot saved to: {output_file}")


def model_comparison_analysis(likelihood, best_fit_params):
    """Perform model comparison between QFD and standard model."""
    print("=" * 60)
    print("Model Comparison Analysis")
    print("=" * 60)
    
    # Compute likelihood for best-fit QFD model
    qfd_loglike = likelihood.log_likelihood(best_fit_params)
    
    # Compute likelihood for standard model (no oscillations)
    standard_params = [best_fit_params[0], 0.0, best_fit_params[2], best_fit_params[3]]
    standard_loglike = likelihood.log_likelihood(standard_params)
    
    # Compute Bayes factor approximation (BIC)
    n_data = len(likelihood.ell_data)
    qfd_bic = -2 * qfd_loglike + 4 * np.log(n_data)  # 4 parameters
    standard_bic = -2 * standard_loglike + 3 * np.log(n_data)  # 3 parameters (no Aosc)
    
    delta_bic = qfd_bic - standard_bic
    
    print(f"QFD model log-likelihood: {qfd_loglike:.2f}")
    print(f"Standard model log-likelihood: {standard_loglike:.2f}")
    print(f"Log-likelihood difference: {qfd_loglike - standard_loglike:.2f}")
    print()
    print(f"QFD model BIC: {qfd_bic:.2f}")
    print(f"Standard model BIC: {standard_bic:.2f}")
    print(f"ΔBIC (QFD - Standard): {delta_bic:.2f}")
    
    if delta_bic < -10:
        evidence = "Very strong"
    elif delta_bic < -6:
        evidence = "Strong"
    elif delta_bic < -2:
        evidence = "Positive"
    elif delta_bic < 2:
        evidence = "Inconclusive"
    else:
        evidence = "Negative"
    
    print(f"Evidence for QFD model: {evidence}")
    print()


def create_fit_comparison_plot(likelihood, best_fit_params, true_params):
    """Create plot comparing data, best-fit, and true models."""
    print("Creating model comparison plot...")
    
    # Compute spectra
    Ctt_data = likelihood.Ctt_data
    Ctt_errors = likelihood.Ctt_errors
    ell_data = likelihood.ell_data
    
    Ctt_best_fit = likelihood.compute_theory_spectrum(best_fit_params)
    
    true_param_values = [true_params['rpsi'], true_params['Aosc'], 
                        true_params['sigma_osc'], true_params['ns']]
    Ctt_true = likelihood.compute_theory_spectrum(true_param_values)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Data with error bars
    plt.errorbar(ell_data, ell_data*(ell_data+1)*Ctt_data, 
                yerr=ell_data*(ell_data+1)*Ctt_errors,
                fmt='ko', capsize=3, label='Mock Data', alpha=0.7)
    
    # Best-fit model
    plt.loglog(ell_data, ell_data*(ell_data+1)*Ctt_best_fit, 
              'r-', linewidth=2, label='Best-fit QFD Model')
    
    # True model
    plt.loglog(ell_data, ell_data*(ell_data+1)*Ctt_true, 
              'b--', linewidth=2, label='True Model')
    
    plt.xlabel(r'$\ell$', fontsize=14)
    plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}$', fontsize=14)
    plt.title('Parameter Fitting Results', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/advanced_fitting_comparison.png', dpi=200)
    plt.show()
    
    print("Model comparison plot saved to: outputs/advanced_fitting_comparison.png")


def main():
    """Run advanced parameter fitting example."""
    print("QFD CMB Module - Advanced Parameter Fitting")
    print("===========================================")
    print()
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Generate mock data
    ell_data, Ctt_data, Ctt_errors, true_params = generate_mock_data()
    
    # Initialize likelihood
    likelihood = QFDLikelihood(ell_data, Ctt_data, Ctt_errors)
    
    # Maximum likelihood fitting
    best_fit_params = maximum_likelihood_fitting(likelihood)
    
    # MCMC sampling
    samples, param_names = mcmc_sampling(likelihood, best_fit_params)
    
    # Create corner plot
    create_corner_plot(samples, param_names, true_params, 
                      'outputs/advanced_corner_plot.png')
    
    # Model comparison
    model_comparison_analysis(likelihood, best_fit_params)
    
    # Create comparison plot
    create_fit_comparison_plot(likelihood, best_fit_params, true_params)
    
    print("Advanced fitting example completed successfully!")
    print("Check the 'outputs' directory for generated plots and results.")


if __name__ == "__main__":
    main()