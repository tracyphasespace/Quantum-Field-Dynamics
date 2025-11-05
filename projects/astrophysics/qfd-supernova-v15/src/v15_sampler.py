"""
V15 MCMC Sampler

Wraps emcee (proven robust sampler from V1) with modern progress tracking.
V13.1: Added multiprocessing pool support for parallel SN evaluation.
"""

from typing import Dict, Callable, Tuple, Optional
from pathlib import Path
import sys
import json
import os
import multiprocessing as mp
import numpy as np
import jax.numpy as jnp
import emcee

from v15_config import V15Config
from v15_model import log_likelihood_single_sn_jax, C_KM_S, C_CM_S

# Environment-tunable walker initialization scale
INIT_SCALE = float(os.environ.get('V15_INIT_SCALE', '1e-4'))


class V15Sampler:
    """
    emcee-based MCMC sampler for QFD global parameters.

    Preserves V1's proven architecture:
    - Independent SN evaluation (no batching)
    - Gradient-free emcee sampler
    - JAX-accelerated physics on GPU

    V13.1 Enhancement:
    - Parallel SN evaluation via multiprocessing pool (optional)
    """

    def __init__(
        self,
        config: V15Config,
        lightcurves_jax: Dict[str, Dict],
        verbose: bool = True,
    ):
        """
        Initialize sampler.

        Args:
            config: V15 configuration
            lightcurves_jax: Dictionary mapping SNID -> JAX-ready photometry data
            verbose: Print progress messages
        """
        self.config = config
        self.lightcurves_jax = lightcurves_jax
        self.verbose = verbose

        self.n_sne = len(lightcurves_jax)
        self.ndim = 3  # k_J, eta_prime, xi
        self.current_step = 0  # Track step for burn-in grace period

        # BUGFIX v7: Informative priors to create posterior curvature
        # Environment-tunable prior widths (sigma), defaults provide weak information
        self.k_J_prior_mean = float(os.environ.get('V15_K_J_PRIOR_MEAN', '70.0'))
        self.k_J_prior_sigma = float(os.environ.get('V15_K_J_PRIOR_SIGMA', '20.0'))
        self.eta_prime_prior_mean = float(os.environ.get('V15_ETA_PRIME_PRIOR_MEAN', '0.01'))
        self.eta_prime_prior_sigma = float(os.environ.get('V15_ETA_PRIME_PRIOR_SIGMA', '0.03'))
        self.xi_prior_mean = float(os.environ.get('V15_XI_PRIOR_MEAN', '30.0'))
        self.xi_prior_sigma = float(os.environ.get('V15_XI_PRIOR_SIGMA', '20.0'))

    def log_prior(self, theta: np.ndarray) -> float:
        """
        Compute log-prior probability.

        BUGFIX v7: Add informative Gaussian priors to create posterior curvature.
        Without priors, flat likelihood (logL~0) gives emcee no gradient.

        Args:
            theta: Global parameters [k_J, eta_prime, xi]

        Returns:
            Log-prior probability
        """
        k_J, eta_prime, xi = theta

        # Hard bounds (reject if outside physics-allowed range)
        if not (self.config.physics.k_J_min < k_J < self.config.physics.k_J_max):
            return -np.inf
        if not (self.config.physics.eta_prime_min < eta_prime < self.config.physics.eta_prime_max):
            return -np.inf
        if not (self.config.physics.xi_min < xi < self.config.physics.xi_max):
            return -np.inf

        # Informative Gaussian priors (create curvature)
        log_prior = 0.0

        # k_J: Weak prior centered at H0~70 km/s/Mpc (cosmology constraint)
        log_prior += -0.5 * ((k_J - self.k_J_prior_mean) / self.k_J_prior_sigma) ** 2

        # eta_prime: Weak prior centered at benchmark value
        log_prior += -0.5 * ((eta_prime - self.eta_prime_prior_mean) / self.eta_prime_prior_sigma) ** 2

        # xi: Weak prior centered at benchmark value
        log_prior += -0.5 * ((xi - self.xi_prior_mean) / self.xi_prior_sigma) ** 2

        return log_prior

    def log_posterior_global(self, theta: np.ndarray, step: int = None) -> float:
        """
        Log-posterior for global parameters.

        Evaluates each SN independently (V1 architecture).
        Uses fixed per-SN parameters (placeholder for future per-SN optimization).

        Args:
            theta: Global parameters [k_J, eta_prime, xi]
            step: Current MCMC step number (for burn-in grace period, uses self.current_step if None)

        Returns:
            Log-posterior probability
        """
        # Use instance step tracker if not explicitly provided
        if step is None:
            step = self.current_step

        k_J, eta_prime, xi = theta

        # BUGFIX v7: Compute informative prior (includes bounds check)
        log_prior = self.log_prior(theta)
        if not np.isfinite(log_prior):
            return -np.inf

        global_params = (k_J, eta_prime, xi)

        # Sum log-likelihoods over all SNe (independently)
        # BUGFIX v6: Improved skip-few guard with proper threshold calculation (cloud.txt)
        # Reset counters per evaluation (critical for multi-walker MCMC)
        sn_items = list(self.lightcurves_jax.items())
        N = len(sn_items)  # Number of SNe (not rows!)
        bad_sn_count = 0
        log_L_total = 0.0
        inspected_count = 0  # For verbose diagnostics
        skipped_snids = []  # Track which SNe are problematic

        for snid, lc_data in sn_items:
            # Extract photometry and redshift
            phot_array = np.column_stack([
                lc_data['mjd'],
                lc_data['wavelength_nm'],
                lc_data['flux_jy'],
                lc_data['flux_err_jy'],
            ])
            phot = jnp.array(phot_array)
            z_obs = lc_data['z']

            # Placeholder per-SN parameters (data-driven estimates)
            # BUGFIX v3: Use data-driven estimates instead of constants
            # to avoid artificial perfect fits that create flat posterior
            t_mjd_peak = float(phot[phot[:, 2].argmax(), 0])  # MJD of peak flux
            t0_guess = t_mjd_peak - 19.0  # Peak ~19 days after explosion

            # BUGFIX: Use fixed H0=70 instead of k_J to break circular dependency
            # OLD (caused degeneracy): D_true_guess = z_obs * C_KM_S / k_J
            D_true_guess = z_obs * C_KM_S / 70.0  # Fixed H0=70 km/s/Mpc

            # Estimate L_peak from observed peak flux and distance
            # BUGFIX v4: Correct f_nu (Jy) to f_lambda conversion with wavelength handling
            # Jy is per Hz, but model works in wavelength space
            idx_peak = int(np.argmax(phot[:, 2]))
            peak_flux_jy = float(phot[idx_peak, 2])
            peak_lambda_cm = float(phot[idx_peak, 1]) * 1e-7  # nm to cm

            # Convert f_nu to f_lambda: f_lambda = f_nu * (c / λ²)
            peak_flux_lambda = peak_flux_jy * 1e-23 * (C_CM_S / peak_lambda_cm**2)  # erg/s/cm²/cm

            # Luminosity: L = f_lambda * 4π * D_L² * Δλ (use λ as effective bandwidth)
            D_L_cm = D_true_guess * 3.0857e24  # Mpc to cm
            L_peak_guess = peak_flux_lambda * 4.0 * np.pi * (D_L_cm ** 2) * peak_lambda_cm  # erg/s

            # BUGFIX v5: Sanity clamps to prevent extreme outliers from dominating posterior
            # Keep L_peak in physically reasonable band for SN Ia
            L_peak_guess = float(np.clip(L_peak_guess, 1e42, 5e44))

            # BUGFIX v9: FREEZE nuisance parameters to reduce per-SN DOF (cloud.txt 2025-11-03)
            # Problem: Even with v7 (priors) + v8 (jitter), 5 DOF per SN still absorb all variance
            # Solution: Freeze 3 params (D_true, A_plasma, β) → only 2 vary (t0, L_peak)
            #
            # This is the critical test: Can single-stage work with reduced per-SN flexibility?
            # - If YES (acceptance > 0.25) → architecture viable with simpler model
            # - If NO (still stuck) → proves need for two-stage optimization

            # FROZEN: A_plasma at typical baseline (no data-driven estimation)
            A_plasma_guess = 0.12  # Constant veil baseline

            # FROZEN: β (wavelength dependence) at canonical value
            beta_guess = 0.5  # Constant wavelength exponent

            # FROZEN: D_true already fixed above at z*c/70 (no circular k_J dependency)

            persn_params = (t0_guess, L_peak_guess, D_true_guess, A_plasma_guess, beta_guess)

            # Evaluate log-likelihood for this SN
            log_L = log_likelihood_single_sn_jax(global_params, persn_params, phot, z_obs)

            # Verbose diagnostics for first 5 SNe to verify parameter ranges
            if self.verbose and inspected_count < 5:
                print(f"  [Diagnostic] SNID={snid}: L_peak={L_peak_guess:.2e} erg/s, "
                      f"A_plasma={A_plasma_guess:.3f}, log_L={float(log_L):.2f}")
                inspected_count += 1

            # Check for numerical issues
            # Skip-few policy: allow a small fraction of problematic SNe
            # BUGFIX v6: Treat both non-finite AND extreme logL as pathological
            # Cloud.txt threshold: |logL| > 1e4 or non-finite
            if not jnp.isfinite(log_L) or abs(float(log_L)) > 1e4:
                bad_sn_count += 1
                skipped_snids.append(snid)
                if self.verbose and bad_sn_count <= 3:
                    print(f"  [Skip] SNID={snid}: log_L={float(log_L):.2f}")
                continue  # Skip this SN, don't add to total

            log_L_total += float(log_L)

        # BUGFIX v6: Proper skip-few threshold calculation (cloud.txt)
        # Use ceil to avoid floor(0.01*43)=0, and provide burn-in grace period
        # Environment-tunable for production flexibility
        if step is not None and step < self.config.sampler.n_burn:
            # During burn-in: default 10% tolerance (tunable via V13_SKIP_CAP_BURN_PCT)
            pct = float(os.environ.get('V15_SKIP_CAP_BURN_PCT', '0.10'))
        else:
            # Post burn-in: default 10% tolerance (tunable via V13_SKIP_CAP_PCT)
            pct = float(os.environ.get('V15_SKIP_CAP_PCT', '0.10'))

        cap = max(1, int(np.ceil(pct * N)))

        if bad_sn_count > cap:
            # Log rejection for debugging (every 50 steps)
            if self.verbose and (step is None or step % 50 == 0):
                print(f"  [SKIP-REJECT] step={step} bad={bad_sn_count}/{N} cap={cap} SNIDs={skipped_snids[:5]}")
            return -np.inf

        # Optional: Log healthy steps periodically
        if self.verbose and step is not None and step % 50 == 0 and bad_sn_count > 0:
            print(f"  [SKIP-OK] step={step} bad={bad_sn_count}/{N} cap={cap} total_logL={log_L_total:.1f} prior={log_prior:.1f}")

        # BUGFIX v7: Return log-posterior = log-likelihood + log-prior
        # Prior creates curvature even when likelihood is flat
        return log_L_total + log_prior

    def run(self) -> Dict:
        """
        WARNING: This sampler uses per-SN light-curve likelihoods and is **legacy**.
        V15 production runs should use Stage-2 α-space MCMC (NumPyro), not this path.
        To proceed anyway, set environment variable V15_ALLOW_LEGACY_SAMPLER=1.

        Run MCMC sampling (emcee architecture from V1).

        V13.1: Optionally uses multiprocessing pool for parallel SN evaluation.

        Returns:
            Dictionary with best-fit parameters and metadata
        """
        import os
        if os.environ.get("V15_ALLOW_LEGACY_SAMPLER","0") != "1":
            raise RuntimeError("Legacy sampler disabled. Use stage2_mcmc_numpyro.py for α-space inference.")

        if self.verbose:
            print(f"\n{'='*80}")
            print("V15 MCMC Sampling")
            print(f"{'='*80}\n")
            print(f"Fitting {self.n_sne} supernovae")
            print(f"  Walkers: {self.config.sampler.n_walkers}")
            print(f"  Steps: {self.config.sampler.n_steps}")
            print(f"  Burn-in: {self.config.sampler.n_burn}")
            if self.config.sampler.n_threads > 1:
                print(f"  Parallel workers: {self.config.sampler.n_threads}")

        # Initial guess (from V1 benchmarks)
        p0 = np.array([
            self.config.physics.k_J_init,
            self.config.physics.eta_prime_init,
            self.config.physics.xi_init,
        ])

        if self.verbose:
            print(f"\nInitial values:")
            print(f"  k_J = {p0[0]:.2f} km/s/Mpc")
            print(f"  eta_prime = {p0[1]:.4e}")
            print(f"  xi = {p0[2]:.2f}")

        # Initialize walkers with small perturbations
        np.random.seed(self.config.random_seed)
        p0_ensemble = p0 + INIT_SCALE * p0 * np.random.randn(
            self.config.sampler.n_walkers, self.ndim
        )

        # Create multiprocessing pool if requested
        pool = None
        if self.config.sampler.n_threads and self.config.sampler.n_threads > 1:
            # Use fork context on Linux (fastest), spawn on Windows/WSL
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context("spawn")

            pool = ctx.Pool(processes=self.config.sampler.n_threads)

            if self.verbose:
                print(f"\nCreated multiprocessing pool with {self.config.sampler.n_threads} workers")

        try:
            # Create emcee sampler
            sampler = emcee.EnsembleSampler(
                self.config.sampler.n_walkers,
                self.ndim,
                self.log_posterior_global,
                pool=pool,
            )

            # Run MCMC with progress reporting
            if self.verbose:
                print("\nRunning MCMC...")
                sys.stdout.flush()

            # Create status file for real-time monitoring
            status_file = self.config.output.output_dir / "_status.txt"

            for i, _ in enumerate(sampler.sample(p0_ensemble, iterations=self.config.sampler.n_steps)):
                # Update step counter for burn-in grace period in skip-few guard
                self.current_step = i

                # Progress reporting every 10%
                if self.verbose and ((i + 1) % max(1, self.config.sampler.n_steps // 10) == 0 or i == 0):
                    pct = 100 * (i + 1) / self.config.sampler.n_steps
                    msg = f"  Progress: {i+1}/{self.config.sampler.n_steps} ({pct:.0f}%)"
                    print(msg)
                    sys.stdout.flush()

                    # Write to status file for easy monitoring
                    import time
                    with open(status_file, 'w') as f:
                        f.write(f"step={i+1}/{self.config.sampler.n_steps}\n")
                        f.write(f"percent={pct:.1f}\n")
                        f.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"status=running\n")

            # Extract results (discard burn-in)
            samples = sampler.get_chain(discard=self.config.sampler.n_burn, flat=True)
            acceptance = np.mean(sampler.acceptance_fraction)

            if self.verbose:
                print(f"\nMCMC complete!")
                print(f"  Total samples: {len(samples)}")
                print(f"  Acceptance fraction: {acceptance:.3f}")

            # Write final status
            import time
            with open(status_file, 'w') as f:
                f.write(f"step={self.config.sampler.n_steps}/{self.config.sampler.n_steps}\n")
                f.write(f"percent=100.0\n")
                f.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"status=completed\n")
                f.write(f"acceptance={acceptance:.3f}\n")
                f.write(f"samples={len(samples)}\n")

        finally:
            # Clean up pool
            if pool is not None:
                pool.close()
                pool.join()

        # Compute statistics
        results = {
            "k_J": float(np.median(samples[:, 0])),
            "k_J_std": float(np.std(samples[:, 0])),
            "eta_prime": float(np.median(samples[:, 1])),
            "eta_prime_std": float(np.std(samples[:, 1])),
            "xi": float(np.median(samples[:, 2])),
            "xi_std": float(np.std(samples[:, 2])),
            "n_sne": self.n_sne,
            "n_steps": self.config.sampler.n_steps,
            "n_burn": self.config.sampler.n_burn,
            "n_threads": self.config.sampler.n_threads,
            "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        }

        # Save results
        self._save_results(results, samples)

        if self.verbose:
            print(f"\nBest-fit global parameters:")
            print(f"  k_J = {results['k_J']:.2f} ± {results['k_J_std']:.2f} km/s/Mpc")
            print(f"  eta_prime = {results['eta_prime']:.4e} ± {results['eta_prime_std']:.4e}")
            print(f"  xi = {results['xi']:.2f} ± {results['xi_std']:.2f}")

        return results

    def _save_results(self, results: Dict, samples: np.ndarray):
        """Save results to JSON and samples to NPY."""
        # Save best-fit parameters
        results_file = self.config.output.output_dir / "v15_best_fit.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save MCMC samples
        if self.config.output.save_chains:
            samples_file = self.config.output.output_dir / "v15_mcmc_samples.npy"
            np.save(samples_file, samples)

        if self.verbose:
            print(f"\nResults saved:")
            print(f"  Best-fit: {results_file}")
            if self.config.output.save_chains:
                print(f"  Samples: {samples_file}")
