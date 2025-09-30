#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qfd_native_distance_fitter.py
QFD-native distance pipeline - derive distances directly from raw light curves.

No ΛCDM assumptions, no SALT2 artifacts. Pure QFD physics from photometry to distance.

Physics Model per epoch i, band b:
    F_obs,i = [L_intrinsic(t_i, b) / (4π D²)] * exp(-τ_plasma(t_i, λ_b; A, τ, β))
              * exp(-τ_FDR(D; η′, ξ)) * exp(-τ_cosmo(D; k_J))

Fitted Parameters (per SN):
    - D: True distance [Mpc] (fundamental QFD quantity)
    - A_plasma: Plasma veil amplitude
    - τ_decay: Plasma veil decay timescale [days]
    - β: Chromatic slope parameter
    - t0: Explosion epoch [MJD]
    - {Δm_b}: Band offset nuisances
    - {σ_jit,b}: Band jitter nuisances

Global Parameters (fixed from Phase 1):
    - k_J: QFD cosmological coupling
    - η′: QFD field strength
    - ξ: QFD coupling ratio

Output: QFD-native Hubble table (z_obs, D_qfd) with zero ΛCDM inheritance
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# QFD physics constants (same as definitive analysis)
C_VAC_M_S = 2.99792458e8
C_VAC_KM_S = 2.99792458e5
MPC_TO_M = 3.086e22
L_SN_WATT = 1e36
E_GAMMA_PULSE_EV = 2.0
E_GAMMA_CMB_EV = 6.34e-4
N_CMB_M3 = 4.11e8
E0_EV = 1e9
L0_M = 1e-18
U_VAC_J_M3 = 1e-15

EV2J = 1.602176634e-19
E_GAMMA_PULSE_J = E_GAMMA_PULSE_EV * EV2J
E_GAMMA_CMB_J = E_GAMMA_CMB_EV * EV2J
E0_J = E0_EV * EV2J

A_FDR_BASE = (L_SN_WATT / (4.0*np.pi*C_VAC_M_S)) / E_GAMMA_PULSE_J
B_FDR_BASE = (L_SN_WATT / (4.0*np.pi*C_VAC_M_S)) / U_VAC_J_M3

class QFDNativeDistanceFitter:
    """
    QFD-native distance fitter using only raw photometry.
    No SALT2, no ΛCDM assumptions anywhere in the pipeline.
    """

    def __init__(self, global_params: Dict[str, float], dl_factor: str = 'D'):
        """
        Initialize with global QFD parameters from Phase 1 analysis.

        Args:
            global_params: Dict with keys k_J, eta_prime, xi, delta_mu0
            dl_factor: Luminosity distance convention - 'D' or 'D*(1+z)'
        """
        self.k_J = global_params['k_J']
        self.eta_prime = global_params['eta_prime']
        self.xi = global_params['xi']
        self.delta_mu0 = global_params.get('delta_mu0', 0.0)
        self.dl_factor = dl_factor

        # Derived QFD quantities
        sigma_drag = self.k_J * (L0_M**2) * (E_GAMMA_CMB_J / E0_J)
        self.alpha0 = N_CMB_M3 * sigma_drag

        # FDR scattering cross section
        self.sigma_scatter = (self.eta_prime * self.xi)**2 * (L0_M**2) * (E_GAMMA_PULSE_J/E0_J)**2

        print(f"🔬 QFD Native Fitter initialized:")
        print(f"   k_J = {self.k_J:.2e}")
        print(f"   η′ = {self.eta_prime:.2e}")
        print(f"   ξ = {self.xi:.3f}")
        print(f"   α₀ = {self.alpha0:.2e} m⁻¹")

    def tau_plasma(self, t_days: np.ndarray, lambda_nm: np.ndarray,
                   A_plasma: float, tau_decay: float, beta: float) -> np.ndarray:
        """QFD plasma veil optical depth."""
        t_days = np.atleast_1d(t_days)
        lambda_nm = np.atleast_1d(lambda_nm)

        # Wavelength factor (chromatic evolution)
        wl_factor = (445.0 / lambda_nm)**beta

        # Time evolution (exponential approach to steady state)
        time_factor = 1.0 - np.exp(-np.maximum(t_days, 0.0) / tau_decay)

        return A_plasma * time_factor[:, np.newaxis] * wl_factor[np.newaxis, :]

    def tau_fdr(self, D_mpc: float) -> float:
        """QFD field-dependent redshift optical depth."""
        D_m = D_mpc * MPC_TO_M
        A = self.sigma_scatter * A_FDR_BASE

        # Integrated optical depth with QFD 1/D and 1/D³ terms
        tau = A * (1.0/np.maximum(D_m, 1e-12)) + (A * self.xi * B_FDR_BASE) / (np.maximum(D_m, 1e-12)**3)
        return tau

    def z_cosmo_from_D(self, D_mpc: float) -> float:
        """QFD cosmological redshift (drag-only)."""
        D_m = D_mpc * MPC_TO_M
        return np.expm1(self.alpha0 * D_m)

    def intrinsic_template(self, t_days: np.ndarray, band: str) -> np.ndarray:
        """
        QFD-native intrinsic SN template (no SALT2).

        For now, use simple spline-based template per band.
        TODO: Replace with QFD-derived template from nuclear physics.
        """
        # Template phases and typical magnitudes (QFD-agnostic baseline)
        template_phases = np.array([-15, -10, -5, 0, 5, 10, 20, 30, 50, 80, 120])

        # Band-dependent template (rough approximation)
        if 'g' in band.lower():
            template_mags = np.array([22, 20, 18, 16.5, 17, 18, 19.5, 20.5, 22, 23, 24])
        elif 'r' in band.lower():
            template_mags = np.array([21.5, 19.5, 17.8, 16.8, 17.2, 18.2, 19.2, 20.2, 21.5, 22.5, 23.5])
        elif 'i' in band.lower():
            template_mags = np.array([21, 19, 17.5, 16.9, 17.3, 18.5, 19.5, 20.5, 21.8, 22.8, 23.8])
        else:
            # Default template
            template_mags = np.array([21.5, 19.5, 17.8, 16.8, 17.2, 18.2, 19.2, 20.2, 21.5, 22.5, 23.5])

        # Spline interpolation
        spline = UnivariateSpline(template_phases, template_mags, s=0, k=3)
        return spline(t_days)

    def model_flux(self, mjd: np.ndarray, band_arr: np.ndarray, lambda_nm_arr: np.ndarray,
                   D_mpc: float, A_plasma: float, tau_decay: float, beta: float,
                   t0: float, band_offsets: Dict[str, float]) -> np.ndarray:
        """
        QFD-native flux model: L_intrinsic * QFD attenuation factors.

        Args:
            mjd: MJD per observation row (N_obs,)
            band_arr: Band per observation row (N_obs,)
            lambda_nm_arr: Effective wavelength per row (N_obs,) - can have NaNs

        Returns:
            Model flux per observation row (N_obs,)
        """
        phases = mjd - t0

        # Stable deterministic band ordering for templates/offsets
        unique_bands = sorted(pd.Series(band_arr, dtype=str).unique().tolist())

        # Build one intrinsic template per band
        band_templates = {}
        for band in unique_bands:
            mask = (band_arr == band)
            # Spline evaluated only at the needed phases for this band
            mag_intr = self.intrinsic_template(phases[mask], band)
            # Apply per-band magnitude offset (defaults to 0.0 if missing)
            mag_intr = mag_intr + float(band_offsets.get(band, 0.0))
            # Convert to (relative) flux
            band_templates[band] = 10.0 ** (-0.4 * mag_intr)

        # FDR optical depth (distance-only; same for all rows)
        tau_f = self.tau_fdr(D_mpc)

        # Compute plasma τ per row (guard wavelength NaNs with default 445 nm)
        lam = np.where(np.isfinite(lambda_nm_arr), lambda_nm_arr, 445.0)
        wl_factor = (445.0 / lam) ** beta
        time_factor = 1.0 - np.exp(-np.maximum(phases, 0.0) / max(tau_decay, 1e-6))
        tau_p = A_plasma * time_factor * wl_factor
        tau_total = tau_p + tau_f

        # Assemble flux per row
        flux_pred = np.empty_like(phases, dtype=float)
        for band in unique_bands:
            mask = (band_arr == band)
            flux_pred[mask] = band_templates[band]  # already per-row length for this band

        # Apply QFD dimming and distance dimming
        flux_pred *= np.exp(-tau_total)

        # Luminosity distance convention (expose as flag)
        if self.dl_factor == 'D*(1+z)':
            # Traditional cosmological luminosity distance
            # Note: would need z_obs passed to model_flux for proper implementation
            flux_pred *= (1.0 / max(D_mpc, 1e-12)**2)
        else:
            # Pure QFD distance (current default)
            flux_pred *= (1.0 / max(D_mpc, 1e-12)**2)

        return flux_pred

    def fit_single_sn(self, lc_data: pd.DataFrame, snid: str, z_obs: float) -> Dict:
        """
        Fit QFD distance and plasma parameters for a single supernova.

        Args:
            lc_data: Light curve DataFrame with QFD-pure photometry
            snid: Supernova identifier
            z_obs: Observed redshift

        Returns:
            Dict with fitted parameters and diagnostics
        """
        print(f"🔬 Fitting QFD-native distance for {snid} (z={z_obs:.3f})")

        # Prepare data
        sn_lc = lc_data[lc_data['snid'] == snid].copy()
        if len(sn_lc) == 0:
            raise ValueError(f"No data found for {snid}")

        # Physical constants for error propagation
        LN10_OVER_2P5 = np.log(10.0) / 2.5

        # Convert to flux with proper error model
        if 'flux_nu_jy' in sn_lc.columns and sn_lc['flux_nu_jy'].notna().any():
            flux_obs = sn_lc['flux_nu_jy'].to_numpy(float)
            flux_err = sn_lc['flux_nu_jy_err'].fillna(np.nan).to_numpy(float)
        else:
            # Convert from magnitude with proper error propagation
            mag = sn_lc['mag'].to_numpy(float)
            mag_err = sn_lc.get('mag_err', pd.Series(np.nan, index=sn_lc.index)).to_numpy(float)
            flux_obs = 10.0 ** (-0.4 * mag)
            # Propagate: σ_f = (ln10/2.5) * f * σ_mag
            flux_err = LN10_OVER_2P5 * flux_obs * np.where(np.isfinite(mag_err), mag_err, 0.1)

        # Floor and guard flux errors
        flux_err = np.where((~np.isfinite(flux_err)) | (flux_err <= 0),
                           0.1 * np.maximum(flux_obs, 1e-12), flux_err)

        mjd = sn_lc['mjd'].values
        bands = sn_lc['band'].values
        wavelengths = sn_lc['wavelength_eff_nm'].values

        # Initial parameter guesses
        t0_guess = mjd[np.argmax(flux_obs)]  # Peak as explosion time

        # Better distance initial guess based on redshift (rough Hubble law with H0=70)
        D_guess = max(z_obs * C_VAC_KM_S / 70.0, 5.0)  # Mpc, with minimum floor

        # Fit parameters: [D_mpc, A_plasma, tau_decay, beta, t0, band_offsets...]
        # Deterministic band ordering to ensure stable parameter mapping
        unique_bands = sorted(pd.Series(bands, dtype=str).unique().tolist())
        n_bands = len(unique_bands)

        def objective(params):
            D_mpc = params[0]
            A_plasma = params[1]
            tau_decay = params[2]
            beta = params[3]
            t0 = params[4]
            band_offsets = {band: params[5+i] for i, band in enumerate(unique_bands)}

            # Bounds checks
            if D_mpc <= 0 or tau_decay <= 0:
                return 1e10

            try:
                # Model prediction (now returns correct shape N_obs)
                model_fluxes = self.model_flux(mjd, bands, wavelengths,
                                             D_mpc, A_plasma, tau_decay, beta, t0, band_offsets)

                # Chi-squared (no flattening needed - already correct shape)
                chi2 = np.sum(((flux_obs - model_fluxes) / flux_err)**2)
                return chi2

            except Exception:
                return 1e10

        # Initial guess
        x0 = [D_guess, 0.1, 30.0, 0.0, t0_guess] + [0.0] * n_bands

        # Bounds
        bounds = [(1e-6, 1e6),     # D_mpc
                  (0, 10),         # A_plasma
                  (1, 1000),       # tau_decay
                  (-2, 2),         # beta
                  (mjd.min()-50, mjd.max()+50)] + [(-5, 5)] * n_bands  # band offsets

        # Optimization
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

            if result.success:
                D_fit = result.x[0]
                A_plasma_fit = result.x[1]
                tau_decay_fit = result.x[2]
                beta_fit = result.x[3]
                t0_fit = result.x[4]
                band_offsets_fit = {band: result.x[5+i] for i, band in enumerate(unique_bands)}

                # Calculate derived quantities
                z_qfd = self.z_cosmo_from_D(D_fit)
                H0_implied = z_obs * C_VAC_KM_S / D_fit  # Empirical H0 check

                fit_result = {
                    'snid': snid,
                    'z_obs': z_obs,
                    'D_qfd_mpc': D_fit,
                    'z_qfd_predicted': z_qfd,
                    'A_plasma': A_plasma_fit,
                    'tau_decay_days': tau_decay_fit,
                    'beta': beta_fit,
                    't0_mjd': t0_fit,
                    'band_offsets': band_offsets_fit,
                    'chi2': result.fun,
                    'ndof': len(flux_obs) - len(result.x),
                    'success': True,
                    'H0_implied_kms_mpc': H0_implied,
                    'qfd_vs_obs_redshift_ratio': z_qfd / z_obs if z_obs > 0 else np.nan
                }

                print(f"   ✅ Success: D = {D_fit:.1f} Mpc, A = {A_plasma_fit:.3f}, χ²/ν = {result.fun/(len(flux_obs)-len(result.x)):.2f}")
                return fit_result

            else:
                print(f"   ❌ Optimization failed: {result.message}")
                return {'snid': snid, 'success': False, 'error': result.message}

        except Exception as e:
            print(f"   ❌ Fitting error: {e}")
            return {'snid': snid, 'success': False, 'error': str(e)}

    def process_light_curve_batch(self, lc_files: List[str], output_dir: str) -> str:
        """
        Process a batch of light curve files to extract QFD-native distances.

        Returns: Path to QFD-native Hubble table
        """
        os.makedirs(output_dir, exist_ok=True)

        all_results = []

        for lc_file in lc_files:
            print(f"\n📊 Processing {lc_file}")

            # Load light curve data
            if lc_file.endswith('.parquet'):
                lc_data = pd.read_parquet(lc_file)
            else:
                lc_data = pd.read_csv(lc_file)

            # Check for QFD purity
            if 'salt2_free' in lc_data.columns and not lc_data['salt2_free'].iloc[0]:
                print(f"⚠️  Warning: {lc_file} may contain SALT2 contamination")

            # Get unique supernovae with redshifts
            sn_list = lc_data[['snid', 'z_helio']].drop_duplicates()
            sn_list = sn_list[sn_list['z_helio'].notna() & (sn_list['z_helio'] > 0)]

            print(f"   Found {len(sn_list)} supernovae with redshifts")

            # Fit each supernova
            for _, row in sn_list.iterrows():
                result = self.fit_single_sn(lc_data, row['snid'], row['z_helio'])
                all_results.append(result)

        # Create QFD-native Hubble table
        successful_fits = [r for r in all_results if r.get('success', False)]

        if successful_fits:
            hubble_df = pd.DataFrame(successful_fits)

            # Add QFD-native distance modulus
            hubble_df['mu_qfd'] = 5 * np.log10(hubble_df['D_qfd_mpc']) + 25 + self.delta_mu0

            # Save results
            hubble_path = os.path.join(output_dir, 'qfd_native_hubble_table.csv')
            hubble_df.to_csv(hubble_path, index=False)

            print(f"\n✅ Created QFD-native Hubble table: {hubble_path}")
            print(f"   {len(successful_fits)} successful distance measurements")
            print(f"   Distance range: {hubble_df['D_qfd_mpc'].min():.1f} - {hubble_df['D_qfd_mpc'].max():.1f} Mpc")

            return hubble_path
        else:
            print("\n❌ No successful fits - check data quality")
            return None

def load_global_params(param_file: str) -> Dict[str, float]:
    """Load global QFD parameters from Phase 1 analysis."""
    with open(param_file, 'r') as f:
        params = json.load(f)

    # Extract parameters (adapt to your Phase 1 output format)
    if 'k_J_map' in params:
        # Format from definitive analysis
        return {
            'k_J': params['k_J_map'],
            'eta_prime': params['eta_prime_map'],
            'xi': params['xi_map'],
            'delta_mu0': params.get('delta_mu0_map', 0.0)
        }
    else:
        # Direct format
        return {
            'k_J': params['k_J'],
            'eta_prime': params['eta_prime'],
            'xi': params['xi'],
            'delta_mu0': params.get('delta_mu0', 0.0)
        }

def main():
    parser = argparse.ArgumentParser(description="QFD-native distance fitter")
    parser.add_argument('--global-params', required=True,
                       help='JSON file with global QFD parameters from Phase 1')
    parser.add_argument('--light-curves', nargs='+', required=True,
                       help='Light curve files (CSV or parquet)')
    parser.add_argument('--output-dir', default='qfd_native_distances',
                       help='Output directory for results')
    parser.add_argument('--dl-factor', choices=['D', 'D*(1+z)'], default='D',
                       help='Luminosity distance convention for QFD flux')

    args = parser.parse_args()

    print("🔬 QFD Native Distance Pipeline")
    print("=" * 50)
    print("🚫 Zero ΛCDM assumptions")
    print("🚫 Zero SALT2 artifacts")
    print("✅ Pure QFD physics: photometry → distance")
    print("=" * 50)

    # Load global parameters
    global_params = load_global_params(args.global_params)

    # Initialize fitter with luminosity distance convention
    fitter = QFDNativeDistanceFitter(global_params, dl_factor=args.dl_factor)

    # Process light curves
    hubble_path = fitter.process_light_curve_batch(args.light_curves, args.output_dir)

    if hubble_path:
        print(f"\n🎉 QFD-native distance analysis complete!")
        print(f"📊 Results: {hubble_path}")
        print(f"🔬 Ready for QFD-native Hubble diagram")

if __name__ == "__main__":
    main()