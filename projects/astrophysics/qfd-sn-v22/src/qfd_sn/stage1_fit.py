"""
Stage 1: Per-Supernova Light Curve Fitting

Fits individual supernova light curves to extract:
    - ln_A: Log amplitude (distance-related parameter)
    - stretch: Light curve time scale (tests time dilation)
    - chi²: Fit quality

This is data-agnostic and works with any Type Ia SN dataset:
    - DES-SN5YR
    - Pantheon+
    - Custom datasets

Physical Model:
    flux(t, λ) = A × template(t/stretch, λ) × exp(-opacity(λ))

Where:
    - A = exp(ln_A) encodes distance and extinction
    - stretch encodes light curve time scale
    - template is a standard SN Ia light curve template
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Tuple, Dict, Optional
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# Standard SN Ia template (Nugent Ia template, simplified)
# In practice, this should be loaded from a proper template file
def get_snia_template():
    """
    Get standard SN Ia light curve template.

    Returns normalized flux as function of (phase, wavelength).
    This is a simplified version - production code should use
    actual SN Ia templates (e.g., Hsiao, Nugent, SALT2).
    """
    # Simplified template: Gaussian in time, power-law in wavelength
    # Phase in days relative to peak, wavelength in Angstroms

    def template_flux(phase_days, wavelength_angstrom):
        """
        Simple SN Ia template: Gaussian in time × power-law in wavelength.

        Real implementation should use empirical templates.
        """
        # Time evolution: rise time ~17 days, decline time ~20 days
        if phase_days < 0:
            time_factor = np.exp(phase_days / 17.0)  # Rising
        else:
            time_factor = np.exp(-phase_days / 20.0)  # Declining

        # Wavelength dependence: power-law (blue is brighter)
        # Reference wavelength: 4400 Å (B-band)
        wavelength_factor = (wavelength_angstrom / 4400.0)**(-0.5)

        return time_factor * wavelength_factor

    return template_flux


def convert_filter_to_wavelength(filter_name: str) -> float:
    """
    Convert filter name to effective wavelength (Å).

    Args:
        filter_name: Filter designation (g, r, i, z, etc.)

    Returns:
        Effective wavelength in Angstroms
    """
    filter_wavelengths = {
        'u': 3543,
        'g': 4770,
        'r': 6231,
        'i': 7625,
        'z': 9134,
        'U': 3560,
        'B': 4390,
        'V': 5490,
        'R': 6580,
        'I': 8060,
    }

    # Handle DES naming convention (desg, desr, etc.)
    if filter_name.startswith('des'):
        filter_name = filter_name[3:]

    base_filter = filter_name.strip().lower()[0]

    if base_filter not in filter_wavelengths:
        # Default to r-band if unknown
        print(f"Warning: Unknown filter '{filter_name}', using r-band (6231 Å)")
        return 6231.0

    return filter_wavelengths[base_filter]


def fit_single_sn_simple(
    sn_name: str,
    z: float,
    mjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    filters: np.ndarray,
    max_iterations: int = 1000
) -> Dict:
    """
    Fit individual SN light curve (simplified robust version).

    Fits:
        - t0: Time of peak (MJD)
        - ln_A: Log amplitude
        - stretch: Time scale factor

    Args:
        sn_name: Supernova identifier
        z: Redshift
        mjd: Modified Julian Dates
        flux: Observed fluxes
        flux_err: Flux uncertainties
        filters: Filter names
        max_iterations: Maximum optimizer iterations

    Returns:
        Dictionary with fit results
    """
    # Filter valid data
    valid = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    if np.sum(valid) < 5:
        return {
            'name': sn_name,
            'z': z,
            'n_epochs': 0,
            'chi2_dof': 1e6,
            'ln_A': 0.0,
            'stretch': 1.0,
            't0': 0.0,
            'success': False,
            'reason': 'insufficient_valid_data'
        }

    mjd = mjd[valid]
    flux = flux[valid]
    flux_err = flux_err[valid]
    filters = filters[valid]

    n_epochs = len(mjd)

    # Convert filters to wavelengths
    wavelengths = np.array([convert_filter_to_wavelength(f) for f in filters])

    # Get template
    template_func = get_snia_template()

    # Initial guesses
    t0_init = mjd[np.argmax(flux)]  # Peak time
    ln_A_init = np.log(np.max(flux) / 1000.0)  # Rough amplitude
    stretch_init = 1.0

    # Define model
    def model(mjd, t0, ln_A, stretch):
        """Light curve model with amplitude, stretch, and peak time."""
        phase = (mjd - t0) / stretch  # Stretched phase

        # Evaluate template at each (phase, wavelength)
        template_values = np.array([
            template_func(p, w)
            for p, w in zip(phase, wavelengths)
        ])

        # Apply amplitude
        return np.exp(ln_A) * template_values

    # Bounds
    t_span = mjd.max() - mjd.min()
    bounds = (
        [mjd.min() - t_span, -30.0, 0.5],  # Lower bounds
        [mjd.max() + t_span, 30.0, 3.0]     # Upper bounds
    )

    try:
        # Fit with curve_fit (uses Levenberg-Marquardt or Trust Region)
        popt, pcov = curve_fit(
            model, mjd, flux,
            p0=[t0_init, ln_A_init, stretch_init],
            sigma=flux_err,
            absolute_sigma=True,
            bounds=bounds,
            max_nfev=max_iterations,
            method='trf'  # Trust Region Reflective
        )

        t0_fit, ln_A_fit, stretch_fit = popt

        # Compute chi²
        model_flux = model(mjd, *popt)
        residuals = (flux - model_flux) / flux_err
        chi2 = np.sum(residuals**2)
        chi2_dof = chi2 / max(1, n_epochs - 3)  # 3 fitted parameters

        return {
            'name': sn_name,
            'z': z,
            'n_epochs': n_epochs,
            'chi2': float(chi2),
            'chi2_dof': float(chi2_dof),
            'ln_A': float(ln_A_fit),
            'stretch': float(stretch_fit),
            't0': float(t0_fit),
            'success': True,
            'reason': 'converged'
        }

    except Exception as e:
        # Fit failed - return fallback values
        return {
            'name': sn_name,
            'z': z,
            'n_epochs': n_epochs,
            'chi2_dof': 1e6,
            'ln_A': 0.0,
            'stretch': 1.0,
            't0': float(t0_init),
            'success': False,
            'reason': f'fit_failed: {str(e)[:100]}'
        }


def load_lightcurves(filepath: str) -> pd.DataFrame:
    """
    Load light curve data from CSV.

    Expected columns:
        - name or snid: Supernova identifier
        - z or redshift: Redshift
        - mjd or MJD: Modified Julian Date
        - flux: Observed flux
        - flux_err or fluxerr: Flux uncertainty
        - filter or band: Filter/band name

    Args:
        filepath: Path to light curve CSV file

    Returns:
        DataFrame with standardized column names
    """
    df = pd.read_csv(filepath)

    # Standardize column names
    column_mapping = {
        'snid': 'name',
        'SNID': 'name',
        'redshift': 'z',
        'Z': 'z',
        'MJD': 'mjd',
        'flux_nu_jy': 'flux',  # DES-SN5YR format
        'flux_nu_jy_err': 'flux_err',  # DES-SN5YR format
        'fluxerr': 'flux_err',
        'FLUXERR': 'flux_err',
        'band': 'filter',
        'BAND': 'filter',
        'FILTER': 'filter'
    }

    df = df.rename(columns=column_mapping)

    # Ensure required columns exist
    required = ['name', 'z', 'mjd', 'flux', 'flux_err', 'filter']
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Expected: {required}"
        )

    print(f"Loaded light curves for {df['name'].nunique()} supernovae")
    print(f"Total epochs: {len(df)}")
    print(f"Redshift range: {df['z'].min():.3f} to {df['z'].max():.3f}")

    return df


def process_single_sn(args):
    """Wrapper for parallel processing."""
    sn_name, sn_data = args

    return fit_single_sn_simple(
        sn_name=sn_name,
        z=sn_data['z'].iloc[0],
        mjd=sn_data['mjd'].values,
        flux=sn_data['flux'].values,
        flux_err=sn_data['flux_err'].values,
        filters=sn_data['filter'].values
    )


def run_stage1(
    lightcurve_file: str,
    output_dir: str,
    n_cores: int = 1,
    max_sne: Optional[int] = None
) -> None:
    """
    Run Stage 1 on all supernovae.

    Args:
        lightcurve_file: Path to light curve CSV
        output_dir: Directory to save results
        n_cores: Number of parallel cores (-1 = all available)
        max_sne: Maximum number of SNe to process (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("STAGE 1: PER-SUPERNOVA LIGHT CURVE FITTING")
    print("="*60)

    # Load data
    df = load_lightcurves(lightcurve_file)

    # Group by supernova
    sne_groups = list(df.groupby('name'))

    if max_sne is not None:
        sne_groups = sne_groups[:max_sne]
        print(f"\nLimiting to first {max_sne} SNe for testing")

    n_sne = len(sne_groups)
    print(f"\nFitting {n_sne} supernovae...")

    # Determine number of cores
    if n_cores == -1:
        import multiprocessing
        n_cores = multiprocessing.cpu_count()

    print(f"Using {n_cores} cores")

    # Process in parallel
    results = []

    if n_cores > 1:
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = {
                executor.submit(process_single_sn, sn_group): sn_group[0]
                for sn_group in sne_groups
            }

            for future in tqdm(as_completed(futures), total=n_sne, desc="Fitting"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    sn_name = futures[future]
                    print(f"\nError processing {sn_name}: {e}")
                    results.append({
                        'name': sn_name,
                        'success': False,
                        'reason': f'exception: {str(e)[:100]}'
                    })
    else:
        # Serial processing
        for sn_group in tqdm(sne_groups, desc="Fitting"):
            try:
                result = process_single_sn(sn_group)
                results.append(result)
            except Exception as e:
                print(f"\nError processing {sn_group[0]}: {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_file = output_path / 'stage1_results.csv'
    results_df.to_csv(output_file, index=False)

    # Print summary
    n_success = results_df['success'].sum()
    n_failed = len(results_df) - n_success

    print(f"\n{'='*60}")
    print("STAGE 1 COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults:")
    print(f"  Total SNe: {len(results_df)}")
    print(f"  Successful fits: {n_success} ({100*n_success/len(results_df):.1f}%)")
    print(f"  Failed fits: {n_failed} ({100*n_failed/len(results_df):.1f}%)")

    if n_success > 0:
        success_df = results_df[results_df['success']]
        print(f"\nFit Quality (successful fits):")
        print(f"  Median chi²/dof: {success_df['chi2_dof'].median():.2f}")
        print(f"  ln_A range: [{success_df['ln_A'].min():.2f}, {success_df['ln_A'].max():.2f}]")
        print(f"  ln_A scatter: {success_df['ln_A'].std():.2f}")
        print(f"  Stretch range: [{success_df['stretch'].min():.2f}, {success_df['stretch'].max():.2f}]")
        print(f"  Stretch mean: {success_df['stretch'].mean():.2f} ± {success_df['stretch'].std():.2f}")

    print(f"\nResults saved to: {output_file}")

    # Create summary JSON
    summary = {
        'n_sne_total': len(results_df),
        'n_success': int(n_success),
        'n_failed': int(n_failed),
        'success_rate': float(n_success / len(results_df)),
        'lightcurve_file': lightcurve_file,
        'output_dir': str(output_dir)
    }

    if n_success > 0:
        summary['fit_statistics'] = {
            'median_chi2_dof': float(success_df['chi2_dof'].median()),
            'ln_A_mean': float(success_df['ln_A'].mean()),
            'ln_A_std': float(success_df['ln_A'].std()),
            'stretch_mean': float(success_df['stretch'].mean()),
            'stretch_std': float(success_df['stretch'].std())
        }

    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    """Main Stage 1 execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Stage 1: Per-Supernova Light Curve Fitting'
    )
    parser.add_argument(
        '--lightcurves', type=str, required=True,
        help='Path to light curve CSV file'
    )
    parser.add_argument(
        '--output', type=str, default='results/stage1',
        help='Output directory'
    )
    parser.add_argument(
        '--ncores', type=int, default=1,
        help='Number of parallel cores (-1 = all available)'
    )
    parser.add_argument(
        '--max-sne', type=int, default=None,
        help='Maximum number of SNe to process (for testing)'
    )

    args = parser.parse_args()

    run_stage1(
        lightcurve_file=args.lightcurves,
        output_dir=args.output,
        n_cores=args.ncores,
        max_sne=args.max_sne
    )


if __name__ == '__main__':
    main()
