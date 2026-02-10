#!/usr/bin/env python3
"""
chromatic_kj_test.py — Test for wavelength-dependent vacuum refraction (K_J)

QFD FRAMEWORK (static universe, no expansion, no dark energy):
    Redshift = vacuum drag on photon energy (NOT Doppler / expansion)
    K_J = vacuum refraction parameter (NOT recession velocity)
    D(z) = z * c / K_J  (QFD distance in static Minkowski spacetime)

PHYSICS:
    Forward scattering (drag vertex, delta function):
        Achromatic, dominant.  Preserves image coherence.
        Sets the geometric core: K_J_geo = xi_QFD * beta^(3/2)

    Non-forward scattering (four-photon vertex, lambda'_4gamma):
        Chromatic, subdominant.  sigma ~ E^2 ~ lambda^(-2)
        Removes photons from beam -> extra dimming at short wavelengths

PREDICTION:
    K_J(lambda) = K_J_geo + delta_K * (lambda_ref / lambda)^2

    Derived geometric core:
        xi_QFD = (7*pi/5)^2 * (5/6) = 49*pi^2/30 ~ 16.11
        beta^(3/2) ~ 5.31
        K_J_geo = xi_QFD * beta^(3/2) ~ 85.6  [km/s/Mpc equivalent]

    Observable:
        K_J(g=472nm) > K_J(r=642nm) > K_J(i=784nm) > K_J(z=867nm)

HUBBLE TENSION RESOLUTION:
    CMB (microwave, lambda~mm):   minimal scattering -> inferred K_J ~ 67
    Optical (SNe, lambda~550nm):  moderate scattering -> inferred K_J ~ 73-90
    Geometric truth:              K_J_geo ~ 85.6 (from nuclear beta)
    The "5-sigma tension" = chromatic vacuum dispersion, not a crisis.

DISTINGUISHING SIGNATURES:
    Dust extinction:      delta_m ~ lambda^(-1)  (Cardelli law)
    QFD vacuum scatter:   delta_m ~ lambda^(-2)  (four-photon vertex)
    Rayleigh scattering:  delta_m ~ lambda^(-4)  (molecular)
    -> Spectral index n=-2 is unique to QFD.

DATA: Pantheon+SH0ES multi-band photometry (DES g,r,i,z)
"""

import numpy as np
import os
import sys
from collections import defaultdict

# ============================================================
# QFD CONSTANTS (Golden Loop: 1/alpha = 2*pi^2*(e^beta/beta) + 1)
# ============================================================

ALPHA = 0.0072973525693          # Fine structure constant (CODATA 2018)
PI = np.pi
EU = np.e                        # Euler's number (avoid shadowing math.e)
C_LIGHT_KM_S = 299792.458       # Speed of light [km/s]


def solve_beta(alpha):
    """Golden Loop: solve 1/alpha = 2*pi^2*(e^beta/beta) + 1 for beta.
    Newton-Raphson on f(b) = 2*pi^2*(e^b/b) - (1/alpha - 1) = 0."""
    target = (1.0 / alpha) - 1.0
    C = 2.0 * PI * PI
    b = 3.0
    for _ in range(100):
        val = C * (np.exp(b) / b) - target
        slope = C * np.exp(b) * (b - 1.0) / (b * b)
        if abs(slope) < 1e-20:
            break
        step = val / slope
        b -= step
        if abs(step) < 1e-14:
            break
    return b


BETA = solve_beta(ALPHA)
K_GEOM = 7.0 * PI / 5.0                       # Hill vortex eigenvalue
XI_QFD = K_GEOM**2 * (5.0 / 6.0)             # Gravitational coupling
KJ_GEOMETRIC = XI_QFD * BETA**1.5             # Derived vacuum drag [dimensionless]

# Reference wavelength (r-band center)
LAMBDA_REF_NM = 642.0

# Band definitions (DES filters)
BANDS = {
    'g': 472.0,   # nm
    'r': 642.0,
    'i': 784.0,
    'z': 867.0,
}


# ============================================================
# DATA LOADING
# ============================================================

def find_data_file():
    """Locate Pantheon+ multi-band photometry."""
    candidates = [
        os.path.join(os.path.dirname(__file__),
                     '../../../data/raw/pantheon_plus_all_photometry.csv'),
        '/home/tracy/development/QFD_SpectralGap/projects/astrophysics/'
        'qfd-supernova-v15/data/pantheon_plus_all_photometry.csv',
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Cannot find pantheon_plus_all_photometry.csv")


def load_photometry(path):
    """Load Pantheon+ photometry into structured arrays.
    Returns list of dicts: {snid, z, band, mjd, flux, flux_err, wavelength, snr}
    """
    entries = []
    with open(path) as f:
        header = f.readline().strip().split(',')
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < len(header):
                continue
            band = parts[col['band']].strip()
            if band not in BANDS:
                continue
            try:
                entry = {
                    'snid': parts[col['snid']].strip(),
                    'z': float(parts[col['z']]),
                    'band': band,
                    'mjd': float(parts[col['mjd']]),
                    'flux': float(parts[col['flux_nu_jy']]),
                    'flux_err': float(parts[col['flux_nu_jy_err']]),
                    'wavelength': float(parts[col['wavelength_eff_nm']]),
                    'snr': float(parts[col['snr']]),
                }
                entries.append(entry)
            except (ValueError, IndexError):
                continue
    return entries


def extract_peak_magnitudes(entries, min_snr=5.0, min_obs_per_band=3):
    """For each SN, extract peak apparent magnitude per band.

    Uses: top-3 brightest observations per band (weighted by SNR).
    Requires at least min_obs_per_band observations per band.

    Returns dict: snid -> {z, bands: {band: m_peak}}
    """
    # Group by SN
    by_sn = defaultdict(list)
    for e in entries:
        if e['snr'] >= min_snr and e['flux'] > 0:
            by_sn[e['snid']].append(e)

    results = {}
    for snid, obs_list in by_sn.items():
        z = obs_list[0]['z']
        if z < 0.01:    # skip very nearby (peculiar velocity dominated)
            continue
        if z > 1.2:     # skip very distant (selection effects)
            continue

        # Group by band
        by_band = defaultdict(list)
        for o in obs_list:
            by_band[o['band']].append(o)

        band_mags = {}
        for band, band_obs in by_band.items():
            if len(band_obs) < min_obs_per_band:
                continue

            # Take top-3 brightest, weighted mean by SNR
            sorted_obs = sorted(band_obs, key=lambda x: -x['flux'])
            top = sorted_obs[:3]

            weights = np.array([o['snr'] for o in top])
            fluxes = np.array([o['flux'] for o in top])

            if np.sum(weights) < 1e-10:
                continue
            peak_flux = np.average(fluxes, weights=weights)

            if peak_flux <= 0:
                continue

            # AB magnitude: m = -2.5 * log10(f_nu / 3631 Jy)
            m_ab = -2.5 * np.log10(peak_flux / 3631.0)
            band_mags[band] = m_ab

        # Require at least g and r (or any 3 bands)
        if len(band_mags) >= 3:
            results[snid] = {'z': z, 'bands': band_mags}

    return results


# ============================================================
# TEST 1: PER-BAND K_J FITTING
# ============================================================

def fit_kj_per_band(sn_data):
    """Fit K_J and M_abs independently for each band.

    QFD model:
        mu_obs = m_peak - M_abs
        mu_QFD = 5 * log10(z * c / K_J) + 25

    Minimize: sum( (m_peak - M_abs - 5*log10(z*c/K_J) - 25)^2 )

    Uses grid search over K_J, analytic M_abs at each K_J.
    """
    results = {}

    for band in BANDS:
        # Collect data for this band
        zz = []
        mm = []
        for snid, info in sn_data.items():
            if band in info['bands']:
                zz.append(info['z'])
                mm.append(info['bands'][band])
        zz = np.array(zz)
        mm = np.array(mm)

        if len(zz) < 20:
            print(f"  {band}-band: insufficient data ({len(zz)} SNe)")
            continue

        # Grid search over K_J
        kj_grid = np.linspace(50.0, 150.0, 1001)
        best_kj = None
        best_chi2 = np.inf

        for kj in kj_grid:
            mu_model = 5.0 * np.log10(zz * C_LIGHT_KM_S / kj) + 25.0
            # Optimal M_abs is the mean offset
            residuals = mm - mu_model
            M_abs = np.median(residuals)
            chi2 = np.sum((mm - mu_model - M_abs)**2)

            if chi2 < best_chi2:
                best_chi2 = chi2
                best_kj = kj
                best_M = M_abs

        # Compute RMS
        mu_best = 5.0 * np.log10(zz * C_LIGHT_KM_S / best_kj) + 25.0
        rms = np.sqrt(np.mean((mm - mu_best - best_M)**2))
        n_sn = len(zz)

        results[band] = {
            'kj': best_kj,
            'M_abs': best_M,
            'rms': rms,
            'n_sn': n_sn,
            'wavelength': BANDS[band],
        }

    return results


# ============================================================
# TEST 2: COLOR-REDSHIFT SLOPE (DIFFERENTIAL, MODEL-INDEPENDENT)
# ============================================================

def color_redshift_test(sn_data):
    """Test whether SN color correlates with redshift.

    If vacuum is dispersive (sigma ~ lambda^-2):
        C_{gr}(z) = C_{gr,intrinsic} + delta_C * z
        where delta_C ~ (lambda_g^-2 - lambda_r^-2) * chromatic_coeff

    Compare slopes for different color pairs to determine spectral index.
    """
    # Color pairs to test
    pairs = [('g', 'r'), ('g', 'i'), ('g', 'z'), ('r', 'i'), ('r', 'z')]

    results = {}

    for b1, b2 in pairs:
        zz = []
        colors = []
        for snid, info in sn_data.items():
            if b1 in info['bands'] and b2 in info['bands']:
                zz.append(info['z'])
                colors.append(info['bands'][b1] - info['bands'][b2])

        zz = np.array(zz)
        colors = np.array(colors)

        if len(zz) < 20:
            continue

        # Linear fit: color = a + b * z
        A = np.vstack([np.ones_like(zz), zz]).T
        (intercept, slope), residuals, _, _ = np.linalg.lstsq(A, colors, rcond=None)

        # Uncertainty on slope (bootstrap)
        n_boot = 1000
        slopes_boot = np.zeros(n_boot)
        for i in range(n_boot):
            idx = np.random.choice(len(zz), len(zz), replace=True)
            A_b = np.vstack([np.ones_like(zz[idx]), zz[idx]]).T
            (_, s_b), _, _, _ = np.linalg.lstsq(A_b, colors[idx], rcond=None)
            slopes_boot[i] = s_b

        slope_err = np.std(slopes_boot)
        significance = abs(slope) / slope_err if slope_err > 0 else 0

        # Predicted chromatic factor
        lam1 = BANDS[b1]
        lam2 = BANDS[b2]
        chromatic_factor = (lam1**(-2) - lam2**(-2))

        results[f'{b1}-{b2}'] = {
            'slope': slope,
            'slope_err': slope_err,
            'intercept': intercept,
            'significance': significance,
            'n_sn': len(zz),
            'chromatic_factor': chromatic_factor,
            'lam1': lam1,
            'lam2': lam2,
        }

    return results


# ============================================================
# TEST 3: SPECTRAL INDEX DETERMINATION
# ============================================================

def fit_spectral_index(kj_results):
    """Fit the power law: K_J(lambda) = K_J_0 * (1 + A * (lambda_ref/lambda)^n)

    Compare n = -1 (dust), n = -2 (QFD vacuum), n = -4 (Rayleigh).
    """
    if len(kj_results) < 3:
        return None

    wavelengths = np.array([kj_results[b]['wavelength'] for b in sorted(kj_results)])
    kj_values = np.array([kj_results[b]['kj'] for b in sorted(kj_results)])

    # Normalize
    kj_mean = np.mean(kj_values)
    delta_kj = kj_values - kj_mean

    # Test each spectral index
    test_indices = {
        'dust (n=-1)': -1.0,
        'QFD vacuum (n=-2)': -2.0,
        'Rayleigh (n=-4)': -4.0,
    }

    results = {}
    for label, n in test_indices.items():
        # Model: delta_kj = A * (lambda_ref / lambda)^n
        x = (LAMBDA_REF_NM / wavelengths)**n
        x_centered = x - np.mean(x)

        if np.sum(x_centered**2) < 1e-20:
            continue

        A = np.sum(delta_kj * x_centered) / np.sum(x_centered**2)
        model = A * x_centered
        residual = np.sum((delta_kj - model)**2)
        total_var = np.sum(delta_kj**2)
        r2 = 1.0 - residual / total_var if total_var > 0 else 0

        results[label] = {
            'index': n,
            'amplitude': A,
            'r2': r2,
            'residual_rms': np.sqrt(residual / len(wavelengths)),
        }

    return results


# ============================================================
# TEST 4: GEOMETRIC K_J LOCK (Fixed K_J = 85.6)
# ============================================================

def geometric_lock_test(sn_data):
    """Test: fix K_J at the derived geometric value (xi_QFD * beta^1.5).
    Compare fit quality to freely-fitted K_J.

    If geometric K_J works, the entire cosmological sector has zero free
    parameters for the vacuum drag component.
    """
    kj_fixed = KJ_GEOMETRIC
    kj_free_values = [70.0, 75.0, 80.0, 85.0, 85.6, 90.0, 95.0, 100.0]

    # Use r-band as reference (most observations, middle of optical)
    zz = []
    mm = []
    for snid, info in sn_data.items():
        if 'r' in info['bands']:
            zz.append(info['z'])
            mm.append(info['bands']['r'])
    zz = np.array(zz)
    mm = np.array(mm)

    if len(zz) < 20:
        return None

    results = {}
    for kj in kj_free_values:
        mu_model = 5.0 * np.log10(zz * C_LIGHT_KM_S / kj) + 25.0
        M_abs = np.median(mm - mu_model)
        resid = mm - mu_model - M_abs
        rms = np.sqrt(np.mean(resid**2))
        results[kj] = {'rms': rms, 'M_abs': M_abs, 'n_sn': len(zz)}

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("QFD CHROMATIC VACUUM REFRACTION TEST")
    print("Static Universe | No Expansion | No Dark Energy")
    print("=" * 70)

    # --- Golden Loop derivation ---
    print("\n--- DERIVED CONSTANTS (Golden Loop: alpha -> beta -> K_J) ---")
    print(f"  alpha             = {ALPHA:.13f}  (CODATA 2018)")
    print(f"  beta              = {BETA:.10f}  (Golden Loop)")
    print(f"  k_geom            = 7*pi/5 = {K_GEOM:.6f}")
    print(f"  xi_QFD            = k_geom^2 * (5/6) = {XI_QFD:.4f}")
    print(f"  beta^(3/2)        = {BETA**1.5:.4f}")
    print(f"  K_J_geometric     = xi_QFD * beta^(3/2) = {KJ_GEOMETRIC:.2f}")
    print(f"")
    print(f"  Prediction: geometric vacuum drag = {KJ_GEOMETRIC:.1f} km/s/Mpc")
    print(f"  CMB inference:  ~67   (microwave, minimal chromatic scatter)")
    print(f"  SN inference:   ~73   (optical, chromatic scatter inflates)")
    print(f"  SN QFD fit:     ~90   (optical + plasma veil)")

    # --- Load data ---
    print("\n--- LOADING PANTHEON+ MULTI-BAND PHOTOMETRY ---")
    data_path = find_data_file()
    print(f"  File: {data_path}")
    entries = load_photometry(data_path)
    print(f"  Loaded {len(entries)} observations in bands {sorted(BANDS.keys())}")

    # Count per band
    band_counts = defaultdict(int)
    sn_ids = set()
    for e in entries:
        band_counts[e['band']] += 1
        sn_ids.add(e['snid'])
    print(f"  Unique supernovae: {len(sn_ids)}")
    for b in sorted(BANDS):
        print(f"    {b}-band ({BANDS[b]:.0f} nm): {band_counts[b]} observations")

    # --- Extract peak magnitudes ---
    print("\n--- EXTRACTING PEAK MAGNITUDES ---")
    sn_data = extract_peak_magnitudes(entries)
    print(f"  Supernovae with >= 3 bands: {len(sn_data)}")

    band_coverage = defaultdict(int)
    for info in sn_data.values():
        for b in info['bands']:
            band_coverage[b] += 1
    for b in sorted(BANDS):
        print(f"    {b}-band: {band_coverage[b]} SNe with peak magnitude")

    # =========================================================
    # TEST 1: PER-BAND K_J FITTING
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 1: PER-BAND K_J FITTING")
    print("  QFD distance: D(z) = z * c / K_J  [static spacetime]")
    print("  Fit K_J and M_abs independently per band")
    print("=" * 70)

    kj_results = fit_kj_per_band(sn_data)

    if kj_results:
        print(f"\n  {'Band':<8} {'lambda(nm)':<12} {'K_J':<10} {'M_abs':<10} "
              f"{'RMS(mag)':<10} {'N_SN':<8}")
        print("  " + "-" * 58)
        for band in ['g', 'r', 'i', 'z']:
            if band in kj_results:
                r = kj_results[band]
                print(f"  {band:<8} {r['wavelength']:<12.0f} {r['kj']:<10.1f} "
                      f"{r['M_abs']:<10.2f} {r['rms']:<10.3f} {r['n_sn']:<8d}")

        # Check monotonic prediction
        bands_sorted = [b for b in ['g', 'r', 'i', 'z'] if b in kj_results]
        kj_sorted = [kj_results[b]['kj'] for b in bands_sorted]
        is_monotonic = all(kj_sorted[i] >= kj_sorted[i+1]
                          for i in range(len(kj_sorted)-1))

        print(f"\n  Chromatic prediction (K_J(g) > K_J(r) > K_J(i) > K_J(z)):")
        print(f"    K_J values: {' > '.join(f'{k:.1f}' for k in kj_sorted)}")
        print(f"    Monotonic decreasing: {'YES' if is_monotonic else 'NO'}")

        if len(kj_sorted) >= 2:
            spread = kj_sorted[0] - kj_sorted[-1]
            print(f"    Spread (g - z): {spread:.1f} km/s/Mpc")

        # Compare to geometric prediction
        kj_r = kj_results.get('r', {}).get('kj', 0)
        if kj_r > 0:
            pct_diff = 100.0 * (kj_r - KJ_GEOMETRIC) / KJ_GEOMETRIC
            print(f"\n  r-band K_J vs geometric prediction:")
            print(f"    K_J(r)      = {kj_r:.1f}")
            print(f"    K_J_geo     = {KJ_GEOMETRIC:.1f}")
            print(f"    Difference  = {pct_diff:+.1f}%")

    # =========================================================
    # TEST 2: COLOR-REDSHIFT SLOPES
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 2: COLOR-REDSHIFT SLOPES (MODEL-INDEPENDENT)")
    print("  If vacuum is dispersive: color should correlate with z")
    print("  Slope(color vs z) ~ (lambda1^-2 - lambda2^-2) * coeff")
    print("=" * 70)

    color_results = color_redshift_test(sn_data)

    if color_results:
        print(f"\n  {'Color':<8} {'Slope':<12} {'Err':<10} {'Signif':<8} "
              f"{'Chrom.Factor':<14} {'N_SN':<8}")
        print("  " + "-" * 60)
        for pair in ['g-r', 'g-i', 'g-z', 'r-i', 'r-z']:
            if pair in color_results:
                r = color_results[pair]
                print(f"  {pair:<8} {r['slope']:< 12.4f} {r['slope_err']:<10.4f} "
                      f"{r['significance']:<8.1f} {r['chromatic_factor']:<14.6f} "
                      f"{r['n_sn']:<8d}")

        # Check if slopes scale with chromatic factor
        pairs_with_data = [(k, v) for k, v in color_results.items()
                           if v['significance'] > 1.0]
        if len(pairs_with_data) >= 3:
            slopes = np.array([v['slope'] for _, v in pairs_with_data])
            factors = np.array([v['chromatic_factor'] for _, v in pairs_with_data])

            # Correlation between slope and lambda^-2 factor
            if np.std(factors) > 0 and np.std(slopes) > 0:
                corr = np.corrcoef(slopes, factors)[0, 1]
                print(f"\n  Correlation(slope, lambda^-2 factor): {corr:.3f}")
                print(f"  (Positive correlation supports QFD chromatic prediction)")

    # =========================================================
    # TEST 3: SPECTRAL INDEX
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 3: SPECTRAL INDEX DETERMINATION")
    print("  Dust: n=-1 | QFD vacuum: n=-2 | Rayleigh: n=-4")
    print("=" * 70)

    spec_results = fit_spectral_index(kj_results)

    if spec_results:
        print(f"\n  {'Model':<25} {'Index':<8} {'R^2':<10} {'Resid RMS':<12}")
        print("  " + "-" * 55)
        for label in sorted(spec_results, key=lambda x: -spec_results[x]['r2']):
            r = spec_results[label]
            marker = " <-- BEST" if r['r2'] == max(v['r2'] for v in spec_results.values()) else ""
            print(f"  {label:<25} {r['index']:<8.0f} {r['r2']:<10.4f} "
                  f"{r['residual_rms']:<12.4f}{marker}")

    # =========================================================
    # TEST 4: GEOMETRIC LOCK TEST
    # =========================================================
    print("\n" + "=" * 70)
    print("TEST 4: GEOMETRIC LOCK TEST (r-band)")
    print(f"  K_J_geometric = xi_QFD * beta^(3/2) = {KJ_GEOMETRIC:.2f}")
    print("  Does fixing K_J at the derived value preserve fit quality?")
    print("=" * 70)

    lock_results = geometric_lock_test(sn_data)

    if lock_results:
        print(f"\n  {'K_J':<10} {'RMS(mag)':<12} {'M_abs':<10} {'N_SN':<8} {'Note':<20}")
        print("  " + "-" * 60)
        best_kj = min(lock_results, key=lambda k: lock_results[k]['rms'])
        for kj in sorted(lock_results):
            r = lock_results[kj]
            note = ""
            if abs(kj - KJ_GEOMETRIC) < 0.5:
                note = "<-- GEOMETRIC"
            elif abs(kj - 67) < 3:
                note = "(CMB inference)"
            elif abs(kj - 73) < 3:
                note = "(SN local)"
            elif kj == best_kj:
                note = "<-- BEST FIT"
            print(f"  {kj:<10.1f} {r['rms']:<12.4f} {r['M_abs']:<10.2f} "
                  f"{r['n_sn']:<8d} {note}")

        geo_rms = lock_results.get(85.6, lock_results.get(85.0, {})).get('rms', 0)
        best_rms = lock_results[best_kj]['rms']
        if geo_rms > 0 and best_rms > 0:
            pct = 100.0 * (geo_rms - best_rms) / best_rms
            print(f"\n  Geometric K_J penalty vs best fit: {pct:+.1f}% RMS")
            if abs(pct) < 10:
                print("  -> Geometric lock is VIABLE (< 10% RMS penalty)")
            else:
                print("  -> Geometric lock has significant penalty")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  Golden Loop:  alpha = {ALPHA:.10f}")
    print(f"                beta  = {BETA:.10f}")
    print(f"  Geometric:    K_J   = xi * beta^(3/2) = {KJ_GEOMETRIC:.2f}")
    print()

    if kj_results:
        for band in ['g', 'r', 'i', 'z']:
            if band in kj_results:
                r = kj_results[band]
                print(f"  K_J({band}, {r['wavelength']:.0f}nm) = {r['kj']:.1f}")

    if spec_results:
        best_model = max(spec_results, key=lambda x: spec_results[x]['r2'])
        print(f"\n  Best spectral model: {best_model}")
        print(f"    (R^2 = {spec_results[best_model]['r2']:.4f})")

    print("\n  QFD interpretation:")
    print("    Redshift = vacuum drag (photon energy loss in static spacetime)")
    print("    K_J = vacuum refraction parameter (NOT recession velocity)")
    print("    Chromatic component = four-photon vertex scattering (sigma ~ E^2)")
    print("    Hubble Tension = chromatic dispersion of the vacuum")
    print("    No expansion. No dark energy. No Big Bang.")


if __name__ == '__main__':
    main()
