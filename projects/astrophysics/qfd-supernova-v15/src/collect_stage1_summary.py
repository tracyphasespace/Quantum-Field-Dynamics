import argparse
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from v15_data import LightcurveLoader # Assuming v15_data is available and has LightcurveLoader

def mu_from_alpha_raw(alpha: float) -> float:
    """
    Build a RAW distance-modulus proxy from the fitted amplitude alpha.
    In V15 the distance lever is alpha (multiplicative in flux, additive in log-flux).
    We convert to magnitudes but DO NOT set an absolute zero-point here.
    A single global c0 will be calibrated in Stage 3 on the CLEAN subset.
    μ_raw = -K * alpha,  where K = 2.5 / ln 10
    """
    K = 2.5 / np.log(10.0)
    return float(-K * alpha)

# Example, simple H0=70 reference (only a baseline; constant offsets cancel in residuals)
# Use your existing v15 fiducial function if available to be perfectly consistent.
def mu_fiducial_from_z(z, c_km_s=299792.458, H0=70.0):
    # Low-z Hubble law DM; good as a fiducial anchor
    # D_L ≈ (c/H0) z  (Mpc), μ = 5 log10(D_L / 10 pc)
    # 10 pc = 1e-5 Mpc  ⇒ μ = 5 log10(D_L) + 25
    D_L_mpc = (c_km_s / H0) * z
    return 5.0 * np.log10(max(D_L_mpc, 1e-6)) + 25.0

def estimate_color_slope(lc_rows):
    """Placeholder for color slope estimation."""
    # This is a placeholder. A proper implementation would involve fitting
    # a trend to multi-band residuals or colors.
    # For now, return a dummy value or a simple metric.
    if len(lc_rows) < 2:
        return 0.0
    # Example: simple difference between two bands if available
    # This needs actual bandpass information, which is not in lc_rows directly here.
    # For a robust solution, one would need to load the full lightcurve data
    # and perform a more sophisticated analysis.
    return np.random.uniform(-0.1, 0.1) # Dummy value

def residual_skew(lc_rows, bf_result):
    """Placeholder for residual skew estimation."""
    # This is a placeholder. A proper implementation would involve:
    # 1. Reconstructing the model lightcurve from bf_result and lc_rows.
    # 2. Calculating residuals (data - model).
    # 3. Computing the skewness of these residuals.
    if len(lc_rows) < 5: # Need enough points for meaningful skew
        return 0.0
    # Dummy skewness for now
    return np.random.uniform(-0.5, 0.5) # Dummy value

def load_stage1_result(sn_dir):
    """Loads Stage 1 results for a single SN."""
    bf_path = sn_dir / "persn_best.npy"
    metrics_path = sn_dir / "metrics.json"

    if not bf_path.exists() or not metrics_path.exists():
        return None

    bf = np.load(bf_path)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    result = {
        'persn_best': bf,
        'chi2': metrics.get('chi2', np.nan),
        'ndof': metrics.get('ndof', np.nan), # Assuming ndof is in metrics.json
        'L_peak': metrics.get('L_peak', np.nan) # kept for reference only
    }
    return result

def main():
    parser = argparse.ArgumentParser(description='Collect Stage 1 Summary')
    parser.add_argument('--stage1', required=True, help='Path to Stage 1 results directory')
    parser.add_argument('--lightcurves', required=True, help='Path to lightcurves CSV file')
    parser.add_argument('--out', required=True, help='Output path for _summary.npz')
    args = parser.parse_args()

    stage1_dir = Path(args.stage1)
    out_path = Path(args.out)

    print(f"Loading lightcurves from {args.lightcurves}...")
    loader = LightcurveLoader(Path(args.lightcurves))
    all_lcs = loader.load()
    lc_by_sn = {str(snid): lc_data for snid, lc_data in all_lcs.items()}

    print(f"Processing SNe from Stage 1 directory: {stage1_dir}")
    
    # Get all SNID directories that have persn_best.npy
    sn_dirs = [d for d in stage1_dir.iterdir() if d.is_dir() and (d / "persn_best.npy").exists()]
    sn_dirs.sort(key=lambda x: int(x.name)) # Sort by SNID

    snids_list = []
    zs = []
    chi2s = []
    ndofs = []
    color_slopes = []
    skew_resids = []
    timespans = []
    alphas = []
    mus_obs = []

    for sn_dir in tqdm(sn_dirs, desc="Collecting Stage 1 results"):
        snid = sn_dir.name
        lc_data = lc_by_sn.get(snid)
        if lc_data is None:
            print(f"Warning: Lightcurve data not found for SNID {snid}. Skipping.")
            continue

        result = load_stage1_result(sn_dir)
        if result is None:
            print(f"Warning: Could not load Stage 1 results for SNID {snid}. Skipping.")
            continue

        # Assuming persn_best is [t0, A_plasma, beta, alpha] (4 params)
        # alpha is at index 3 for 4-parameter model
        alpha = float(result['persn_best'][3])
        mu_raw = mu_from_alpha_raw(alpha)
        
        # Get z from lightcurve data
        z_obs = lc_data.z

        # Calculate ndof (number of observations - number of fitted parameters)
        # Assuming 4 fitted parameters (t0, A_plasma, beta, alpha)
        # and ndof is not directly in metrics.json
        num_observations = len(lc_data.mjd) # Assuming mjd is a good proxy for num_observations
        num_fitted_params = 4
        ndof_val = num_observations - num_fitted_params
        if ndof_val <= 0:
            ndof_val = 1 # Avoid division by zero or negative ndof

        # Populate lists
        snids_list.append(int(snid))
        zs.append(z_obs)
        chi2s.append(result['chi2'])
        ndofs.append(ndof_val)
        color_slopes.append(estimate_color_slope(lc_data)) # Placeholder
        skew_resids.append(residual_skew(lc_data, result['persn_best'])) # Placeholder
        timespans.append(lc_data.mjd.max() - lc_data.mjd.min())
        alphas.append(alpha)
        mus_obs.append(mu_raw)

    # Sanity checks for mu_obs
    mu_array = np.array(mus_obs)
    rng = mu_array.max() - mu_array.min()
    assert np.isfinite(mu_array).all(), "μ_obs contains NaN/inf"
    assert rng > 0.5, f"μ_obs range too small ({rng:.3f}); alpha→μ mapping likely broken"

    np.savez_compressed(
        out_path,
        snid=np.array(snids_list, dtype=np.int64),
        z=np.array(zs, dtype=np.float64),
        mu_obs=np.array(mus_obs, dtype=np.float64), # raw; Stage 3 will calibrate c0
        chi2=np.array(chi2s, dtype=np.float64),
        ndof=np.array(ndofs, dtype=np.float64),
        color_slope=np.array(color_slopes, dtype=np.float64),
        skew_resid=np.array(skew_resids, dtype=np.float64),
        timespan=np.array(timespans, dtype=np.float64),
        alpha=np.array(alphas, dtype=np.float64),
    )

    print(f"Successfully created summary file: {out_path}")
    print(f"  Total SNe processed: {len(snids_list)}")
    print(f"  μ_obs range: [{mu_array.min():.3f}, {mu_array.max():.3f}]")

if __name__ == '__main__':
    main()