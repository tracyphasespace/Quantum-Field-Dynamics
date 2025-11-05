#!/usr/bin/env python3
"""
v15_metrics.py — Build per-SN metrics for V15 gating.
Outputs: results/v15/per_sn_metrics.csv
"""
import os, sys, csv, json, math, argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np

# Optional project modules (if present in PYTHONPATH)
try:
    import v15_data
except Exception:
    v15_data = None
try:
    import v15_model
except Exception:
    v15_model = None

def read_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)

def second_derivative_three_point(x: np.ndarray, y: np.ndarray, idx: int) -> float:
    if idx <= 0 or idx >= len(x)-1:
        return float("nan")
    # quadratic fit to (idx-1, idx, idx+1)
    X = np.array([[1, x[idx-1], x[idx-1]**2],
                  [1, x[idx],   x[idx]**2],
                  [1, x[idx+1], x[idx+1]**2]], dtype=float)
    coef = np.linalg.lstsq(X, y[idx-1:idx+1+1], rcond=None)[0]  # a0 + a1 x + a2 x^2
    return float(2*coef[2])

def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return 0.0, 0.0
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)

def skewness(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mu = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    m3 = np.mean((x - mu)**3)
    g1 = m3 / (s**3 + 1e-12)
    return float((np.sqrt(n*(n-1))/(n-2)) * g1)

def load_stage1_variant(stage1_dir: Path, snid: str, variant: str) -> Dict[str, Any]:
    p = stage1_dir / snid / f"persn_{variant}.json"
    if p.exists():
        return read_json(p)
    return {}

def load_stage1_metrics(stage1_dir: Path, snid: str) -> Dict[str, Any]:
    p = stage1_dir / snid / "metrics.json"
    if p.exists():
        return read_json(p)
    return {}

def load_photometry(csv_path: Path, snid: str):
    # Prefer project loader if available
    if v15_data is not None and hasattr(v15_data, "load_lightcurves_numpy"):
        try:
            lcs = v15_data.load_lightcurves_numpy(str(csv_path), sn_filter={snid})
            return lcs.get(snid, None)
        except Exception:
            pass
    # Fallback: parse CSV
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("snid") == snid:
                try:
                    mjd = float(row.get("mjd") or row.get("MJD") or row.get("time") or 0.0)
                    flux = float(row.get("flux_nu_jy") or row.get("flux") or 0.0)
                    lam  = float(row.get("wavelength_A") or row.get("lambda_A") or 5000.0)
                    err  = float(row.get("flux_err") or row.get("flux_nu_jy_err") or 1e-6)
                except Exception:
                    continue
                rows.append([mjd, flux, lam, err])
    if not rows:
        return None
    return np.array(rows, dtype=np.float64)

def logL_single_sn(global_params, persn_params, phot, z_obs):
    # Use model if available
    if v15_model is not None and hasattr(v15_model, "log_likelihood_single_sn_numpy"):
        try:
            return float(v15_model.log_likelihood_single_sn_numpy(global_params, persn_params, phot, z_obs))
        except Exception:
            pass
    # Fallback surrogate likelihood (achromatic power law vs wavelength)
    mjd = phot[:,0]; flux = phot[:,1]; lam = phot[:,2]; err = np.maximum(phot[:,3], 1e-12)
    alpha = float(persn_params.get("alpha", 0.0))
    beta  = float(persn_params.get("beta", 0.0))
    l0 = 5000.0
    model = np.exp(alpha) * (np.maximum(lam,1.0)/l0)**beta
    chi2 = np.sum(((flux - model)/err)**2)
    return float(-0.5*chi2 - 1e-6*(alpha**2 + beta**2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True, help="Stage-1 directory with per-SN JSONs")
    ap.add_argument("--lightcurves", required=True, help="Unified CSV (min3)")
    ap.add_argument("--globals", default="70,0.01,30", help="kJ,eta_prime,xi for kJ scan (comma)")
    ap.add_argument("--out", required=True, help="Output CSV: per_sn_metrics.csv")
    ap.add_argument("--kmin", type=float, default=20.0)
    ap.add_argument("--kmax", type=float, default=200.0)
    ap.add_argument("--kstep", type=float, default=1.5)
    args = ap.parse_args()

    stage1 = Path(args.stage1)
    csv_path = Path(args.lightcurves)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        kJ_fix, eta_fix, xi_fix = [float(x) for x in args.globals.split(",")]
    except Exception:
        print("ERROR: --globals must be like '70,0.01,30'", file=sys.stderr)
        sys.exit(2)

    snids = [p.name for p in stage1.iterdir() if p.is_dir()]
    grid_k = np.arange(args.kmin, args.kmax + 1e-12, args.kstep)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snid","chi2_ndof","kJ_star","kJ_curv","color_slope","alpha","skew_resid","nobs"])
        for snid in snids:
            base = load_stage1_variant(stage1, snid, "base")
            if not base:
                # fall back to bbh/lens if present
                base = load_stage1_variant(stage1, snid, "bbh") or load_stage1_variant(stage1, snid, "lens")
                if not base:
                    continue

            metrics = load_stage1_metrics(stage1, snid)
            chi2 = metrics.get("chi2", np.nan)
            ndof = metrics.get("ndof", max(metrics.get("nobs", 1)-len(base), 1))
            chi2_ndof = float(chi2/ndof) if (isinstance(ndof,(int,float)) and ndof>0 and math.isfinite(chi2)) else np.nan
            alpha = float(base.get("alpha", 0.0))
            z_obs = float(base.get("z_obs", 0.0))

            phot = load_photometry(csv_path, snid)
            if phot is None or len(phot) < 6:
                w.writerow([snid, f"{chi2_ndof:.6g}", "", "", "", f"{alpha:.6g}", "", 0 if phot is None else len(phot)])
                continue

            # k_J scan
            Lvals = np.empty_like(grid_k)
            for i, kJ in enumerate(grid_k):
                Lvals[i] = logL_single_sn((kJ, eta_fix, xi_fix), base, phot, z_obs)
            imax = int(np.argmax(Lvals))
            kJ_star = float(grid_k[imax])
            kJ_curv = second_derivative_three_point(grid_k, Lvals, imax)

            # residual diagnostics (surrogate model)
            mjd = phot[:,0]; flux = phot[:,1]; lam = phot[:,2]; err = np.maximum(phot[:,3], 1e-12)
            beta = float(base.get("beta", 0.0))
            l0 = 5000.0
            model = np.exp(alpha) * (np.maximum(lam,1.0)/l0)**beta
            resid = (flux - model)/err
            color_slope, _ = linear_regression(np.log(np.maximum(lam,1.0)), resid)
            skew_resid = skewness(resid)

            w.writerow([snid, f"{chi2_ndof:.6g}", f"{kJ_star:.3f}", f"{kJ_curv:.6g}", f"{color_slope:.6g}", f"{alpha:.6g}", f"{skew_resid:.6g}", len(phot)])

if __name__ == "__main__":
    main()
