#!/usr/bin/env python3
"""
v15_gate.py — Gate SNe into BASE/BBH/LENS using per_sn_metrics.csv
Outputs: flagged_bbh.txt, flagged_lens.txt
"""
import csv, argparse, math
from pathlib import Path
from typing import List, Tuple
import numpy as np

def read_metrics(path: Path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

# Simple 1D 2-component EM for GMM over kJ_star (ignores NaNs)
def gmm_1d_fit(x: np.ndarray, mu0=(70.0, 140.0), sigma0=(15.0, 20.0), iters: int = 25) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.array(mu0), np.array(sigma0), np.array([0.5,0.5])
    mu = np.array(mu0, dtype=float)
    sig = np.array(sigma0, dtype=float)
    pi = np.array([0.5, 0.5], dtype=float)
    for _ in range(iters):
        # E-step
        p = np.vstack([
            pi[k] * (1.0/(np.sqrt(2*np.pi)*sig[k])) * np.exp(-0.5*((x-mu[k])/sig[k])**2)
            for k in range(2)
        ])  # shape (2, N)
        denom = np.sum(p, axis=0) + 1e-12
        r = p / denom  # responsibilities

        Nk = np.sum(r, axis=1)
        # M-step
        mu = np.sum(r*x, axis=1)/Nk
        sig = np.sqrt(np.sum(r*(x-mu[:,None])**2, axis=1)/Nk + 1e-9)
        pi = Nk/np.sum(Nk)
    return mu, sig, pi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="per_sn_metrics.csv from v15_metrics.py")
    ap.add_argument("--out-dir", required=True, help="directory for flagged lists")
    ap.add_argument("--bbh_cut", type=float, default=110.0, help="hard boundary between modes")
    ap.add_argument("--chi2_bbh", type=float, default=1.6, help="chi2/ndof threshold for BBH cross-check")
    ap.add_argument("--alpha_q", type=float, default=0.10, help="alpha quantile for dimming check")
    ap.add_argument("--color_tau", type=float, default=0.08, help="abs(color_slope) threshold")
    ap.add_argument("--chi2_lens", type=float, default=2.0, help="chi2/ndof threshold for lens")
    ap.add_argument("--color_lens_lo", type=float, default=0.04)
    ap.add_argument("--color_lens_hi", type=float, default=0.10)
    ap.add_argument("--skew_lens", type=float, default=0.8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metrics(Path(args.metrics))
    # Collect arrays
    snids = np.array([r["snid"] for r in rows])
    def fget(key, default=np.nan):
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(key, default)))
            except Exception:
                vals.append(default)
        return np.array(vals, dtype=float)

    chi2_ndof = fget("chi2_ndof")
    kJ_star = fget("kJ_star")
    color_slope = fget("color_slope")
    alpha = fget("alpha")
    skew_resid = fget("skew_resid")

    # Fit GMM on kJ_star (finite only)
    finite_mask = np.isfinite(kJ_star)
    mu, sig, pi = gmm_1d_fit(kJ_star[finite_mask], mu0=(70.0,140.0), sigma0=(15.0,20.0), iters=30)

    # Responsibilities for all SNe (default to component 0 if NaN)
    def comp_pdf(x, m, s):
        return (1.0/(np.sqrt(2*np.pi)*s))*np.exp(-0.5*((x-m)/s)**2)
    p1 = comp_pdf(kJ_star, mu[0], sig[0])*pi[0]
    p2 = comp_pdf(kJ_star, mu[1], sig[1])*pi[1]
    r2 = p2 / (p1 + p2 + 1e-12)  # prob of high-k component

    # Alpha 10th percentile for dimming check
    alpha_q10 = np.nanpercentile(alpha[np.isfinite(alpha)], args.alpha_q*100.0) if np.any(np.isfinite(alpha)) else -1e9

    # BBH rule
    bbh_mask = (kJ_star > args.bbh_cut) & (r2 > 0.5) & (
        (chi2_ndof > args.chi2_bbh) | (alpha < alpha_q10) | (np.abs(color_slope) > args.color_tau)
    )

    # LENS rule (from remaining)
    lens_mask = (~bbh_mask) & (
        (chi2_ndof > args.chi2_lens) & (
            (np.abs(color_slope) >= args.color_lens_lo) & (np.abs(color_slope) <= args.color_lens_hi)
        ) | (skew_resid > args.skew_lens)
    )

    flagged_bbh = snids[bbh_mask].tolist()
    flagged_lens = snids[lens_mask].tolist()

    (out_dir / "flagged_bbh.txt").write_text("\n".join(flagged_bbh) + ("\n" if flagged_bbh else ""))
    (out_dir / "flagged_lens.txt").write_text("\n".join(flagged_lens) + ("\n" if flagged_lens else ""))

    # Also write a small JSON summary of the GMM fit
    gmm_info = {
        "mu": mu.tolist(),
        "sigma": sig.tolist(),
        "pi": pi.tolist(),
        "bbh_cut": args.bbh_cut,
        "counts": {"bbh": int(len(flagged_bbh)), "lens": int(len(flagged_lens)), "total": int(len(snids))},
    }
    (out_dir / "gmm_summary.json").write_text(json.dumps(gmm_info, indent=2))

if __name__ == "__main__":
    main()
