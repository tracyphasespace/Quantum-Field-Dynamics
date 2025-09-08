#!/usr/bin/env python3
# stability_predictor.py
#
# Post-processing lifetime fit for QFD triplet runs.
# Model:  tau = k0 * G / (1 + k_csr * chi)
# where   G   = U^aU * (R/R_e)^aR * (I/I_e)^aI * (K_e/K)^aK
#         chi = |L * Q| / I
#
# Usage:
#   python stability_predictor.py \
#       --electron runs_vortex/tmp/electron \
#       --muon     runs_vortex/tmp/muon \
#       --tau      runs_vortex/tmp/tau \
#       --out      runs_vortex
#
# Optional knobs: --aU --aR --aI --aK --U0 --tau_mu --tau_tau --csv

import argparse, json, os, sys
from typing import Tuple, Dict

# ---------- I/O helpers ----------

def _load_results(dirpath: str) -> Dict:
    """Read results.json from a run directory."""
    path = os.path.join(dirpath, "results.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing results.json at: {path}")
    with open(path, "r") as f:
        res = json.load(f)
    parts = res.get("energy_parts", {}) or {}
    return {
        "dir": os.path.abspath(dirpath),
        "U": float(res.get("U", 0.99)),
        "R_eff": float(res.get("R_eff", 1.0)),
        "Q": float(res.get("charge_proxy", 0.0)),
        "L": float(res.get("spin_proxy", 0.0)),
        "I": float(res.get("inertia_proxy", 1.0)),
        # curvature/stress proxy from energy parts; small floor to avoid zeros
        "K": float(parts.get("E_curve", 1e-12)),
    }

# ---------- physics handles ----------

def _geom_index(feat: Dict, ref: Dict, aU: float, aR: float, aI: float, aK: float,
                U0: float, eps: float = 1e-12) -> float:
    """
    G = (U/U0)^aU * (R/R_ref)^aR * (I/I_ref)^aI * (K_ref/K)^aK
    Electron 'ref' anchors the radius/inertia scale; U0 anchors velocity.
    """
    Ufac = max(feat["U"], eps) / max(U0, eps)
    Rfac = max(feat["R_eff"], eps) / max(ref["R_eff"], eps)
    Ifac = max(feat["I"], eps) / max(ref["I"], eps)
    Kfac = max(ref["K"], eps) / max(feat["K"], eps)
    return (Ufac**aU) * (Rfac**aR) * (Ifac**aI) * (Kfac**aK)

def _csr_handle(feat: Dict, eps: float = 1e-12) -> float:
    """chi = |L*Q| / I  (charge–spin reinforcement handle)."""
    return abs(feat["L"] * feat["Q"]) / max(feat["I"], eps)

# ---------- two-point analytic fit ----------

def _solve_kcsr_and_k0(tau_mu: float, tau_tau: float,
                       G_mu: float, G_tau: float,
                       chi_mu: float, chi_tau: float,
                       eps: float = 1e-18) -> Tuple[float, float, bool]:
    """
    Solve for k_csr and k0 from:
      tau_mu  = k0 * G_mu  / (1 + k_csr * chi_mu)
      tau_tau = k0 * G_tau / (1 + k_csr * chi_tau)
    Returns (k_csr, k0, ok_flag).
    """
    # Derivation gives:  A = (tau_tau/tau_mu) * (G_mu/G_tau)
    # Then  k_csr = (1 - A) / (A*chi_tau - chi_mu)
    A = (tau_tau / tau_mu) * (G_mu / G_tau)
    denom = (A * chi_tau - chi_mu)
    if abs(denom) < eps:
        # Degenerate: fall back to k_csr=0, k0 by μ equation.
        kcsr = 0.0
        k0 = tau_mu / max(G_mu, eps)
        return kcsr, k0, False
    kcsr = (1.0 - A) / denom
    k0 = tau_mu * (1.0 + kcsr * chi_mu) / max(G_mu, eps)
    return kcsr, k0, True

def _predict_tau(k0: float, kcsr: float, G: float, chi: float, eps: float = 1e-12) -> float:
    return k0 * G / (1.0 + kcsr * chi + eps)

# ---------- formatting ----------

def _fmt_s(x: float) -> str:
    if x == 0.0:
        return "0"
    mag = abs(x)
    if 1e-3 <= mag <= 1e4:
        return f"{x:.6g}"
    return f"{x:.3e}"

def _print_row(label: str, meas: float, pred: float):
    delta = pred - meas
    print(f"{label:<6}  τ(meas)={_fmt_s(meas)} s   τ(pred)={_fmt_s(pred)} s   Δ={_fmt_s(delta)} s")

# ---------- main CLI ----------

def main():
    p = argparse.ArgumentParser(
        description="Fit k_csr and k0 from (μ, τ) lifetimes using geometric stability indices."
    )
    p.add_argument("--electron", required=True, help="Path to electron run dir (contains results.json)")
    p.add_argument("--muon",     required=True, help="Path to muon run dir (contains results.json)")
    p.add_argument("--tau",      required=True, help="Path to tau run dir (contains results.json)")
    p.add_argument("--out",      default=None,  help="Directory to write stability_summary.json (default: electron parent)")
    # physical anchors
    p.add_argument("--tau_mu",   type=float, default=2.196981e-6, help="Measured muon lifetime [s]")
    p.add_argument("--tau_tau",  type=float, default=2.903e-13,   help="Measured tau lifetime [s]")
    p.add_argument("--U0",       type=float, default=0.99,        help="Reference U anchor for G")
    # exponents for G
    p.add_argument("--aU", type=float, default=4.0, help="Exponent for U/U0 in G")
    p.add_argument("--aR", type=float, default=1.0, help="Exponent for R/R_e in G")
    p.add_argument("--aI", type=float, default=1.0, help="Exponent for I/I_e in G")
    p.add_argument("--aK", type=float, default=1.0, help="Exponent for K_e/K in G")
    p.add_argument("--csv", action="store_true", help="Also emit stability_summary.csv")
    args = p.parse_args()

    # Load features
    f_e = _load_results(args.electron)
    f_m = _load_results(args.muon)
    f_t = _load_results(args.tau)

    # Build indices (electron anchors R,I,K; U anchored by U0)
    G_e = _geom_index(f_e, f_e, args.aU, args.aR, args.aI, args.aK, args.U0)
    G_m = _geom_index(f_m, f_e, args.aU, args.aR, args.aI, args.aK, args.U0)
    G_t = _geom_index(f_t, f_e, args.aU, args.aR, args.aI, args.aK, args.U0)

    chi_e = _csr_handle(f_e)
    chi_m = _csr_handle(f_m)
    chi_t = _csr_handle(f_t)

    # Two-point analytic solve (μ, τ)
    kcsr, k0, ok = _solve_kcsr_and_k0(args.tau_mu, args.tau_tau, G_m, G_t, chi_m, chi_t)

    # Predictions
    tau_m_pred = _predict_tau(k0, kcsr, G_m, chi_m)
    tau_t_pred = _predict_tau(k0, kcsr, G_t, chi_t)
    tau_e_pred = _predict_tau(k0, kcsr, G_e, chi_e)  # model value; electron physically stable

    # Report
    print("\n=== Stability fit (two-point analytic) ===")
    print(f"Dirs:\n  e: {f_e['dir']}\n  μ: {f_m['dir']}\n  τ: {f_t['dir']}")
    print(f"Exponents: aU={args.aU}, aR={args.aR}, aI={args.aI}, aK={args.aK}   |   U0={args.U0}")
    print(f"G:   e={_fmt_s(G_e)}  μ={_fmt_s(G_m)}  τ={_fmt_s(G_t)}")
    print(f"chi: e={_fmt_s(chi_e)} μ={_fmt_s(chi_m)} τ={_fmt_s(chi_t)}")
    print(f"k_csr={_fmt_s(kcsr)}   k0={_fmt_s(k0)}   solved={ok}")
    _print_row("muon", args.tau_mu,  tau_m_pred)
    _print_row("tau",  args.tau_tau, tau_t_pred)
    print(f"(electron model τ)≈{_fmt_s(tau_e_pred)} s  [diagnostic only]\n")

    # Write artifacts
    out_dir = args.out or os.path.dirname(os.path.dirname(os.path.abspath(args.electron))) or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "inputs": {
            "electron_dir": f_e["dir"], "muon_dir": f_m["dir"], "tau_dir": f_t["dir"],
            "tau_mu_meas_s": args.tau_mu, "tau_tau_meas_s": args.tau_tau,
            "U0": args.U0, "exponents": {"aU": args.aU, "aR": args.aR, "aI": args.aI, "aK": args.aK}
        },
        "features": {"electron": f_e, "muon": f_m, "tau": f_t},
        "geom_indices": {"G_e": G_e, "G_mu": G_m, "G_tau": G_t},
        "csr_handles": {"chi_e": chi_e, "chi_mu": chi_m, "chi_tau": chi_t},
        "fit": {"k_csr": kcsr, "k0": k0, "two_point_solution": bool(ok)},
        "predicted_lifetimes_s": {
            "electron_model": tau_e_pred, "muon": tau_m_pred, "tau": tau_t_pred
        },
        "errors_s": {"muon": tau_m_pred - args.tau_mu, "tau": tau_t_pred - args.tau_tau}
    }
    with open(os.path.join(out_dir, "stability_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stability] wrote {os.path.join(out_dir, 'stability_summary.json')}")

    if args.csv:
        import csv
        csv_path = os.path.join(out_dir, "stability_summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["species", "U", "R_eff", "I", "Q", "L", "K", "G", "chi",
                        "tau_meas_s", "tau_pred_s", "tau_error_s"])
            w.writerow(["electron", f_e["U"], f_e["R_eff"], f_e["I"], f_e["Q"], f_e["L"], f_e["K"],
                        G_e, chi_e, "", tau_e_pred, ""])
            w.writerow(["muon",    f_m["U"], f_m["R_eff"], f_m["I"], f_m["Q"], f_m["L"], f_m["K"],
                        G_m, chi_m, args.tau_mu,  tau_m_pred, tau_m_pred - args.tau_mu])
            w.writerow(["tau",     f_t["U"], f_t["R_eff"], f_t["I"], f_t["Q"], f_t["L"], f_t["K"],
                        G_t, chi_t, args.tau_tau, tau_t_pred, tau_t_pred - args.tau_tau])
        print(f"[stability] wrote {csv_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[stability] ERROR: {e}", file=sys.stderr)
        sys.exit(1)