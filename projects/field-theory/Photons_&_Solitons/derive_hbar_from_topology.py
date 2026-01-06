#!/usr/bin/env python3
"""
QFD Photon Toroid Test: Helicity-locked scaling => E ∝ ω

This script constructs a localized toroidal vector potential A(x) (a "smoke-ring"),
computes B = curl A, and evaluates:

  Helicity: H = ∫ A·B dV
  Energy  : E = 1/2 ∫ |B|^2 dV   (magnetic sector proxy for |F|^2 scaling)

Then it enforces the "topological lock" by rescaling amplitude so that H = H_target
for each geometry scale. Under that constraint, the Lean theorem predicts:

  E ∝ k_eff      and with ω = c k_eff, we get E = ħ_eff ω

We estimate k_eff from the fields (RMS ratio):
  k_eff ≈ sqrt( ∫|B|^2 dV / ∫|A|^2 dV )

We also compute a second estimator:
  k_eff2 ≈ sqrt( ∫|curl B|^2 dV / ∫|B|^2 dV )

and optionally compare with a geometric proxy k_geom ≈ 1/a.

Usage:
  python derive_hbar_from_topology.py --N 128 --L 6.0 --R0 1.6 --a0 0.35 --twist 0.6 --c 1.0 \
      --scales 0.8 1.0 1.25 1.5 2.0 --Htarget 1.0 --plot

Notes:
- Domain is a cube [-L, L]^3 with N^3 samples.
- Boundary effects: choose L big enough so the torus field decays to ~0 at edges.
- This is not a PDE-solved soliton; it is an ansatz + constraint test of the scaling logic.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FieldStats:
    H: float
    E: float
    A2: float
    B2: float
    C2: float
    divB_rms: float
    B_rms: float
    k_eff: float
    omega: float
    hbar_eff: float
    k_eff2: float
    omega2: float
    hbar_eff2: float
    beltrami_resid: float
    kappa_opt: float
    beltrami_opt: float
    beltrami_corr: float


def make_grid(N: int, L: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Create a uniform grid on [-L, L]^3 with N points per axis.
    Returns X,Y,Z (shape N,N,N) and spacing dx.
    """
    x = np.linspace(-L, L, N, dtype=np.float64)
    dx = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return X, Y, Z, dx


def toroidal_frame(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, R0: float, eps: float = 1e-12):
    """
    Construct toroidal unit vectors in Cartesian coordinates:
      e_rho  : radial from z-axis
      e_phi  : azimuthal around z-axis
      e_theta: poloidal tangent around the minor circle (in plane spanned by e_rho and e_z)

    Also returns:
      rho = sqrt(x^2+y^2)
      s   = distance to ring centerline (minor radius coordinate): sqrt((rho-R0)^2+z^2)
    """
    rho = np.sqrt(X**2 + Y**2) + eps
    e_rho_x = X / rho
    e_rho_y = Y / rho
    e_rho_z = np.zeros_like(rho)

    e_phi_x = -Y / rho
    e_phi_y = X / rho
    e_phi_z = np.zeros_like(rho)

    # minor-circle distance to ring centerline
    s = np.sqrt((rho - R0) ** 2 + Z**2) + eps

    # poloidal tangent in the (e_rho, e_z) plane: perpendicular to the minor-radius direction
    # minor-radius direction is (rho-R0)*e_rho + z*e_z
    # tangent is (-z)*e_rho + (rho-R0)*e_z, normalized by s
    e_theta_x = (-Z / s) * e_rho_x
    e_theta_y = (-Z / s) * e_rho_y
    e_theta_z = (rho - R0) / s

    return (e_rho_x, e_rho_y, e_rho_z), (e_phi_x, e_phi_y, e_phi_z), (e_theta_x, e_theta_y, e_theta_z), rho, s


def tube_envelope(s: np.ndarray, a: float, kind: str) -> np.ndarray:
    """
    Radial envelope around the ring centerline distance s.

    - "gaussian": exp(-(s^2)/a^2) (infinite tail; can introduce mild scale drift in finite boxes)
    - "bump": compact support C^∞ bump:
        u = s/a
        g(u) = exp(1 - 1/(1-u^2)) for u<1 else 0
      normalized so g(0)=1 and g→0 smoothly as u→1-.
    """
    kind = kind.lower()
    if kind == "gaussian":
        return np.exp(-(s**2) / (a**2))
    if kind == "bump":
        u = s / max(a, 1e-300)
        g = np.zeros_like(u)
        mask = u < 1.0
        um = u[mask]
        g[mask] = np.exp(1.0 - 1.0 / (1.0 - um**2))
        return g
    raise ValueError(f"Unknown envelope kind: {kind!r} (use 'gaussian' or 'bump').")


def build_vector_potential(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    R0: float,
    a: float,
    amp: float,
    twist: float,
    envelope: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a localized toroidal vector potential:
      A = amp * g(s) * [ e_phi + twist * e_theta ]
    where g(s) is a Gaussian envelope around the ring centerline.

    - e_phi term gives toroidal circulation
    - e_theta term introduces linked/poloidal structure -> nonzero A·(curl A) helicity in general
    """
    (_, _, _), (ephi_x, ephi_y, ephi_z), (ethe_x, ethe_y, ethe_z), _, s = toroidal_frame(X, Y, Z, R0)

    g = tube_envelope(s, a, envelope)

    Ax = amp * g * (ephi_x + twist * ethe_x)
    Ay = amp * g * (ephi_y + twist * ethe_y)
    Az = amp * g * (ephi_z + twist * ethe_z)
    return Ax, Ay, Az

def curl(Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Curl on a uniform grid using np.gradient (2nd order central in interior).
    Axes correspond to x,y,z -> axis 0,1,2.
    """
    dFz_dy = np.gradient(Fz, dx, axis=1)
    dFy_dz = np.gradient(Fy, dx, axis=2)

    dFx_dz = np.gradient(Fx, dx, axis=2)
    dFz_dx = np.gradient(Fz, dx, axis=0)

    dFy_dx = np.gradient(Fy, dx, axis=0)
    dFx_dy = np.gradient(Fx, dx, axis=1)

    Cx = dFz_dy - dFy_dz
    Cy = dFx_dz - dFz_dx
    Cz = dFy_dx - dFx_dy
    return Cx, Cy, Cz

def divergence(Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray, dx: float) -> np.ndarray:
    dFx_dx = np.gradient(Fx, dx, axis=0)
    dFy_dy = np.gradient(Fy, dx, axis=1)
    dFz_dz = np.gradient(Fz, dx, axis=2)
    return dFx_dx + dFy_dy + dFz_dz

def integrate_scalar(f: np.ndarray, dx: float) -> float:
    return float(np.sum(f) * (dx**3))

def compute_stats(Ax, Ay, Az, Bx, By, Bz, dx: float, c: float) -> FieldStats:
    divB = divergence(Bx, By, Bz, dx)
    B2 = Bx**2 + By**2 + Bz**2
    A2 = Ax**2 + Ay**2 + Az**2

    # Second "spectral" estimator: curl B
    Cx, Cy, Cz = curl(Bx, By, Bz, dx)
    C2 = Cx**2 + Cy**2 + Cz**2

    H = integrate_scalar(Ax * Bx + Ay * By + Az * Bz, dx)
    E = 0.5 * integrate_scalar(B2, dx)  # magnetic energy proxy

    A2_int = integrate_scalar(A2, dx)
    B2_int = integrate_scalar(B2, dx)
    C2_int = integrate_scalar(C2, dx)

    BC_int = integrate_scalar(Bx*Cx + By*Cy + Bz*Cz, dx)

    # k_eff estimate: B ~ k A  => k ~ sqrt(∫|B|^2 / ∫|A|^2)
    k_eff = math.sqrt(B2_int / max(A2_int, 1e-300))

    omega = c * k_eff
    hbar_eff = E / max(omega, 1e-300)

    # k_eff2 estimate: curlB ~ k B => k ~ sqrt(∫|curlB|^2 / ∫|B|^2)
    k_eff2 = math.sqrt(C2_int / max(B2_int, 1e-300))
    omega2 = c * k_eff2
    hbar_eff2 = E / max(omega2, 1e-300)

    # Beltrami/force-free residual: ||curlB - k_eff2 * B|| / ||B||
    Rx = Cx - k_eff2 * Bx
    Ry = Cy - k_eff2 * By
    Rz = Cz - k_eff2 * Bz
    R2_int = integrate_scalar(Rx**2 + Ry**2 + Rz**2, dx)
    beltrami_resid = math.sqrt(R2_int / max(B2_int, 1e-300))

    # Optimal least-squares κ for curlB ≈ κ B
    kappa_opt = BC_int / max(B2_int, 1e-300)
    # minimized residual: ||curlB - κ_opt B||/||B||
    # = sqrt( <C,C>/<B,B> - κ_opt^2 )
    beltrami_opt = math.sqrt(max(C2_int / max(B2_int, 1e-300) - kappa_opt**2, 0.0))
    # correlation cos(angle) between B and curlB
    beltrami_corr = BC_int / max(math.sqrt(B2_int*C2_int), 1e-300)

    divB_rms = float(np.sqrt(np.mean(divB**2)))
    B_rms = float(np.sqrt(np.mean(B2)))

    return FieldStats(
        H=H,
        E=E,
        A2=A2_int,
        B2=B2_int,
        C2=C2_int,
        divB_rms=divB_rms,
        B_rms=B_rms,
        k_eff=k_eff,
        omega=omega,
        hbar_eff=hbar_eff,
        k_eff2=k_eff2,
        omega2=omega2,
        hbar_eff2=hbar_eff2,
        beltrami_resid=beltrami_resid,
        kappa_opt=kappa_opt,
        beltrami_opt=beltrami_opt,
        beltrami_corr=beltrami_corr,
    )

def rescale_to_target_helicity(Ax, Ay, Az, Bx, By, Bz, dx: float, H_target: float):
    """
    Since B = curl A is linear in A, both E and H scale as:
      A -> sA  =>  B -> sB
      H = ∫ A·B -> s^2 H
      E = 1/2 ∫ |B|^2 -> s^2 E
    Thus we can enforce H_target by multiplying A by sqrt(H_target/H).
    """
    H = integrate_scalar(Ax * Bx + Ay * By + Az * Bz, dx)
    if abs(H) < 1e-300:
        raise RuntimeError("Helicity is ~0; adjust twist or geometry (you need linked structure).")

    # Amplitude rescaling cannot flip helicity sign; require same sign.
    if H_target * H <= 0:
        raise RuntimeError(
            f"Helicity sign mismatch: measured H={H:.3e}, target H_target={H_target:.3e}. "
            "Flip the sign of 'twist' or the handedness of the ansatz to match."
        )
    scale = math.sqrt(H_target / H)
    Ax2, Ay2, Az2 = scale * Ax, scale * Ay, scale * Az
    Bx2, By2, Bz2 = scale * Bx, scale * By, scale * Bz
    return Ax2, Ay2, Az2, Bx2, By2, Bz2, scale

def run_sweep(args):
    X, Y, Z, dx = make_grid(args.N, args.L)
    results = []

    # For a stable numerical test, keep H_target sign consistent with the computed sign.
    H_target = float(args.Htarget)

    for s in args.scales:
        R = args.R0 * s
        a = args.a0 * s

        # Initial unscaled amplitude (1.0), then we will helicity-lock by rescaling.
        Ax, Ay, Az = build_vector_potential(X, Y, Z, R0=R, a=a, amp=1.0, twist=args.twist, envelope=args.envelope)
        Bx, By, Bz = curl(Ax, Ay, Az, dx)

        # helicity-lock
        Ax, Ay, Az, Bx, By, Bz, amp_scale = rescale_to_target_helicity(
            Ax, Ay, Az, Bx, By, Bz, dx, H_target
        )

        st = compute_stats(Ax, Ay, Az, Bx, By, Bz, dx, c=args.c)

        # geometric proxy
        k_geom = 1.0 / max(a, 1e-300)
        omega_geom = args.c * k_geom
        hbar_geom = st.E / max(omega_geom, 1e-300)

        results.append((s, R, a, amp_scale, st, k_geom, omega_geom, hbar_geom))

    return results, dx

def print_table(results):
    print("\nQFD Photon Toroid Test (helicity-locked scaling)")
    print("--------------------------------------------------------------------------")
    header = (
        "scale   R       a       A_scale   H           E           k_eff     "
        "E/(c*k_eff)   k_eff2    E/(c*k_eff2)  beltrami  kappa_opt  b_opt   corr    divB_rms/B_rms   k_geom   E/(c*k_geom)"
    )
    print(header)
    print("-" * len(header))

    for (s, R, a, A_scale, st, k_geom, omega_geom, hbar_geom) in results:
        div_rel = st.divB_rms / max(st.B_rms, 1e-300)
        print(
            f"{s:5.2f}  {R:6.3f}  {a:6.3f}  {A_scale:7.4f}  "
            f"{st.H:11.4e}  {st.E:11.4e}  {st.k_eff:8.4f}  "
            f"{st.hbar_eff:11.4e}  {st.k_eff2:8.4f}  {st.hbar_eff2:11.4e}  "
            f"{st.beltrami_resid:9.3e}  "
            f"{st.kappa_opt:8.4f}  {st.beltrami_opt:7.3f}  {st.beltrami_corr:6.3f}  "
            f"{div_rel:14.4e}  "
            f"{k_geom:7.3f}  {hbar_geom:12.4e}"
        )

    # summarize invariance
    hbars = np.array([st.hbar_eff for (_, _, _, _, st, _, _, _) in results], dtype=np.float64)
    hbars2 = np.array([st.hbar_eff2 for (_, _, _, _, st, _, _, _) in results], dtype=np.float64)
    print("\nSummary:")
    print(f"  ħ_eff (field k_eff): mean={hbars.mean():.6e}, std={hbars.std():.6e}, CV={hbars.std()/max(hbars.mean(),1e-300):.3e}")
    print(f"  ħ_eff (field k_eff2): mean={hbars2.mean():.6e}, std={hbars2.std():.6e}, CV={hbars2.std()/max(hbars2.mean(),1e-300):.3e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128, help="grid points per axis")
    ap.add_argument("--L", type=float, default=8.0, help="half-width of cubic domain [-L,L] (increase to reduce boundary drift)")
    ap.add_argument("--R0", type=float, default=1.6, help="base major radius")
    ap.add_argument("--a0", type=float, default=0.35, help="base minor radius")
    ap.add_argument("--twist", type=float, default=0.6, help="relative poloidal component in A (controls helicity)")
    ap.add_argument("--envelope", type=str, default="bump", choices=["gaussian", "bump"],
                    help="tube envelope profile; 'bump' has compact support to reduce finite-box drift")
    ap.add_argument("--c", type=float, default=1.0, help="vacuum wave speed used to compute omega=c*k")
    ap.add_argument("--Htarget", type=float, default=1.0, help="target helicity value to enforce (sign matters)")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.8, 1.0, 1.25, 1.5, 2.0], help="geometry scale factors")
    ap.add_argument("--plot", action="store_true", help="plot E vs k_eff and E/omega vs scale")
    args = ap.parse_args()

    results, dx = run_sweep(args)
    print_table(results)

    if args.plot:
        import os
        from datetime import datetime
        import matplotlib.pyplot as plt

        # --- Create results directory and timestamp ---
        results_dir = "results/derive_hbar_from_topology"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        scales = np.array([s for (s, *_rest) in results], dtype=np.float64)
        k_eff = np.array([st.k_eff for (_s, _R, _a, _A, st, _kg, _og, _hg) in results], dtype=np.float64)
        E = np.array([st.E for (_s, _R, _a, _A, st, _kg, _og, _hg) in results], dtype=np.float64)
        hbar = np.array([st.hbar_eff for (_s, _R, _a, _A, st, _kg, _og, _hg) in results], dtype=np.float64)
        k_eff2 = np.array([st.k_eff2 for (_s, _R, _a, _A, st, _kg, _og, _hg) in results], dtype=np.float64)
        hbar2 = np.array([st.hbar_eff2 for (_s, _R, _a, _A, st, _kg, _og, _hg) in results], dtype=np.float64)

        # --- Plot 1: E vs k_eff ---
        fig1 = plt.figure()
        plt.plot(k_eff, E, marker="o")
        plt.xlabel("k_eff (field estimate)")
        plt.ylabel("Energy E")
        plt.title("Helicity-locked scaling: E vs k_eff")
        plt.grid(True)
        fig1_path = os.path.join(results_dir, f"{timestamp}_E_vs_keff.png")
        plt.savefig(fig1_path)
        plt.close(fig1)
        print(f"Saved plot: {fig1_path}")

        # --- Plot 2: hbar_eff vs scale ---
        fig2 = plt.figure()
        plt.plot(scales, hbar, marker="o")
        plt.xlabel("geometry scale")
        plt.ylabel("E/omega (ħ_eff estimate)")
        plt.title("Invariance check: ħ_eff (k_eff) across scales")
        plt.grid(True)
        fig2_path = os.path.join(results_dir, f"{timestamp}_hbar_vs_scale.png")
        plt.savefig(fig2_path)
        plt.close(fig2)
        print(f"Saved plot: {fig2_path}")

        # --- Plot 3: hbar_eff2 vs scale ---
        fig3 = plt.figure()
        plt.plot(scales, hbar2, marker="o")
        plt.xlabel("geometry scale")
        plt.ylabel("E/(c*k_eff2) (ħ_eff2 estimate)")
        plt.title("Invariance check: ħ_eff2 (k_eff2) across scales")
        plt.grid(True)
        fig3_path = os.path.join(results_dir, f"{timestamp}_hbar2_vs_scale.png")
        plt.savefig(fig3_path)
        plt.close(fig3)
        print(f"Saved plot: {fig3_path}")


if __name__ == "__main__":
    main()
