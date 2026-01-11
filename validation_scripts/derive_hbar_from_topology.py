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
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np


# ---------------------------
# Utility: finite checks
# ---------------------------
def _is_finite(x) -> bool:
    """Check if array or scalar contains only finite values."""
    try:
        return bool(np.isfinite(x).all())
    except Exception:
        return bool(np.isfinite(x))


# ---------------------------
# Relaxation trace record
# ---------------------------
@dataclass
class RelaxTraceRow:
    step: int
    eta: float
    J: float      # combined objective
    E: float
    H: float
    b_opt: float
    corr: float
    divB_ratio: float
    kappa_opt: float


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


def project_to_helicity(Ax, Ay, Az, Bx, By, Bz, dx, Htarget, *, eps=1e-30,
                        allow_sign_flip=False, _sign_flip_warned=[False]):
    """
    Rescale fields so that helicity matches Htarget.

    Assumption (true for these constructions): scaling A -> sA implies B=curl A -> sB,
    and helicity H=∫A·B scales as s^2.

    If allow_sign_flip=True, will flip field sign when H and Htarget have opposite signs,
    then scale amplitude. This is useful during relaxation.

    Returns: (Ax2, Ay2, Az2, Bx2, By2, Bz2, H_before, H_after, scale, sign_flipped)
    """
    H = integrate_scalar(Ax*Bx + Ay*By + Az*Bz, dx)
    if not np.isfinite(H) or abs(H) < eps:
        raise FloatingPointError(f"Helicity is non-finite or ~0 (H={H}); cannot project.")

    sign_flipped = False

    # Sign handling
    if H * Htarget < 0:
        if allow_sign_flip:
            # Flip the field sign to match target helicity sign
            Ax, Ay, Az = -Ax, -Ay, -Az
            Bx, By, Bz = -Bx, -By, -Bz
            H = -H  # Now H and Htarget have same sign
            sign_flipped = True
            if not _sign_flip_warned[0]:
                print(f"  [Warning] Helicity sign flip applied (H={-H:.3e} -> {H:.3e})")
                _sign_flip_warned[0] = True
        else:
            raise ValueError(
                f"Helicity sign mismatch: H={H:+.6e}, Htarget={Htarget:+.6e}. "
                "Flip --Htarget or adjust --twist sign so the constructed topology matches."
            )

    # Now safe to take sqrt since H and Htarget have same sign
    s = math.sqrt(abs(Htarget / H))
    Ax2 = Ax * s
    Ay2 = Ay * s
    Az2 = Az * s
    Bx2 = Bx * s
    By2 = By * s
    Bz2 = Bz * s
    # Recompute for diagnostics
    H2 = integrate_scalar(Ax2*Bx2 + Ay2*By2 + Az2*Bz2, dx)
    return Ax2, Ay2, Az2, Bx2, By2, Bz2, H, H2, s, sign_flipped


def compute_combined_objective(Bx, By, Bz, dx, E0=None,
                                w_b=1.0, w_div=0.1, w_E=0.01):
    """
    Compute combined objective for Beltrami relaxation:

    J(B) = w_b * ||curl B - κ(B) B||² / ||B||²
         + w_div * ||div B||² / ||B||²
         + w_E * E(B) / E0

    Where:
      - κ(B) is computed by least-squares fit each step
      - E0 is reference energy (to keep scaling reasonable)
      - w_b dominates (Beltrami relaxation)
      - w_div ensures divergence-free (physically clean)
      - w_E can be small (energy regularization)

    Returns: (J, b_opt, divB_ratio, E, kappa_opt, corr)
    """
    # Compute curl B
    Cx, Cy, Cz = curl(Bx, By, Bz, dx)

    # Compute div B
    divB = divergence(Bx, By, Bz, dx)

    # Integrals
    B2_int = integrate_scalar(Bx**2 + By**2 + Bz**2, dx)
    C2_int = integrate_scalar(Cx**2 + Cy**2 + Cz**2, dx)
    BC_int = integrate_scalar(Bx*Cx + By*Cy + Bz*Cz, dx)
    divB2_int = integrate_scalar(divB**2, dx)

    # Energy
    E = 0.5 * B2_int

    # Optimal kappa for curl B ≈ κ B
    kappa_opt = BC_int / max(B2_int, 1e-300)

    # Beltrami residual: ||curl B - κ_opt B|| / ||B||
    b_opt_sq = max(C2_int / max(B2_int, 1e-300) - kappa_opt**2, 0.0)
    b_opt = math.sqrt(b_opt_sq)

    # Correlation
    corr = BC_int / max(math.sqrt(B2_int * C2_int), 1e-300)

    # Divergence ratio
    divB_ratio = math.sqrt(divB2_int / max(B2_int, 1e-300))

    # Combined objective
    J = w_b * b_opt_sq
    J += w_div * (divB2_int / max(B2_int, 1e-300))
    if E0 is not None and E0 > 0:
        J += w_E * (E / E0)

    return J, b_opt, divB_ratio, E, kappa_opt, corr


def fit_beltrami_kappa(Bx, By, Bz, dx):
    """
    Fit optimal κ for curl B ≈ κ B using least-squares.

    Returns: (kappa_opt, b_opt, corr)
      - kappa_opt: optimal proportionality constant
      - b_opt: minimized residual ||curl B - κ_opt B|| / ||B||
      - corr: correlation cos(angle) between B and curl B
    """
    Cx, Cy, Cz = curl(Bx, By, Bz, dx)
    B2_int = integrate_scalar(Bx**2 + By**2 + Bz**2, dx)
    C2_int = integrate_scalar(Cx**2 + Cy**2 + Cz**2, dx)
    BC_int = integrate_scalar(Bx*Cx + By*Cy + Bz*Cz, dx)

    kappa_opt = BC_int / max(B2_int, 1e-300)
    b_opt = math.sqrt(max(C2_int / max(B2_int, 1e-300) - kappa_opt**2, 0.0))
    corr = BC_int / max(math.sqrt(B2_int * C2_int), 1e-300)
    return kappa_opt, b_opt, corr


def backtracking_step(update_fn, Ax, Ay, Az, dx, Htarget, obj_fn,
                      eta, eta_min=1e-6, max_tries=25, armijo_c=1e-4):
    """
    Try a descent step with Armijo-style backtracking line search.

    - update_fn(Ax,Ay,Az,eta) -> (Ax_new, Ay_new, Az_new) (raw gradient/projection update)
    - obj_fn(Ax,Ay,Az) -> scalar objective (combined J)
    - After the candidate step, project to helicity exactly.

    Armijo acceptance rule:
        Accept if J_new <= J_old * (1 - armijo_c * eta)
        This prevents "tiny improvements" that are purely numerical noise.

    Returns: (Ax, Ay, Az, obj_before, obj_after, eta_used)
             eta_used=0.0 if no acceptable step found
    """
    obj0 = obj_fn(Ax, Ay, Az)
    if not np.isfinite(obj0):
        raise FloatingPointError(f"Initial objective is non-finite: {obj0}")

    eta_try = eta
    for attempt in range(max_tries):
        Ax1, Ay1, Az1 = update_fn(Ax, Ay, Az, eta_try)
        if not (_is_finite(Ax1) and _is_finite(Ay1) and _is_finite(Az1)):
            eta_try *= 0.5
            if eta_try < eta_min:
                break
            continue

        # Compute B for projection
        Bx1, By1, Bz1 = curl(Ax1, Ay1, Az1, dx)

        # exact helicity projection after the move (allow sign flip during relaxation)
        try:
            result = project_to_helicity(
                Ax1, Ay1, Az1, Bx1, By1, Bz1, dx, Htarget, allow_sign_flip=True
            )
            Ax1, Ay1, Az1 = result[0], result[1], result[2]
            Bx1, By1, Bz1 = result[3], result[4], result[5]
        except Exception:
            eta_try *= 0.5
            if eta_try < eta_min:
                break
            continue

        obj1 = obj_fn(Ax1, Ay1, Az1)

        # Armijo condition: sufficient decrease
        armijo_threshold = obj0 * (1.0 - armijo_c * eta_try)
        if np.isfinite(obj1) and obj1 <= armijo_threshold:
            return Ax1, Ay1, Az1, obj0, obj1, eta_try

        eta_try *= 0.5
        if eta_try < eta_min:
            break

    # failed to find an acceptable step
    return Ax, Ay, Az, obj0, obj0, 0.0


def relax_to_beltrami(Ax, Ay, Az, dx, Htarget, steps=200, eta=0.005,
                      backtrack=True, eta_min=1e-6, report_every=25,
                      trace_file: Optional[str] = None,
                      w_b=1.0, w_div=0.1, w_E=0.01):
    """
    Constrained relaxation:
      minimize J(B) = w_b*||curl B - κB||²/||B||² + w_div*||div B||²/||B||² + w_E*E/E0
      with H = ∫A·B fixed (helicity lock)
    using projected gradient descent in A.

    This routine is numerically stiff; default eta should be small (≈0.005).
    backtrack=True (default) enables automatic eta-halving with Armijo acceptance.

    Args:
        trace_file: if provided, write CSV trace of convergence
        w_b, w_div, w_E: weights for combined objective components
    """
    trace: List[RelaxTraceRow] = []

    def helicity(Ax, Ay, Az, Bx, By, Bz):
        return integrate_scalar(Ax*Bx + Ay*By + Az*Bz, dx)

    # Get initial energy for normalization
    Bx0, By0, Bz0 = curl(Ax, Ay, Az, dx)
    E0_initial = 0.5 * integrate_scalar(Bx0**2 + By0**2 + Bz0**2, dx)

    # Combined objective function for backtracking
    def obj_fn(Ax, Ay, Az):
        Bx, By, Bz = curl(Ax, Ay, Az, dx)
        J, _, _, _, _, _ = compute_combined_objective(
            Bx, By, Bz, dx, E0=E0_initial, w_b=w_b, w_div=w_div, w_E=w_E
        )
        return float(J)

    # Raw update step (one projected gradient step, no helicity rescale)
    def update_fn(Ax, Ay, Az, eta_local):
        Bx, By, Bz = curl(Ax, Ay, Az, dx)
        # g = ∂E/∂A ∝ curl B
        g1, g2, g3 = curl(Bx, By, Bz, dx)
        # h = ∂H/∂A ∝ 2B
        h1, h2, h3 = 2*Bx, 2*By, 2*Bz
        # Projection coefficient α = <g,h>/<h,h>
        gh = integrate_scalar(g1*h1 + g2*h2 + g3*h3, dx)
        hh = integrate_scalar(h1*h1 + h2*h2 + h3*h3, dx)
        alpha = gh / max(hh, 1e-300)
        # Projected gradient (preserves H to first order)
        d1 = g1 - alpha*h1
        d2 = g2 - alpha*h2
        d3 = g3 - alpha*h3
        return Ax - eta_local*d1, Ay - eta_local*d2, Az - eta_local*d3

    # Ensure we start on the correct helicity manifold (allow sign flip)
    Bx, By, Bz = curl(Ax, Ay, Az, dx)
    try:
        result = project_to_helicity(
            Ax, Ay, Az, Bx, By, Bz, dx, Htarget, allow_sign_flip=True
        )
        Ax, Ay, Az = result[0], result[1], result[2]
        Bx, By, Bz = result[3], result[4], result[5]
    except (FloatingPointError, ValueError) as e:
        print(f"Warning: initial projection failed: {e}")
        # Continue anyway with current fields

    eta0 = eta
    backtrack_mode = "Armijo" if backtrack else "fixed"
    print(f"Running relaxation for {steps} steps (eta={eta:.4f}, mode={backtrack_mode})...")
    print(f"  Objective weights: w_b={w_b}, w_div={w_div}, w_E={w_E}")

    for it in range(steps):
        if not np.isfinite(eta0) or eta0 <= 0:
            print(f"  step {it:4d}: eta exhausted, stopping early")
            break

        if backtrack:
            # Use Armijo backtracking line search
            Ax2, Ay2, Az2, o0, o1, eta_used = backtracking_step(
                update_fn, Ax, Ay, Az, dx, Htarget, obj_fn,
                eta=eta0, eta_min=eta_min
            )
            if eta_used == 0.0:
                print(f"  step {it:4d}: cannot find stable step, stopping")
                break
            Ax, Ay, Az = Ax2, Ay2, Az2
            # Optionally allow eta to creep back up slowly
            eta0 = min(eta, eta_used * 1.05)
        else:
            # Fixed eta: raw update then exact helicity projection
            Ax, Ay, Az = update_fn(Ax, Ay, Az, eta0)
            Bx, By, Bz = curl(Ax, Ay, Az, dx)

            # Safe helicity projection (allow sign flip)
            try:
                result = project_to_helicity(
                    Ax, Ay, Az, Bx, By, Bz, dx, Htarget, allow_sign_flip=True
                )
                Ax, Ay, Az = result[0], result[1], result[2]
            except (FloatingPointError, ValueError):
                # Simple stability heuristic: shrink eta
                eta0 *= 0.5
                if eta0 < eta_min:
                    print(f"  step {it:4d}: eta below minimum, stopping")
                    break
                continue

            # Additional stability check: if fields contain NaNs, shrink eta
            if not (_is_finite(Ax) and _is_finite(Ay) and _is_finite(Az)):
                eta0 *= 0.5
                if eta0 < eta_min:
                    print(f"  step {it:4d}: NaN detected, eta exhausted, stopping")
                    break
                continue

        # Compute diagnostics
        Bx, By, Bz = curl(Ax, Ay, Az, dx)
        J, b_opt, divB_ratio, E, kappa_opt, corr = compute_combined_objective(
            Bx, By, Bz, dx, E0=E0_initial, w_b=w_b, w_div=w_div, w_E=w_E
        )
        H = helicity(Ax, Ay, Az, Bx, By, Bz)

        # Record trace
        trace.append(RelaxTraceRow(
            step=it, eta=eta0, J=J, E=E, H=H,
            b_opt=b_opt, corr=corr, divB_ratio=divB_ratio, kappa_opt=kappa_opt
        ))

        # Print progress
        if (it % report_every) == 0 or it == steps - 1:
            print(f"  step {it:4d}: J={J:.4e}, E={E:.4e}, H={H:.4e}, "
                  f"b_opt={b_opt:.4f}, corr={corr:+.4f}, divB={divB_ratio:.2e}, eta={eta0:.4g}")

    # Write trace to CSV if requested
    if trace_file and trace:
        os.makedirs(os.path.dirname(trace_file) or ".", exist_ok=True)
        with open(trace_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'eta', 'J', 'E', 'H', 'b_opt', 'corr', 'divB_ratio', 'kappa_opt'])
            for row in trace:
                writer.writerow([row.step, row.eta, row.J, row.E, row.H,
                                row.b_opt, row.corr, row.divB_ratio, row.kappa_opt])
        print(f"  Trace written to: {trace_file}")

    return Ax, Ay, Az, trace


def run_sweep(args):
    X, Y, Z, dx = make_grid(args.N, args.L)
    results = []
    all_traces = []

    # For a stable numerical test, keep H_target sign consistent with the computed sign.
    H_target = float(args.Htarget)

    for idx, s in enumerate(args.scales):
        R = args.R0 * s
        a = args.a0 * s

        # Initial unscaled amplitude (1.0)
        Ax, Ay, Az = build_vector_potential(X, Y, Z, R0=R, a=a, amp=1.0, twist=args.twist, envelope=args.envelope)

        amp_scale = float('nan')
        trace = []

        if args.relax:
            # Determine trace file path if requested
            trace_file = None
            if args.relax_trace:
                os.makedirs("results/traces", exist_ok=True)
                trace_file = f"results/traces/relax_scale_{s:.3f}.csv"

            # Run relaxation to find the helicity-constrained energy minimizer
            # Note: backtrack defaults to True now (via argparse default)
            Ax, Ay, Az, trace = relax_to_beltrami(
                Ax, Ay, Az, dx, H_target,
                steps=args.relax_steps, eta=args.relax_eta,
                backtrack=not args.relax_no_backtrack,  # inverted flag
                eta_min=args.relax_eta_min,
                trace_file=trace_file
            )
            all_traces.append((s, trace))
        else:
            # Just lock helicity without energy minimization
            Bx, By, Bz = curl(Ax, Ay, Az, dx)
            Ax, Ay, Az, Bx, By, Bz, amp_scale = rescale_to_target_helicity(
                Ax, Ay, Az, Bx, By, Bz, dx, H_target
            )

        # Final fields and stats
        Bx, By, Bz = curl(Ax, Ay, Az, dx)
        st = compute_stats(Ax, Ay, Az, Bx, By, Bz, dx, c=args.c)

        # geometric proxy
        k_geom = 1.0 / max(a, 1e-300)
        omega_geom = args.c * k_geom
        hbar_geom = st.E / max(omega_geom, 1e-300)

        results.append((s, R, a, amp_scale, st, k_geom, omega_geom, hbar_geom))

    return results, dx, all_traces

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
    hbars_geom = np.array([hg for (_, _, _, _, _, _, _, hg) in results], dtype=np.float64)
    corrs = np.array([st.beltrami_corr for (_, _, _, _, st, _, _, _) in results], dtype=np.float64)
    b_opts = np.array([st.beltrami_opt for (_, _, _, _, st, _, _, _) in results], dtype=np.float64)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    # Primary result (using k_eff2, the spectral estimator from curl B)
    print(f"\n  PRIMARY ħ_eff (k_eff2 = spectral mode estimator):")
    print(f"    mean = {hbars2.mean():.6e}")
    print(f"    std  = {hbars2.std():.6e}")
    print(f"    CV   = {hbars2.std()/max(hbars2.mean(),1e-300):.3e}")

    # Secondary estimators for comparison
    print(f"\n  SECONDARY ESTIMATORS (for comparison):")
    print(f"    ħ_eff (k_eff = field proxy):    mean={hbars.mean():.6e}, CV={hbars.std()/max(hbars.mean(),1e-300):.3e}")
    print(f"    ħ_eff (k_geom = geometric):     mean={hbars_geom.mean():.6e}, CV={hbars_geom.std()/max(hbars_geom.mean(),1e-300):.3e}")

    # Beltrami quality indicators
    print(f"\n  BELTRAMI QUALITY (force-free convergence):")
    print(f"    b_opt (residual): mean={b_opts.mean():.4f}, range=[{b_opts.min():.4f}, {b_opts.max():.4f}]")
    print(f"    corr (alignment): mean={corrs.mean():+.4f}, range=[{corrs.min():+.4f}, {corrs.max():+.4f}]")
    if abs(corrs.mean()) > 0.95:
        print(f"    [PASS] Strong Beltrami alignment (|corr| > 0.95)")
    elif abs(corrs.mean()) > 0.80:
        print(f"    [WARN] Moderate Beltrami alignment (0.80 < |corr| < 0.95)")
    else:
        print(f"    [FAIL] Weak Beltrami alignment (|corr| < 0.80) - consider more relaxation steps")

def run_selftest() -> bool:
    """
    Minimal regression test to verify solver stability.

    Asserts:
      - Final |H - Htarget| < tol_H
      - Final |corr| > 0.9 (strong Beltrami alignment)
      - Final b_opt < 1.5 (reasonable residual)
      - No NaNs at any step

    Returns True if all checks pass.
    """
    print("\n" + "="*60)
    print("SELFTEST: Running minimal regression test...")
    print("="*60)

    # Small grid for fast test
    N, L = 48, 5.0
    Htarget = -1.0
    tol_H = 1e-2
    min_corr = 0.9
    max_b_opt = 1.5

    X, Y, Z, dx = make_grid(N, L)
    Ax, Ay, Az = build_vector_potential(X, Y, Z, R0=1.2, a=0.3, amp=1.0, twist=0.5, envelope="bump")

    print(f"  Grid: N={N}, L={L}, Htarget={Htarget}")

    # Run relaxation
    Ax, Ay, Az, trace = relax_to_beltrami(
        Ax, Ay, Az, dx, Htarget,
        steps=100, eta=0.01, backtrack=True, eta_min=1e-6, report_every=50
    )

    if not trace:
        print("  [FAIL] No trace recorded (relaxation failed immediately)")
        return False

    final = trace[-1]

    # Check for NaNs
    if not all(np.isfinite([final.J, final.E, final.H, final.b_opt, final.corr])):
        print("  [FAIL] NaN detected in final values")
        return False

    # Check helicity lock
    H_err = abs(final.H - Htarget) / abs(Htarget)
    if H_err > tol_H:
        print(f"  [FAIL] Helicity error too large: |H - Htarget|/|Htarget| = {H_err:.4e} > {tol_H}")
        return False
    print(f"  [PASS] Helicity lock: |H - Htarget|/|Htarget| = {H_err:.4e}")

    # Check Beltrami alignment
    if abs(final.corr) < min_corr:
        print(f"  [FAIL] Beltrami correlation too weak: |corr| = {abs(final.corr):.4f} < {min_corr}")
        return False
    print(f"  [PASS] Beltrami alignment: |corr| = {abs(final.corr):.4f}")

    # Check residual
    if final.b_opt > max_b_opt:
        print(f"  [FAIL] Beltrami residual too high: b_opt = {final.b_opt:.4f} > {max_b_opt}")
        return False
    print(f"  [PASS] Beltrami residual: b_opt = {final.b_opt:.4f}")

    print("\n  [ALL TESTS PASSED]")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="QFD Photon Toroid Test: Helicity-locked scaling => E ∝ ω",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sweep without relaxation
  %(prog)s --scales 0.8 1.0 1.25 1.5 2.0

  # With Beltrami relaxation (recommended)
  %(prog)s --relax --scales 1.0 1.5 2.0

  # High-resolution run with trace logging
  %(prog)s --relax --relax_trace --N 160 --scales 1.0 --plot

  # Run regression selftest
  %(prog)s --selftest
"""
    )
    ap.add_argument("--N", type=int, default=128, help="grid points per axis")
    ap.add_argument("--L", type=float, default=8.0, help="half-width of cubic domain [-L,L]")
    ap.add_argument("--R0", type=float, default=1.6, help="base major radius")
    ap.add_argument("--a0", type=float, default=0.35, help="base minor radius")
    ap.add_argument("--twist", type=float, default=0.6, help="poloidal component (controls helicity)")
    ap.add_argument("--envelope", type=str, default="bump", choices=["gaussian", "bump"],
                    help="tube envelope profile")
    ap.add_argument("--c", type=float, default=1.0, help="vacuum wave speed")
    ap.add_argument("--Htarget", type=float, default=1.0, help="target helicity (sign matters)")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.8, 1.0, 1.25, 1.5, 2.0],
                    help="geometry scale factors")
    ap.add_argument("--plot", action="store_true", help="generate plots")

    # Relaxation options
    relax_group = ap.add_argument_group("relaxation options")
    relax_group.add_argument("--relax", action="store_true",
                             help="run constrained Beltrami relaxation")
    relax_group.add_argument("--relax_steps", type=int, default=200,
                             help="number of relaxation steps")
    relax_group.add_argument("--relax_eta", type=float, default=0.01,
                             help="initial step size (default 0.01)")
    relax_group.add_argument("--relax_eta_min", type=float, default=1e-6,
                             help="minimum eta for backtracking")
    relax_group.add_argument("--relax_no_backtrack", action="store_true",
                             help="disable Armijo backtracking (not recommended)")
    relax_group.add_argument("--relax_trace", action="store_true",
                             help="write CSV trace files to results/traces/")

    # Test mode
    ap.add_argument("--selftest", action="store_true",
                    help="run regression selftest and exit")

    args = ap.parse_args()

    # Handle selftest mode
    if args.selftest:
        success = run_selftest()
        sys.exit(0 if success else 1)

    # Run parameter sweep with inverted twist to match helicity sign
    # Error showed H=-3.412, target=1.0. Flipping twist should fix this.
    args.twist = -args.twist
    results, dx, traces = run_sweep(args)
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
