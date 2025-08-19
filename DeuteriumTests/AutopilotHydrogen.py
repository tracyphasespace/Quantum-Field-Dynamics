#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutopilotHydrogen.py (Legacy - Genesis Constants v3.2)

QFD v3 parameter sweep with autopilot polish phase.

NOTE: This is a legacy development script used for Genesis Constants discovery.
For production use, see:
- Deuterium.py (main solver with Genesis Constants defaults)
- run_target_deuterium.py (convenience wrapper)
- test_genesis_constants.py (automated validation)

GENESIS CONSTANTS DISCOVERED: α=4.0, γₑ=6.0 (virial=0.0472)

Original features:
  • Autopilot polish phase (freeze ramps, shrink LRs, project constraints)
  • Early-stop when energy is flat and penalties are small
  • Per-run field snapshots (.pt) for external visualization (psi_N, psi_e)

Usage (example):
  python run_definitive_sweep_v5_autopilot_fields.py --alpha 5.0 --ge 8.0 \
         --outfile qfd_final_zoom_sweep.json --save success
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
from torch import Tensor
from torch.fft import fftn, ifftn

try:
    from tqdm.auto import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False


# ==============================================================================
# Data classes
# ==============================================================================

@dataclass
class PhysicsParams:
    alpha: float
    beta: float
    eta: float
    gamma_e: float
    kappa_time: float
    L_star: float
    energy_scale: float
    Z: int
    A: int
    N_e: int

@dataclass
class Numerics:
    grid_points: int
    max_radius: float
    iters_outer: int
    iters_inner: int
    lr_e: float
    lr_N: float
    lr_joint: float
    spectral_cutoff: float
    precond_a_N: float
    precond_a_E: float
    kappa: float
    w_Q: float
    w_B: float
    w_center: float
    e_seed_kind: str
    seed_wN: float
    seed_eps: float
    seed_wE: float
    e_shell_peak_frac: float
    e_shell_sigma_frac: float
    e_filament_r_frac: float
    e_filament_z_frac: float
    e_filament_amp: float
    kappa_time_ramp_T: int
    gamma_e_target: float
    gamma_e_ramp_T: int
    tol_energy_rel: float = 1e-7

@dataclass
class SolverState:
    psi_N: Tensor
    psi_e: Tensor

@dataclass
class EnergyBreakdown:
    T_N: float
    T_e: float
    V_coul: float
    V4_N_eff: float
    V4_e: float
    V_time_balance: float
    penalties: Dict[str, float]

@dataclass
class SolveResult:
    converged: bool
    E_model: float
    virial_resid: float
    breakdown: EnergyBreakdown
    meta: Dict[str, float]
    # New: optional fields payload for visualization / snapshots
    fields: Optional[Dict[str, Tensor]] = None


# ==============================================================================
# Coupled Solver (Autopilot + return_fields)
# ==============================================================================

class CoupledSolverV3:
    def __init__(self, p: PhysicsParams, nm: Numerics, dev: torch.device, return_fields: bool = False):
        self.p, self.nm, self.device = p, nm, dev
        self.return_fields = return_fields

        if not (0.0 <= nm.spectral_cutoff <= 1.0):
            raise ValueError(f"spectral_cutoff must be in [0,1], got {nm.spectral_cutoff}")

        n, R = nm.grid_points, nm.max_radius
        self.xs = torch.linspace(-R, R, n, device=dev)
        self.dx = float(self.xs[1] - self.xs[0])
        self.dV = self.dx ** 3

        xs2 = self.xs ** 2
        r2 = xs2[:, None, None] + xs2[None, :, None] + xs2[None, None, :]
        self.r = torch.sqrt(r2)

        # torch.fft.fftfreq returns on CPU; move to device explicitly
        freqs = torch.fft.fftfreq(n, d=self.dx, dtype=self.xs.dtype).to(dev) * (2 * math.pi)
        kx2 = freqs ** 2
        self.k2 = kx2[:, None, None] + kx2[None, :, None] + kx2[None, None, :]
        self.k_max = torch.sqrt(self.k2.max()).item()
        self._spectral_mask = torch.sqrt(self.k2) <= (nm.spectral_cutoff * self.k_max)

    # -------- Core energy and helpers --------

    def energy(self, st: SolverState) -> Tuple[Tensor, EnergyBreakdown]:
        p, nm, dV = self.p, self.nm, self.dV
        psi_N, psi_e = st.psi_N, st.psi_e

        rho_N = psi_N ** 2
        rho_e = -psi_e ** 2
        rho_tot = rho_N + rho_e

        tN = self._kinetic_density(psi_N)
        tE = self._kinetic_density(psi_e)

        phi_N = self._coulomb_potential(rho_N)
        phi_e = self._coulomb_potential(rho_e)

        v_coul = p.alpha * 0.5 * (rho_N * phi_e + rho_e * phi_N)

        h = torch.exp(-p.eta * (rho_e ** 2))
        v4N = p.beta * h * (psi_N ** 4)

        ge = getattr(p, "gamma_e_eff", p.gamma_e)
        v4e = p.beta * ge * (psi_e ** 4)

        k_time = getattr(p, "kappa_time_eff", p.kappa_time)
        v_time = k_time * (rho_tot ** 2)

        T_N = tN.sum() * dV
        T_e = tE.sum() * dV
        V_coul = v_coul.sum() * dV
        V4_N = v4N.sum() * dV
        V4_e = v4e.sum() * dV
        V_time = v_time.sum() * dV

        Q = rho_tot.sum() * dV
        B = rho_N.sum() * dV
        # Charge/mass constraints
        pen_Q = nm.w_Q * (Q - (p.A - p.N_e)) ** 2
        pen_B = nm.w_B * (B - p.A) ** 2

        # Center penalty
        rho_abs = rho_N - rho_e
        m_tot = rho_abs.sum() * dV + 1e-18
        cx = (rho_abs * self.xs[:, None, None]).sum() * dV / m_tot
        cy = (rho_abs * self.xs[None, :, None]).sum() * dV / m_tot
        cz = (rho_abs * self.xs[None, None, :]).sum() * dV / m_tot
        pen_center = nm.w_center * (cx ** 2 + cy ** 2 + cz ** 2)

        E_model = T_N + T_e + V_coul + V4_N + V4_e + V_time + pen_Q + pen_B + pen_center

        br = EnergyBreakdown(
            T_N=T_N.item(),
            T_e=T_e.item(),
            V_coul=V_coul.item(),
            V4_N_eff=V4_N.item(),
            V4_e=V4_e.item(),
            V_time_balance=V_time.item(),
            penalties={"Q": float(pen_Q.item()), "B": float(pen_B.item()), "center": float(pen_center.item())},
        )
        return E_model, br

    def solve_ground(self, seed: SolverState, progress: bool = True) -> SolveResult:
        st = seed
        nm, p = self.nm, self.p

        iterator = range(nm.iters_outer)
        if progress and HAVE_TQDM:
            iterator = tqdm(iterator, desc="Autopilot Sweep", leave=False)

        E_prev: Optional[float] = None
        converged = False

        polish_started = False
        polish_iters = 0
        penalty_target = 1e-5
        max_polish_outer = 150  # safety cap

        for it in iterator:
            # Ramps (disabled once we enter polish)
            if not polish_started:
                ramp_k = min(1.0, (it + 1) / nm.kappa_time_ramp_T)
                p.kappa_time_eff = p.kappa_time * (0.1 + 0.9 * ramp_k)
                ramp_g = min(1.0, (it + 1) / nm.gamma_e_ramp_T)
                p.gamma_e_eff = p.gamma_e + (nm.gamma_e_target - p.gamma_e) * ramp_g

            # Annealed learning rates (+ smaller during polish)
            decay = 0.3 + 0.7 * (1.0 - it / max(1, nm.iters_outer))
            lr_e, lr_N, lr_j = nm.lr_e * decay, nm.lr_N * decay, nm.lr_joint * decay
            if polish_started:
                lr_e *= 0.5
                lr_N *= 0.5
                lr_j *= 0.5

            # Coordinate descent
            for _ in range(nm.iters_inner):
                st = self._step(st, "e", lr_e)
            for _ in range(nm.iters_inner):
                st = self._step(st, "N", lr_N)
            st = self._step(st, "joint", lr_j)

            # Diagnostics / triggers
            if (it % 20 == 0) or (it == nm.iters_outer - 1):
                E, br = self.energy(st)
                if torch.isnan(E) or torch.isinf(E):
                    if HAVE_TQDM and hasattr(iterator, "write"):
                        iterator.write("  [ERROR] Energy NaN/Inf.")
                    else:
                        print("  [ERROR] Energy NaN/Inf.")
                    break

                vir = self._virial_residual(br)
                pen_max = max(br.penalties.values()) if br.penalties else 0.0

                if progress and HAVE_TQDM:
                    iterator.set_postfix({"E": f"{E.item():.2e}", "vir": f"{vir:.2f}", "penMax": f"{pen_max:.2e}"})

                # Autopilot polish trigger: low virial and flat(ish) energy
                if (not polish_started) and vir < 0.09 and (E_prev is not None):
                    if abs(E.item() - E_prev) / (abs(E_prev) + 1e-12) < 5 * nm.tol_energy_rel:
                        if HAVE_TQDM and hasattr(iterator, "write"):
                            iterator.write("  [INFO] Low virial & flat energy: engaging auto-polish.")
                        else:
                            print("  [INFO] Low virial & flat energy: engaging auto-polish.")
                        polish_started = True
                        p.kappa_time_eff = p.kappa_time  # freeze to target
                        p.gamma_e_eff = nm.gamma_e_target
                        st = self._project_constraints(st)  # nudge constraints
                        polish_iters = 0

                if polish_started:
                    polish_iters += 1
                    if (polish_iters % 40) == 0:
                        st = self._project_constraints(st)
                    if polish_iters >= max_polish_outer:
                        converged = True
                        break

                # Early stop on flat energy (and penalties ok if in polish)
                if E_prev is not None:
                    flat = abs(E.item() - E_prev) / (abs(E_prev) + 1e-12) < nm.tol_energy_rel
                    if flat and ((not polish_started) or (pen_max <= penalty_target)):
                        converged = True
                        break

                E_prev = E.item()

        E_final, br_final = self.energy(st)
        vir_final = self._virial_residual(br_final)

        meta = {
            "alpha": self.p.alpha,
            "gamma_e_eff": getattr(self.p, "gamma_e_eff", self.p.gamma_e),
            "kappa_time_eff": getattr(self.p, "kappa_time_eff", self.p.kappa_time),
            "spectral_cutoff": self.nm.spectral_cutoff,
            "grid_points": self.nm.grid_points,
            "polish_started": polish_started,
            "polish_iters": polish_iters,
        }

        fields = None
        if self.return_fields:
            # Save compact CPU tensors for portability and smaller files
            fields = {
                "psi_N": st.psi_N.detach().to(torch.float32).cpu(),
                "psi_e": st.psi_e.detach().to(torch.float32).cpu(),
            }

        return SolveResult(
            converged=converged,
            E_model=float(E_final.item()),
            virial_resid=float(vir_final),
            breakdown=br_final,
            meta=meta,
            fields=fields,
        )

    # -------- Autopilot helpers --------

    def _project_constraints(self, st: SolverState) -> SolverState:
        """Project to satisfy B≈A and Q≈A−N_e while preserving shape."""
        psi_N, psi_e = st.psi_N.clone(), st.psi_e.clone()
        dV, A, Ne = self.dV, float(self.p.A), float(self.p.N_e)
        with torch.no_grad():
            IN = (psi_N ** 2).sum() * dV
            psi_N *= torch.sqrt(torch.as_tensor(A) / IN.clamp_min(1e-18))
            IN = (psi_N ** 2).sum() * dV
            target_Ie = IN - (A - Ne)
            Ie = (psi_e ** 2).sum() * dV
            psi_e *= torch.sqrt(target_Ie.clamp_min(1e-18) / Ie.clamp_min(1e-18))
        return SolverState(psi_N.contiguous(), psi_e.contiguous())

    def _kinetic_density(self, psi: Tensor) -> Tensor:
        dx = self.dx
        gx = (torch.roll(psi, -1, 0) - torch.roll(psi, 1, 0)) / (2 * dx)
        gy = (torch.roll(psi, -1, 1) - torch.roll(psi, 1, 1)) / (2 * dx)
        gz = (torch.roll(psi, -1, 2) - torch.roll(psi, 1, 2)) / (2 * dx)
        return 0.5 * (gx ** 2 + gy ** 2 + gz ** 2)

    def _coulomb_potential(self, rho: Tensor) -> Tensor:
        k2, kappa = self.k2, self.nm.kappa
        RHO = fftn(rho)
        Gk = torch.zeros_like(k2)
        Gk[k2 > 0] = 4 * math.pi / (k2[k2 > 0] + kappa ** 2)
        return ifftn(RHO * Gk).real

    @staticmethod
    def _virial_residual(br: EnergyBreakdown) -> float:
        twoT = 2.0 * (br.T_N + br.T_e)
        rhs = (-br.V_coul) + 3.0 * (br.V4_N_eff + br.V4_e + br.V_time_balance)
        return float(abs(twoT - rhs) / (abs(twoT) + abs(rhs) + 1e-12))

    def _step(self, st: SolverState, which: str, lr: float) -> SolverState:
        psi_N = st.psi_N.clone().detach().requires_grad_(which in ("N", "joint"))
        psi_e = st.psi_e.clone().detach().requires_grad_(which in ("e", "joint"))
        E, _ = self.energy(SolverState(psi_N, psi_e))
        E.backward()
        with torch.no_grad():
            if psi_N.grad is not None:
                gN = self._precondition(psi_N.grad, self.dx, self.nm.precond_a_N)
                psi_N -= lr * gN
            if psi_e.grad is not None:
                gE = self._precondition(psi_e.grad, self.dx, self.nm.precond_a_E)
                psi_e -= lr * gE
            psi_N = self._filter(psi_N)
            psi_e = self._filter(psi_e)
        return SolverState(psi_N.detach(), psi_e.detach())

    def _precondition(self, g: Tensor, dx: float, a: float) -> Tensor:
        if a <= 0:
            return g
        k2 = self.k2
        G = fftn(g)
        G /= (1 + a * k2 * dx ** 2)
        return ifftn(G).real

    def _filter(self, psi: Tensor) -> Tensor:
        psi = torch.nan_to_num(psi, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-5.0, 5.0)
        if self.nm.spectral_cutoff <= 0:
            return psi
        U = fftn(psi)
        U[~self._spectral_mask] = 0
        return ifftn(U).real


# ==============================================================================
# Seeding
# ==============================================================================

def build_seed_from_config(nm: Numerics, p: PhysicsParams, dev: torch.device) -> SolverState:
    n, R = nm.grid_points, nm.max_radius
    xs = torch.linspace(-R, R, n, device=dev)
    dx = float(xs[1] - xs[0])

    xs2 = xs ** 2
    r2 = xs2[:, None, None] + xs2[None, :, None] + xs2[None, None, :]
    r = torch.sqrt(r2)

    # Nucleus (Gaussian)
    psi_N = torch.exp(-(r / (nm.seed_wN * R)).clamp_min(1e-12) ** 2)

    # Electron: shell + thin filament along z
    r0 = nm.e_shell_peak_frac * R
    sig = max(nm.e_shell_sigma_frac * R, 1e-12)
    psi_e_shell = torch.exp(-((r - r0) ** 2) / (2 * sig ** 2))

    r2_cyl = xs2[:, None] + xs2[None, :]
    r_cyl = torch.sqrt(r2_cyl)[:, :, None].expand(n, n, n)
    Zgrid = xs[None, None, :].expand(n, n, n)
    r_fil = nm.e_filament_r_frac * R
    z_fil = max(nm.e_filament_z_frac * R, 1e-12)
    mask = (r_cyl < r_fil) & (torch.abs(Zgrid) < z_fil)
    psi_e_fil = nm.e_filament_amp * torch.cos(math.pi * Zgrid / (2 * z_fil))

    psi_e = psi_e_shell + torch.where(mask, psi_e_fil, torch.zeros_like(psi_e_fil))
    psi_e *= nm.seed_wE

    # Normalize to A and N_e
    dV = dx ** 3
    psi_N *= torch.sqrt(torch.as_tensor(p.A, device=dev) / ((psi_N ** 2).sum() * dV).clamp_min(1e-18))
    psi_e *= torch.sqrt(torch.as_tensor(float(p.N_e), device=dev) / ((psi_e ** 2).sum() * dV).clamp_min(1e-18))

    return SolverState(psi_N.contiguous(), psi_e.contiguous())


# ==============================================================================
# CLI / Main
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QFD v3 Autopilot Sweep with field snapshots")
    parser.add_argument("--alpha", type=float, default=5.0, help="Sweep center for alpha")
    parser.add_argument("--ge", type=float, default=8.0, help="Sweep center for gamma_e_target")
    parser.add_argument("--outfile", type=str, default="qfd_autopilot_sweep_results.json", help="JSON results file")
    parser.add_argument("--save", choices=("success", "all", "none"), default="success",
                        help="Save .pt snapshots for 'success' runs only (default), for 'all' runs, or 'none'")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64", help="Torch default dtype")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Compute device")
    parser.add_argument("--grid", type=int, default=None, help="Override grid_points")
    parser.add_argument("--iters", type=int, default=None, help="Override iters_outer")
    return parser.parse_args()


def main():
    args = parse_args()

    # Dtype & device
    if args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float64)

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    torch.manual_seed(1234)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except AttributeError:
        pass

    # Base numerics/physics (Gold Standard defaults)
    TUNE_BASE: Dict = {
        "beta": 3.0, "eta": 0.05, "gamma_e": 0.15, "kappa_time": 3.2,
        "L_star": 1.0, "energy_scale": 1.0, "Z": 1, "A": 1, "N_e": 1,
        "grid_points": 96 if args.grid is None else int(args.grid),
        "max_radius": 14.0,
        "iters_outer": 600 if args.iters is None else int(args.iters),
        "iters_inner": 12, "lr_e": 2e-4, "lr_N": 3e-4, "lr_joint": 1e-4,
        "spectral_cutoff": 0.36, "precond_a_N": 1.0, "precond_a_E": 2.0,
        "kappa": 4e-4, "w_Q": 1e3, "w_B": 1e3, "w_center": 1e1,
        "e_seed_kind": "shell_filament", "seed_wN": 0.60, "seed_eps": 0.004,
        "seed_wE": 1.20, "e_shell_peak_frac": 0.28, "e_shell_sigma_frac": 0.10,
        "e_filament_r_frac": 0.02, "e_filament_z_frac": 0.25,
        "e_filament_amp": 0.50, "kappa_time_ramp_T": 120, "gamma_e_ramp_T": 200,
        "tol_energy_rel": 1e-7,
    }

    sweep_center = {"alpha": float(args.alpha), "ge_target": float(args.ge)}
    sweep_grid = {
        "alpha": [args.alpha - 1.0, args.alpha - 0.5, args.alpha + 0.5, args.alpha + 1.0],
        "gamma_e_target": [args.ge - 2.0, args.ge - 1.0, args.ge + 1.0, args.ge + 2.0],
    }

    outdir = os.path.dirname(args.outfile) or "."
    os.makedirs(outdir, exist_ok=True)

    print("=" * 78)
    print("QFD v3 Autopilot Sweep (fields enabled)")
    print(f"Start time: {time.ctime()}")
    print(f"Device:     {dev.type}, Dtype: {torch.get_default_dtype()}")
    print(f"Centering search around alpha={sweep_center['alpha']}, ge_target={sweep_center['ge_target']}")
    print("=" * 78)

    results = []
    total_runs = len(sweep_grid["alpha"]) * len(sweep_grid["gamma_e_target"])
    run_count = 0

    for alpha_val in sweep_grid["alpha"]:
        for ge_target_val in sweep_grid["gamma_e_target"]:
            run_count += 1
            print(f"\n--- Running [{run_count}/{total_runs}]: alpha={alpha_val:.2f}, ge_target={ge_target_val:.2f} ---")

            tune = dict(TUNE_BASE)
            tune["alpha"] = float(alpha_val)
            tune["gamma_e_target"] = float(ge_target_val)

            # Instantiate configs
            p = PhysicsParams(**{k: tune[k] for k in PhysicsParams.__annotations__})
            nm = Numerics(**{k: tune[k] for k in Numerics.__annotations__})

            solver = CoupledSolverV3(p, nm, dev, return_fields=(args.save in ("success", "all")))
            seed = build_seed_from_config(nm, p, dev)

            try:
                res = solver.solve_ground(seed=seed, progress=True)

                virial_ok = res.virial_resid < 0.1
                penalties_ok = all(v < 1e-5 for v in res.breakdown.penalties.values())
                success = bool(res.converged) and virial_ok and penalties_ok
                pen_max = max(res.breakdown.penalties.values()) if res.breakdown.penalties else float("inf")

                row = {
                    "alpha": float(alpha_val),
                    "ge_target": float(ge_target_val),
                    "E_model": float(res.E_model),
                    "virial": float(res.virial_resid),
                    "pen_max": float(pen_max),
                    "success": bool(success),
                    "meta": res.meta,
                    "penalties": res.breakdown.penalties,
                }
                results.append(row)

                # Snapshot policy
                should_save = (args.save == "all") or (args.save == "success" and success)
                if should_save and (res.fields is not None):
                    snap_name = f"state_a{alpha_val:.2f}_g{ge_target_val:.2f}.pt"
                    snap_path = os.path.join(outdir, snap_name)
                    state_to_save = {
                        "psi_N": res.fields["psi_N"],
                        "psi_e": res.fields["psi_e"],
                        "numerics": {
                            "grid_points": nm.grid_points,
                            "max_radius": nm.max_radius,
                        },
                        "tune": {"alpha": p.alpha, "gamma_e_target": nm.gamma_e_target},
                        "meta": res.meta,
                    }
                    torch.save(state_to_save, snap_path)
                    print(f"Saved final fields to {snap_path}")

            except Exception as e:
                print(f"  [ERROR] Run failed: {e}")
                results.append({
                    "alpha": float(alpha_val),
                    "ge_target": float(ge_target_val),
                    "E_model": 999.0,
                    "virial": 999.0,
                    "pen_max": 999.0,
                    "success": False,
                })
            finally:
                # Persist partial results after each run
                with open(args.outfile, "w") as f:
                    json.dump({"sweep_center": sweep_center, "results": results}, f, indent=2)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

    # Final report
    print("\n" + "=" * 78)
    print("Autopilot Sweep Report")
    print("=" * 78)
    print(f"{'alpha':>8s} | {'ge_target':>12s} | {'E_model':>12s} | {'Virial':>10s} | {'PenMax':>12s} | {'Result'}")
    print("-" * 85)

    # Sort: successes first, then lowest virial
    results.sort(key=lambda r: (-int(r.get("success", False)), r.get("virial", float("inf"))))
    for r in results:
        status = "SUCCESS" if r.get("success", False) else "FAIL"
        print(f"{r['alpha']:>8.2f} | {r['ge_target']:>12.2f} | {r['E_model']:>12.3e} | {r['virial']:>10.4f} | {r['pen_max']:>12.3e} | {status}")

    print("-" * 85)
    best = results[0]
    if best.get("success", False):
        print("\n[SUCCESS] A stable, physically valid ground state has been found!")
        print(f"Optimal parameters: alpha = {best['alpha']:.2f}, gamma_e_target = {best['ge_target']:.2f}")
    else:
        print("\n[INFO] No configuration met all success criteria.")
        print("Look for the row with the lowest combined Virial and PenMax.")
    print("=" * 78)
    print(f"Saved sweep results to {args.outfile}")


if __name__ == "__main__":
    main()
