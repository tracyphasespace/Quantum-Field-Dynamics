#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AllNightLong.py (Legacy - Genesis Constants v3.2)

Extended parameter sweep runner for QFD v3.

NOTE: This is a legacy development script used for Genesis Constants discovery.
For production use, see:
- Deuterium.py (main solver with Genesis Constants defaults)
- run_target_deuterium.py (convenience wrapper)
- test_genesis_constants.py (automated validation)

GENESIS CONSTANTS DISCOVERED: α=4.0, γₑ=6.0 (virial=0.0472)
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
# DATA CLASSES
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
    tol_energy_rel: float = 1e-6


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
    final_state: SolverState  # NEW: return the final fields


# ==============================================================================
# QFD SOLVER (Autopilot + Convergence Override + Checkpointing)
# ==============================================================================

class CoupledSolverV3:
    def __init__(self, p: PhysicsParams, nm: Numerics, dev: torch.device):
        self.p, self.nm, self.device = p, nm, dev

        if not (0.0 <= nm.spectral_cutoff <= 1.0):
            raise ValueError(f"spectral_cutoff must be in [0,1], got {nm.spectral_cutoff}")

        # Grid and derived quantities
        n, R = nm.grid_points, nm.max_radius
        self.xs = torch.linspace(-R, R, n, device=dev)
        self.dx = float(self.xs[1] - self.xs[0])
        self.dV = self.dx ** 3

        xs2 = self.xs ** 2
        r2 = xs2[:, None, None] + xs2[None, :, None] + xs2[None, None, :]
        self.r = torch.sqrt(r2)

        freqs = torch.fft.fftfreq(n, d=self.dx, dtype=self.xs.dtype).to(dev) * (2 * math.pi)
        kx2 = freqs ** 2
        self.k2 = kx2[:, None, None] + kx2[None, :, None] + kx2[None, None, :]
        self.k_max = torch.sqrt(self.k2.max()).item()
        self._spectral_mask = torch.sqrt(self.k2) <= (nm.spectral_cutoff * self.k_max)

        # Convergence override state
        self._chk_psi_N: Optional[Tensor] = None
        self._chk_psi_e: Optional[Tensor] = None

        # Optional checkpointing (configured via set_checkpointing)
        self._ckpt_every = 0
        self._ckpt_dir = None
        self._ckpt_tag = None

    # ---------- public helpers ----------

    def set_checkpointing(self, every: int = 0, out_dir: Optional[str] = None, tag: Optional[str] = None):
        self._ckpt_every = max(0, int(every))
        self._ckpt_dir = out_dir
        self._ckpt_tag = tag

    # ---------- energy & physics ----------

    def energy(self, st: SolverState) -> Tuple[Tensor, EnergyBreakdown]:
        p, nm, dV = self.p, self.nm, self.dV
        psi_N, psi_e = st.psi_N, st.psi_e

        rho_N = psi_N ** 2
        rho_e = -psi_e ** 2
        rho_tot = rho_N - (-rho_e)  # i.e., psi_N^2 - psi_e^2

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

        pen_Q = nm.w_Q * (Q - (p.A - p.N_e)) ** 2
        pen_B = nm.w_B * (B - p.A) ** 2

        # center of mass penalty (using |rho|)
        rho_abs = rho_N - rho_e  # since rho_e is negative
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
            penalties={"Q": pen_Q.item(), "B": pen_B.item(), "center": pen_center.item()},
        )
        return E_model, br

    def solve_ground(
        self,
        seed: SolverState,
        progress: bool = True,
        engage_autopilot: bool = True,
        max_polish_outer: int = 300,
        polish_reproj_every: int = 40,
    ) -> SolveResult:
        """Main optimizer loop with ramps + optional autopilot polish phase and
        convergence override based on ΔE & Δψ.

        Returns SolveResult including final_state (fields).
        """
        st = seed
        nm, p = self.nm, self.p

        iterator = range(nm.iters_outer)
        if progress and HAVE_TQDM:
            iterator = tqdm(iterator, desc="AllNightLong", leave=False)

        E_prev = None
        converged = False

        polish_started = False
        polish_iters = 0
        penalty_target = 1e-5  # threshold for penalties_ok

        for it in iterator:
            # Ramps (unless polishing has engaged)
            if not polish_started:
                ramp_k = min(1.0, (it + 1) / nm.kappa_time_ramp_T)
                p.kappa_time_eff = p.kappa_time * (0.1 + 0.9 * ramp_k)
                ramp_g = min(1.0, (it + 1) / nm.gamma_e_ramp_T)
                p.gamma_e_eff = p.gamma_e + (nm.gamma_e_target - p.gamma_e) * ramp_g

            # Decayed step sizes
            decay = 0.3 + 0.7 * (1.0 - it / max(1, nm.iters_outer))
            lr_e = nm.lr_e * decay
            lr_N = nm.lr_N * decay
            lr_j = nm.lr_joint * decay

            if polish_started:
                # smaller steps during polish
                lr_e *= 0.5
                lr_N *= 0.5
                lr_j *= 0.5

            # Inner steps
            for _ in range(nm.iters_inner):
                st = self._step(st, "e", lr_e)
            for _ in range(nm.iters_inner):
                st = self._step(st, "N", lr_N)
            st = self._step(st, "joint", lr_j)

            # Periodic checkpoints
            if self._ckpt_every > 0 and (it % self._ckpt_every == 0) and self._ckpt_dir and self._ckpt_tag:
                self._save_checkpoint(st, it)

            # Periodic monitoring
            if (it % 20 == 0) or (it == nm.iters_outer - 1):
                E, br = self.energy(st)
                if torch.isnan(E) or torch.isinf(E):
                    print("  [ERROR] Energy NaN/Inf. Aborting this run.")
                    break

                vir = self._virial_residual(br)
                pen_max = max(br.penalties.values()) if br.penalties else 0.0

                if progress and HAVE_TQDM:
                    iterator.set_postfix(
                        {"E": f"{E.item():.2e}", "vir": f"{vir:.2f}", "penMax": f"{pen_max:.2e}"}
                    )

                # Autopilot trigger: low virial + flat energy
                if engage_autopilot and (not polish_started) and vir < 0.10 and (E_prev is not None):
                    if abs(E.item() - E_prev) / (abs(E_prev) + 1e-12) < 5 * nm.tol_energy_rel:
                        if progress:
                            print("  [INFO] Low virial & flat ΔE → engaging autopilot polish.")
                        polish_started = True
                        polish_iters = 0
                        p.kappa_time_eff = p.kappa_time  # freeze ramps
                        p.gamma_e_eff = nm.gamma_e_target
                        # Optional: gently stiffen penalties during polish only
                        # self.nm.w_B *= 2.0; self.nm.w_Q *= 2.0
                        st = self._project_constraints(st)  # immediate nudge

                # Convergence override: flat ΔE + flat Δψ and already good virial
                delta_psi = None
                if self._chk_psi_N is not None:
                    delta_N = self._rel_change(st.psi_N, self._chk_psi_N)
                    delta_e = self._rel_change(st.psi_e, self._chk_psi_e)
                    delta_psi = max(delta_N, delta_e)

                if it % 40 == 0:
                    self._chk_psi_N = st.psi_N.detach().clone()
                    self._chk_psi_e = st.psi_e.detach().clone()

                flat_E = (E_prev is not None) and (abs(E.item() - E_prev) / (abs(E_prev) + 1e-12) < nm.tol_energy_rel)
                flat_psi = (delta_psi is not None) and (delta_psi < 3 * nm.tol_energy_rel)

                # If polishing, cap polish length and allow early-stop on penalties
                if polish_started:
                    polish_iters += 1
                    if (polish_iters % max(1, polish_reproj_every)) == 0:
                        st = self._project_constraints(st)
                    if polish_iters >= max_polish_outer:
                        converged = True
                        break
                    if flat_E and (pen_max <= penalty_target):
                        converged = True
                        break

                # If not polishing, and it's flat E + flat ψ and vir already good → done
                if (not polish_started) and flat_E and flat_psi and (vir < 0.08):
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
        }
        return SolveResult(converged, E_final.item(), vir_final, br_final, meta, st)

    # ---------- lower-level helpers ----------

    def _save_checkpoint(self, st: SolverState, it: int):
        if not (self._ckpt_dir and self._ckpt_tag):
            return
        os.makedirs(self._ckpt_dir, exist_ok=True)
        path = os.path.join(self._ckpt_dir, f"ckpt_{self._ckpt_tag}_it{it:04d}.pt")
        torch.save(
            {"psi_N": st.psi_N.detach().cpu(), "psi_e": st.psi_e.detach().cpu()},
            path,
        )

    @staticmethod
    def _virial_residual(br: EnergyBreakdown) -> float:
        twoT = 2.0 * (br.T_N + br.T_e)
        rhs = (-br.V_coul) + 3.0 * (br.V4_N_eff + br.V4_e + br.V_time_balance)
        return abs(twoT - rhs) / (abs(twoT) + abs(rhs) + 1e-12)

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
        mask = k2 > 0
        Gk[mask] = 4 * math.pi / (k2[mask] + kappa ** 2)
        return ifftn(RHO * Gk).real

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
        G = fftn(g)
        G /= (1 + a * self.k2 * dx ** 2)
        return ifftn(G).real

    def _filter(self, psi: Tensor) -> Tensor:
        psi = torch.nan_to_num(psi, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-5.0, 5.0)
        if self.nm.spectral_cutoff <= 0:
            return psi
        U = fftn(psi)
        U[~self._spectral_mask] = 0
        return ifftn(U).real

    def _project_constraints(self, st: SolverState) -> SolverState:
        psi_N, psi_e = st.psi_N.clone(), st.psi_e.clone()
        dV, A, Ne = self.dV, float(self.p.A), float(self.p.N_e)
        with torch.no_grad():
            IN = (psi_N ** 2).sum() * dV
            psi_N *= torch.sqrt(A / IN.clamp_min(1e-18))
            IN = (psi_N ** 2).sum() * dV
            target_Ie = IN - (A - Ne)
            Ie = (psi_e ** 2).sum() * dV
            psi_e *= torch.sqrt(target_Ie.clamp_min(1e-18) / Ie.clamp_min(1e-18))
        return SolverState(psi_N.contiguous(), psi_e.contiguous())

    @staticmethod
    def _rel_change(x_new: Tensor, x_old: Tensor, eps: float = 1e-18) -> float:
        num = torch.norm((x_new - x_old).reshape(-1))
        den = torch.norm(x_old.reshape(-1)) + eps
        return (num / den).item()


# ==============================================================================
# SEED
# ==============================================================================

def build_seed_from_config(nm: Numerics, p: PhysicsParams, dev: torch.device) -> SolverState:
    n, R = nm.grid_points, nm.max_radius
    xs = torch.linspace(-R, R, n, device=dev)
    dx = float(xs[1] - xs[0])
    xs2 = xs ** 2
    r2 = xs2[:, None, None] + xs2[None, :, None] + xs2[None, None, :]
    r = torch.sqrt(r2)

    # nucleus
    psi_N = torch.exp(-(r / (nm.seed_wN * R)).clamp_min(1e-12) ** 2)

    # electron: shell + filament
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

    dV = dx ** 3
    psi_N *= torch.sqrt(p.A / ((psi_N ** 2).sum() * dV).clamp_min(1e-18))
    psi_e *= torch.sqrt(float(p.N_e) / ((psi_e ** 2).sum() * dV).clamp_min(1e-18))

    return SolverState(psi_N.contiguous(), psi_e.contiguous())


# ==============================================================================
# DEFAULT TUNE BASE
# ==============================================================================

def default_tune_base() -> Dict:
    return {
        "beta": 3.0, "eta": 0.05, "gamma_e": 0.15, "kappa_time": 3.2,
        "L_star": 1.0, "energy_scale": 1.0, "Z": 1, "A": 1, "N_e": 1,
        "grid_points": 96, "max_radius": 14.0,
        "iters_outer": 1200,   # long overnight default for single
        "iters_inner": 12,
        "lr_e": 2e-4, "lr_N": 3e-4, "lr_joint": 1e-4,
        "spectral_cutoff": 0.36,
        "precond_a_N": 1.0, "precond_a_E": 2.0,
        "kappa": 4e-4,
        "w_Q": 1e3, "w_B": 1e3, "w_center": 1e1,
        "e_seed_kind": "shell_filament", "seed_wN": 0.60, "seed_eps": 0.004, "seed_wE": 1.20,
        "e_shell_peak_frac": 0.28, "e_shell_sigma_frac": 0.10,
        "e_filament_r_frac": 0.02, "e_filament_z_frac": 0.25, "e_filament_amp": 0.50,
        "kappa_time_ramp_T": 120, "gamma_e_ramp_T": 200,
        "tol_energy_rel": 1e-8,  # strict for overnight
    }


# ==============================================================================
# CLI / MAIN
# ==============================================================================

def parse_args():
    ap = argparse.ArgumentParser(description="AllNightLong QFD v3 runner")
    ap.add_argument("--mode", choices=("single", "sweep"), default="single",
                    help="Run a single point (default) or a 4x4 sweep around center.")
    ap.add_argument("--alpha", type=float, default=4.0, help="Center alpha (and single alpha).")
    ap.add_argument("--ge", type=float, default=6.0, help="Center gamma_e_target (and single ge).")
    ap.add_argument("--iters", type=int, default=1200, help="Override iters_outer.")
    ap.add_argument("--tol", type=float, default=1e-8, help="Override tol_energy_rel.")
    ap.add_argument("--grid", type=int, default=None, help="Override grid_points.")
    ap.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--outdir", type=str, default="runs_allnight", help="Directory to save states/ckpts/summary.")
    ap.add_argument("--outfile", type=str, default="allnight_summary.json", help="Summary JSON filename (inside outdir).")
    ap.add_argument("--save-every", type=int, default=0, help="Checkpoint every N outer its (0=off).")
    return ap.parse_args()


def success_from_physics(virial: float, penalties: Dict[str, float]) -> bool:
    virial_ok = float(virial) < 0.10
    penalties_ok = all(float(v) < 1e-5 for v in penalties.values())
    return virial_ok and penalties_ok


def run_point(dev, alpha_val: float, ge_target_val: float, tune_base: Dict, outdir: str,
              iters_override: Optional[int], tol_override: Optional[float],
              grid_override: Optional[int], save_every: int) -> Dict:
    tune = dict(tune_base)
    tune["alpha"] = alpha_val
    tune["gamma_e_target"] = ge_target_val
    if iters_override is not None:
        tune["iters_outer"] = iters_override
    if tol_override is not None:
        tune["tol_energy_rel"] = tol_override
    if grid_override is not None:
        tune["grid_points"] = grid_override

    # Instantiate params
    p = PhysicsParams(**{k: tune[k] for k in PhysicsParams.__annotations__})
    nm = Numerics(**{k: tune[k] for k in Numerics.__annotations__})

    solver = CoupledSolverV3(p, nm, dev)
    tag = f"a{alpha_val:.2f}_g{ge_target_val:.2f}"
    solver.set_checkpointing(every=save_every, out_dir=outdir, tag=tag)

    seed = build_seed_from_config(nm, p, dev)

    try:
        res = solver.solve_ground(seed=seed, progress=True, engage_autopilot=True)
    except RuntimeError as e:
        print(f"  [ERROR] Solver crashed: {e}")
        if "CUDA out of memory" in str(e):
            print("  Hint: reduce --grid or use --dtype float32.")
        raise

    # Save final fields
    os.makedirs(outdir, exist_ok=True)
    state_path = os.path.join(outdir, f"state_{tag}.pt")
    torch.save(
        {
            "psi_N": res.final_state.psi_N.detach().cpu(),
            "psi_e": res.final_state.psi_e.detach().cpu(),
            "meta": res.meta,
            "alpha": alpha_val,
            "gamma_e_target": ge_target_val,
            "numerics": {k: tune[k] for k in Numerics.__annotations__},
            "physics": {k: tune[k] for k in PhysicsParams.__annotations__},
            "E_model": res.E_model,
            "virial": res.virial_resid,
            "penalties": res.breakdown.penalties,
        },
        state_path,
    )

    row = {
        "alpha": alpha_val,
        "ge_target": ge_target_val,
        "E_model": res.E_model,
        "virial": res.virial_resid,
        "penalties": res.breakdown.penalties,
        "pen_max": max(res.breakdown.penalties.values()) if res.breakdown.penalties else 0.0,
        "success": success_from_physics(res.virial_resid, res.breakdown.penalties),
        "converged_flag": bool(res.converged),  # informational
        "meta": res.meta,
        "state_path": state_path,
    }
    return row


def main():
    args = parse_args()

    # dtype & device
    torch.set_default_dtype(torch.float64 if args.dtype == "float64" else torch.float32)
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

    # Base tune
    tune_base = default_tune_base()
    tune_base["iters_outer"] = int(args.iters)
    tune_base["tol_energy_rel"] = float(args.tol)
    if args.grid is not None:
        tune_base["grid_points"] = int(args.grid)

    # I/O
    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, args.outfile)

    print("=" * 78)
    print(f"AllNightLong QFD v3 runner  |  Mode: {args.mode}")
    print(f"Start time: {time.ctime()}")
    print(f"Device:     {dev.type}, dtype: {torch.get_default_dtype()}")
    print(f"Out dir:    {args.outdir}")
    print("=" * 78)

    results = []
    if args.mode == "single":
        print(f"Single-point run at alpha={args.alpha}, ge_target={args.ge}")
        try:
            row = run_point(
                dev,
                alpha_val=args.alpha,
                ge_target_val=args.ge,
                tune_base=tune_base,
                outdir=args.outdir,
                iters_override=args.iters,
                tol_override=args.tol,
                grid_override=args.grid,
                save_every=args.save_every,
            )
            results.append(row)
        except Exception:
            # still write partial summary
            with open(summary_path, "w") as f:
                json.dump({"mode": "single", "center": {"alpha": args.alpha, "ge_target": args.ge}, "results": results}, f, indent=2)
            raise
        # Save summary
        with open(summary_path, "w") as f:
            json.dump({"mode": "single", "center": {"alpha": args.alpha, "ge_target": args.ge}, "results": results}, f, indent=2)

    else:  # sweep: 4x4 around center
        sweep_center = {"alpha": args.alpha, "ge_target": args.ge}
        sweep_grid = {
            "alpha": [args.alpha - 1.0, args.alpha - 0.5, args.alpha + 0.5, args.alpha + 1.0],
            "gamma_e_target": [args.ge - 2.0, args.ge - 1.0, args.ge + 1.0, args.ge + 2.0],
        }
        total_runs = len(sweep_grid["alpha"]) * len(sweep_grid["gamma_e_target"])
        run_idx = 0

        try:
            for a in sweep_grid["alpha"]:
                for g in sweep_grid["gamma_e_target"]:
                    run_idx += 1
                    print(f"\n--- Running [{run_idx}/{total_runs}]: alpha={a:.2f}, ge_target={g:.2f} ---")
                    row = run_point(
                        dev,
                        alpha_val=a,
                        ge_target_val=g,
                        tune_base=tune_base,
                        outdir=args.outdir,
                        iters_override=args.iters,
                        tol_override=args.tol,
                        grid_override=args.grid,
                        save_every=args.save_every,
                    )
                    results.append(row)
                    # incremental save
                    with open(summary_path, "w") as f:
                        json.dump({"mode": "sweep", "center": sweep_center, "grid": sweep_grid, "results": results}, f, indent=2)
        except Exception:
            with open(summary_path, "w") as f:
                json.dump({"mode": "sweep", "center": sweep_center, "grid": sweep_grid, "results": results}, f, indent=2)
            raise

        # Final report to console
        print("\n" + "=" * 78)
        print("AllNightLong Sweep Report")
        print("=" * 78)
        print(f"{'alpha':>8s} | {'ge_target':>12s} | {'E_model':>12s} | {'Virial':>10s} | {'PenMax':>12s} | {'Result'}")
        print("-" * 85)
        results_sorted = sorted(results, key=lambda r: (-int(r["success"]), r["virial"]))
        for r in results_sorted:
            status = "SUCCESS" if r["success"] else "FAIL"
            print(f"{r['alpha']:>8.2f} | {r['ge_target']:>12.2f} | {r['E_model']:>12.3e} | {r['virial']:>10.4f} | {r['pen_max']:>12.3e} | {status}")
        print("-" * 85)
        best = results_sorted[0] if results_sorted else None
        if best and best["success"]:
            print("\n[SUCCESS] Physically valid ground state found!")
            print(f"Optimal parameters: alpha = {best['alpha']:.2f}, gamma_e_target = {best['ge_target']:.2f}")
        else:
            print("\n[INFO] No configuration met both thresholds. Pick lowest Virial + PenMax and re-run single with longer patience.")
        print("=" * 78)

        # Final save (again)
        with open(summary_path, "w") as f:
            json.dump({"mode": "sweep", "center": sweep_center, "grid": sweep_grid, "results": results}, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Exiting cleanly.")
        sys.exit(130)
