#!/usr/bin/env python3
"""Phase-8 SCF solver with compounding cohesion (v4 - Merged from ChatSolver)."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# ---- PYTHONPATH shim ------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIR = Path(__file__).resolve().parent
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

try:
    from qfd_effective_potential_models import compute_alpha_eff
except ImportError:
    # Fallback if the module is not found in the path
    def compute_alpha_eff(A, Z, c_v2_base, c_v2_iso, c_v2_mass):
        A13 = max(1.0, A)**(1.0/3.0)
        iso_term  = Z*(Z-1.0)/(A13 + 1e-12)
        expo_arg  = max(min(c_v2_mass*A, 5.0), -5.0)
        return (c_v2_base + c_v2_iso*iso_term) * math.exp(expo_arg)

# ---- Utilities ------------------------------------------------------------

def torch_det_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def central_grad_sq3(f: torch.Tensor, dx: float) -> torch.Tensor:
    """Sum of squared central differences across x, y, z (periodic BCs)."""
    fxp = torch.roll(f, -1, dims=0)
    fxm = torch.roll(f,  1, dims=0)
    fyp = torch.roll(f, -1, dims=1)
    fym = torch.roll(f,  1, dims=1)
    fzp = torch.roll(f, -1, dims=2)
    fzm = torch.roll(f,  1, dims=2)
    gx = (fxp - fxm) / (2.0 * dx)
    gy = (fyp - fym) / (2.0 * dx)
    gz = (fzp - fzm) / (2.0 * dx)
    return gx * gx + gy * gy + gz * gz

# ---- Rotor terms ---------------------------------------------------------

class RotorParams(torch.nn.Module):
    def __init__(self, lambda_R2: float = 0.0, lambda_R3: float = 0.0, B_target: float = 0.0):
        super().__init__()
        self.lambda_R2 = float(lambda_R2)
        self.lambda_R3 = float(lambda_R3)
        self.B_target  = float(B_target)

class RotorTerms:
    def __init__(self, dx: float, dV: float, params: RotorParams):
        self.dx, self.dV, self.p = dx, dV, params

    @staticmethod
    def hodge_dual(B: torch.Tensor) -> torch.Tensor:
        return torch.stack((B[1], B[2], B[0]), dim=0)

    def T_rotor(self, B: torch.Tensor) -> torch.Tensor:
        if self.p.lambda_R2 == 0.0:
            return B.new_tensor(0.0)
        gsum = central_grad_sq3(B[0], self.dx) + central_grad_sq3(B[1], self.dx) + central_grad_sq3(B[2], self.dx)
        return self.p.lambda_R2 * gsum.sum() * self.dV

    def V_rotor(self, B: torch.Tensor) -> torch.Tensor:
        if self.p.lambda_R3 == 0.0:
            return B.new_tensor(0.0)
        Bmag = torch.sqrt(1e-24 + B[0]*B[0] + B[1]*B[1] + B[2]*B[2])
        return self.p.lambda_R3 * ((Bmag - self.p.B_target)**2).sum() * self.dV

# ---- Core model ----------------------------------------------------------

class Phase8Model(torch.nn.Module):
    def __init__(self, A:int, Z:int, grid:int, dx:float,
                 c_v2_base:float, c_v2_iso:float, c_v2_mass:float,
                 c_v4_base:float, c_v4_size:float,
                 rotor: RotorParams,
                 device: str = "cpu",
                 coulomb_mode:str="spectral", alpha_coul: float = 1.0,
                 mass_penalty_N: float = 0.0, mass_penalty_e: float = 0.0,
                 project_mass_each: bool = False, project_e_each: bool = False,
                 kappa_rho: float = 0.0,
                 alpha_e_scale: float = 0.5, beta_e_scale: float = 0.5,
                 alpha_model: str = "exp",
                 coulomb_twopi: bool = False,
                 c_sym: float = 0.0):
        super().__init__()
        self.A, self.Z = int(A), int(Z)
        self.N, self.dx = int(grid), float(dx)
        self.dV = self.dx**3
        self.device = torch.device(device)

        A13 = max(1.0, A)**(1.0/3.0)
        if alpha_model == "exp":
            self.alpha_eff = float(compute_alpha_eff(A, Z, c_v2_base, c_v2_iso, c_v2_mass))
        else:
            iso_term = Z*(Z-1.0)/(A13 + 1e-12)
            self.alpha_eff = float(c_v2_base + c_v2_iso*iso_term + c_v2_mass * A)
        self.beta_eff  = float(c_v4_base + c_v4_size * A13)
        self.coeffs_raw = dict(c_v2_base=c_v2_base, c_v2_iso=c_v2_iso,
                               c_v2_mass=c_v2_mass,
                               c_v4_base=c_v4_base, c_v4_size=c_v4_size, A13=A13)
        self.alpha_model = alpha_model

        self.alpha_e = float(alpha_e_scale) * self.alpha_eff
        self.beta_e  = float(beta_e_scale) * self.beta_eff

        shape = (self.N, self.N, self.N)
        self.psi_N = torch.zeros(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        self.psi_e = torch.zeros(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        self.B_N   = torch.zeros((3, *shape), dtype=torch.float32, device=self.device, requires_grad=True)

        self.rotor_terms = RotorTerms(self.dx, self.dV, rotor)

        self.coulomb_mode = str(coulomb_mode)
        self.alpha_coul   = float(alpha_coul)
        self.mass_penalty_N = float(mass_penalty_N)
        self.mass_penalty_e = float(mass_penalty_e)
        self.project_mass_each = bool(project_mass_each)
        self.project_e_each    = bool(project_e_each)
        self.kappa_rho = float(kappa_rho)
        self.coulomb_twopi = bool(coulomb_twopi)

        # Symmetry energy term: penalizes N-Z asymmetry
        self.c_sym = float(c_sym)

        self.kN = 1.0
        self.ke = 1.0

    @torch.no_grad()
    def initialize_fields(self, seed:int=0, init_mode:str="gauss"):
        g = torch.Generator(device=self.device).manual_seed(seed)
        r2 = self._r2_grid()
        if init_mode == "gauss":
            R0 = (1.2 * max(1.0, self.A) ** (1.0/3.0))
            sigma2_n = (0.60*R0)**2
            sigma2_e = (1.00*R0)**2
            gn = torch.exp(-r2/(2.0*sigma2_n))
            ge = torch.exp(-r2/(2.0*sigma2_e))
            torch.manual_seed(seed)
            self.psi_N.copy_(gn + 1e-3*torch.randn_like(gn))
            self.psi_e.copy_(ge + 1e-3*torch.randn_like(ge))
        else:
            torch.manual_seed(seed)
            for T in (self.psi_N, self.psi_e):
                T.uniform_(0.0, 1.0)
                T.mul_(torch.exp(-0.25 * r2))
                T.add_(1e-3 * torch.randn_like(T))
        self.B_N.zero_().add_(1e-3 * torch.randn_like(self.B_N))
        nN = torch.sqrt((self.psi_N*self.psi_N).sum() * self.dV + 1e-24)
        ne = torch.sqrt((self.psi_e*self.psi_e).sum() * self.dV + 1e-24)
        if float(nN) > 0:
            self.psi_N.mul_((0.5 * self.A) / float(nN))
        if float(ne) > 0:
            self.psi_e.mul_((0.5 * self.Z) / float(ne))

    def _r2_grid(self) -> torch.Tensor:
        n = self.N
        ax = torch.arange(n, device=self.device, dtype=torch.float32)
        ax = (ax - (n-1)/2.0) * self.dx
        X, Y, Z = torch.meshgrid(ax, ax, ax, indexing="ij")
        return X*X + Y*Y + Z*Z

    def kinetic_scalar(self, psi: torch.Tensor, kcoef: float) -> torch.Tensor:
        return 0.5 * kcoef * central_grad_sq3(psi, self.dx).sum() * self.dV

    def potential_from_density(self, rho: torch.Tensor, alpha: float, beta: float) -> tuple:
        V4 = 0.5 * alpha * (rho * rho).sum() * self.dV
        V6 = (1.0/6.0) * beta * (rho * rho * rho).sum() * self.dV
        return V4, V6

    def surface_energy(self, rho: torch.Tensor) -> torch.Tensor:
        if self.kappa_rho == 0.0:
            return rho.new_tensor(0.0)
        grad_sq = central_grad_sq3(rho, self.dx)
        return self.kappa_rho * grad_sq.sum() * self.dV

    def _spectral_phi(self, rho: torch.Tensor) -> torch.Tensor:
        if self.coulomb_mode != "spectral" or self.alpha_coul == 0.0:
            return torch.zeros_like(rho)
        Rk = torch.fft.fftn(rho)
        n = self.N
        kx = torch.fft.fftfreq(n, d=self.dx).to(self.device) * (2.0*math.pi)
        ky = kx; kz = kx
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing="ij")
        k2 = KX*KX + KY*KY + KZ*KZ
        green = torch.zeros_like(k2)
        mask = k2 > 1e-12
        green[mask] = 1.0 / k2[mask]
        phi_k = 4.0 * math.pi * Rk * green
        if self.coulomb_twopi:
            phi_k *= (8.0 * math.pi)
        return torch.fft.ifftn(phi_k).real

    def coulomb_cross_energy(self, rhoN: torch.Tensor, rhoE: torch.Tensor) -> torch.Tensor:
        if self.coulomb_mode != "spectral" or self.alpha_coul == 0.0:
            return rhoN.new_tensor(0.0)
        phi_e = self._spectral_phi(rhoE)
        phi_n = self._spectral_phi(rhoN)
        return self.alpha_coul * 0.5 * ((rhoN * phi_e + rhoE * phi_n).sum() * self.dV)

    def mass_penalty(self, rho: torch.Tensor, target: float, penalty: float) -> torch.Tensor:
        if penalty == 0.0:
            return rho.new_tensor(0.0)
        total = rho.sum() * self.dV
        return penalty * (total - target)**2

    def projections(self) -> None:
        if not (self.project_mass_each or self.project_e_each):
            return
        with torch.no_grad():
            if self.project_mass_each:
                rho_N = self.nucleon_density()
                total = float(rho_N.sum() * self.dV)
                if total > 1e-12:
                    scale = math.sqrt(self.A / total)
                    self.psi_N.mul_(scale)
                    self.B_N.mul_(scale)
            if self.project_e_each:
                rho_e = self.electron_density()
                total = float(rho_e.sum() * self.dV)
                if total > 1e-12:
                    scale = math.sqrt(self.Z / total)
                    self.psi_e.mul_(scale)

    def nucleon_density(self) -> torch.Tensor:
        return self.psi_N*self.psi_N + (self.B_N*self.B_N).sum(dim=0)

    def electron_density(self) -> torch.Tensor:
        return self.psi_e*self.psi_e

    def symmetry_energy(self) -> torch.Tensor:
        """
        QFD Charge Asymmetry Energy: E_sym = c_sym × (N-Z)² / A^(1/3)

        Penalizes deviation from charge-balanced soliton configurations.
        Critical for predicting correct stability trends in QFD field theory.

        NOT based on nucleon counting - this is a soliton field effect.
        The A^(1/3) scaling reflects surface-dominated field dynamics,
        not bulk nucleon interactions.

        Parameters:
        - N = A - Z (number of neutral field quanta in configuration)
        - Z = charged field quanta
        - N-Z = charge asymmetry parameter
        - c_sym ≈ 20-30 MeV (to be calibrated against experimental data)

        Large |N-Z| → high field energy cost for maintaining asymmetric configuration
        Example: Fe-56 (N=30, Z=26, N-Z=4) vs Fe-45 (N=19, Z=26, N-Z=-7)
                 Fe-56 has lower charge asymmetry → lower V_sym → more stable
        """
        if self.c_sym == 0.0:
            return torch.tensor(0.0, device=self.device)

        N = self.A - self.Z  # Neutral field component count
        A13 = max(1.0, self.A) ** (1.0 / 3.0)

        # Charge asymmetry penalty: quadratic in (N-Z)
        # Surface scaling: / A^(1/3) for field boundary effects
        E_sym = self.c_sym * (N - self.Z)**2 / A13

        return torch.tensor(E_sym, device=self.device)

    def energies(self) -> Dict[str, torch.Tensor]:
        psiN = self.psi_N
        psiE = self.psi_e
        rhoN = self.nucleon_density()
        rhoE = self.electron_density()

        T_N = self.kinetic_scalar(psiN, self.kN)
        T_e = self.kinetic_scalar(psiE, self.ke)
        T_rotor = self.rotor_terms.T_rotor(self.B_N)
        V4_N, V6_N = self.potential_from_density(rhoN, self.alpha_eff, self.beta_eff)
        V4_e, V6_e = self.potential_from_density(rhoE, self.alpha_e, self.beta_e)

        rho_tot = rhoN
        V_surf = self.surface_energy(rho_tot)

        V_coul = self.coulomb_cross_energy(rhoN, rhoE)

        V_mass_N = self.mass_penalty(rhoN, float(self.A), self.mass_penalty_N)
        V_mass_e = self.mass_penalty(rhoE, float(self.Z), self.mass_penalty_e)

        V_rotor = self.rotor_terms.V_rotor(self.B_N)

        # Symmetry energy: penalizes N-Z asymmetry
        V_sym = self.symmetry_energy()

        return dict(
            T_N=T_N, T_e=T_e, T_rotor=T_rotor,
            V4_N=V4_N, V6_N=V6_N, V4_e=V4_e, V6_e=V6_e,
            V_iso=torch.tensor(0.0, device=self.device),
            V_rotor=V_rotor,
            V_surf=V_surf,
            V_coul_cross=V_coul,
            V_mass_N=V_mass_N, V_mass_e=V_mass_e,
            V_sym=V_sym,
        )

    def virial(self, energies: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = sum(energies.values())
        kinetic = energies["T_N"] + energies["T_e"] + energies["T_rotor"]
        return 2.0 * kinetic + total

# ---- SCF loop ------------------------------------------------------------

def scf_minimize(model: Phase8Model, iters_outer:int=360, lr_psi:float=1e-2, lr_B:float=1e-2,
                 early_stop_vir:float=0.2, verbose:bool=False) -> tuple[Dict[str,float], float, Dict[str,torch.Tensor]]:
    optim = torch.optim.Adam([
        {"params": [model.psi_N], "lr": lr_psi},
        {"params": [model.psi_e], "lr": lr_psi},
        {"params": [model.B_N], "lr": lr_B},
    ])
    best = dict(E=float("inf"), vir=float("inf"))
    best_state = None
    best_energies = None
    for it in range(1, iters_outer+1):
        optim.zero_grad()
        energies = model.energies()
        total = sum(energies.values())
        vir = model.virial(energies)
        loss = total + 10.0 * vir*vir
        loss.backward()
        optim.step()
        model.projections()
        e_val = float(total.detach())
        vir_val = float(vir.detach())
        if abs(vir_val) < abs(best.get("vir", float("inf"))):
            best = dict(E=e_val, vir=vir_val)
            best_state = [
                model.psi_N.detach().clone(),
                model.psi_e.detach().clone(),
                model.B_N.detach().clone(),
            ]
            best_energies = {k: v.detach().clone() for k, v in energies.items()}
        if verbose and it % 60 == 0:
            print(f"[{it:04d}] E={e_val:+.6e} |vir|={abs(vir_val):.3f}")
        if abs(vir_val) <= early_stop_vir:
            break
    if best_state is not None:
        with torch.no_grad():
            model.psi_N.copy_(best_state[0])
            model.psi_e.copy_(best_state[1])
            model.B_N.copy_(best_state[2])
    return best, best.get("vir", float("nan")), best_energies or energies

# ---- CLI -----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Phase-8 SCF solver (v4 - Merged from ChatSolver)")
    p.add_argument("--A", type=int, required=True)
    p.add_argument("--Z", type=int, required=True)
    p.add_argument("--c-v2-base", type=float, default=2.2)
    p.add_argument("--c-v2-iso",  type=float, default=0.027)
    p.add_argument("--c-v2-mass", type=float, default=0.0,
                   help="Mass-dependent cohesion coefficient")
    p.add_argument("--c-v4-base", type=float, default=5.28)
    p.add_argument("--c-v4-size", type=float, default=-0.085)
    p.add_argument("--alpha-model", choices=["exp","linear"], default="exp",
                   help="Choose exponential (default) or linear cohesion scaling")
    p.add_argument("--lambda-R2", type=float, default=3e-4)
    p.add_argument("--lambda-R3", type=float, default=1e-3)
    p.add_argument("--B-target",  type=float, default=0.01)
    p.add_argument("--grid-points", type=int, default=48)
    p.add_argument("--dx", type=float, default=1.0)
    p.add_argument("--iters-outer", type=int, default=600)
    p.add_argument("--lr-psi", type=float, default=1e-2)
    p.add_argument("--lr-B",   type=float, default=1e-2)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--early-stop-vir", type=float, default=0.20)
    p.add_argument("--coulomb", choices=["off","spectral"], default="spectral")
    p.add_argument("--coulomb-charge-mode", choices=["gemini","simple"], default="gemini")
    p.add_argument("--alpha-coul", type=float, default=1.0)
    p.add_argument("--alpha0", type=float, default=None)
    p.add_argument("--gamma", type=float, default=0.8)
    p.add_argument("--kappa-rho", type=float, default=0.0)
    p.add_argument("--kappa-rho0", type=float, default=None)
    p.add_argument("--kappa-rho-exp", type=float, default=1.0)
    p.add_argument("--alpha-e-scale", type=float, default=0.5)
    p.add_argument("--beta-e-scale", type=float, default=0.5)
    p.add_argument("--c-sym", type=float, default=0.0,
                   help="QFD charge asymmetry coefficient (MeV). Penalizes deviation from "
                        "charge-balanced soliton configurations. Typical values: 20-30. "
                        "E_sym = c_sym × (N-Z)² / A^(1/3). NOT nucleon-based - field effect.")
    p.add_argument("--mass-penalty-N", type=float, default=0.0)
    p.add_argument("--mass-penalty-e", type=float, default=0.0)
    p.add_argument("--project-mass-each", action="store_true")
    p.add_argument("--project-e-each", action="store_true")
    p.add_argument("--lambda-scale", type=float, default=650.0)
    p.add_argument("--tol", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--init", choices=["gauss","random"], default="gauss")
    p.add_argument("--emit-json", action="store_true")
    p.add_argument(
        "--out-json", type=str, default="",
        help="Write solver output to this JSON path (requires --emit-json)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--meta-json", type=str, default=None, help="Path to a meta parameters JSON file.")
    p.add_argument("--coulomb-legacy", choices=["current","tol015"], default="current",
                   help="Reproduce tol015 spectral Coulomb scaling (adds 2π factor).")
    return p.parse_args()

# ---- JSON helpers --------------------------------------------------------

def write_json(payload: Dict[str, Any], path: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))

# ---- Main ----------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t_start = time.time()

    if args.meta_json:
        print(f"Loading meta parameters from: {args.meta_json}")
        with open(args.meta_json, 'r') as f:
            payload = json.load(f)
        required = {"alpha0", "gamma", "kappa_rho0", "alpha_e_scale", "c_v2_mass"}
        if payload.get("schema") != "qfd.merged5.v1" or not required.issubset(payload):
            raise ValueError("Meta JSON is not a valid 'qfd.merged5.v1' schema; refusing to proceed.")
        
        # Override args with values from JSON
        args.alpha0 = payload.get("alpha0", args.alpha0)
        args.gamma = payload.get("gamma", args.gamma)
        args.kappa_rho0 = payload.get("kappa_rho0", args.kappa_rho0)
        args.alpha_e_scale = payload.get("alpha_e_scale", args.alpha_e_scale)
        args.c_v2_mass = payload.get("c_v2_mass", args.c_v2_mass)
        decoder = payload.get("decoder", {})
        if decoder:
            args.coulomb_legacy = decoder.get("coulomb_legacy", args.coulomb_legacy)
            if "alpha_model" in decoder and args.alpha_model == "exp":
                # prefer decoder's alpha model when user left default
                args.alpha_model = decoder["alpha_model"]

    try:
        torch_det_seed(args.seed)
        device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

        alpha_coul_used = args.alpha_coul
        if args.alpha0 is not None:
            alpha_coul_used = args.alpha0 * (args.Z / args.A) ** args.gamma

        kappa_rho_used = args.kappa_rho
        if args.kappa_rho0 is not None:
            A23 = args.A ** (2.0/3.0)
            kappa_rho_used = args.kappa_rho0 * (A23 ** args.kappa_rho_exp)

        rotor = RotorParams(args.lambda_R2, args.lambda_R3, args.B_target)
        model = Phase8Model(
            A=args.A, Z=args.Z, grid=args.grid_points, dx=args.dx,
            c_v2_base=args.c_v2_base, c_v2_iso=args.c_v2_iso, c_v2_mass=args.c_v2_mass,
            c_v4_base=args.c_v4_base, c_v4_size=args.c_v4_size,
            rotor=rotor, device=str(device),
            coulomb_mode=args.coulomb, alpha_coul=alpha_coul_used,
            mass_penalty_N=args.mass_penalty_N, mass_penalty_e=args.mass_penalty_e,
            project_mass_each=args.project_mass_each, project_e_each=args.project_e_each,
            kappa_rho=kappa_rho_used,
            alpha_e_scale=args.alpha_e_scale, beta_e_scale=args.beta_e_scale,
            alpha_model=args.alpha_model,
            coulomb_twopi=(args.coulomb_legacy == "tol015"),
            c_sym=args.c_sym,
        )
        model.initialize_fields(seed=args.seed, init_mode=args.init)

        br, vir, energy_terms = scf_minimize(
            model,
            iters_outer=args.iters_outer,
            lr_psi=args.lr_psi,
            lr_B=args.lr_B,
            early_stop_vir=args.early_stop_vir,
            verbose=args.verbose,
        )
        with torch.no_grad():
            energy_terms = {k: float(v.detach()) for k, v in model.energies().items()}
        elapsed = time.time() - t_start

        payload = {
            "status": "ok",
            "A": int(args.A),
            "Z": int(args.Z),
            "alpha_eff": float(model.alpha_eff),
            "beta_eff": float(model.beta_eff),
            "coeffs_raw": model.coeffs_raw,
            "E_model": float(br["E"]),
            "virial": float(vir),
            "virial_abs": float(abs(vir)),
            "coulomb": args.coulomb,
            "coulomb_charge_mode": args.coulomb_charge_mode,
            "alpha_coul": float(alpha_coul_used),
            "alpha0": args.alpha0,
            "gamma": float(args.gamma),
            "c_v2_base": float(args.c_v2_base),
            "c_v2_iso": float(args.c_v2_iso),
            "c_v2_mass": float(args.c_v2_mass),
            "c_v4_base": float(args.c_v4_base),
            "c_v4_size": float(args.c_v4_size),
            "kappa_rho": float(kappa_rho_used),
            "kappa_rho0": args.kappa_rho0,
            "kappa_rho_exp": float(args.kappa_rho_exp),
            "alpha_e_scale": float(args.alpha_e_scale),
            "beta_e_scale": float(args.beta_e_scale),
            "c_sym": float(args.c_sym),
            "alpha_model": args.alpha_model,
            "coulomb_legacy": args.coulomb_legacy,
            "lambda_R2": float(args.lambda_R2),
            "lambda_R3": float(args.lambda_R3),
            "B_target": float(args.B_target),
            "grid_points": args.grid_points,
            "dx": float(args.dx),
            "iters_outer": args.iters_outer,
            "lr_psi": float(args.lr_psi),
            "lr_B": float(args.lr_B),
            "seed": int(args.seed),
            "init": args.init,
            "tol": float(args.tol),
            "elapsed_sec": float(elapsed),
            "physical_success": bool((br["E"] < 0.0) and (abs(vir) <= args.tol)),
        }
        payload.update({k: energy_terms.get(k) for k in [
            "T_N","T_e","T_rotor","V4_N","V6_N","V4_e","V6_e",
            "V_iso","V_rotor","V_surf","V_coul_cross","V_mass_N","V_mass_e","V_sym"
        ]})
        if args.emit_json or args.out_json:
            write_json(payload, args.out_json)
        else:
            print(json.dumps(payload, indent=2))
    except Exception as exc:
        err = {
            "status": "error",
            "error": str(exc),
        }
        if args.emit_json or args.out_json:
            write_json(err, args.out_json)
        else:
            print(json.dumps(err, indent=2))
        raise

if __name__ == "__main__":
    main()
