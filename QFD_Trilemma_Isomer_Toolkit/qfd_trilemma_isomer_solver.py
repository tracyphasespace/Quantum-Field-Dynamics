
from __future__ import annotations
import math
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import torch
torch.set_default_dtype(torch.float64)

@dataclass
class Couplings:
    kC: float = 1.0
    k_grad: float = 1.0
    k_mass: float = 0.05
    k_area: float = 0.0
    lambda_stab: float = 0.7
    alpha_h: float = 0.6
    eps_sat: float = 1.0
    shape_area_coeff: float = 0.6
    shape_cap_coeff: float = 0.5

@dataclass
class SearchConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    nr: int = 384
    r_pad: float = 6.0
    s_max: float = 0.6
    lbfgs_steps: int = 120
    starts: int = 8
    seed: int = 1234

@dataclass
class Theta:
    uR: torch.Tensor
    uT: torch.Tensor
    uPsi0: torch.Tensor
    uS: torch.Tensor

    @staticmethod
    def from_values(R: float, T: float, Psi0: float, s: float, s_max: float, device: str):
        def inv_softplus(x):
            xe = torch.tensor(x, dtype=torch.float64, device=device)
            return torch.log(torch.expm1(xe.clamp(min=1e-8)))
        def inv_tanh(x):
            val = max(min(x / s_max, 0.999999), -0.999999)
            return 0.5*math.log((1+val)/(1-val))
        return Theta(
            uR=inv_softplus(R),
            uT=inv_softplus(T),
            uPsi0=inv_softplus(Psi0),
            uS=torch.tensor(inv_tanh(s), dtype=torch.float64, device=device)
        ).to(device)

    def to(self, device: str):
        self.uR = self.uR.to(device)
        self.uT = self.uT.to(device)
        self.uPsi0 = self.uPsi0.to(device)
        self.uS = self.uS.to(device)
        return self

    def clamp_copy(self):
        sp = torch.nn.Softplus(beta=1.0, threshold=20.0)
        R = sp(self.uR) + 1e-9
        T = sp(self.uT) + 1e-9
        Psi0 = sp(self.uPsi0) + 1e-12
        return R, T, Psi0

class QFDTrilemmaModel:
    def __init__(self, couplings: Couplings, cfg: SearchConfig):
        self.c = couplings
        self.cfg = cfg
        self.device = torch.device(cfg.device)

    def _shape_factors(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_area = 1.0 + self.c.shape_area_coeff * (s*s)
        f_cap = 1.0 + self.c.shape_cap_coeff * (s*s)
        return f_area, f_cap

    def _psi_shell(self, r: torch.Tensor, R: torch.Tensor, T: torch.Tensor, Psi0: torch.Tensor) -> torch.Tensor:
        x = (r - R) / T
        return Psi0 * torch.exp(-x*x)

    def _eps_density(self, r: torch.Tensor, psi: torch.Tensor, dpsi_dr: torch.Tensor) -> torch.Tensor:
        return self.c.k_grad * (dpsi_dr**2) + self.c.k_mass * (psi**2)

    def _h_of_eps(self, eps: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.c.alpha_h * (eps / (1.0 + eps / self.c.eps_sat))

    def _integrate_radial(self, r: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        dr = r[1] - r[0]
        integrand = 4.0*math.pi * (r*r) * f
        return torch.trapz(integrand, dx=dr)

    def energy_terms(self, Q: float, theta: Theta) -> Dict[str, torch.Tensor]:
        device = self.device
        s_max = self.cfg.s_max

        R, T, Psi0 = theta.clamp_copy()
        s = torch.tanh(theta.uS) * s_max

        r_max = R + self.cfg.r_pad * T
        nr = self.cfg.nr
        r = torch.linspace(0.0, r_max.item(), nr, dtype=torch.float64, device=device)
        r = torch.clamp(r, min=1e-12)

        psi = self._psi_shell(r, R, T, Psi0)
        dr = r[1]-r[0]
        dpsi_dr = torch.zeros_like(psi)
        dpsi_dr[1:-1] = (psi[2:] - psi[:-2]) / (2*dr)
        dpsi_dr[0] = (psi[1] - psi[0]) / dr
        dpsi_dr[-1] = (psi[-1] - psi[-2]) / dr

        eps = self._eps_density(r, psi, dpsi_dr)
        h = self._h_of_eps(eps)

        f_area, f_cap = self._shape_factors(s)

        E_coul = self.c.kC * (Q**2) / (R * f_cap)
        E_surf = self._integrate_radial(r, self.c.k_grad * (dpsi_dr**2))
        E_area = self.c.k_area * (R*R) * f_area
        E_mass = self._integrate_radial(r, self.c.k_mass * (psi**2))
        E_quag = - self.c.lambda_stab * self._integrate_radial(r, (h - 1.0)**2)

        E_total = E_coul + E_surf + E_area + E_mass + E_quag

        return {
            "E_coul": E_coul,
            "E_surf": E_surf,
            "E_area": E_area,
            "E_mass": E_mass,
            "E_quag": E_quag,
            "E_total": E_total,
            "R": R, "T": T, "Psi0": Psi0, "s": s,
        }

    def _random_theta(self, rng: torch.Generator) -> Theta:
        R0 = float(torch.rand(1, generator=rng) * 4.0 + 0.8)
        T0 = float(torch.rand(1, generator=rng) * 0.9 + 0.06)
        Psi0 = float(torch.rand(1, generator=rng) * 2.0 + 0.5)
        s0 = float((torch.rand(1, generator=rng) - 0.5) * 2.0 * 0.4)
        return Theta.from_values(R0, T0, Psi0, s0, self.cfg.s_max, device=str(self.device))

    def minimize(self, Q: float, theta: Theta) -> Tuple[float, Dict[str, float]]:
        theta = theta
        params = [theta.uR.clone().detach().requires_grad_(True),
                  theta.uT.clone().detach().requires_grad_(True),
                  theta.uPsi0.clone().detach().requires_grad_(True),
                  theta.uS.clone().detach().requires_grad_(True)]
        opt = torch.optim.LBFGS(params, lr=0.7, max_iter=self.cfg.lbfgs_steps, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            th = Theta(params[0], params[1], params[2], params[3])
            terms = self.energy_terms(Q, th)
            loss = terms["E_total"]
            loss.backward()
            return loss

        final_loss = opt.step(closure)
        th = Theta(params[0].detach(), params[1].detach(), params[2].detach(), params[3].detach())
        terms = self.energy_terms(Q, th)
        out = {k: float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v) for k,v in terms.items()}
        return float(terms["E_total"].detach().cpu().item()), out

    def multistart_isomers(self, Q: float, starts: int | None = None) -> List[Dict[str, float]]:
        starts = starts or self.cfg.starts
        rng = torch.Generator(device=self.device).manual_seed(self.cfg.seed)

        minima: List[Dict[str, float]] = []
        seen: List[Tuple[float, float, float, float]] = []

        def is_duplicate(o: Dict[str, float]) -> bool:
            key = (round(o["R"], 3), round(o["T"], 3), round(o["Psi0"], 3), round(o["s"], 3))
            for k in seen:
                if key == k:
                    return True
            seen.append(key)
            return False

        for i in range(starts):
            th0 = self._random_theta(rng)
            E, desc = self.minimize(Q, th0)
            if not is_duplicate(desc):
                minima.append(desc)

        minima.sort(key=lambda d: d["E_total"])
        return minima

def run_demo():
    Q = 43.0  # Technetium Z
    couplings = Couplings(
        kC=0.06,
        k_grad=1.0,
        k_mass=0.04,
        k_area=0.02,
        lambda_stab=0.9,
        alpha_h=0.8,
        eps_sat=1.2,
        shape_area_coeff=0.6,
        shape_cap_coeff=0.5,
    )
    cfg = SearchConfig(nr=320, lbfgs_steps=80, starts=8)
    model = QFDTrilemmaModel(couplings, cfg)

    minima = model.multistart_isomers(Q=Q, starts=cfg.starts)

    calib = {}
    if len(minima) >= 2:
        dE_model = minima[1]["E_total"] - minima[0]["E_total"]
        target_keV = 142.0
        scale_keV_per_unit = target_keV / dE_model if dE_model != 0 else float('nan')
        calib = {
            "target_transition_keV": target_keV,
            "dE_model_units": dE_model,
            "keV_per_model_unit": scale_keV_per_unit,
            "predicted_levels_keV": [ (m["E_total"] - minima[0]["E_total"]) * scale_keV_per_unit for m in minima]
        }

    result = {
        "Q": Q,
        "device": str(model.device),
        "couplings": asdict(couplings),
        "config": asdict(cfg),
        "minima": minima,
        "calibration": calib
    }
    out_path = "/mnt/data/qfd_trilemma_demo_tc99m.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Pretty print
    lines = []
    lines.append("Found minima (sorted by energy):")
    for i, m in enumerate(minima[:6]):
        lines.append(f"  [{i}] E_total={m['E_total']:.6f}  R={m['R']:.3f}  T={m['T']:.3f}  s={m['s']:.3f}"
                     f"  | E_coul={m['E_coul']:.6f}  E_quag={m['E_quag']:.6f}")
    if calib:
        lines.append("\\nCalibration against Tc-99m (≈142 keV):")
        lines.append(f"  Model ΔE_01 = {calib['dE_model_units']:.6f} model-units")
        lines.append(f"  Scale: 1 model-unit ≈ {calib['keV_per_model_unit']:.3f} keV")
        lines.append("  Predicted excited-state ladder (keV above ground):")
        for i, ek in enumerate(calib["predicted_levels_keV"][:6]):
            lines.append(f"    Level {i}: {ek:.2f} keV")

    return out_path, "\\n".join(lines)

if __name__ == "__main__":
    p, text = run_demo()
    print(text)
    print(f"\\nJSON saved: {p}")
