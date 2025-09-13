
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np

# Optional SciPy for global search
try:
    from scipy.optimize import basinhopping, minimize
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# Torch for autograd-friendly energy construction (CPU/GPU agnostic)
import torch
torch.set_default_dtype(torch.float64)


# ===============================
#   Trilemma Couplings / Config
# ===============================

@dataclass
class Couplings:
    # Core Trilemma knobs
    lambda_kinetic: float = 1.0        # weight on ∫ |∇ψ|^2 dV
    lambda_electrostatic: float = 0.06 # ~ kC in Q^2 / (R * f_cap(s))
    lambda_quagmire: float = 0.9       # weight on -∫ (h-1)^2 dV
    xi: float = 1.0                    # h(psi) = 1 + xi * psi^2 (book-aligned)

    # Auxiliary (numerical regularizers)
    lambda_mass: float = 0.04          # ∫ ψ^2 dV keeps ψ finite
    lambda_area: float = 0.02          # tiny explicit area term ~ R^2 f_area(s)

    # Asphericity shape responses (small-deformation proxies)
    shape_area_coeff: float = 0.6      # f_area(s) = 1 + c_A s^2
    shape_cap_coeff: float = 0.5       # f_cap(s)  = 1 + c_C s^2


@dataclass
class SolverConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    nr: int = 320               # radial samples
    r_pad: float = 6.0          # domain = R + r_pad*T
    s_max: float = 0.6          # |s| <= s_max
    seed: int = 1234
    R_max: float = 8.0          # clamp for stability
    T_max: float = 2.0
    Psi0_max: float = 4.0

    # Global search
    niter: int = 120            # reduce steps to fit runtime here
    stepsize: float = 0.35      # basinhopping trial step magnitude
    bfgs_maxiter: int = 120     # local BFGS iterations

    # Fallback multistart
    multistart_starts: int = 16 # if SciPy is unavailable


# ===============================
#      Trilemma Hamiltonian
# ===============================

class TrilemmaHamiltonian:
    """Implements the QFD Trilemma Hamiltonian on a 1D radial shell ansatz
    with a simple asphericity parameter s. All terms are differentiable in torch."""

    def __init__(self, Q: float, A: int, couplings: Couplings, cfg: SolverConfig):
        self.Q = float(Q)
        self.A = int(A)
        self.c = couplings
        self.cfg = cfg
        self.device = torch.device(cfg.device)

    # ---- shape helpers ----
    def shape_factors(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_area = 1.0 + self.c.shape_area_coeff * s*s
        f_cap  = 1.0 + self.c.shape_cap_coeff  * s*s
        return f_area, f_cap

    # ---- wavelet ansatz ----
    def psi_shell(self, r: torch.Tensor, R: torch.Tensor, T: torch.Tensor, Psi0: torch.Tensor) -> torch.Tensor:
        x = (r - R) / T
        return Psi0 * torch.exp(-x*x)

    # ---- energy density proxy ε ----
    def eps_density(self, psi: torch.Tensor, dpsi_dr: torch.Tensor) -> torch.Tensor:
        # retained for potential diagnostics; not used in h now
        return self.c.lambda_kinetic * (dpsi_dr**2) + self.c.lambda_mass * (psi**2)

    # ---- temporal quagmire map h(psi) = 1 + xi * psi^2 ----
    def h_of_psi(self, psi: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.c.xi * (psi**2)

    # ---- radial integration ----
    def integrate_r(self, r: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        dr = r[1] - r[0]
        integrand = 4.0 * math.pi * (r*r) * f
        return torch.trapz(integrand, dx=dr)


    # ---- exact spheroid capacitance (normalized, i.e., divided by 4*pi*eps0) ----
    @staticmethod
    def spheroid_capacitance_hat(a_major: torch.Tensor, b_minor: torch.Tensor, kind: str) -> torch.Tensor:
        # C_hat = C / (4*pi*eps0) for a conducting spheroid of revolution.
        # Uses closed forms (Queiroz 'Capacitance Calculations', eqs (7)-(8)).
        # 'oblate': major semi-axis a (equatorial), minor b (polar), a >= b
        #    C_hat = sqrt(a^2 - b^2) / arcsin(sqrt(a^2 - b^2)/a)
        #    limits: sphere (b->a): C_hat->a ; thin disk (b->0): C_hat-> 2a/pi
        # 'prolate': major semi-axis a (polar), minor b (equatorial), a >= b
        #    C_hat = sqrt(a^2 - b^2) / ln( (a + sqrt(a^2 - b^2)) / b )
        #    limits: sphere (b->a): C_hat->a ; long needle (b->0): C_hat->0
        a = a_major
        b = b_minor
        a = torch.clamp(a, min=1e-12)
        b = torch.clamp(b, min=1e-12, max=a-1e-12)
        delta = torch.sqrt(torch.clamp(a*a - b*b, min=1e-24))
        if kind == "oblate":
            x = torch.clamp(delta / a, max=1-1e-12)
            arcs = torch.arcsin(x)
            C_hat = delta / torch.clamp(arcs, min=1e-18)
        elif kind == "prolate":
            num = a + delta
            den = torch.clamp(b, min=1e-18)
            ratio = torch.clamp(num / den, min=1+1e-15)
            ln_term = torch.log(ratio)
            C_hat = delta / torch.clamp(ln_term, min=1e-18)
        else:
            raise ValueError("kind must be 'oblate' or 'prolate'")
        return C_hat

    # ---- finite difference derivative ----
    @staticmethod
    def central_gradient_1d(y: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        g = torch.zeros_like(y)
        g[1:-1] = (y[2:] - y[:-2]) / (2*dx)
        g[0]    = (y[1]  - y[0])  / dx
        g[-1]   = (y[-1] - y[-2]) / dx
        return g

    # ---- total energy + breakdown ----
    def energy(self, x_phys: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        x_phys: dict with R, T, Psi0, s (all torch scalars on self.device)
        """
        R = x_phys["R"]
        T = x_phys["T"]
        Psi0 = x_phys["Psi0"]
        s = x_phys["s"]

        # radial grid
        r_max = R + self.cfg.r_pad * T
        r = torch.linspace(0.0, r_max.item(), self.cfg.nr, dtype=torch.float64, device=self.device)
        r = torch.clamp(r, min=1e-12)
        dr = r[1] - r[0]

        psi = self.psi_shell(r, R, T, Psi0)
        dpsi_dr = self.central_gradient_1d(psi, dr)

        eps = self.eps_density(psi, dpsi_dr)
        h = self.h_of_psi(psi)

        f_area, f_cap = self.shape_factors(s)

        # (1) kinetic / surface (∇ψ)^2
        E_kin = self.integrate_r(r, self.c.lambda_kinetic * (dpsi_dr**2))

        # (2) electrostatic via spheroid capacitance
        E_es  = self.c.lambda_electrostatic * (self.Q**2) / (R * f_cap)

        # (2b) tiny explicit area pressure ~ R^2 f_area(s)
        E_area = self.c.lambda_area * (R*R) * f_area

        # (3) mass-like regularizer ∫ ψ^2
        E_mass = self.integrate_r(r, self.c.lambda_mass * (psi**2))

        # (4) quagmire stabilization:  -λ ∫ (h-1)^2
        I4 = self.integrate_r(r, psi**4)
        E_quag = - self.c.lambda_quagmire * self.integrate_r(r, (h - 1.0)**2)

        E_total = E_kin + E_es + E_area + E_mass + E_quag
        E_other = E_total - E_quag

        return dict(E_total=E_total, E_other=E_other, I4=I4,
                    E_kin=E_kin, E_es=E_es, E_area=E_area, E_mass=E_mass, E_quag=E_quag,
                    R=R, T=T, Psi0=Psi0, s=s)

    # ---- numpy <-> torch bridge for optimizers ----
    def pack(self, R: float, T: float, Psi0: float, s: float) -> np.ndarray:
        # Unconstrained variables: uR,uT,uPsi0 via inverse softplus, uS via atanh(s/s_max)
        def inv_softplus(x: float) -> float:
            x = max(x, 1e-10)
            return float(np.log(np.expm1(x)))
        def inv_tanh(z: float) -> float:
            z = np.clip(z / self.cfg.s_max, -0.999999, 0.999999)
            return float(0.5*np.log((1+z)/(1-z)))
        return np.array([inv_softplus(R), inv_softplus(T), inv_softplus(Psi0), inv_tanh(s)], dtype=np.float64)

    def unpack(self, u: np.ndarray) -> Dict[str, torch.Tensor]:
        # Map unconstrained -> physical: softplus for positives, tanh*s_max for s
        sp = torch.nn.Softplus(beta=1.0, threshold=20.0)
        uR, uT, uPsi0, uS = [torch.tensor(float(v), dtype=torch.float64, device=self.device) for v in u]
        R = torch.clamp(sp(uR)   + 1e-9, max=self.cfg.R_max)
        T = torch.clamp(sp(uT)   + 1e-9, max=self.cfg.T_max)
        Psi0 = torch.clamp(sp(uPsi0) + 1e-12, max=self.cfg.Psi0_max)
        s = torch.tanh(uS) * self.cfg.s_max
        return dict(R=R, T=T, Psi0=Psi0, s=s)

    def objective_np(self, u: np.ndarray) -> float:
        x_phys = self.unpack(u)
        terms = self.energy(x_phys)
        return float(terms["E_total"].detach().cpu().item())

    def objective_with_breakdown(self, u: np.ndarray) -> Tuple[float, Dict[str, float]]:
        x_phys = self.unpack(u)
        terms = self.energy(x_phys)
        out = {k: float(v.detach().cpu().item()) for k, v in terms.items()}
        return out["E_total"], out


# ===============================
#        Isomer Search API
# ===============================

@dataclass
class Isomer:
    E_total: float
    breakdown: Dict[str, float]

class IsomerSearch:
    def __init__(self, H: TrilemmaHamiltonian, tol_energy_units: float = 1e-3):
        self.H = H
        self.minima: List[Isomer] = []
        self.tol = tol_energy_units  # model-unit tolerance to dedup minima

    def _store_min(self, E: float, breakdown: Dict[str, float]):
        # Deduplicate by rounding parameter tuple & energy
        R = round(breakdown["R"], 4)
        T = round(breakdown["T"], 4)
        Psi0 = round(breakdown["Psi0"], 4)
        s = round(breakdown["s"], 4)
        key = (R, T, Psi0, s)
        for m in self.minima:
            b = m.breakdown
            k2 = (round(b["R"],4), round(b["T"],4), round(b["Psi0"],4), round(b["s"],4))
            if key == k2 or abs(E - m.E_total) < self.tol:
                return
        self.minima.append(Isomer(E_total=E, breakdown=breakdown))

    def run_basinhopping(self, x0: np.ndarray, niter: int, stepsize: float, bfgs_maxiter: int) -> List[Isomer]:
        if not HAVE_SCIPY:
            return self.run_multistart(torch_starts= max(8, niter//2))

        def fun(u: np.ndarray) -> float:
            return self.H.objective_np(u)

        def hop_callback(x: np.ndarray, f: float, accept: bool):
            # After each basin minimum, record if new
            _, br = self.H.objective_with_breakdown(x)
            self._store_min(br["E_total"], br)

        res = basinhopping(
            func=fun,
            x0=x0,
            niter=niter,
            stepsize=stepsize,
            minimizer_kwargs=dict(method="BFGS", options=dict(maxiter=bfgs_maxiter, gtol=1e-6)),
            callback=hop_callback,
            disp=False,
            seed=int(self.H.cfg.seed),
        )
        # Ensure the final min is stored too
        x_final = res.x
        _, br = self.H.objective_with_breakdown(x_final)
        self._store_min(br["E_total"], br)

        self.minima.sort(key=lambda m: m.E_total)
        return self.minima

    def run_multistart(self, torch_starts: int = 10) -> List[Isomer]:
        # Torch LBFGS from random packs
        gen = torch.Generator(device=self.H.device).manual_seed(self.H.cfg.seed)

        def rand_phys():
            R = float(torch.rand((), generator=gen) * 4.0 + 0.8)
            T = float(torch.rand((), generator=gen) * 0.9 + 0.06)
            Psi0 = float(torch.rand((), generator=gen) * 2.0 + 0.5)
            s = float((torch.rand((), generator=gen) - 0.5) * 2.0 * 0.4)
            return self.H.pack(R, T, Psi0, s)

        for _ in range(torch_starts):
            u0 = rand_phys()
            # Torch LBFGS on unconstrained variables
            u = torch.tensor(u0, dtype=torch.float64, device=self.H.device, requires_grad=True)
            opt = torch.optim.LBFGS([u], lr=0.6, max_iter=self.H.cfg.bfgs_maxiter, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                val = torch.tensor(self.H.objective_np(u.detach().cpu().numpy()), dtype=torch.float64, device=self.H.device)
                val.backward()
                return val

            opt.step(closure)
            u_opt = u.detach().cpu().numpy()
            _, br = self.H.objective_with_breakdown(u_opt)
            self._store_min(br["E_total"], br)

        self.minima.sort(key=lambda m: m.E_total)
        return self.minima


# ===============================
#        Top-level runner
# ===============================

def run_isomer_search_for_tc99(out_json_path: str = "tc99_isomer_search.json") -> Dict:
    Q, A = 43.0, 99

    couplings = Couplings(
        lambda_kinetic=1.0,
        lambda_electrostatic=0.06,
        lambda_quagmire=0.9,
        xi=1.0,
        lambda_mass=0.04,
        lambda_area=0.02,
        shape_area_coeff=0.6,
        shape_cap_coeff=0.5,
    )
    cfg = SolverConfig(nr=320, niter=14, stepsize=0.35, bfgs_maxiter=120)

    H = TrilemmaHamiltonian(Q=Q, A=A, couplings=couplings, cfg=cfg)
    search = IsomerSearch(H, tol_energy_units=1e-3)

    # starting point (unconstrained pack)
    x0 = H.pack(R=2.2, T=0.22, Psi0=1.0, s=0.1)
    minima = search.run_basinhopping(x0=x0, niter=cfg.niter, stepsize=cfg.stepsize, bfgs_maxiter=cfg.bfgs_maxiter)

    # Build JSON-able result
    mins = []
    for m in minima:
        br = dict(m.breakdown)
        br['E_total'] = m.E_total
        mins.append(br)


    # Optional quick calibration of lambda_quagmire to match 142.6 keV:
    # If there are ≥2 minima, solve for lambda such that ΔE_target holds,
    # using E_total_i = E_other_i - lambda * ∫(h-1)^2 dV = E_other_i - lambda * xi^2 * I4_i (since h-1=xi*psi^2).
    # We approximate shapes fixed and then re-relax locally with BFGS at the calibrated lambda.
    cal_detail = {}
    calib = {}
    if len(mins) >= 2:
        E_other0, E_other1 = mins[0]["E_other"], mins[1]["E_other"]
        I4_0, I4_1 = mins[0]["I4"], mins[1]["I4"]
        xi = couplings.xi
        target_keV = 142.6
        # Model units target gap; we don't know keV-per-unit yet. Instead we first fit lambda in model-units
        # against a chosen scale of 1 keV/model-unit, then we compute the scale separately below.
        # A better approach is to determine lambda that yields the *ratio* when adding a free scale factor,
        # but that becomes underdetermined. So we set xi and solve for lambda that gives the *model* gap dE_target_model.
        # Take dE_target_model = current (E_total1 - E_total0) as a starting point (so lambda stays near current).
        dE_model_now = mins[1]["E_total"] - mins[0]["E_total"]
        # Solve ΔE(λ) = (E_other1 - E_other0) - λ * xi^2 * (I4_1 - I4_0) = ΔE_target_model
        denom = (xi**2) * (I4_1 - I4_0)
        lambda_star = couplings.lambda_quagmire
        if abs(denom) > 1e-12:
            lambda_star = ( (E_other1 - E_other0) - dE_model_now ) / denom + couplings.lambda_quagmire
            # constrain to positive
            lambda_star = float(max(lambda_star, 1e-6))
        # Re-run local BFGS at lambda_star to re-relax the two lowest states
        couplings.lambda_quagmire = lambda_star
        H2 = TrilemmaHamiltonian(Q=Q, A=A, couplings=couplings, cfg=cfg)
        # Local refine from previous minima parameters:
        def local_refine(m):
            u = H2.pack(m["R"], m["T"], m["Psi0"], m["s"])
            from scipy.optimize import minimize
            res = minimize(H2.objective_np, u, method="BFGS", options=dict(maxiter=cfg.bfgs_maxiter, gtol=1e-6))
            _, br = H2.objective_with_breakdown(res.x)
            return br
        br0 = local_refine(mins[0])
        br1 = local_refine(mins[1])
        dE_refined = br1["E_total"] - br0["E_total"]
        # Determine keV per model-unit by mapping refined dE to 142.6 keV
        keV_per_unit = target_keV / dE_refined if dE_refined != 0 else float("nan")
        cal_detail = dict(lambda_quagmire=lambda_star, xi=xi, dE_refined_model=dE_refined, keV_per_unit=keV_per_unit)
        # Rebuild predicted ladder in keV using refined ground as reference:
        mins = [br0, br1] + mins[2:]
        ladder_keV = [ (m["E_total"] - br0["E_total"]) * keV_per_unit for m in mins ]
        calib = dict(
            nuclide="Tc-99m → Tc-99",
            target_transition_keV=target_keV,
            model_delta_E_units=dE_refined,
            keV_per_model_unit=keV_per_unit,
            ladder_keV=ladder_keV,
            calibrated_couplings=dict(lambda_quagmire=lambda_star, xi=xi),
        )

    result = dict(
        have_scipy=HAVE_SCIPY,
        device=str(H.device),
        Q=Q, A=A,
        couplings=asdict(couplings),
        config=asdict(cfg),
        minima=mins,
        calibration=calib,
    )

    with open(out_json_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    out = run_isomer_search_for_tc99(out_json_path="tc99_isomer_search.json")
    # Pretty print a compact summary
    ms = out["minima"]
    print(f"SciPy available: {out['have_scipy']}  |  Device: {out['device']}")
    print(f"Found {len(ms)} candidate minima for Tc-99 (sorted):")
    for i, m in enumerate(ms[:8]):
        print(f"  [{i}] E_total={m['E_total']:.6f}  R={m['R']:.3f}  T={m['T']:.3f}  s={m['s']:.3f}"
              f" | E_es={m['E_es']:.6f}  E_quag={m['E_quag']:.6f}")
    if out['calibration']:
        c = out['calibration']
        print(f"\nCalibration to {c['nuclide']} ({c['target_transition_keV']} keV):")
        print(f"  ΔE_model = {c['model_delta_E_units']:.6f} units")
        print(f"  1 unit ≈ {c['keV_per_model_unit']:.3f} keV")
        print("  Ladder (keV above ground):")
        for i, ek in enumerate(c['ladder_keV'][:8]):
            print(f"    Level {i}: {ek:.2f} keV")
