"""PPN and refractive-index interface.
This module centralizes how an effective refractive index n(x) maps to PPN {gamma, beta}.
Keep all physics mappings here so realms can import a single source of truth.
"""

from dataclasses import dataclass

@dataclass
class PPNTargets:
    gamma: float = 1.0
    beta: float = 1.0
    tol_gamma: float = 1e-5
    tol_beta: float = 1e-4

@dataclass
class VacuumConstraints:
    n_vac: float = 1.0
    allow_dispersion: bool = False

def validate_vacuum(n_vac: float, allow_dispersion: bool) -> list[str]:
    problems = []
    if abs(n_vac - 1.0) > 0.0:
        problems.append(f"Vacuum refractive index must be exactly 1.0, got {n_vac}")
    if allow_dispersion:
        # You may relax this in media, but *vacuum* dispersion must be off
        problems.append("Vacuum dispersion must be disabled; set allow_dispersion=False")
    return problems

def map_index_to_ppn(a1: float, a2: float):
    """Minimal placeholder mapping from n = 1 + a1*(Phi/c^2) + a2*(Phi/c^2)^2
    to effective PPN-like coefficients.

    IMPORTANT: You should replace this with your derived QFD->PPN mapping.
    For now, we *enforce* the GR-consistent values by definition:
      - a1 == -2  -> gamma = 1
      - a2 chosen to match beta = 1 (left as direct target)
    """
    gamma_pred = 1.0 if abs(a1 + 2.0) < 1e-12 else 1.0 + 0.5*(a1 + 2.0)  # penalize deviation
    # Without a validated analytic relation, treat beta as a target matched by a2=beta-1 proxy
    beta_pred = 1.0 + 0.0*a2  # neutral until you insert the real mapping
    return gamma_pred, beta_pred
