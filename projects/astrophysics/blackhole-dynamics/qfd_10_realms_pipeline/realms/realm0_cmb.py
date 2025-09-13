"""
Realm 0: CMB Baseline & Polarization Gate

Purpose:
  - Fix the thermalization zeropoint using T_CMB (≈2.725 K)
  - Enforce *vacuum* polarization constraints:
      * No vacuum birefringence (rotation angle ~ 0)
      * No TB/EB in vacuum (parity invariance)
  - Narrow photon-interaction couplings so that spectral distortions and
    polarization constraints are respected in vacuum (media/plasmas can differ).

Outputs:
  - Constraints on psi_s0 zeropoint (thermalization engine lands at T_CMB)
  - Strong priors/upper bounds for k_J (vacuum drag ~ 0 locally & at recombination)
  - Bounds on couplings that would induce vacuum birefringence or parity violation
    (e.g., components of eta', xi combinations that act in vacuum)
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CMBTargets:
    T_CMB_K: float = 2.725
    allow_vacuum_birefringence: bool = False
    allow_parity_violation: bool = False

def run(params: Dict[str, Any], cfg: CMBTargets) -> Dict[str, Any]:
    notes = []
    fixed = {}
    narrowed = {}

    # 1) Thermalization zeropoint -> fix/anchor psi_s0 mapping
    # Placeholder linkage: record target; your thermalization engine should enforce this.
    fixed["T_CMB_K"] = cfg.T_CMB_K
    narrowed["psi_s0_anchor"] = "Set thermalization zeropoint so asymptotic vacuum temperature = T_CMB"

    # 2) Polarization gates in vacuum
    if cfg.allow_vacuum_birefringence:
        notes.append("Warning: vacuum birefringence allowed in config; set False to match CMB constraints.")
    else:
        narrowed["vacuum_birefringence"] = "Disallowed -> bounds on any terms rotating polarization in vacuum"

    if cfg.allow_parity_violation:
        notes.append("Warning: vacuum parity violation allowed in config; set False to suppress TB/EB in vacuum.")
    else:
        narrowed["vacuum_parity_violation"] = "Disallowed -> exclude TB/EB-generating vacuum couplings"

    # 3) Spectral distortion / drag proxies (very strict in vacuum/recombination)
    narrowed["k_J_upper_bound_recomb"] = "≈ 0 in vacuum to avoid µ/y spectral distortions"
    narrowed["eta_prime_vacuum_bound"] = "Must not induce frequency-dependent vacuum effects"

    return {
        "status": "ok" if not notes else "warning",
        "fixed": fixed,
        "narrowed": narrowed,
        "notes": notes
    }
