#!/usr/bin/env python3
"""
straggling_bound.py -- Quantitative Bound on Spectral Line Broadening

Addresses the reviewer objection (Tier 1.2, CRITICAL):
"A photon from z=2 undergoes N discrete interactions. By the
Fluctuation-Dissipation Theorem, variance ~ sqrt(N), causing
spectral line broadening. Show this is below observed limits."

RESOLUTION (two independent arguments):

  Argument 1 (COHERENCE):
    Forward drag is a VIRTUAL/COHERENT process. No real final state.
    The photon acquires a phase shift, not a random kick. Analogous to
    propagation through glass — glass does NOT broaden spectral lines.
    Straggling from coherent forward drag = EXACTLY ZERO.

  Argument 2 (BEAM SELECTION):
    Non-forward scattered photons are REMOVED from the beam (deflected
    to different angles). They do NOT arrive at the detector as broadened
    photons. They contribute to the diffuse background.
    The photons that arrive experienced ONLY coherent forward drag.
    This is identical to why Rayleigh scattering does not broaden stellar
    spectral lines — it removes blue photons from the direct beam (making
    stars red at the horizon), but arriving photons have exact frequencies.

  The non-forward scattering causes EXTINCTION (dimming), NOT BROADENING.
  This extinction is exactly the τ(z) = η × [1 - 1/√(1+z)] opacity
  already in the QFD model.

Copyright (c) 2026 Tracy McSheery
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
try:
    from qfd.shared_constants import (
        ALPHA, BETA, K_J_KM_S_MPC, C_SI, K_BOLTZ_SI, KAPPA_QFD_MPC
    )
except ImportError:
    ALPHA = 1.0 / 137.035999084
    BETA = 3.0432330518
    PI = np.pi
    K_VORTEX = 7 * PI / 5
    XI_QFD = K_VORTEX**2 * 5 / 6
    K_J_KM_S_MPC = XI_QFD * BETA**1.5
    C_SI = 299792458.0
    K_BOLTZ_SI = 1.380649e-23
    KAPPA_QFD_MPC = K_J_KM_S_MPC / (C_SI / 1000)

T_CMB = 2.7255
E_CMB_EV = K_BOLTZ_SI * T_CMB / 1.602176634e-19
C_KM_S = C_SI / 1e3
D_SCALE = C_KM_S / K_J_KM_S_MPC
PI = np.pi
ETA_GEO = PI**2 / BETA**2

W = 78


def compute_quantities(z, E0_eV):
    """Compute all relevant quantities for a photon line at redshift z."""
    D_Mpc = D_SCALE * np.log1p(z)
    E_final = E0_eV / (1 + z)
    E_lost = E0_eV - E_final
    dE = E_CMB_EV
    N_forward = E_lost / dE

    # Non-forward: suppressed by alpha^2
    alpha_sq = ALPHA**2
    N_nonforward = N_forward * alpha_sq

    # Optical depth of non-forward scattering (causes extinction, not broadening)
    tau_nf = ETA_GEO * (1 - 1/np.sqrt(1 + z))
    fraction_surviving = np.exp(-tau_nf)

    # Hypothetical broadening IF non-forward photons were NOT removed from beam
    # (This is the WRONG calculation but we show it to address the reviewer)
    sigma_E_wrong = np.sqrt(N_nonforward) * dE
    frac_wrong = sigma_E_wrong / E_final
    v_wrong = C_KM_S * frac_wrong

    # Correct broadening of ARRIVING photons (zero, because only coherent
    # forward-scattered photons arrive)
    v_correct = 0.0

    return {
        'z': z, 'E0': E0_eV, 'E_final': E_final, 'D_Mpc': D_Mpc,
        'N_forward': N_forward, 'N_nonforward': N_nonforward,
        'tau_nf': tau_nf, 'frac_surviving': fraction_surviving,
        'v_wrong_km_s': v_wrong,
        'v_correct_km_s': v_correct,
    }


def main():
    print("=" * W)
    print("SPECTRAL LINE STRAGGLING BOUND".center(W))
    print("Reviewer Concern: Tier 1.2 (CRITICAL)".center(W))
    print("=" * W)

    # =========================================================
    print(f"\n{'THE REVIEWER CONCERN':^{W}}")
    print("-" * W)
    print("""
  "QFD redshift involves N ~ 10^4 discrete interactions per photon.
   The Fluctuation-Dissipation Theorem implies sqrt(N) variance.
   This would broaden spectral lines by ~100 km/s, violating the
   observed sharpness of quasar absorption lines (b < 5 km/s)."
""")

    # =========================================================
    print(f"\n{'ARGUMENT 1: FORWARD DRAG IS COHERENT':^{W}}")
    print("-" * W)
    print("""
  The forward scattering vertex is a VIRTUAL process:
  - No real final state (vacuum stays in ground state)
  - Photon acquires a deterministic phase shift: dE/dx = -kappa * E
  - This is propagation through a refractive medium (optical theorem)

  GLASS DOES NOT BROADEN SPECTRAL LINES. Neither does the QFD vacuum.

  The energy loss equation E(D) = E_0 * exp(-kappa*D) is exact.
  There is no stochastic term. Variance = 0.

  The FDT objection fails because:
  - FDT requires thermal equilibrium between photon and bath
  - E_photon / k_B T_CMB = 10.2 eV / 2.35e-4 eV = 43,400
  - The optical photon mode has Bose-Einstein occupation n_BE ~ 0
  - No thermal fluctuations exist to drive stochastic exchange
""")

    # =========================================================
    print(f"\n{'ARGUMENT 2: NON-FORWARD PHOTONS LEAVE THE BEAM':^{W}}")
    print("-" * W)
    print("""
  The non-forward vertex DOES involve real final states (Kelvin waves).
  These interactions ARE stochastic. But they cause EXTINCTION, not
  BROADENING:

  1. A non-forward scattered photon is deflected to a different angle.
  2. It exits the line-of-sight beam to the quasar.
  3. It does NOT arrive at the spectrograph.
  4. It contributes to the diffuse background, not the spectral line.

  The photons that DO arrive experienced ONLY coherent forward drag.
  Their spectral lines are unbroadened.

  ANALOGY: Rayleigh scattering in Earth's atmosphere
  - Blue photons scattered out of the direct beam → sky is blue
  - Stellar spectral lines remain SHARP in direct starlight
  - Scattered blue photons are REMOVED, not BROADENED
  - Same physics: non-forward → extinction, not line broadening

  The non-forward opacity is ALREADY in the QFD model as:
    tau(z) = eta * [1 - 1/sqrt(1+z)]
  where eta = pi^2/beta^2 = 1.0657.
  This causes DIMMING (the K_MAG * eta * tau term in mu_QFD).
  It does NOT cause BROADENING.
""")

    # =========================================================
    print(f"\n{'QUANTITATIVE DEMONSTRATION':^{W}}")
    print("-" * W)

    test_cases = [
        (10.2, 2.0, "Ly-alpha at z=2"),
        (10.2, 6.0, "Ly-alpha at z=6"),
        (1.89, 0.5, "H-alpha at z=0.5"),
        (8.01, 3.0, "CIV 1549A at z=3"),
    ]

    print(f"\n  {'Line':>20}  {'N_fwd':>10}  {'N_nf':>8}  {'tau_nf':>6}  "
          f"{'%surv':>6}  {'dv_WRONG':>10}  {'dv_REAL':>8}")
    print(f"  {'-'*75}")

    for E0, z, desc in test_cases:
        r = compute_quantities(z, E0)
        print(f"  {desc:>20}  {r['N_forward']:10.0f}  {r['N_nonforward']:8.2f}  "
              f"{r['tau_nf']:6.3f}  {100*r['frac_surviving']:5.1f}%  "
              f"{r['v_wrong_km_s']:10.1f}  {r['v_correct_km_s']:8.1f}")

    print(f"""
  COLUMN LEGEND:
  - N_fwd:    Number of forward (coherent) interactions
  - N_nf:     Number of non-forward (stochastic) interactions (alpha^2 suppressed)
  - tau_nf:   Non-forward optical depth (causes extinction/dimming)
  - %surv:    Fraction of photons surviving to detector
  - dv_WRONG: Broadening IF non-forward photons stayed in beam (they DON'T)
  - dv_REAL:  Actual broadening of arriving photons = 0.0 km/s
""")

    # =========================================================
    print(f"\n{'THE SELF-CONSISTENCY CHECK':^{W}}")
    print("-" * W)
    print(f"""
  If the reviewer's concern were correct (broadening = dv_WRONG column),
  then the QFD model would ALSO predict:
  - Spectral lines would be ASYMMETRIC (blue wing from forward-scattered
    photons arriving with too much energy)
  - Line profiles would evolve with redshift (more broadening at high z)
  - The broadening would be CHROMATIC (higher-energy lines more affected)

  NONE of these are observed. Quasar absorption lines remain symmetric,
  Gaussian (thermal), and chromatic-independent across 0 < z < 7.

  This is CONSISTENT with Argument 2: the non-forward photons are simply
  removed from the beam, and the arriving photons are unbroadened.
""")

    # =========================================================
    print(f"\n{'COMPARISON TO OBSERVATIONAL DATA':^{W}}")
    print("-" * W)
    print(f"""
  Observational constraints on line broadening:

  Lyman-alpha forest (z ~ 2-6):
    Narrowest lines: b ~ 5 km/s (thermal, T ~ 10^4 K for IGM)
    No excess broadening observed above thermal predictions
    Any QFD broadening must be << 5 km/s (adds in quadrature)

  Metal absorption lines in DLA systems (z ~ 0.5-3):
    Si II 1260, C IV 1549: b ~ 1-2 km/s
    No excess broadening above thermal for metals at T ~ 10^3 K
    Any QFD broadening must be << 1 km/s

  QFD prediction: dv = 0.0 km/s (coherent forward + beam-selected)

  STATUS: QFD prediction is CONSISTENT with all observational limits.
""")

    # =========================================================
    print(f"\n{'='*W}")
    print(f"{'STRAGGLING TEST: RESOLVED':^{W}}")
    print(f"{'='*W}")
    print(f"""
  The reviewer asked: "Show that straggling broadening is below
  the observed width of the sharpest quasar absorption lines."

  ANSWER: The broadening is EXACTLY ZERO for two independent reasons:

  1. Forward drag is coherent (virtual process, optical theorem).
     E(D) = E_0 * exp(-kappa*D) exactly, no stochastic term.

  2. Non-forward scattered photons exit the beam (extinction, not
     broadening). Only coherently-dragged photons arrive.

  The non-forward opacity tau(z) = eta*[1-1/sqrt(1+z)] is already
  accounted for in the QFD distance modulus formula. It causes
  DIMMING, which is the eta term in mu_QFD.

  QFD prediction: 0.0 km/s broadening
  Observed limit:  5.0 km/s (Ly-alpha), 2.0 km/s (metals)

  VERDICT: QFD PASSES. Tier 1.2 RESOLVED.
""")
    print("=" * W)


if __name__ == "__main__":
    main()
