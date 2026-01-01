"""
Compute the "Goldilocks" scattering coefficient η′ from Tolman's surface
brightness law and FIRAS limits.

Assumptions:
- Survival fraction S(z) ≈ exp(-η′ z) (leading-order optical depth)
- SN dimming Δμ = -2.5 log10 S ⇒ η′_SN = Δμ * ln 10 / (2.5 z)
- FIRAS y-distortion ~ y_eff ≈ η′ * ξ_eff, but in practice we cap η′ by
  |ΔT/T|_FIRAS ~ 5e-5 (conservative, since detailed mapping comes from
  RadiativeTransferParams in Lean).

Outputs both the SN-required η′ and the FIRAS cap; final η′ is the minimum.
References:
- QFD/Cosmology/ScatteringBias.lean (survival/dimming proofs)
- QFD/Cosmology/RadiativeTransfer.lean (energy conservation, FIRAS constraints)
"""

from dataclasses import dataclass
import math

CMB_T0 = 2.7255  # K
FIRAS_DELTA_T_OVER_T = 5e-5  # ~50 ppm limit
XI_STAGE2_DEFAULT = 6.45  # |xi| from Stage-2 SNe fits (see qfd-sn-v22)

@dataclass
class EtaPrimeResult:
    eta_prime_final: float
    eta_prime_sn: float
    eta_prime_firas: float
    sn_dimming_mag: float
    redshift_ref: float


def solve_eta_prime(sn_mag: float = 0.25,
                    z_ref: float = 1.0,
                    xi_eff: float = XI_STAGE2_DEFAULT,
                    delta_T_over_T_max: float = FIRAS_DELTA_T_OVER_T) -> EtaPrimeResult:
    # Tolman dimming → required optical depth.
    eta_from_sn = sn_mag * math.log(10.0) / (2.5 * z_ref)
    # FIRAS: y_eff = eta_prime * xi_eff must be below limit.
    xi_abs = abs(xi_eff)
    eta_from_firas = delta_T_over_T_max / xi_abs if xi_abs > 0 else float('inf')
    eta_final = min(eta_from_sn, eta_from_firas)
    return EtaPrimeResult(
        eta_prime_final=eta_final,
        eta_prime_sn=eta_from_sn,
        eta_prime_firas=eta_from_firas,
        sn_dimming_mag=sn_mag,
        redshift_ref=z_ref,
    )


if __name__ == "__main__":
    res = solve_eta_prime()
    print("η'_SN =", res.eta_prime_sn)
    print("η'_FIRAS =", res.eta_prime_firas)
    print("η' (final) =", res.eta_prime_final)
