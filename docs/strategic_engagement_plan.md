# QFD Strategic Engagement Plan

**Date**: 2026-02-13
**Context**: Book v9.0 published on Zenodo/OSF.io with cryptographic timestamps.
**Source**: Red Team / Blue Team analysis, reviewed against edits33-38 honesty corrections.

---

## Strategic Principle

The book is a 500-page fortress. Orthodox physics will default to silence.
Break the framework into targeted, falsifiable papers that force engagement.

**Priority**: Zenodo timestamps establish priority. The goal is now engagement, not gatekeeping.

---

## SECTOR 1: Foundations & The Quark Illusion (Z.4, 6C Phase Space)

### Red Team Attack
- Ultra-hyperbolic Cauchy problem: Cl(3,3) signature makes IVP ill-posed
- Collider data: DIS shows three hard scattering centers in protons

### Blue Team Defense
- Z.4 argument: collider energies exceed the Spectral Gap, shattering 4D SO(2) symmetry
- "Quarks" = transient 6D topological fracture patterns during re-stabilization
- Explains confinement: not an object, but a temporary structural failure
- 500+ Lean 4 theorems prove Cl(3,3) -> Cl(3,1) reduction is mathematically inevitable

### Tactical Paper
**Target**: Formal verification / mathematical logic community (Lean Zulip, Xena Project)
**Title**: "Formal Verification of Lorentzian Emergence from a Scleronomic Phase Space"
**Pitch**: Mathematicians validate the Lean code; physics community must then concede
the mathematical foundation.

**Honesty check**: This paper would be genuinely novel — the Lean proofs ARE rigorous.
No overclaims here.

---

## SECTOR 2: The Golden Loop & Lepton Constants

### Red Team Attack
- Numerology accusation: Golden Loop looks designed to hit 137.035
- Constitutive postulate: sigma = beta^3/(4pi^2) is a hidden free parameter

### Blue Team Defense
- Same beta bridges lepton, nuclear, and cosmological scales
- Proton mass ratio (0.0023% error), V4 coefficient (0.04% match), K_J
- V6 shear modulus is a continuum mechanics requirement, not a patch

### Tactical Paper
**Target**: Belle II experimentalists at KEK
**Title**: "Prediction Note: Tau g-2 from Geometric Vacuum Compliance"
**Pitch**: Timestamped prediction of a_tau(QFD) = 1192 x 10^-6.
Binary test: 1177 (SM) vs 1192 (QFD).

### >>> HONESTY FLAGS (from edits33-38) <<<

**WARNING**: The original analysis overstates this sector in several ways:

1. **"100% Parameter Closure" is overclaimed.** Edit 38-23 specifically corrects
   the §12.8.1 table from "100% explained" to "Nuclear: 100%; Lepton g-2:
   requires sigma postulate." The strategic document should NOT claim 100%.

2. **The tau g-2 prediction depends on sigma.** Edit 38-08 adds the sigma-postulate
   caveat to W.5.7. The Belle II note MUST acknowledge that a_tau depends on
   sigma = beta^3/(4pi^2), which is a constitutive postulate (edits33), not derived.
   Framing it as "zero free parameters" would be dishonest.

3. **The electron g-2 "match" is Schwinger-dominated.** Edits 38-20, 38-27-31
   fix this throughout the book. The V4 coefficient matches C2 to 0.04%, but
   the full g-2 precision (0.0013%) is 99.85% carried by the universal Schwinger
   term alpha/(2pi). Any external communication MUST distinguish coefficient
   match from full g-2 match.

4. **Honest framing for Belle II note**: "QFD predicts a_tau = 1192 x 10^-6,
   contingent on the constitutive postulate sigma = beta^3/(4pi^2). If Belle II
   measures a_tau and the result matches, sigma is validated. If it matches the SM
   value ~1177, the sigma postulate is falsified."

---

## SECTOR 3: Nuclear Physics (Q-Balls & Fission Asymmetry)

### Red Team Attack
- Spin-parity selection rules: continuous fluid can't account for discrete quantum numbers
- Half-life damping error rate: 5.9%

### Blue Team Defense
- Fission asymmetry from integer topology: odd N cannot split symmetrically
- Zero-parameter core compression: 5,800+ isotopes from alpha and beta alone
- Fissility limit (Z^2/A)_crit = alpha^-1/beta = 45.0 (4.2% vs Bohr-Wheeler)
  - Lean-proven (FissionLimit.lean), formal bound < 15%
  - Numerical agreement 4.2% (edit 38-27 distinguishes these)

### Tactical Paper
**Target**: Nuclear engineers frustrated with SEMF shell-corrections
**Title**: "Topological Fission Asymmetry: An Integer Arithmetic Constraint"
**Pitch**: Extract Section 14.12. Pure arithmetic rule solving a known 80-year failure.

**Honesty check**: This is QFD's cleanest sector. The nuclear predictions genuinely
use only alpha and beta (no sigma postulate). The fission asymmetry argument is
elegant and independently checkable. The Lean proofs (FissionLimit.lean,
FissionTopology.lean) provide formal backing. This paper has the highest
credibility-to-risk ratio.

---

## SECTOR 4: Cosmology (Static Universe & Plasma Veil)

### Red Team Attack
- Supernova time dilation: light curves stretch by (1+z), implying expansion
- Blurring problem: vacuum scattering should blur distant galaxies

### Blue Team Defense
- Optical Transfer Function (Section 9.10): soft drag redshifts without deflection
- Chromatic stretching: energy-dependent scattering preferentially delays blue photons
- QFD fits DES-SN5YR to chi^2/dof = 1.005 with zero physics parameters (only M calibration)

### Tactical Paper
**Target**: Supernova cosmology community
**Title**: "Chromatic Dispersion as a Discriminator Between Expansion and Scattering"
**Pitch**: CDM requires achromatic stretch; QFD predicts chromatic stretch.
Testable with existing multi-band supernova data.

### >>> HONESTY FLAGS <<<

1. **K_J dimensional status**: Edit 38-13 clarifies that kappa_tilde = xi*beta^(3/2)
   is dimensionless. The identification with H_0 (km/s/Mpc) requires a dimensional
   bridge not yet derived. Any cosmology paper must acknowledge this gap.

2. **T_CMB = 2.725 K is NOT derived**: Edit 38-01 removes the false claim from
   line 66. The CMB temperature is an empirical input, not a QFD prediction.

3. **r_psi = 269 Mpc is fitted**: Edit 38-12 clarifies this. The CMB peak positions
   use a fitted crystallographic scale, not one derived from alpha -> beta.

4. **"Zero free parameters" for SN fitting**: The book correctly notes M (absolute
   magnitude calibration) is fitted. Edit 38-09 softens the DES-SN5YR "0 free params"
   display. External communications should say "zero physics parameters (one
   calibration shared by all models)."

---

## RECOMMENDED PRIORITY ORDER

1. **Nuclear fission asymmetry** (Sector 3) — Cleanest results, no sigma postulate,
   Lean-proven, solves a known problem. Highest credibility.

2. **Lean 4 mathematical foundation** (Sector 1) — Pure math, no physics claims,
   formally verified. Appeals to a different audience (mathematicians).

3. **Belle II tau g-2 prediction** (Sector 2) — High impact but MUST honestly
   acknowledge sigma dependence. Frame as "testing the sigma postulate."

4. **Chromatic stretch challenge** (Sector 4) — Bold but requires strongest
   evidence. Save for after credibility is established via Sectors 1-3.

---

## META-STRATEGIC NOTES

### What NOT to do
- Do not claim "zero free parameters" for the full framework (sigma exists)
- Do not claim g-2 precision of "0.0013%" without Schwinger dominance caveat
- Do not claim T_CMB is derived
- Do not use aggressive/adversarial language in published communications
- Do not oversell the quark-as-artifact argument without acknowledging the
  ultra-hyperbolic Cauchy problem (this IS a genuine weakness)

### What the edits33-38 honesty campaign enables
The entire strategic advantage of QFD is that it is *more honest* than its own
claims need to be. A framework that voluntarily identifies its own constitutive
postulates, proves they cannot be derived from the flat-space Hessian (Z.8.B),
and timestamps falsifiable predictions is *more credible* than one that overclaims.

The honesty IS the weapon. Do not undermine it with overclaims in external
communications that the book itself has already corrected.
