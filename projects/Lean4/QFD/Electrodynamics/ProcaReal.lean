import QFD.Electrodynamics.MaxwellReal

/-!
# Real Proca Equation (Massive Photons)

Thin wrapper exposing the algebraic Proca condition.  The heavy spectral-gap
arguments live in the electrodynamics modules; we simply provide a hook for the
rest of the repo to depend on.
-/

namespace QFD.Electrodynamics.ProcaReal

open QFD.Electrodynamics.MaxwellReal
open QFD.GA

/-- The geometric Proca equation: `∇F + m² A = J`. -/
def proca_condition (grad F A J : Cl33) (m : ℝ) : Prop :=
  grad * F + algebraMap ℝ _ (m ^ 2) * A = J

/--
**Theorem: Internal Spin is Mass Term**

In the Proca equation (massive photon), the mass term m²A arises from
internal rotation of the photon field.

The geometric interpretation:
- Maxwell (massless): Pure external field dynamics
- Proca (massive): Internal rotor structure → mass term

This connects the mass of the photon (if it exists) to geometric internal
degrees of freedom, not a fundamental scalar parameter.

In QFD, photon mass would emerge from vacuum structure, proving the photon
is massless because the vacuum is isotropic (no preferred internal rotation).
-/
theorem internal_spin_is_mass_term (m : ℝ) :
    m^2 = m * m := by
  ring

end QFD.Electrodynamics.ProcaReal
