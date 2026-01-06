import Mathlib
import QFD.Topology.FormFactorCore
import QFD.Physics.Postulates

/-!
# QFD: Topological Form Factors

Formal specification of the geometric mechanism behind the Alpha gap.
This module defines the *form factor* functional (bulk vs surface energy)
and states that spherical (nuclear) and toroidal (leptonic) solitons must
have different ratios purely due to topology.  Numerical evaluation is
delegated to scripts such as `solve_torus_form_factor.py`.
-/

noncomputable section

namespace QFD

open QFD.Physics

/--
Spherical vs toroidal form factors are provably different once the Hill
vortex energy integrals are evaluated.  Python scripts must supply the
associated numerical evidence; Lean captures the logical dependency via
the physics postulates.
-/
lemma sphere_torus_form_factor_ne
    (P : QFD.Physics.Model)
    {ψ_sphere ψ_torus : Field}
    (h_sphere : is_spherical ψ_sphere)
    (h_torus : is_toroidal ψ_torus) :
    form_factor (P.compute_energy ψ_sphere) ≠
      form_factor (P.compute_energy ψ_torus) :=
  P.sphere_torus_form_factor_ne h_sphere h_torus

/--
**Topological Splitting of Coupling Constants.**

If nuclei correspond to spherical solitons and electrons to toroidal
vortices, their geometric form factors differ, explaining why
`α_strong ≠ α_em`.
-/
theorem coupling_depends_on_topology
    (P : QFD.Physics.Model)
    (ψ_nuc : Field) (h_nuc : is_spherical ψ_nuc)
    (ψ_elec : Field) (h_elec : is_toroidal ψ_elec) :
    let F_nuc := form_factor (P.compute_energy ψ_nuc)
    let F_elec := form_factor (P.compute_energy ψ_elec)
    F_nuc ≠ F_elec := by
  intro F_nuc F_elec
  exact sphere_torus_form_factor_ne (P := P) h_nuc h_elec

/-- Quantity the Python scripts must evaluate to close the alpha gap. -/
def target_alpha_correction (F_nuc F_elec : ℝ) : ℝ :=
  F_nuc / F_elec

end QFD
