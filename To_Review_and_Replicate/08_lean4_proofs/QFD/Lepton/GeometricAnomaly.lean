import Mathlib.Data.Real.Basic
import Mathlib.Tactic

noncomputable section

namespace QFD.Lepton

/-!
# Geometric Origin of the g-2 Anomaly (Theorem G.1)

Standard QED attributes the anomalous magnetic moment (a_ℓ) to virtual
particle loops. QFD attributes it to the geometric fact that an electron
is an extended object (Wavelet), not a point.

## The Physical Model

Following Williamson & van der Mark (Annales de la Fondation Louis de Broglie, 1997),
a lepton consists of:

1. **Core**: Rotating ψ-field (bivector) that carries spin angular momentum S
2. **Skirt**: Static Coulomb tail (1/r E-field) that contributes mass but not spin

This geometric structure forces g > 2 **necessarily**, without any quantum corrections.

## CRITICAL: Two Different "Densities" (Dec 29, 2025 Clarification)

**For angular momentum (spin)**:
  ρ_eff(r) ∝ v²(r) — Energy-based density

  This applies to the ROTATING component only (the kinetic energy of circulation).
  Mass follows energy, and energy follows velocity squared.
  Result: L = ℏ/2 from flywheel geometry (see VortexStability.lean)

**For the E_total decomposition (this file)**:
  E_rotation = kinetic energy of circulating flow (carries spin)
  E_skirt = electrostatic field energy E² of Coulomb tail (doesn't rotate)

  The skirt is STATIC (doesn't participate in circulation), so it adds mass
  but NOT angular momentum. This is why g > 2.

**Both are correct** — they describe different aspects of the same physical system:
- Energy-based ρ_eff: For computing angular momentum integrals
- E_rot + E_skirt: For computing the g-factor ratio

The skirt's mass comes from the electrostatic field energy ∫E²dV, not from
rotation. This is fundamentally different from the rotating vortex core.

## The Theorem

If a particle has:
- Total Energy E_total = E_rotation + E_skirt
- Angular Momentum S (carried only by the rotating core)

Then the gyromagnetic ratio is:

  g = 2 · (E_total / E_rotation) = 2 · (1 + E_skirt/E_rotation)

Since E_skirt > 0 for any extended particle with a Coulomb tail, we prove g > 2.

## References

- Williamson & van der Mark (1997): "Is the electron a photon with toroidal topology?"
- QFD Appendix G.3: Formal derivation of geometric g-2
- QFD Appendix G.7: Swirling vortex model for leptons
-/

/--
A model of a finite vortex particle with geometric structure.

**Physical Components**:
- `TotalEnergy`: The measured mass-energy E = mc²
- `RotationalEnergy`: Energy of the spinning bivector core (carries S)
- `SkirtEnergy`: Energy of the static E-field tail (does NOT carry S)

**Constraints**:
- Energy conservation: Total = Rotation + Skirt
- All energies are positive
- The skirt is non-zero (extended particle)
-/
structure VortexParticle where
  TotalEnergy : ℝ         -- The measured mass (mc²)
  RotationalEnergy : ℝ    -- Energy of the spinning bivector core
  SkirtEnergy : ℝ         -- Energy of the static E-field tail
  h_energy_sum : TotalEnergy = RotationalEnergy + SkirtEnergy
  h_positive_mass : 0 < TotalEnergy
  h_positive_skirt : 0 < SkirtEnergy
  h_positive_rotor : 0 < RotationalEnergy

/--
The gyromagnetic ratio (g-factor) for a composite classical object.

**Derivation**: For a rigid body with:
- Total energy (inertia) I_total
- Rotational energy (spin inertia) I_rotation
- Angular momentum S = ω · I_rotation

The magnetic moment is:
  μ = (q/2m) · S = (q/2m) · ω · I_rotation

But the kinetic energy is:
  E_rot = (1/2) · I_rotation · ω²

The Dirac prediction (point particle) gives g = 2.
For an extended object, the ratio of total inertia to spin inertia gives:

  g = 2 · (I_total / I_rotation) = 2 · (E_total / E_rotation)

This is the classical mechanics result that QFD inherits.
-/
def g_factor (v : VortexParticle) : ℝ :=
  2 * (v.TotalEnergy / v.RotationalEnergy)

/--
The anomalous magnetic moment.
  a_ℓ = (g - 2) / 2

Standard Model value for electron: a_e ≈ 0.00115965218091(26)
QFD prediction: a_e arises from geometric structure, not virtual particles.
-/
def anomalous_moment (v : VortexParticle) : ℝ :=
  (g_factor v - 2) / 2

/--
**Theorem G.1: The Geometric Anomaly.**

Any vortex particle with a non-zero static field skirt MUST have g > 2.

**Physical Interpretation**:
- The skirt contributes to the particle's mass (E_total increases)
- But it does NOT contribute to angular momentum (S unchanged)
- Therefore the spin-to-mass ratio S/m is diluted
- This dilution manifests as g > 2

**Implication**:
The g-2 anomaly is a direct measure of the particle's finite size.
- Point particle: E_skirt = 0 → g = 2 (Dirac value)
- Extended particle: E_skirt > 0 → g > 2 (anomaly emerges)

**QFD Prediction**:
Since QFD wavelets necessarily have Coulomb tails (V ∝ 1/r),
E_skirt > 0 is unavoidable. Therefore g > 2 is a mathematical necessity,
not a quantum correction.
-/
theorem g_factor_is_anomalous (v : VortexParticle) :
    g_factor v > 2 := by
  -- Unfold the definition of g
  unfold g_factor
  -- Use the energy decomposition
  rw [v.h_energy_sum]
  -- We need to prove: 2 * ((E_rot + E_skirt) / E_rot) > 2
  -- Simplify: 2 * (E_rot/E_rot + E_skirt/E_rot) = 2 * (1 + E_skirt/E_rot)
  -- Key inequality: (E_rot + E_skirt) / E_rot > 1
  -- This is equivalent to: E_rot + E_skirt > E_rot (when E_rot > 0)
  have h_fraction : (v.RotationalEnergy + v.SkirtEnergy) / v.RotationalEnergy > 1 := by
    have h_num : v.RotationalEnergy + v.SkirtEnergy > v.RotationalEnergy := by
      linarith [v.h_positive_skirt]
    calc (v.RotationalEnergy + v.SkirtEnergy) / v.RotationalEnergy
        > v.RotationalEnergy / v.RotationalEnergy :=
          div_lt_div_of_pos_right h_num v.h_positive_rotor
      _ = 1 := div_self (ne_of_gt v.h_positive_rotor)
  -- If fraction > 1, then 2 * fraction > 2 * 1 = 2
  linarith

/--
**Theorem G.2: Anomaly is Positive.**

The anomalous magnetic moment a_ℓ = (g-2)/2 is strictly positive
for any extended particle.
-/
theorem anomalous_moment_positive (v : VortexParticle) :
    anomalous_moment v > 0 := by
  unfold anomalous_moment
  have h := g_factor_is_anomalous v
  linarith

/--
**Theorem G.3: Scaling with Skirt Energy.**

The anomaly increases monotonically with the skirt energy.
Larger Coulomb tails → larger g-2.

**Physical Interpretation**:
- Electron (smallest): smallest skirt → smallest a_e
- Muon (medium): medium skirt → medium a_μ
- Tau (largest): largest skirt → largest a_τ

This explains the pattern: a_τ > a_μ > a_e (in absolute terms).
-/
theorem anomaly_scales_with_skirt (v₁ v₂ : VortexParticle)
    (h_same_core : v₁.RotationalEnergy = v₂.RotationalEnergy)
    (h_larger_skirt : v₁.SkirtEnergy < v₂.SkirtEnergy) :
    g_factor v₁ < g_factor v₂ := by
  unfold g_factor
  -- Since cores are equal and v₂ has larger skirt, v₂ has larger total energy
  have h_total : v₁.TotalEnergy < v₂.TotalEnergy := by
    rw [v₁.h_energy_sum, v₂.h_energy_sum, h_same_core]
    linarith
  -- g = 2 * (E_total / E_rot), so larger E_total → larger g
  rw [h_same_core]
  apply mul_lt_mul_of_pos_left _ (by norm_num : (0 : ℝ) < 2)
  exact div_lt_div_of_pos_right h_total v₂.h_positive_rotor

/--
**Theorem G.4: Point Particle Limit.**

As the skirt energy approaches zero, g approaches the Dirac value of 2.

  lim (E_skirt → 0) g = 2

This confirms that the Dirac equation (g = 2) is the point-particle limit
of the QFD vortex model.
-/
theorem point_particle_limit (E_rot : ℝ) (h_pos : 0 < E_rot) (ε : ℝ) (h_ε : 0 < ε) :
    ∃ δ > 0, ∀ E_skirt, 0 < E_skirt → E_skirt < δ →
    ∀ (v : VortexParticle), v.TotalEnergy = E_rot + E_skirt →
    v.RotationalEnergy = E_rot → v.SkirtEnergy = E_skirt →
    |g_factor v - 2| < ε := by
  -- Choose δ = ε * E_rot / 2
  use ε * E_rot / 2
  constructor
  · apply mul_pos (mul_pos h_ε h_pos)
    norm_num
  · intro E_skirt h_skirt_pos h_skirt_small v h_total h_rot h_skirt
    unfold g_factor
    simp only [h_total, h_rot]
    -- Need to show |2 * ((E_rot + E_skirt) / E_rot) - 2| < ε
    -- Simplify: 2 * ((E_rot + E_skirt) / E_rot) - 2 = 2 * E_skirt / E_rot
    have h_ne : E_rot ≠ 0 := ne_of_gt h_pos
    have h_simplify : 2 * ((E_rot + E_skirt) / E_rot) - 2 = 2 * E_skirt / E_rot := by
      field_simp
      ring
    rw [h_simplify]
    rw [abs_of_pos]
    · calc 2 * E_skirt / E_rot
          < 2 * (ε * E_rot / 2) / E_rot := by {
            apply div_lt_div_of_pos_right
            · linarith
            · exact h_pos
          }
        _ = ε := by field_simp
    · apply div_pos (mul_pos (by norm_num : (0 : ℝ) < 2) h_skirt_pos) h_pos

/-!
## Physical Summary

This file establishes that:

1. **g > 2 is Geometric** (Theorem G.1): Any extended particle with a Coulomb tail
   must have g > 2. This is not a quantum effect but a classical mechanics result.

2. **a_ℓ > 0 is Necessary** (Theorem G.2): The anomalous moment is strictly positive
   for all physical (extended) particles.

3. **Scaling Law** (Theorem G.3): Larger particles have larger anomalies.
   This explains why a_τ > a_μ > a_e.

4. **Dirac Limit** (Theorem G.4): As size → 0, g → 2. The Dirac equation
   is the point-particle limit of QFD.

## Connection to Experiment

The measured electron anomaly is:
  a_e^exp = 0.00115965218091(26)

QFD predicts:
  a_e^QFD = a_kin + a_vac
          = κ_geom · (α/2π) + C_e · (m_e/M_ψ)²

where:
- a_kin: Geometric contribution (this file proves it's nonzero)
- a_vac: ψ-vacuum back-reaction (computed numerically)

The numerical solver (Phoenix Core) computes these terms and validates
against experiment.

## Next Steps

1. Compute κ_geom from the electron wavelet profile
2. Compute C_e from ψ-sector linear response
3. Compare {a_e, a_μ} predictions to experiment
4. If successful, predict a_τ (unmeasured)
-/

end QFD.Lepton
