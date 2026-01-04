import Mathlib

set_option autoImplicit false

namespace QFD

/-!
  # QFD Photon Sector: Kinematic Soliton Model

  This update bridges the gap between "Bookkeeping" and "Dynamics".
  It models the Photon not just as an energy quantum, but as a
  spatial Soliton with:
  1. Wavenumber k (geometry)
  2. Momentum p = ℏk (mechanical recoil)
  3. Linear Dispersion ω = c|k| (stiffness constraint)
  4. Shape Invariance (soliton stability)
-/

universe u

variable {Point : Type u}

structure PsiField (Point : Type u) where
  ψ : Point → ℝ  -- Placeholder for Cl(3,3) multivector field

structure Config (Point : Type u) where
  charge : ℤ
  energy : ℝ
  /-- The spatial "footprint" or wavelength scale of the configuration -/
  scale  : ℝ

structure QFDModel (Point : Type u) where
  Ψ : PsiField Point

  -- The Three Constants + Planck
  /-- Fine-structure coupling (gear mesh strength) -/
  α : ℝ
  /-- Vacuum stiffness (controls c and dispersion suppression) -/
  β : ℝ
  /-- Saturation scale (nonlinear focusing) -/
  λ_sat : ℝ
  /-- Angular impulse of the electron vortex -/
  ℏ : ℝ

  /-- Speed of light: Terminal velocity of vacuum shear waves.
      In QFD, c ≈ sqrt(β / density). -/
  c_vac : ℝ

  -- Existence Predicates
  PhaseClosed : Config Point → Prop
  OnShell : Config Point → Prop
  FiniteEnergy : Config Point → Prop

  /-- The Soliton Stability Predicate.
      Represents the "Soliton Balance": Dispersion (spreading) is exactly
      cancelled by Nonlinear Focusing (λ_sat).
      Mathematically: d(Width)/dt = 0. -/
  ShapeInvariant : Config Point → Prop

  -- Interaction Predicates
  Bound : Config Point → Config Point → Prop
  ELevel : ℕ → ℝ

namespace QFDModel

variable {M : QFDModel Point}

/- 1) The Photon: Now with Geometry and Momentum -/

/--
  A Photon is defined by its spatial wavenumber `k`.
  Frequency and Energy are derived via the Vacuum constraints.
-/
structure Photon where
  /-- Wavenumber k = 2π / λ -/
  k : ℝ
  /-- Wavenumber must be positive for a propagating wave -/
  hk_pos : k > 0

namespace Photon

/-- Wavelength λ = 2π / k -/
noncomputable def wavelength (γ : Photon) : ℝ := (2 * Real.pi) / γ.k

/-- Linear Momentum p = ℏk.
    This fulfills the user requirement: p ∝ 1/λ.
    Physical meaning: The "kick" delivered by the retro-rocket. -/
def momentum (M : QFDModel Point) (γ : Photon) : ℝ := M.ℏ * γ.k

/-- Frequency ω = c|k|.
    This assumes the "Stiff Vacuum" limit where dispersion is negligible. -/
def frequency (M : QFDModel Point) (γ : Photon) : ℝ := M.c_vac * γ.k

/-- Energy E = ℏω.
    Standard quantization derived from the vortex angular impulse. -/
def energy (M : QFDModel Point) (γ : Photon) : ℝ := M.ℏ * (frequency M γ)

/-- The Dispersion Relation Lemma:
    Proves that Energy is proportional to Momentum times Speed of Light.
    E = p * c
-/
theorem energy_momentum_relation (γ : Photon) :
    energy M γ = (momentum M γ) * M.c_vac := by
  simp [energy, frequency, momentum]
  ring

end Photon

/- 2) Solitons: Now with Stability Requirements -/

/-- A Soliton is now PhaseClosed, OnShell, FiniteEnergy AND ShapeInvariant. -/
def Soliton (M : QFDModel Point) : Type u :=
  { c : Config Point //
    M.PhaseClosed c ∧
    M.OnShell c ∧
    M.FiniteEnergy c ∧
    M.ShapeInvariant c } -- Added stability requirement

instance : Coe (Soliton M) (Config Point) := ⟨Subtype.val⟩

def Electron (M : QFDModel Point) : Type u :=
  { s : Soliton M // (s : Config Point).charge = (-1) }

def Proton (M : QFDModel Point) : Type u :=
  { s : Soliton M // (s : Config Point).charge = (1) }

/- 3) Hydrogen and "Lock and Key" Absorption -/

structure Hydrogen (M : QFDModel Point) where
  e : Electron M
  p : Proton M
  bound : M.Bound (e : Config Point) (p : Config Point)

structure HState (M : QFDModel Point) where
  H : Hydrogen M
  n : ℕ

/--
  Absorption defined by Kinematic Conservation.
  Ideally, this would also include conservation of momentum (p_photon transferred to H),
  but here we focus on the Energy "Tooth Matching".
-/
def Absorbs (M : QFDModel Point) (s : HState M) (γ : Photon) (s' : HState M) : Prop :=
  s'.H = s.H ∧
  s.n < s'.n ∧
  M.ELevel s'.n = M.ELevel s.n + Photon.energy M γ

/--
  The "Gear Mesh" Theorem.
  If the photon's spatial geometry (k) produces an energy (ℏck) that exactly matches
  the gap between Hydrogen gears (E_m - E_n), absorption is valid.
-/
theorem absorption_geometric_match
    {M : QFDModel Point} {H : Hydrogen M} {n m : ℕ} (hnm : n < m)
    (γ : Photon)
    -- The geometric condition: k must scale to the energy gap
    (hGeo : M.ℏ * (M.c_vac * γ.k) = M.ELevel m - M.ELevel n) :
    Absorbs M ⟨H, n⟩ γ ⟨H, m⟩ := by
  refine ⟨rfl, hnm, ?_⟩
  simp [Photon.energy, Photon.frequency] at *
  linarith [hGeo]

end QFDModel
end QFD
