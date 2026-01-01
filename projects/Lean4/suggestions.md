# Suggestions / Build Notes

1. **Targeted builds** – Please run module-level builds for the remaining files that need fixes:
   - `lake build QFD.Conservation.NeutrinoMixing`  # ❌ Mathlib Matrix issue
   - `lake build QFD.Nuclear.BoundaryCondition`    # ❌ Schema.Constraints
   - `lake build QFD.Nuclear.MagicNumbers`         # ❌ Schema.Constraints
   - `lake build QFD.Nuclear.DeuteronFit`          # ❌ Schema.Constraints
   - `lake build QFD.QM_Translation.Zitterbewegung`  # ❌ SchrodingerEvolution
   - `lake build QFD.QM_Translation.PauliExclusion`  # ❌ Namespace errors
   - `lake build QFD.Relativity.LorentzRotors`     # ❌ Build failure
   - `lake build QFD.Cosmology.ZeroPointEnergy`    # ❌ Schema.Constraints
   - `lake build QFD.Weak.GeometricBosons`         # ❌ NeutrinoID
   - `lake build QFD.Nuclear.IsomerDecay`          # ❌ Schema.Constraints
   - `lake build QFD.Soliton.BreatherModes`        # ❌ YukawaDerivation
   - `lake build QFD.Gravity.TorsionContribution`
   - `lake build QFD.Lepton.Antimatter`
   - `lake build QFD.Unification.FieldGradient`
   - `lake build QFD.Electrodynamics.ComptonScattering`
   - `lake build QFD.Lepton.PairProduction`
   - `lake build QFD.Vacuum.Screening`
   - `lake build QFD.Matter.Superconductivity`
   - `lake build QFD.Cosmology.ArrowOfTime`
   - `lake build QFD.Electrodynamics.NoMonopoles`
   - `lake build QFD.Nuclear.Confinement`
   - `lake build QFD.Gravity.FrozenStarRadiation`
   - `lake build QFD.Computing.RotorLogic`
   - `lake build QFD.Cosmology.DarkMatterDensity`
   - `lake build QFD.Electrodynamics.LambShift`
   - `lake build QFD.Gravity.GravitationalWaves`
   - `lake build QFD.Weak.DoubleBetaDecay`
   - `lake build QFD.Nuclear.StabilityLimit`
   - `lake build QFD.Relativity.TimeDilationMechanism`
   - `lake build QFD.Electrodynamics.AharonovBohm`
   - `lake build QFD.Nuclear.QCDLattice`
   - `lake build QFD.Cosmology.AxisOfEvil`
   Running the individual modules avoids triggering a repo-wide rebuild.

2. **Koide relation** – The new `PhaseAlignedTriplet` record captures the phase-
   aligned mass ansatz and exposes the Koide constraint via the
   `koide_relation` field.  A future pass needs to supply the actual geometric
   proof that populates this field.

3. **Gravitational observables** – `PerihelionShift` and `SnellLensing` encode
   the symbolic statements while delegating the heavy orbital integral to the
   existing physics notebooks.  Fill in the derivations when that analysis is
   available.

4. **Boundary/Deuteron models** – The nuclear files now provide smooth toy
   profiles so other modules can depend on them without `sorry`.  Replace these
   placeholders with the full Yukawa/soliton expressions when ready.

5. **Cosmology drift law** – `HubbleDrift.redshift_is_exponential_decay` keeps
   the exponential solution shape wired up but still assumes the solution form.
   Hook it up to the real ODE proof (likely via the MVT lemma) before locking
   down publication artifacts.

6. **Cl33 infrastructure** – `QFD/GA/Cl33Instances.lean` now provides
   `Nontrivial`/`zero_ne_one` instances, injectivity of `algebraMap`, and
   non-vanishing of the basis generators.  Import this file wherever the physics
   proofs need access to those lemmas.

7. **LEAN_CODING_GUIDE review** – Re-read the guide (covers failure patterns,
   namespace rules, Unicode, automation). No new changes needed right now; just
   keep future edits consistent with the documented patterns and checklist.

8. **REFACTORING_TASKS.md** – For the other AI: Detailed refactoring instructions
   for `YukawaDerivation.lean` (blocks BreatherModes). Main issue: `lambda` is a
   reserved keyword in Lean 4 - must be renamed to `lam` throughout. Also includes
   doc-string formatting fixes and equation theorem rewrites. See file for complete
   checklist and expected outcomes.
