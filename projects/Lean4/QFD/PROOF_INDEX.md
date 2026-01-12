# Proof Index - Meat & Potatoes

**Version**: 2.1
**Last Updated**: 2026-01-11
**Total Proven**: 1,133 (914 theorems + 219 lemmas)
**Lean Files**: 240 (synced: internal = public)
**Sorries**: 0
**Axioms**: 11 (centralized in Physics/Postulates.lean)

---

## Summary by Category

| Category | Proofs | Description |
|----------|--------|-------------|
| **MEAT** | **785** | Core physics theorems |
| **POTATOES** | **348** | Infrastructure & scaffolding |
| **Total** | **1,133** | |

---

## MEAT: Core Physics (785 proofs)

These are the substantive physics results - the actual claims about nature.

### Nuclear Physics (209 proofs)
**Location**: `Nuclear/`

Core compression law, binding energies, magic numbers, fission asymmetry.

| File | Proofs | Key Results |
|------|--------|-------------|
| CoreCompressionLaw.lean | 45+ | β-derived binding formula |
| SymmetryEnergy*.lean | 30+ | Isospin dependence |
| TimeCliff.lean | 20+ | Stability boundaries |
| BindingMassScale.lean | 15+ | Mass-energy relations |

### Lepton Sector (137 proofs)
**Location**: `Lepton/`

Mass spectrum, g-2 anomaly, generation structure, Koide relation.

| File | Proofs | Key Results |
|------|--------|-------------|
| GeometricG2.lean | 25+ | Muon g-2 from geometry |
| MassSpectrum.lean | 20+ | e/μ/τ mass ratios |
| VortexStability.lean | 30+ | Hill vortex stability |
| KoideRelation.lean | 15+ | Q = 2/3 theorem |

### Cosmology (91 proofs)
**Location**: `Cosmology/`

CMB axis alignment, quadrupole/octupole uniqueness, vacuum drag.

| File | Proofs | Key Results |
|------|--------|-------------|
| AxisExtraction.lean | 25+ | Quadrupole axis uniqueness (IT.1) |
| OctupoleExtraction.lean | 20+ | Octupole axis uniqueness (IT.2) |
| CoaxialAlignment.lean | 15+ | Axis coincidence (IT.4) |
| Polarization.lean | 10+ | Sign-flip falsifiability |

### Spacetime Emergence (65 proofs)
**Location**: Top-level files

The central result: 4D Minkowski emerges from Cl(3,3) + internal rotation.

| File | Proofs | Key Results |
|------|--------|-------------|
| EmergentAlgebra.lean | 30+ | Centralizer = Cl(3,1) |
| SpectralGap.lean | 20+ | Extra dimensions suppressed |
| GoldenLoop.lean | 15+ | α → β derivation |

### Hydrogen/Photon (62 proofs)
**Location**: `Hydrogen/`

Photon soliton structure, energy levels, absorption/emission.

| File | Proofs | Key Results |
|------|--------|-------------|
| PhotonSoliton.lean | 20+ | Soliton stability gates |
| SpeedOfLight.lean | 10+ | c = √(β/ρ) derivation |
| HbarDerivation.lean | 5+ | ℏ scale invariance |

### Soliton Theory (52 proofs)
**Location**: `Soliton/`

Topological stability, mass-energy density, quantization conditions.

| File | Proofs | Key Results |
|------|--------|-------------|
| TopologicalStability.lean | 20+ | Winding number conservation |
| MassEnergyDensity.lean | 15+ | E = mc² emergence |
| Quantization.lean | 10+ | Discrete charge spectrum |

### Gravity (42 proofs)
**Location**: `Gravity/`

Schwarzschild metric, geodesics, horizon structure.

| File | Proofs | Key Results |
|------|--------|-------------|
| Geodesic.lean | 15+ | Free-fall equations |
| Schwarzschild.lean | 10+ | Metric derivation |

### Black Holes (17 proofs)
**Location**: `BlackHole/`

Saturation limits, horizon thermodynamics.

### Conservation Laws (26 proofs)
**Location**: `Conservation/`

Energy, momentum, charge conservation from geometry.

### QM Translation (25 proofs)
**Location**: `QM_Translation/`

Pauli matrices, Dirac equation, phase as rotation.

| File | Proofs | Key Results |
|------|--------|-------------|
| DiracRealization.lean | 10+ | γ-matrices from Cl(3,3) |
| RealDiracEquation.lean | 8+ | Mass = internal momentum |
| SchrodingerEvolution.lean | 5+ | e^{iθ} → e^{Bθ} |

### Charge Quantization (23 proofs)
**Location**: `Charge/`

Topological origin of discrete charge values.

### Electrodynamics (14 proofs)
**Location**: `Electrodynamics/`

Maxwell equations from Cl(3,3), field tensor.

---

## POTATOES: Infrastructure (348 proofs)

Supporting lemmas, algebraic scaffolding, type definitions.

### Geometric Algebra (52 proofs)
**Location**: `GA/`

Cl(3,3) foundations - basis products, anticommutation, conjugation.

| File | Proofs | Key Results |
|------|--------|-------------|
| BasisOperations.lean | 20+ | e_i² = ±1, anticommutation |
| PhaseCentralizer.lean | 15+ | B² = -1, phase structure |
| Conjugation.lean | 10+ | Reversal, grade involution |

### Math Scaffolding (52 proofs)
**Location**: `Math/`

Pure mathematics supporting physics derivations.

### Physics Postulates (33 proofs)
**Location**: `Physics/`

Axiom centralization, constraint structures.

### Other Infrastructure (211 proofs)

| Directory | Proofs | Purpose |
|-----------|--------|---------|
| Vacuum | 13 | Vacuum field properties |
| Rift | 15 | Boundary dynamics |
| Topology | 10 | Topological foundations |
| Atomic | 9 | Chaos/resonance |
| Thermodynamics | 9 | Statistical mechanics |
| Classical | 7 | Classical limits |
| Weak | 6 | Weak interaction sketches |
| Schema | 3 | Type definitions |
| Test | 1 | Build verification |

---

## Quick Lookup

### Method 1: Search CLAIMS_INDEX.txt
```bash
# Find all theorems about charge
grep -i "charge" QFD/CLAIMS_INDEX.txt

# Find theorems in Nuclear/
grep "Nuclear/" QFD/CLAIMS_INDEX.txt | head -20

# Count by directory
grep "Lepton/" QFD/CLAIMS_INDEX.txt | wc -l
```

### Method 2: Browse by File
```bash
# List all theorem names in a file
grep "^theorem" QFD/Cosmology/AxisExtraction.lean
```

### Method 3: ProofLedger.lean
Maps plain-English claims to specific theorems with line numbers.

---

## Verification Commands

```bash
# Total proofs
wc -l QFD/CLAIMS_INDEX.txt  # → 1,133

# Theorem vs lemma split
grep "^theorem" QFD/CLAIMS_INDEX.txt | wc -l  # → 914
grep "^lemma" QFD/CLAIMS_INDEX.txt | wc -l   # → 219

# Sorry check (should be 0)
grep -r "sorry" QFD/ --include="*.lean" | wc -l  # → 0

# Build verification
lake build QFD  # Should complete with 0 errors
```
