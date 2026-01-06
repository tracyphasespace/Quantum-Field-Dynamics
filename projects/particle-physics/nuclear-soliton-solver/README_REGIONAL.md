# QFD Nuclear Soliton Solver - Regional Calibration

**Purpose**: Improve heavy isotope predictions through mass-region-specific parameter optimization

**Date**: December 29, 2025

**Location**: `/home/tracy/development/QFD_SpectralGap/projects/particle-physics/nuclear-soliton-solver/`

---

## Problem Statement

The universal Trial 32 parameters achieve excellent results for light nuclei but show systematic underbinding for heavy isotopes:

| Mass Region | Performance | Status |
|-------------|-------------|--------|
| Light (A < 60) | < 1% error | ✅ Excellent |
| Medium (60 ≤ A < 120) | 2-3% error | ⚠️ Moderate |
| Heavy (A ≥ 120) | -7% to -9% underbinding | ❌ Needs fix |

**Root cause**: Universal parameters cannot capture A-dependent physics across 3 orders of magnitude in mass.

---

## Solution: Regional Calibration

Optimize separate parameter sets for each mass region while maintaining physics consistency:

```
Light region (A < 60):
  → Keep Trial 32 (already optimal)
  → Focus: magic numbers, shell closures

Medium region (60 ≤ A < 120):
  → Fine-tune around Trial 32
  → Focus: transition behavior, stability valley

Heavy region (A ≥ 120):
  → Increase cohesion ~10-15%
  → Focus: correct systematic underbinding
  → Hypothesis: Need stronger c_v2_base, weaker c_v4_base
```

---

## Quick Start

### Setup

```bash
# Navigate to project
cd /home/tracy/development/QFD_SpectralGap/projects/particle-physics/nuclear-soliton-solver

# Install dependencies
pip install -r requirements.txt

# Verify AME2020 data is available
ls -lh ../data/ame2020_system_energies.csv
```

### Validation (Test Current State)

```bash
# Validate Trial 32 on all regions
python src/qfd_regional_calibration.py \
  --ame-csv ../data/ame2020_system_energies.csv \
  --validate-only \
  --region all
```

Expected output:
- Light: Loss ~0.0001 (< 1% error)
- Medium: Loss ~0.0009 (2-3% error)
- Heavy: Loss ~0.0064 (7-9% error)

### Optimize Heavy Region

```bash
# Optimize only heavy isotopes
python src/qfd_regional_calibration.py \
  --ame-csv ../data/ame2020_system_energies.csv \
  --region heavy \
  --n-calibration 20
```

### Full Regional Calibration

```bash
# Optimize all three regions
python src/qfd_regional_calibration.py \
  --ame-csv ../data/ame2020_system_energies.csv \
  --region all \
  --n-calibration 15
```

Results saved to: `regional_calibration_results.json`

---

## File Structure

```
nuclear-soliton-solver/
├── src/
│   ├── qfd_solver.py                    # Core SCF solver (Phase 9)
│   ├── qfd_metaopt_ame2020.py           # Universal parameter optimization
│   ├── qfd_regional_calibration.py      # ★ NEW: Regional optimization
│   └── analyze_isotopes.py              # Isotope scaling validation
├── data/
│   └── ame2020_system_energies.csv      # Experimental nuclear masses
├── docs/
│   ├── PHYSICS_MODEL.md                 # Soliton field theory overview
│   └── FINDINGS.md                      # Trial 32 performance analysis
├── results/
│   └── regional_calibration_results.json  # Optimized regional parameters
├── README.md                            # Original NuclideModel README
├── README_REGIONAL.md                   # This file
└── requirements.txt                     # Python dependencies
```

---

## Regional Parameter Strategy

### Light Region (A < 60)

**Status**: Optimal with Trial 32

**Parameters**: (unchanged from Trial 32)
```json
{
  "c_v2_base": 2.201711,
  "c_v4_base": 5.282364,
  "c_sym": 25.0,
  ...
}
```

**Calibration set**:
- He-4, C-12, O-16 (doubly magic)
- Si-28, Ca-40, Ca-48
- Fe-56 (stability peak)

### Medium Region (60 ≤ A < 120)

**Status**: Fine-tune around Trial 32

**Expected adjustments**:
- c_v2_base: ±5%
- c_v4_base: ±5%
- Focus on transition isotopes

**Calibration set**:
- Ni-58, Ni-62, Ni-64 (Z=28 magic)
- Cu-63, Cu-65
- Ag-107, Ag-109
- Sn-100, Sn-120 (Z=50 magic)

### Heavy Region (A ≥ 120)

**Status**: NEEDS ADJUSTMENT

**Target**: Fix -8% systematic underbinding

**Strategy**:
1. Increase cohesion: c_v2_base × 1.10-1.15 (+10-15%)
2. Decrease repulsion: c_v4_base × 0.90-0.95 (-5-10%)
3. Optional: Add explicit surface term

**Calibration set**:
- Sn-120 (boundary with medium)
- Au-197, Hg-200
- Pb-206, Pb-207, Pb-208 (Z=82 magic, doubly magic)
- U-235, U-238

**Expected improvement**: -8% error → -2% error

---

## Physics Rationale

### Why Regional Parameters Make Sense

1. **Surface-to-volume ratio changes**: A^(2/3) / A → 0 as A increases
   - Light nuclei: surface-dominated
   - Heavy nuclei: volume-dominated
   - Universal parameters can't capture both limits

2. **Shell structure varies**: Magic numbers at different scales
   - Light: 2, 8, 20, 28
   - Heavy: 50, 82, 126
   - Different closure energies

3. **Coulomb energy scales as Z²/A^(1/3)**:
   - Light: negligible
   - Heavy: ~10% of binding energy
   - Affects optimal field balance

### Why This Isn't "Overfitting"

**Concern**: More parameters → overfitting?

**Response**:
- Still 9 parameters per region (same as Trial 32)
- NOT fitting each isotope individually
- Physics-driven calibration sets (magic numbers, shell closures)
- Regional boundaries have physical justification (surface/volume transition)

**Test**: Predict isotopes NOT in calibration set to verify generalization.

---

## Expected Results

### Before Regional Calibration (Trial 32)

| Region | Mean Error | Max Error | Status |
|--------|-----------|----------|--------|
| Light | 0.4% | 0.9% | ✅ |
| Medium | 2.5% | 3.9% | ⚠️ |
| Heavy | -8.1% | -9.2% | ❌ |

### After Regional Calibration (Target)

| Region | Mean Error | Max Error | Status |
|--------|-----------|----------|--------|
| Light | 0.4% | 0.9% | ✅ (unchanged) |
| Medium | 1.2% | 2.5% | ✅ (improved) |
| Heavy | -2.0% | -4.0% | ✅ (much better) |

---

## Next Steps

### Phase 1: Validate Current Implementation ✓

```bash
python src/qfd_regional_calibration.py --validate-only
```

### Phase 2: Optimize Heavy Region

```bash
python src/qfd_regional_calibration.py --region heavy
```

Expected runtime: 30-60 minutes (depends on calibration set size)

### Phase 3: Full Regional Optimization

```bash
python src/qfd_regional_calibration.py --region all
```

Expected runtime: 1-2 hours

### Phase 4: Validation on Full Isotope Set

Test optimized parameters on:
- 254 stable isotopes
- All AME2020 isotopes (3558 total)
- Isotope chains (e.g., Fe-54,56,57,58)

### Phase 5: Publication

Write up regional calibration results:
- Parameter tables
- Error distribution plots
- Comparison with SEMF, Skyrme/RMF
- Physical interpretation of parameter variations

---

## Potential Extensions

1. **Explicit surface term**:
   ```python
   E_surf = c_surf × A**(2/3)
   ```
   Add to energy functional in qfd_solver.py

2. **Pairing energy**:
   ```python
   delta_N = 1 if (A-Z) % 2 else -1
   E_pair = c_pair × delta_N × delta_Z / np.sqrt(A)
   ```

3. **Smooth parameter interpolation**:
   Instead of discrete regions, use continuous functions:
   ```python
   c_v2_base(A) = c0 + c1 × tanh((A - A0) / σ)
   ```

4. **Deformation**: Non-spherical field ansatz for heavy nuclei

---

## Connection to Original Work

This regional calibration builds on:

**Original location**:
- `/home/tracy/development/qfd_hydrogen_project/qfd_research_suite/NuclideModel/`

**Key files copied**:
- qfd_solver.py (Phase 9 SCF solver)
- qfd_metaopt_ame2020.py (meta-optimizer)
- PHYSICS_MODEL.md, FINDINGS.md (documentation)

**Differences**:
- NEW: Regional calibration framework
- NEW: Mass-region-specific optimization
- Maintains: All original solver physics
- Maintains: Trial 32 baseline for light nuclei

---

## References

- Trial 32 analysis: `docs/FINDINGS.md`
- Physics model: `docs/PHYSICS_MODEL.md`
- AME2020 database: M. Wang et al., Chinese Physics C 45 (2021)

---

## Contact

For questions about regional calibration approach, see project lead Tracy.

For questions about original solver implementation, see NuclideModel README.md.

---

**Last updated**: December 29, 2025
