# QFD Nuclear Soliton Solver - Project Overview

**Created**: December 29, 2025
**Location**: `/home/tracy/development/QFD_SpectralGap/projects/particle-physics/nuclear-soliton-solver/`
**Purpose**: Regional calibration to fix heavy isotope underbinding

---

## What This Is

A **field-theoretic nuclear mass calculator** using soliton configurations instead of nucleon-based models.

**Key innovation**: Nuclei are coherent field structures (solitons), not collections of protons/neutrons.

---

## Current Status

### Copied from Original Location

**Source**: `/home/tracy/development/qfd_hydrogen_project/qfd_research_suite/NuclideModel/`

**Files copied** (December 29, 2025):
- ✅ `qfd_solver.py` - Phase 9 SCF solver (560 lines)
- ✅ `qfd_metaopt_ame2020.py` - Parameter optimization (544 lines)
- ✅ `analyze_isotopes.py` - Isotope validation (140 lines)
- ✅ Documentation: PHYSICS_MODEL.md, FINDINGS.md
- ✅ AME2020 data: 3558 experimental nuclear masses

### New in This Repo

**Created for regional calibration**:
- ✅ `qfd_regional_calibration.py` - Mass-region-specific optimization (NEW!)
- ✅ `README_REGIONAL.md` - Regional calibration guide
- ✅ `validate_setup.sh` - Quick setup checker
- ✅ `PROJECT_OVERVIEW.md` - This file

---

## The Problem We're Solving

### Trial 32 Performance (Universal Parameters)

| Mass Region | Performance | Example Isotopes |
|-------------|-------------|------------------|
| **Light** (A < 60) | **< 1% error** ✅ | He-4: +0.12%, O-16: +0.46%, Ca-40: -0.88% |
| **Medium** (60-119) | **2-3% error** ⚠️ | Fe-56: -2.72%, Ni-62: -3.72%, Cu-63: -3.86% |
| **Heavy** (A ≥ 120) | **-7% to -9% error** ❌ | Pb-208: -8.39%, Au-197: -8.75%, U-238: -7.71% |

### Root Cause

**Universal parameters can't handle**:
- Surface-to-volume ratio transition (A^(2/3) / A → 0 as A increases)
- Different shell structure at different mass scales
- Coulomb energy scaling (Z²/A^(1/3) becomes significant for heavy nuclei)

### Solution: Regional Calibration

Optimize **separate parameter sets** for each mass region:
- Light: Keep Trial 32 (already optimal)
- Medium: Fine-tune around Trial 32
- Heavy: **Increase cohesion 10-15%** to fix systematic underbinding

**Expected improvement**: Heavy isotope errors -8% → -2%

---

## Quick Reference

### Directory Structure

```
nuclear-soliton-solver/
├── src/
│   ├── qfd_solver.py                      # Core solver
│   ├── qfd_metaopt_ame2020.py             # Universal optimization
│   ├── qfd_regional_calibration.py        # ★ Regional optimization (NEW)
│   └── analyze_isotopes.py                # Validation
├── data/
│   └── ame2020_system_energies.csv        # 3558 experimental masses
├── docs/
│   ├── PHYSICS_MODEL.md                   # Field theory details
│   └── FINDINGS.md                        # Trial 32 results
├── results/                                # Output directory
├── README.md                              # Original NuclideModel README
├── README_REGIONAL.md                     # Regional calibration guide
├── PROJECT_OVERVIEW.md                    # This file
├── requirements.txt                       # Python dependencies
└── validate_setup.sh                      # Setup checker
```

### Key Commands

```bash
# Validate setup
./validate_setup.sh

# Install dependencies
pip install -r requirements.txt

# Validate Trial 32 on all regions (see current performance)
python src/qfd_regional_calibration.py --validate-only --region all

# Optimize heavy region only
python src/qfd_regional_calibration.py --region heavy --n-calibration 20

# Full regional optimization
python src/qfd_regional_calibration.py --region all --n-calibration 15
```

---

## Physics Model Summary

### Energy Functional

```
E_total = T_kinetic + V_cohesion + V_repulsion + V_coulomb + V_surface + V_asymmetry

Where:
  T_kinetic = ∫ |∇ψ|² dV                    (field gradients)
  V_cohesion = -α_eff ∫ ψ² dV                (attractive, promotes binding)
  V_repulsion = β_eff ∫ ψ⁴ dV                (repulsive, prevents collapse)
  V_coulomb = ∫ ρ_charge V_coulomb dV        (spectral solver)
  V_asymmetry = c_sym (N-Z)²/A^(1/3)         (charge imbalance penalty)
```

### Key Parameters (Trial 32)

| Parameter | Value | Physical Meaning |
|-----------|-------|------------------|
| c_v2_base | 2.2017 | Baseline cohesion strength |
| c_v4_base | 5.2824 | Baseline quartic repulsion |
| c_sym | 25.0 | Charge asymmetry coefficient |
| kappa_rho | 0.0298 | Density-dependent coupling |

**Regional calibration will optimize these independently for light/medium/heavy.**

---

## Expected Outcomes

### Phase 1: Validation (Completed)

✅ Setup complete
✅ Files copied
✅ AME2020 data available
✅ Regional calibration framework created

### Phase 2: Heavy Region Optimization (Next)

**Target**: Fix -8% systematic underbinding

**Strategy**:
- Increase c_v2_base by 10-15% (stronger cohesion)
- Decrease c_v4_base by 5-10% (weaker repulsion)
- Optimize c_sym for heavy nuclei

**Test isotopes**: Pb-208, Au-197, U-238

**Expected runtime**: 30-60 minutes

### Phase 3: Full Regional Calibration

**Optimize all three regions**:
- Light: Validate Trial 32 (no changes expected)
- Medium: Fine-tune (5-10% adjustments)
- Heavy: Significant adjustment (10-15%)

**Expected runtime**: 1-2 hours

### Phase 4: Validation

**Test on**:
- 254 stable isotopes
- All 3558 AME2020 isotopes
- Isotope chains (Fe-54,56,57,58)

**Publication-ready results**:
- Parameter tables
- Error distribution plots
- Comparison with other models (SEMF, Skyrme/RMF)

---

## Scientific Context

### How This Fits in QFD Framework

**QFD (Quantum Field Dynamics)** proposes:
1. Vacuum has stiffness parameter β ≈ 3.043233053
2. Particles are topological structures in this medium
3. Mass arises from energy balance in these structures

**This solver**:
- Models nuclei as soliton field configurations
- Uses QFD-native symmetry energy (not SEMF)
- Tests if β appears in nuclear binding (via calibrated parameters)

**Connection to other QFD work**:
- Lepton masses: Hill vortex model (β = 3.043233053)
- Cosmology: CMB axis of evil formalization
- Lean proofs: 500+ theorems on QFD framework

### Relation to Other Nuclear Models

| Model | Approach | Light Error | Heavy Error |
|-------|----------|-------------|-------------|
| **This solver** | Soliton fields | < 1% | -8% → -2% (target) |
| SEMF | Phenomenology | ~5% | ~2% |
| Skyrme/RMF | DFT mean-field | ~1% | ~0.5% |

**Advantage**: Conceptually novel (no nucleon assumptions)
**Challenge**: Needs regional calibration to match DFT accuracy

---

## Collaboration Notes

### Working with Original Code

**Original location preserved**:
- `/home/tracy/development/qfd_hydrogen_project/qfd_research_suite/NuclideModel/`

**This repo is a COPY, not a move**:
- Original code unchanged
- Can sync improvements back if desired
- Independent development path

### Code Quality

**Strengths**:
- Clean, well-documented Python
- Comprehensive error handling
- GPU-ready (PyTorch/CUDA)
- Physics-driven calibration

**Minor issues noted**:
- Phase version mismatch (header says "Phase-8", docs say "Phase 9")
- Global state in evaluate_parameters (functional, but unconventional)
- Hardcoded solver path (fragile if restructured)

**These don't affect functionality** - can be cleaned up later if desired.

---

## Next Actions

### Immediate (Today/Tomorrow)

1. **Run validation**:
   ```bash
   ./validate_setup.sh
   python src/qfd_regional_calibration.py --validate-only --region all
   ```

2. **Optimize heavy region**:
   ```bash
   python src/qfd_regional_calibration.py --region heavy
   ```

3. **Analyze results**:
   - Check if errors improve from -8% to -2%
   - Verify virial convergence
   - Inspect parameter changes

### Near-term (This Week)

4. **Full regional calibration**: Optimize all three regions

5. **Validation sweep**: Test on 254 stable isotopes

6. **Documentation**: Write up results in results/REGIONAL_CALIBRATION_REPORT.md

### Long-term (Next Month)

7. **Add explicit surface term**: E_surf = c_surf × A^(2/3)

8. **Add pairing energy**: Even-odd mass staggering

9. **Publication**: Submit regional calibration results

---

## Questions to Answer

### Scientific Questions

1. **Does regional calibration work?**
   - Can we get heavy errors < -2%?
   - Do optimized parameters make physical sense?

2. **How do parameters vary across regions?**
   - Is c_v2_base(heavy) ≈ 1.15 × c_v2_base(light)?
   - What about c_sym, kappa_rho?

3. **Can we predict the parameter trends?**
   - Use physics arguments to derive functional forms?
   - c_v2_base(A) = c0 + c1 × f(A)?

### Technical Questions

4. **Do we need smooth interpolation at boundaries?**
   - A=59 vs A=60: discrete jump or smooth transition?

5. **Should we add more physics terms?**
   - Explicit surface energy
   - Pairing energy
   - Deformation

---

## Resources

### Documentation in This Repo

- `README_REGIONAL.md` - Complete regional calibration guide
- `docs/PHYSICS_MODEL.md` - Soliton field theory details
- `docs/FINDINGS.md` - Trial 32 performance analysis

### External References

- AME2020: M. Wang et al., Chinese Physics C 45 (2021)
- QFD Framework: `/home/tracy/development/QFD_SpectralGap/CLAUDE.md`
- Original solver: `/home/tracy/development/qfd_hydrogen_project/qfd_research_suite/NuclideModel/`

---

## Contact

**Project Lead**: Tracy
**Location**: `/home/tracy/development/QFD_SpectralGap/projects/particle-physics/nuclear-soliton-solver/`
**Date Created**: December 29, 2025

---

**Summary**: This repo contains a soliton-based nuclear mass solver with a NEW regional calibration framework to fix systematic heavy isotope underbinding. Ready to optimize heavy region parameters and improve -8% errors to -2% target.
