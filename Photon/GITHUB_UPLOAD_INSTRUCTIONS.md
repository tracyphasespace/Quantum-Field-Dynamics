# GitHub Upload Instructions

**Target Repository**: https://github.com/tracyphasespace/Quantum-Field-Dynamics
**Target Directory**: `projects/field-theory/Photons_&_Solitons/`
**Date**: 2026-01-04

---

## Files to Upload

### Core Documentation (4 files)

```
GITHUB_README.md  →  README.md  (rename on upload)
MASTER_INDEX.md
QFD_VALIDATION_STATUS.md
g-2.md
```

### Analysis Scripts (19 files)

All from `analysis/` directory:

```
analysis/validate_g2_prediction.py
analysis/validate_zeeman_vortex_torque.py
analysis/validate_spinorbit_chaos.py
analysis/validate_lyapunov_predictability_horizon.py
analysis/validate_hydrodynamic_c_hbar_bridge.py
analysis/validate_all_constants_as_material_properties.py
analysis/validate_vortex_force_law.py
analysis/validate_chaos_alignment_decay.py
analysis/validate_hbar_scaling.py
analysis/derive_constants.py
analysis/integrate_hbar.py
analysis/dimensional_audit.py
analysis/validate_hydrodynamic_c.py
analysis/soliton_balance_simulation.py
analysis/validate_unified_forces.py
analysis/validate_fine_structure_scaling.py
analysis/validate_lepton_isomers.py
analysis/lepton_stability_3param.py
analysis/lepton_energy_partition.py
```

### Support Files (2 files)

```
requirements.txt
run_all.py
```

---

## Upload Steps

### Option 1: Command Line (Recommended)

```bash
# Navigate to your local repo
cd /path/to/Quantum-Field-Dynamics

# Create target directory
mkdir -p projects/field-theory/Photons_&_Solitons

# Copy files from Photon directory
SOURCE=/home/tracy/development/QFD_SpectralGap/Photon
TARGET=projects/field-theory/Photons_&_Solitons

# Copy documentation
cp $SOURCE/GITHUB_README.md $TARGET/README.md
cp $SOURCE/MASTER_INDEX.md $TARGET/
cp $SOURCE/QFD_VALIDATION_STATUS.md $TARGET/
cp $SOURCE/g-2.md $TARGET/

# Copy analysis scripts
cp $SOURCE/analysis/*.py $TARGET/analysis/

# Copy support files
cp $SOURCE/requirements.txt $TARGET/
cp $SOURCE/run_all.py $TARGET/

# Add to git
git add projects/field-theory/Photons_&_Solitons/
git commit -m "Add QFD Photon Sector: g-2 prediction validated to 0.45%

Key results:
- Lepton anomalous magnetic moment from vortex geometry
- V₄ = -ξ/β predicts A₂ coefficient to 0.45% (Tier A independent observable)
- Zeeman splitting exact match (0.000% error)
- Complete validation suite (19 scripts)
- Publication-ready g-2 paper draft

See README.md for quick start and validation details."

# Push to GitHub
git push origin main
```

### Option 2: GitHub Web Interface

1. Navigate to: https://github.com/tracyphasespace/Quantum-Field-Dynamics
2. Go to `projects/field-theory/`
3. Click "Add file" → "Create new file"
4. Name it: `Photons_&_Solitons/README.md`
5. Copy-paste content from `GITHUB_README.md`
6. Click "Commit new file"
7. Repeat for other documentation files
8. Create `analysis/` subdirectory and upload scripts

---

## Verification Checklist

After upload, verify:

- [ ] README.md displays correctly on GitHub
- [ ] All 19 analysis scripts uploaded to `analysis/`
- [ ] requirements.txt present
- [ ] run_all.py present
- [ ] g-2.md formatted correctly (equations may need LaTeX rendering)
- [ ] MASTER_INDEX.md links work
- [ ] QFD_VALIDATION_STATUS.md complete

---

## GitHub Repository Structure (After Upload)

```
Quantum-Field-Dynamics/
└── projects/
    └── field-theory/
        └── Photons_&_Solitons/
            ├── README.md                     ← GITHUB_README.md (renamed)
            ├── MASTER_INDEX.md
            ├── QFD_VALIDATION_STATUS.md
            ├── g-2.md                        ← Publication draft
            │
            ├── analysis/                     ← All validation scripts
            │   ├── validate_g2_prediction.py ← ⭐ Main result
            │   ├── validate_zeeman_vortex_torque.py
            │   ├── validate_spinorbit_chaos.py
            │   ├── validate_lyapunov_predictability_horizon.py
            │   ├── validate_hydrodynamic_c_hbar_bridge.py
            │   ├── validate_all_constants_as_material_properties.py
            │   ├── validate_vortex_force_law.py
            │   ├── validate_chaos_alignment_decay.py
            │   ├── derive_constants.py
            │   ├── integrate_hbar.py
            │   ├── dimensional_audit.py
            │   ├── validate_hydrodynamic_c.py
            │   ├── validate_hbar_scaling.py
            │   ├── soliton_balance_simulation.py
            │   ├── validate_unified_forces.py
            │   ├── validate_fine_structure_scaling.py
            │   ├── validate_lepton_isomers.py
            │   ├── lepton_stability_3param.py
            │   └── lepton_energy_partition.py
            │
            ├── requirements.txt
            └── run_all.py
```

---

## Post-Upload Actions

### 1. Test Installation

Clone your repo and test:

```bash
git clone https://github.com/tracyphasespace/Quantum-Field-Dynamics.git
cd Quantum-Field-Dynamics/projects/field-theory/Photons_&_Solitons
pip install -r requirements.txt
python3 run_all.py
```

**Expected**: 19/19 scripts pass

### 2. Create Release Tag

```bash
git tag -a v1.0-g2-validation -m "QFD Photon Sector v1.0: g-2 prediction validated to 0.45%"
git push origin v1.0-g2-validation
```

### 3. Add Topics (GitHub Web)

Add these topics to your repository for discoverability:
- `quantum-field-theory`
- `vortex-dynamics`
- `anomalous-magnetic-moment`
- `qed`
- `lepton-physics`
- `superfluid-vacuum`

### 4. Create DOI (Optional)

Link to Zenodo for citeable DOI:
1. Go to https://zenodo.org/
2. Link your GitHub repository
3. Create release
4. Get DOI
5. Add badge to README

---

## Commit Message Template

```
Add QFD Photon Sector: g-2 prediction validated to 0.45%

This commit adds the complete QFD Photon Sector validation suite,
demonstrating prediction of the lepton anomalous magnetic moment
(g-2) from vortex geometry.

Key Results:
- Tier A: g-2 prediction (0.45% error) - independent observable ✅
- Tier A: Zeeman splitting (0.000% error) - exact match ✅
- Tier B: Chaos origin (λ = 0.023 > 0) - deterministic mechanism ✅
- Tier B: c-ℏ coupling (ℏ/√β constant) - scaling verified ✅

Files Added:
- Complete validation suite (19 Python scripts)
- Publication-ready g-2 paper draft
- Comprehensive documentation (Tier A/B/C/D)
- Quick start guide with run_all.py

Scientific Significance:
This represents the first non-circular prediction of a QED coefficient
(vacuum polarization A₂) from a classical vortex model. The 0.45%
agreement validates that vacuum polarization can be interpreted as
the ratio of surface tension to bulk compression in vortex geometry.

Reference: g-2.md (draft for Physical Review Letters)
Status: Publication-ready
Date: 2026-01-04
```

---

## Archive Handling (Optional)

The `archive/` directories contain historical documents from development sessions. You can:

**Option 1**: Don't upload (keep local only)
- Pro: Cleaner GitHub repo
- Con: Lose historical context

**Option 2**: Upload to separate `archive/` subdirectory
- Pro: Complete historical record
- Con: More files in repo

**Recommendation**: Don't upload archives initially. Can always add later if needed for transparency.

---

## README Badge Suggestions

Add these to the top of README.md after upload:

```markdown
![Status](https://img.shields.io/badge/status-publication--ready-brightgreen)
![g-2 Error](https://img.shields.io/badge/g--2%20error-0.45%25-blue)
![Scripts](https://img.shields.io/badge/validation%20scripts-19-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
```

---

## Social Media Announcement Template

Once uploaded:

> 🎯 New: QFD predicts lepton g-2 to 0.45% accuracy!
>
> We modeled leptons as vortices in a superfluid vacuum. The ratio of surface tension to bulk stiffness (V₄ = -ξ/β) independently predicts the QED vacuum polarization coefficient (A₂) without free parameters.
>
> ✅ Non-circular (different observable)
> ✅ 0.45% error (publication-ready)
> ✅ Mechanistic interpretation (gradient energy ↔ virtual particles)
>
> Code & paper: [GitHub link]
> #physics #QED #vortexdynamics #g2anomaly

---

## Contact

If you encounter issues with the upload:
1. Check file paths are correct
2. Verify analysis/ subdirectory created
3. Ensure requirements.txt has all dependencies
4. Test run_all.py locally before pushing

---

**Created**: 2026-01-04
**Status**: Ready for upload
**Next**: Push to GitHub, create release, announce result
