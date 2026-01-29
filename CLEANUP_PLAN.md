# QFD Repository Cleanup Plan

**Created**: 2026-01-29
**Purpose**: Safe removal of deprecated/redundant directories while preserving capabilities
**Estimated savings**: ~2GB

---

## Executive Summary

The repository contains multiple version iterations (v15 → v17 → v18 → V21 → V22) with significant overlap. This plan ensures V22 (current) is self-contained before removing older versions.

### Key Principle
> **Work backwards from newest**: Verify V22 stands alone before removing anything it might depend on.

---

## Version Dependency Graph

```
V22 (Current - Dec 2025)
├── V22_Lepton_Analysis/      ← ACTIVE (6.4 MB)
├── V22_Nuclear_Analysis/     ← ACTIVE (1.9 MB)
├── V22_Supernova_Analysis/   ← ACTIVE (676 KB)
│   └── Uses: data/raw/des5yr_v21_*.csv (already local)
│
├── V22_Lepton_Analysis_V2/   ← DEPRECATED (640 KB) ⚠️
│   └── Writes TO main V22 (dependency inversion)
│
V21 (Incomplete - 66 MB)
├── projects/astrophysics/V21 Supernova Analysis package/
│   ├── Has: v17_*.py modules (duplicate)
│   ├── Missing: Stage2/Stage3 from v18
│   └── Status: Incomplete, superseded by V22
│
v18 (Working Reference - 176 KB)
├── v18/pipeline/
│   ├── core/v17_*.py (foundation modules)
│   └── stages/stage[1-3]_*.py (MCMC pipeline)
│
v17 (Foundation Modules - embedded in v18, V21, summary/)
├── v17_data.py, v17_lightcurve_model.py, v17_qfd_model.py
│   └── Duplicated in 3 locations!
│
v15 (Archived - 1.8 GB!)
├── projects/astrophysics/qfd-supernova-v15/
│   ├── data/DES-SN5YR-1.2/ (1.8 GB - mostly unused)
│   └── Only 2 files actually used from entire 1.8GB
│
archive/ (23 MB)
├── LaGrangianSolitons_deprecated/
└── v15/ (code only, no data)
```

---

## Space Analysis

| Directory | Size | Status | Action |
|-----------|------|--------|--------|
| `qfd-supernova-v15/data/DES-SN5YR-1.2/` | **1.8 GB** | 99% unused | Clean to ~20 MB |
| `V21 Supernova Analysis package/` | 66 MB | Incomplete | Archive after assessment |
| `archive/` | 23 MB | Historical | Review for removal |
| `V22_Lepton_Analysis_V2/` | 640 KB | Deprecated | Remove |
| `summary/` (v17 duplicates) | ~80 KB | Redundant | Consolidate |
| **Total potential savings** | **~1.9 GB** | | |

---

## Cleanup Steps (In Order)

### Phase 1: Document & Verify (No Deletions)

#### Step 1.1: Verify V22 Self-Containment
```bash
# Check V22 Supernova external dependencies
grep -r "SupernovaSrc\|v18/\|v21/" V22_Supernova_Analysis/ --include="*.py"

# Check V22 Lepton external dependencies
grep -r "/home/tracy" V22_Lepton_Analysis/ --include="*.py" | grep -v "QFD_SpectralGap"

# Check V22 Nuclear external dependencies
grep -r "/home/tracy" V22_Nuclear_Analysis/ --include="*.py" | grep -v "QFD_SpectralGap"
```

**Expected result**: V22 should only reference `data/raw/` for external data.

#### Step 1.2: Verify data/raw/ Has Required Files
```bash
ls -la data/raw/des5yr_*.csv
# Should have: des5yr_v21_exact.csv, des5yr_v21_SIGN_CORRECTED.csv, etc.
```

### Phase 2: Safe Removals (Low Risk)

#### Step 2.1: Remove V22_Lepton_Analysis_V2 (640 KB)
**Reason**: Deprecated Dec 23 AM snapshot. All scripts write to MAIN V22 directory anyway.

```bash
# First, document what's there
ls -la V22_Lepton_Analysis_V2/ > /tmp/v2_contents_backup.txt
ls -la projects/particle-physics/V22_Lepton_Analysis_V2/ >> /tmp/v2_contents_backup.txt

# Remove both locations (they're duplicates)
rm -rf V22_Lepton_Analysis_V2/
rm -rf projects/particle-physics/V22_Lepton_Analysis_V2/
```

#### Step 2.2: Clean DES-SN5YR-1.2 (Save ~1.7 GB)
**Reason**: Only 2 files used from entire 1.8GB directory.

**Files actually used** (keep these):
- `4_DISTANCES_COVMAT/DES-SN5YR_HD.csv`
- `4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv`

```bash
cd projects/astrophysics/qfd-supernova-v15/data/DES-SN5YR-1.2/

# Backup the needed files
cp -r 4_DISTANCES_COVMAT/ ../DES_distances_backup/

# Remove unused subdirectories (CAREFUL!)
rm -rf 0_DATA/ 1_SIMULATIONS/ 2_LCFIT_MODEL/ 3_CLASSIFICATION/
rm -rf 5_COSMOLOGY/ 6_DCR_CORRECTIONS/ 7_PIPPIN_FILES/

# Or safer: move to archive
mkdir -p /tmp/des_archive
mv 0_DATA 1_SIMULATIONS 2_LCFIT_MODEL 3_CLASSIFICATION \
   5_COSMOLOGY 6_DCR_CORRECTIONS 7_PIPPIN_FILES /tmp/des_archive/
```

### Phase 3: Consolidate Duplicates

#### Step 3.1: Consolidate v17 Modules
**Current locations** (all identical or near-identical):
1. `v18/pipeline/core/v17_*.py` ← **CANONICAL**
2. `projects/astrophysics/V21 Supernova Analysis package/v17_*.py`
3. `summary/v17_*.py`

**Action**: Keep v18 as canonical, remove others OR symlink.

```bash
# Verify they're identical
diff v18/pipeline/core/v17_data.py summary/v17_data.py
diff v18/pipeline/core/v17_lightcurve_model.py summary/v17_lightcurve_model.py

# If identical, remove duplicates
rm summary/v17_*.py
rm "projects/astrophysics/V21 Supernova Analysis package/v17_*.py"

# Update any imports in summary/ to use v18 path
```

### Phase 4: Archive Assessment (Higher Risk)

#### Step 4.1: Assess V21 Package
**Question**: Is V21 still needed given V22 exists?

**Check for unique content**:
```bash
# Files in V21 not in V22
diff -rq "projects/astrophysics/V21 Supernova Analysis package/" V22_Supernova_Analysis/
```

**If V21 has no unique capability**: Archive to `archive/V21_deprecated/`

#### Step 4.2: Assess v18 Pipeline
**Question**: Does V22 Supernova actually need v18?

**Current dependencies**:
- V22 uses `data/raw/des5yr_v18_working.csv` (already extracted)
- V22 validates against v18 params but doesn't import v18 code

**If V22 is truly standalone**: Archive v18 to `archive/v18/`

#### Step 4.3: Assess v15 Remainder
After DES cleanup, `qfd-supernova-v15/` will be ~25MB (mostly figures and small CSV).

**Check if anything references it**:
```bash
grep -r "qfd-supernova-v15" . --include="*.py" | grep -v ".lake"
```

**If only external refs (SupernovaSrc)**: Can archive.

### Phase 5: Final Cleanup

#### Step 5.1: Remove archive/LaGrangianSolitons_deprecated
```bash
rm -rf archive/LaGrangianSolitons_deprecated/
```

#### Step 5.2: Clean Empty Directories
```bash
find . -type d -empty -delete
```

---

## Files That Must NOT Be Deleted

### Active V22 Work
- `V22_Lepton_Analysis/` (all contents)
- `V22_Nuclear_Analysis/` (all contents)
- `V22_Supernova_Analysis/` (all contents)

### Data Dependencies
- `data/raw/des5yr_*.csv` (V22 needs these)
- `data/raw/ame2020*.csv` (nuclear data)

### Lean Formalization
- `projects/Lean4/` (entire directory - PROTECTED)

### Core Library
- `qfd/` (Python library)

### Documentation
- `CLAUDE.md`, `Lepton.md`, `README.md`
- All `*_SUMMARY.md` files

---

## Verification After Cleanup

Run these tests after each phase:

```bash
# 1. Lean still builds
cd projects/Lean4 && lake build QFD.GA.Cl33

# 2. V22 Lepton validation works
python3 validate_koide_beta3058.py

# 3. No broken imports in V22
cd V22_Lepton_Analysis && python3 -c "import sys; sys.path.insert(0, '.'); import v22_lepton_mass_solver"
cd V22_Supernova_Analysis && python3 -c "import sys; sys.path.insert(0, 'scripts'); import v22_qfd_fit_proper"

# 4. Git status clean (no accidentally deleted tracked files)
git status
```

---

## Rollback Plan

Before any deletions:
```bash
# Create safety backup
tar -czf ~/qfd_pre_cleanup_backup.tar.gz \
    V22_Lepton_Analysis_V2 \
    projects/particle-physics/V22_Lepton_Analysis_V2 \
    projects/astrophysics/qfd-supernova-v15/data/DES-SN5YR-1.2 \
    summary/v17_*.py
```

If something breaks:
```bash
# Restore from backup
tar -xzf ~/qfd_pre_cleanup_backup.tar.gz
```

---

## Summary Checklist

- [ ] **Phase 1**: Verify V22 self-contained (no broken refs)
- [ ] **Phase 2.1**: Remove V22_Lepton_Analysis_V2 (640 KB)
- [ ] **Phase 2.2**: Clean DES-SN5YR-1.2 unused dirs (~1.7 GB)
- [ ] **Phase 3.1**: Consolidate v17 duplicates (~80 KB)
- [ ] **Phase 4.1**: Archive V21 if not needed (66 MB)
- [ ] **Phase 4.2**: Archive v18 if V22 standalone (176 KB)
- [ ] **Phase 4.3**: Archive v15 remainder (~25 MB)
- [ ] **Phase 5**: Final cleanup (empty dirs, deprecated archives)
- [ ] **Verify**: All tests pass after cleanup

**Total estimated savings**: ~1.9 GB
