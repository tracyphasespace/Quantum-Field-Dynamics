# Cross-Project Manifest: QFD ↔ NSE ↔ Riemann

**Purpose**: Track file dependencies and sync requirements between the three Lean projects.
**Last Updated**: 2025-01-14
**Maintainer**: AI assistants should update this file when syncing changes.

---

## Project Locations

| Project | Path | GitHub |
|---------|------|--------|
| **Master QFD** | `/home/tracy/development/QFD_SpectralGap/projects/Lean4/` | tracyphasespace/QFD-Universe |
| **Navier-Stokes** | `/home/tracy/development/GeminiTest3/NavierStokesPaper/Lean/` | (private) |
| **Riemann** | `/home/tracy/development/Riemann/Lean/Riemann/` | (private) |

---

## Sync Direction Legend

- **⬇️ BORROW**: File flows FROM Master TO project (project uses Master's version)
- **⬆️ CONTRIBUTE**: File flows FROM project TO Master (project has improvements)
- **🔄 BIDIRECTIONAL**: Both directions possible (diverged, needs manual merge)
- **🆕 PROJECT-ONLY**: File exists only in project (project-specific)
- **📦 INSIGHT**: General insight extracted and added to Master

---

## Navier-Stokes Project

### Borrowed Files (NSE ← Master)

These files in NSE are copies/adaptations from Master QFD. When Master updates, NSE may need updates.

| NSE File | Master Source | Namespace | Status |
|----------|---------------|-----------|--------|
| `Phase1_Foundation/Cl33.lean` | `QFD/GA/Cl33.lean` | Changed to local | ⬇️ Copy |
| `Phase1_Foundation/BasisOperations.lean` | `QFD/GA/BasisOperations.lean` | Changed | ⬇️ Copy |
| `Phase1_Foundation/BasisProducts.lean` | `QFD/GA/BasisProducts.lean` | Changed | ⬇️ Copy |
| `Phase1_Foundation/BasisReduction.lean` | `QFD/GA/BasisReduction.lean` | Changed | ⬇️ Copy |
| `Phase1_Foundation/HodgeDual.lean` | `QFD/GA/HodgeDual.lean` | Changed | ⬇️ Copy |
| `Phase1_Foundation/PhaseCentralizer.lean` | `QFD/GA/PhaseCentralizer.lean` | Changed | ⬇️ Copy |
| `Phase2_Operators/DiracRealization.lean` | `QFD/QM_Translation/DiracRealization.lean` | Changed | ⬇️ Copy |
| `Phase2_Operators/SchrodingerEvolution.lean` | `QFD/QM_Translation/SchrodingerEvolution.lean` | Changed | ⬇️ Copy |

### Contributed Files (NSE → Master)

These insights from NSE have been merged into Master QFD.

| NSE Source | Master Destination | Date | Status |
|------------|-------------------|------|--------|
| `Phase2_Projection/Conservation_Exchange.lean` | `QFD/Insights/ScleronomicConservation.lean` | 2025-01-14 | 📦 Merged |
| `Phase3_Advection/Advection_Pressure.lean` | `QFD/Insights/CommutatorDecomposition.lean` | 2025-01-14 | 📦 Merged |
| `suggested_for_removal/QFD/Physics/Postulates.lean` | `QFD/Physics/Postulates.lean` | 2025-01-14 | ⬆️ 4 axioms→theorems |

### NSE-Only Files (No Sync Needed)

| Directory | Content | Notes |
|-----------|---------|-------|
| `NavierStokes_Core/` | NS-specific operators | Project-specific |
| `Phase0_Analysis/` | Phase space setup | Project-specific |
| `Phase3_Isomorphism/` | Clifford-Beltrami | Could generalize later |
| `Phase4_Regularity/` | Global existence | NS-specific |
| `Phase5_Equivalence/` | Clay equivalence | NS-specific |
| `Phase6_Cauchy/` | Scleronomic lift | Could generalize later |
| `Phase7_Density/` | Analytic spaces | NS-specific |

---

## Riemann Project

### Borrowed Files (Riemann ← Master)

| Riemann File | Master Source | Namespace | Status |
|--------------|---------------|-----------|--------|
| `GA/Cl33.lean` | `QFD/GA/Cl33.lean` | `Riemann.GA` | 🔄 Diverged (has unique theorems) |

### Contributed Files (Riemann → Master)

| Riemann Source | Master Destination | Date | Status |
|----------------|-------------------|------|--------|
| `GA/Cl33.lean` (B_internal theorems) | `QFD/Insights/ComplexEmbedding.lean` | 2025-01-14 | 📦 Merged |

### Riemann-Only Files (No Sync Needed)

| Directory | Content | Notes |
|-----------|---------|-------|
| `GA/Cl33Ops.lean` | SpectralParam, exp_B | Riemann-specific |
| `ZetaSurface/` | All 16 files | Riemann-specific |

---

## Master QFD Insights Directory

New directory `QFD/Insights/` contains general results extracted from project work.

| File | Origin | Content |
|------|--------|---------|
| `ComplexEmbedding.lean` | Riemann | B² = -1, ℂ subalgebra embedding |
| `ScleronomicConservation.lean` | NSE | D² = 0 ⟺ Δ_q = Δ_p exchange |
| `CommutatorDecomposition.lean` | NSE | [A,B] + {A,B} = 2AB decomposition |

---

## Sync Procedures

### When Master QFD Updates GA Files

1. Check if NSE Phase1_Foundation needs update:
   ```bash
   diff QFD_SpectralGap/projects/Lean4/QFD/GA/Cl33.lean \
        GeminiTest3/NavierStokesPaper/Lean/Phase1_Foundation/Cl33.lean
   ```
2. If significant changes, update NSE copies (change namespace back to local)
3. Rebuild NSE: `cd NavierStokesPaper && lake build`

### When NSE Develops New General Insights

1. Identify if the insight is NS-specific or general Cl(3,3) result
2. If general, extract to `QFD/Insights/` with proper attribution
3. Update this manifest with the new entry
4. Rebuild Master: `cd QFD_SpectralGap/projects/Lean4 && lake build`

### When Riemann Develops New General Insights

1. Identify if the insight is Riemann-specific or general Cl(3,3) result
2. If general, extract to `QFD/Insights/` with proper attribution
3. Update this manifest with the new entry

### Periodic Full Sync (Recommended Monthly)

1. Review all borrowed files for drift
2. Check for new axiom→theorem conversions in projects
3. Update manifest with any new project-only files
4. Run full build on all three projects

---

## Change Log

| Date | Change | By |
|------|--------|-----|
| 2025-01-14 | Initial manifest created | Claude |
| 2025-01-14 | Merged 4 axioms→theorems from NSE to Master Postulates.lean | Claude |
| 2025-01-14 | Created QFD/Insights/ with 3 files from NSE/Riemann | Claude |
| 2025-01-14 | Deleted NSE suggested_for_removal/ (redundant copies) | Claude |

---

## AI Assistant Instructions

**When starting a session involving multiple projects:**

1. Read this manifest FIRST to understand file relationships
2. Check the Change Log for recent updates
3. Before modifying shared files, verify which project "owns" the canonical version

**When you improve a borrowed file:**

1. Make the improvement in the PROJECT (not Master) first
2. Test that it builds in the project
3. If the improvement is general, propose merging to Master
4. Update this manifest with the change

**When you see duplicate code:**

1. Check this manifest - it may be an intentional copy with different namespace
2. Do NOT delete without checking if it's a borrowed file
3. Improvements should flow UPSTREAM to Master, not downstream

**Red Flags (ask user before proceeding):**

- Modifying any file listed as "⬇️ BORROW" in a project
- Deleting files without checking this manifest
- Creating new files that duplicate Master functionality
