# QFD Projects Migration Plan

**Date**: 2025-12-20
**Backup**: `projects_backup_20251219.tar.gz` (1.4GB)
**Target**: Migrate all projects to Grand Solver v1.1 schema + Lean4 formalism

## Backup Status ✅

```bash
-rw-r--r-- 1 tracy tracy 20M Dec 20 09:32 projects_backup_clean_20251219.tar.gz
```

**Contents**: Source code backup (excludes .lake, build, __pycache__, Mathlib artifacts)

**Excluded**: `.lake/`, `build/`, `__pycache__/`, `.git/`, Mathlib dependencies

## Project Inventory

### 1. Lean4 (4.6GB) ✅ COMPLETE

**Status**: Already migrated and verified
- Formalism complete: EmergentAlgebra.lean, Cl33.lean, SpectralGap.lean
- Schema definitions: DimensionalAnalysis.lean, Couplings.lean, Constraints.lean
- Consistency checker: Validated against Grand Solver v1.1

**Action**: ✅ None - This is the reference formalism

---

### 2. astrophysics/ (71MB)

#### 2a. blackhole-dynamics/qfd_10_realms_pipeline

**Purpose**: 10-realm cascade solver for coupling constant estimation

**Current Structure**:
- `coupling_constants/` - Parameter registry and validation
- `realms/` - realm0 (CMB) through realm10 (isotopes)
- `common/physics.py`, `common/solvers.py`
- **Own parameter system** with JSON registry

**Files of Interest**:
- `coupling_constants/registry/parameter_registry.py`
- `realms/realm0_cmb.py` through `realm10_isotopes.py`
- `dependency_graph.json`, `final_parameter_state.json`

**Migration Needed**:
- [ ] Map 10-realm parameters to Grand Solver v1.1 schema
- [ ] Create RunSpec.json for each realm
- [ ] Convert realm solvers to observable adapters
- [ ] Integrate with `qfd.adapters.cosmo`, `qfd.adapters.nuclear`, etc.

**Priority**: **HIGH** - This is a multi-domain parameter estimation system

---

#### 2b. redshift-analysis/RedShift

**Purpose**: CMB power spectrum analysis (QFD vs ΛCDM comparison)

**Current Structure**:
- `qfd_cmb/` - kernels, ppsi_models, projector, visibility
- `fit_planck.py` - CMB fitting script
- `hubble_constant_validation.py`
- **Hardcoded QFD parameters**

**Files of Interest**:
- `qfd_cmb/ppsi_models.py`
- `fit_planck.py`
- `examples/basic_analysis.py`

**Migration Needed**:
- [ ] Create `qfd.adapters.cosmo.predict_cmb_power` adapter
- [ ] Extract CMB physics into RunSpec.json
- [ ] Map hardcoded params (A_plasma, rho_vac, w_dark) to schema
- [ ] Create Lean4 constraints for cosmological parameters

**Priority**: **HIGH** - Direct Grand Solver integration candidate

---

#### 2c. V21 Supernova Analysis package

**Purpose**: Supernova distance modulus analysis (BBH model)

**Current Structure**:
- `v17_qfd_model.py`, `v18_bbh_model.py`
- `stage1_v20_fullscale.py` - Full pipeline
- `bbh_robust_optimizer.py`
- **Custom optimizer**, **hardcoded parameters**

**Files of Interest**:
- `v18_bbh_model.py` - BBH physics
- `bbh_robust_optimizer.py`
- `QFD_PHYSICS.md` - Documentation

**Migration Needed**:
- [ ] Create `qfd.adapters.cosmo.predict_sn_distance_modulus` adapter
- [ ] Extract BBH parameters to RunSpec.json
- [ ] Integrate optimizer with Grand Solver
- [ ] Create Lean4 constraints for BBH model

**Priority**: **MEDIUM** - Secondary cosmology observable

---

### 3. particle-physics/ (508KB)

#### 3a. nuclide-prediction

**Purpose**: Nuclear binding energy predictions (Core Compression Law)

**Current Structure**:
- `2ndTryCoreCompressionlaw.py`
- `test_fit.py`
- **Direct CCL implementation**

**Files of Interest**:
- `2ndTryCoreCompressionlaw.py`

**Migration Needed**:
- [ ] **Already implemented!** → `qfd.adapters.nuclear.predict_binding_energy`
- [ ] Verify parameters match Lean4 `NuclearParams`
- [ ] Create RunSpec for AME2020 fitting
- [ ] Replace standalone script with Grand Solver call

**Priority**: **IMMEDIATE** - This is the first production target

**Note**: Core Compression Law is already in Grand Solver v1.1!

---

#### 3b. deuterium-tests

**Purpose**: Hydrogen/deuterium binding energy calculations

**Current Structure**:
- `Deuterium.py`, `AllNightLong.py`, `AutopilotHydrogen.py`
- `qfd_result_schema.py` - Custom result schema
- `test_genesis_constants.py`

**Files of Interest**:
- `qfd_result_schema.py` - May inform Grand Solver schema
- `test_genesis_constants.py` - Parameter validation

**Migration Needed**:
- [ ] Create `qfd.adapters.nuclear.predict_deuteron_binding` adapter
- [ ] Map genesis constants to Lean4 `NuclearParams`
- [ ] Create RunSpec for deuterium tests
- [ ] Validate against `test_genesis_constants.py`

**Priority**: **MEDIUM** - Nuclear physics validation

---

#### 3c. lepton-isomers

**Purpose**: Lepton mass predictions (electron, muon, tau)

**Current Structure**:
- `src/solvers/` - hamiltonian, phoenix_solver, backend
- `src/constants/` - electron.json, muon.json, tau.json
- `validate_all_particles.py`
- **Standalone solver system**

**Files of Interest**:
- `src/constants/*.json` - Lepton parameter sets
- `src/solvers/phoenix_solver.py`
- `validate_all_particles.py`

**Migration Needed**:
- [ ] Create `qfd.adapters.particle.predict_lepton_mass` adapter
- [ ] Extract constants to ParameterSpec.json
- [ ] Map to Lean4 `ParticleParams`
- [ ] Create RunSpec for lepton mass fitting

**Priority**: **LOW** - Particle physics extension

---

### 4. field-theory/ (76KB)

#### 4a. trilemma-toolkit

**Purpose**: Resonant atom isomer solver (Tc-99m, etc.)

**Current Structure**:
- `qfd_trilemma_isomer_solver.py`
- `qfd_resonant_atom_v4_2.py`
- `qfd_trilemma_demo_tc99m.json`, `tc99_isomer_search.json`

**Files of Interest**:
- `qfd_trilemma_demo_tc99m.json` - Parameter format
- `qfd_trilemma_isomer_solver.py`

**Migration Needed**:
- [ ] Create `qfd.adapters.nuclear.predict_isomer_energy` adapter
- [ ] Convert JSON configs to RunSpec format
- [ ] Map trilemma parameters to schema
- [ ] Lean4 constraints for isomer physics

**Priority**: **LOW** - Specialized nuclear physics

---

#### 4b. poynting-vectors

**Purpose**: Field theory symbolic derivations

**Current Structure**:
- `PoyntingVectors.py`
- `SymbolicDerivationsUsingGAlbebra.py`
- **Symbolic math**, **not fitting/optimization**

**Migration Needed**:
- [ ] None - This is symbolic derivation, not parameter fitting
- [ ] May inform Lean4 field theory formalizations

**Priority**: **NONE** - Keep as-is (symbolic toolkit)

---

## Migration Priority Order

### Phase 1: Immediate (Production-Ready)

1. **nuclide-prediction** → Already implemented in Grand Solver v1.1!
   - Use existing `qfd.adapters.nuclear.predict_binding_energy`
   - Create RunSpec for AME2020 dataset
   - **This can run TODAY**

2. **redshift-analysis/RedShift** → CMB observable adapter
   - Create `qfd.adapters.cosmo.predict_cmb_power`
   - Map existing `ppsi_models.py` to adapter
   - Create RunSpec for Planck2018 data

### Phase 2: High-Priority (Multi-Domain Integration)

3. **blackhole-dynamics/qfd_10_realms_pipeline** → Convert to Grand Solver
   - This is a **parameter cascade system** that needs careful integration
   - Each realm becomes a RunSpec component
   - Validate against existing `dependency_graph.json`

4. **V21 Supernova Analysis** → BBH distance modulus adapter
   - Create `qfd.adapters.cosmo.predict_sn_distance_modulus`
   - Integrate BBH physics

### Phase 3: Secondary (Validation & Extension)

5. **deuterium-tests** → Nuclear validation adapter
6. **lepton-isomers** → Particle physics extension

### Phase 4: Specialized (Keep As-Is or Low Priority)

7. **trilemma-toolkit** → Isomer physics (low priority)
8. **poynting-vectors** → No migration (symbolic toolkit)

---

## Schema Compliance Requirements

All projects must conform to:

### 1. Lean4 Formalism
- Parameters defined in `projects/Lean4/QFD/Schema/Couplings.lean`
- Constraints defined in `projects/Lean4/QFD/Schema/Constraints.lean`
- Dimensional analysis via `DimensionalAnalysis.lean`

### 2. Grand Solver v1.1 Schema
- RunSpec format: `schema/v0/RunSpec_v03.schema.json`
- Parameter spec: `schema/v0/ParameterSpec.schema.json`
- Observable adapters: `qfd/adapters/<domain>/<observable>.py`

### 3. Validation Requirements
- Consistency check: `python schema/v0/check_lean_json_consistency.py <runspec>`
- Test suite: Automated tests for each adapter
- Provenance tracking: All results include schema hashes + environment

---

## Migration Workflow Template

For each project:

### Step 1: Parameter Extraction
```bash
# Identify all physics parameters
grep -r "param\|constant\|coupling" <project_dir>/*.py
```

### Step 2: Create Lean4 Definitions
```lean
-- Add to projects/Lean4/QFD/Schema/Couplings.lean
structure <Domain>Params where
  param1 : Quantity <Dimensions>
  param2 : Quantity <Dimensions>
  ...
```

### Step 3: Create Observable Adapter
```python
# qfd/adapters/<domain>/<observable>.py
def predict_<observable>(df, params, config=None):
    # Extract parameters
    p1 = params.get("domain.param1", params.get("param1"))

    # Physics calculation
    y_pred = ...

    return y_pred
```

### Step 4: Create RunSpec
```json
{
  "schema_version": "v0.1",
  "experiment_id": "exp_<project>_<date>",
  "model": {"id": "qfd.<domain>.<model>"},
  "parameters": [...],
  "datasets": [...],
  "objective": {
    "type": "chi_squared",
    "components": [{
      "dataset_id": "...",
      "observable_adapter": "qfd.adapters.<domain>.predict_<observable>"
    }]
  },
  "solver": {"method": "scipy.minimize", "options": {"algo": "L-BFGS-B"}}
}
```

### Step 5: Validate
```bash
# Consistency check
python schema/v0/check_lean_json_consistency.py schema/v0/experiments/<project>.json

# Run solver
./schema/v0/run_solver.sh schema/v0/experiments/<project>.json
```

---

## Next Steps

### Immediate Actions

1. **Start with nuclide-prediction** (already implemented!)
   - Acquire AME2020 dataset
   - Create RunSpec: `experiments/ccl_ame2020.json`
   - Run: `./run_solver.sh experiments/ccl_ame2020.json`
   - **This validates the entire infrastructure**

2. **Create CMB adapter** (redshift-analysis)
   - Port `ppsi_models.py` to `qfd.adapters.cosmo.predict_cmb_power`
   - Create RunSpec for Planck2018
   - **This demonstrates multi-domain capability**

3. **Analyze qfd_10_realms_pipeline**
   - This is the most complex migration
   - May inform Grand Solver architecture improvements
   - Consider creating "meta-RunSpec" for cascading solvers

### Documentation Needed

- [ ] Migration guide for each project type
- [ ] Adapter writing tutorial (expand ADAPTER_GUIDE.md)
- [ ] Lean4 ↔ Python parameter mapping reference
- [ ] Example RunSpecs for each domain

---

## Success Criteria

Migration is complete when:

1. ✅ All physics projects run via Grand Solver v1.1
2. ✅ All parameters defined in Lean4 `Couplings.lean`
3. ✅ All constraints defined in Lean4 `Constraints.lean`
4. ✅ Consistency checker passes for all RunSpecs
5. ✅ Complete provenance tracking for all results
6. ✅ Multi-domain optimization works (Nuclear + CMB + BBH)

---

## Backup Restoration

If needed:
```bash
tar -xzf projects_backup_clean_20251219.tar.gz
```

**Backup location**: `/home/tracy/development/QFD_SpectralGap/projects_backup_clean_20251219.tar.gz`

**Backup date**: 2025-12-20 09:32
**Size**: 20MB (source code only, no build artifacts)

---

**Status**: Migration planning complete. Backup secured. Ready to begin Phase 1.
