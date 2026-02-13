# QFD-Universe Repository Skeleton
# Researcher-facing public repo: clean, runnable, citable

```
QFD-Universe/
│
├── README.md                          # 1-page: what QFD is, key results, how to replicate
├── CITATION.cff                       # How to cite this work
├── LICENSE
├── requirements.txt                   # Single pip install for everything
├── run_all.py                         # Master script: validates ALL sectors, prints summary table
│
├── docs/
│   ├── QFD_Whitepaper.md             # Theory overview (non-technical)
│   ├── PARAMETER_GLOSSARY.md         # β, δ, α relationships explained
│   ├── REPLICATION_GUIDE.md          # Step-by-step: clone → install → validate
│   └── llms.txt                      # Machine-readable index (already exists)
│
│
│ ═══════════════════════════════════════════════════════════════
│  SECTOR 1: LEPTON PHYSICS
│  "From α to electron, muon, and tau masses"
│ ═══════════════════════════════════════════════════════════════
│
├── lepton/
│   ├── README.md                     # Sector overview, key claims, parameters
│   ├── run_sector.py                 # Runs all lepton validations
│   │
│   ├── golden-loop/                  # ★ V22 Lepton Analysis (α → β → masses)
│   │   ├── README.md                # What this proves, how to interpret
│   │   ├── validate_leptons.py      # Entry: test_all_leptons_beta_from_alpha.py
│   │   └── expected_results.json    # Reference output for comparison
│   │
│   ├── g2-prediction/               # ★ Photon sector g-2 (0.45% accuracy)
│   │   ├── README.md
│   │   ├── validate_g2.py           # Core g-2 prediction script
│   │   ├── run_all.py               # Full 12-script validation suite
│   │   └── analysis/                # Supporting validation scripts
│   │
│   └── koide-relation/              # Geometric Koide Q=2/3 proof
│       ├── README.md                # Connection between δ and β
│       └── validate_koide.py        # Numerical validation
│
│
│ ═══════════════════════════════════════════════════════════════
│  SECTOR 2: NUCLEAR PHYSICS
│  "Nuclei as coherent soliton configurations"
│ ═══════════════════════════════════════════════════════════════
│
├── nuclear/
│   ├── README.md
│   ├── run_sector.py
│   │
│   ├── soliton-solver/              # ★ 3,558 nuclear masses from soliton model
│   │   ├── README.md
│   │   ├── src/
│   │   │   ├── qfd_solver.py        # Single-nucleus solver
│   │   │   └── qfd_metaopt_ame2020.py  # Full AME2020 optimization
│   │   ├── docs/
│   │   │   ├── PHYSICS_MODEL.md
│   │   │   └── CALIBRATION_GUIDE.md
│   │   └── data/
│   │       └── trial32_params.json  # Calibrated parameters
│   │
│   └── nuclide-scaling/             # ★ 2-param universal law (R²=0.98)
│       ├── README.md
│       ├── run_all.py               # Full prediction + validation
│       └── data/
│           └── NuMass.csv           # Input dataset (5,842 isotopes)
│
│
│ ═══════════════════════════════════════════════════════════════
│  SECTOR 3: COSMOLOGY
│  "Photon scattering replaces dark energy"
│ ═══════════════════════════════════════════════════════════════
│
├── cosmology/
│   ├── README.md
│   ├── run_sector.py
│   │
│   ├── supernova-fit/               # ★ V22: χ²/ν=0.939 on 1,829 SNe
│   │   ├── README.md
│   │   ├── v22_qfd_fit.py           # Lean-constrained fit
│   │   └── data/                    # DES-SN5YR subset or pointer
│   │
│   ├── sn-transparency/             # ★ Raw photometry → cosmology (full pipeline)
│   │   ├── README.md
│   │   ├── reproduce_quick.sh       # 30-min filtered path
│   │   ├── reproduce_full.sh        # 3-4 hour raw path
│   │   └── scripts/
│   │
│   └── blackhole-dynamics/          # ★ 4-mechanism escape simulation
│       ├── README.md
│       ├── main.py
│       └── simulation.py
│
│
│ ═══════════════════════════════════════════════════════════════
│  SECTOR 4: CROSS-SCALE VALIDATION
│  "Same β across all sectors"
│ ═══════════════════════════════════════════════════════════════
│
├── cross-scale/
│   ├── README.md                    # β universality argument
│   ├── run_sector.py
│   │
│   └── ten-realms/                  # ★ Full α → β → predictions pipeline
│       ├── README.md
│       ├── test_golden_loop.py      # 30-second validation
│       └── realm_configs/           # Per-realm configuration
│
│
│ ═══════════════════════════════════════════════════════════════
│  FORMAL PROOFS (optional deep-dive)
│ ═══════════════════════════════════════════════════════════════
│
├── lean4/
│   ├── README.md                    # What's proven, what's sorry, how to build
│   ├── lakefile.toml
│   ├── lean-toolchain
│   └── QFD/
│       ├── GA/                      # Geometric algebra (Cl(3,3))
│       ├── Lepton/                  # Koide relation proofs
│       ├── Cosmology/               # CMB, adjoint stability
│       └── Nuclear/                 # Binding energy formalization
│
│
│ ═══════════════════════════════════════════════════════════════
│  EXPLORATION LAB (secondary projects)
│ ═══════════════════════════════════════════════════════════════
│
├── exploration/
│   ├── README.md                    # "These are active investigations, not validated"
│   │
│   ├── energy-functional/           # MCMC β-convergence test
│   │   ├── README.md
│   │   └── mcmc_stage1_gradient.py
│   │
│   ├── lepton-mcmc/                 # Bayesian parameter estimation
│   │   ├── README.md
│   │   └── run_mcmc.py
│   │
│   ├── photon-solitons/             # CMB derivations, ℏ from topology
│   │   ├── README.md
│   │   └── run_all.py
│   │
│   ├── deuterium/                   # Simplest nuclear test case
│   │   └── run_target_deuterium.py
│   │
│   ├── lepton-isomers/              # Novel internal structure hypothesis
│   │   └── README.md
│   │
│   ├── redshift-analysis/           # Hubble constant validation
│   │   └── README.md
│   │
│   ├── sne-model-comparison/        # Head-to-head 4-model comparison
│   │   └── README.md
│   │
│   ├── lagrangian-solitons/         # Formal Lagrangian derivations
│   │   └── README.md
│   │
│   ├── nuclear-solver-alt/          # Alternative nuclear solver
│   │   └── README.md
│   │
│   └── trilemma-toolkit/            # Resonant atom model
│       └── README.md
│
│
│ ═══════════════════════════════════════════════════════════════
│  META
│ ═══════════════════════════════════════════════════════════════
│
├── .github/
│   └── workflows/
│       ├── validate.yml             # CI: run_all.py on every push
│       └── pages.yml                # Deploy docs to GitHub Pages
│
└── data/
    └── experimental/                # PDG values, AME2020 reference, SNe datasets
        ├── pdg_lepton_masses.json
        ├── pdg_g2_values.json
        └── README.md               # Data provenance and citations
```

## Design Principles

### 1. Three-Command Replication
```bash
git clone https://github.com/tracyphasespace/QFD-Universe.git
pip install -r requirements.txt
python run_all.py
```
Prints a summary table:
```
QFD Validation Summary
═══════════════════════════════════════════════════════════
Sector          Test                    Result    Status
───────────────────────────────────────────────────────────
Lepton          e,μ,τ masses (β=3.04)   χ²<1e-11  PASS
Lepton          g-2 anomaly             0.45% err PASS
Nuclear         AME2020 (3558 nuclei)   <1% light PASS
Nuclear         Universal scaling       R²=0.98   PASS
Cosmology       SNe fit (1829)          χ²/ν=0.94 PASS
Cross-scale     Golden loop (α→β→m)     validated PASS
═══════════════════════════════════════════════════════════
```

### 2. Each Sector is Self-Contained
- Own README, own run_sector.py, own expected results
- A researcher interested only in nuclear physics clones the repo
  and runs `python nuclear/run_sector.py`

### 3. Clear Epistemic Tiers
- `lepton/`, `nuclear/`, `cosmology/`, `cross-scale/` = **validated claims**
- `exploration/` = **active investigations, not yet validated**
- `lean4/` = **formal proofs** (mathematical, not physical)
- Each README states what's proven vs what's fitted

### 4. Exploration Lab for the Curious
The 10 secondary projects live under `exploration/` with a clear
banner: "These are research tools, not validated predictions."
Researchers who want to dig deeper can explore; those who just
want to replicate the core claims can ignore this directory entirely.

### 5. Sync Strategy (SpectralGap → Universe)
Option A: Manual curation (copy validated scripts, strip internals)
Option B: `scripts/publish.sh` that rsyncs from SpectralGap paths:
  - V22_Lepton_Analysis/          → lepton/golden-loop/
  - Photon/                       → lepton/g2-prediction/
  - projects/pp/nuclear-soliton/  → nuclear/soliton-solver/
  - projects/pp/nuclide-predict/  → nuclear/nuclide-scaling/
  - V22_Supernova_Analysis/       → cosmology/supernova-fit/
  - projects/astro/qfd-sn-v22/   → cosmology/sn-transparency/
  - projects/astro/blackhole/     → cosmology/blackhole-dynamics/
  - projects/astro/10_realms/     → cross-scale/ten-realms/
  (strips __pycache__, results/, session logs, AI briefings)
