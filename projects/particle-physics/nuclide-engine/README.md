# QFD Nuclide Engine (AI Instance 2)

Topological nuclear model predicting stability, decay modes, and half-lives
from a single geometric field with two constants: alpha (fine structure) and
beta (vacuum stiffness, derived from alpha via the Golden Loop).

**Author**: Tracy McSheery
**Data**: NUBASE2020 (Kondev et al. 2021, Chinese Physics C 45 030001)

## Quick Start

```bash
pip install -r requirements.txt
python run_all.py
```

Results are written to `results/`. Each script produces `<name>_output.txt`.

## Using Your Own Data

The parser reads NUBASE2020 fixed-width format directly. To use your own
downloaded copy:

1. **Download NUBASE2020** from the IAEA:
   https://www-nds.iaea.org/amdc/  (look for `nubase_4.mas20.txt` or similar)

2. **Place the file** as `data/raw/nubase2020_raw.txt` (replacing the included copy)

3. **Or point to any path** — the `load_nubase()` parser in
   `model_nuclide_topology.py` accepts any file path:

   ```python
   from model_nuclide_topology import load_nubase
   entries = load_nubase('/path/to/your/nubase2020_raw.txt')
   # Returns list of dicts: {A, Z, N, is_stable, dominant_mode,
   #                         half_life_s, state, exc_energy_keV, jpi}
   ```

4. **AME2020 mass excess data** (used by AI Instance 1 for Q-values):
   https://www-nds.iaea.org/amdc/  (look for `mass_1.mas20.txt`)

### Parser Details

`load_nubase(path, include_isomers=False)` parses the NUBASE2020 fixed-width
format (columns 0-3: A, 4-7: ZZZi, 69-78: half-life, 78-80: unit, 88-102:
spin-parity). It handles all time units from yoctoseconds to yottayears,
strips comment markers (#), and classifies decay modes. Set
`include_isomers=True` to get all excited states (IAS entries excluded).

## Script Inventory

### Core Engine
| Script | Description | Size |
|--------|-------------|------|
| `model_nuclide_topology.py` | Core engine: valley prediction, mode classification, clock, NUBASE parser | 150 KB |
| `channel_analysis.py` | Per-channel half-life fits, rate competition test | 146 KB |
| `isomer_clock_analysis.py` | Isomer-specific clocks with EM selection rules | 47 KB |

### Test Scripts
| Script | Description |
|--------|-------------|
| `test_177_algebraic.py` | Derives N_max = 2*pi*beta^3 = 177 (neutron ceiling) |
| `test_density_shells.py` | Density shell structure in clock residuals |
| `test_diameter_ceiling.py` | Maximum nuclear diameter from vacuum geometry |
| `test_lyapunov_clock.py` | Lyapunov exponent / Dzhanibekov effect on half-lives |
| `test_overflow_unified.py` | Overflow (drip line) unified test |
| `test_resonance_spacing.py` | Resonance spacing patterns in nuclear data |
| `test_tennis_racket.py` | Tennis racket theorem applied to nuclear isomers |

### Visualization
| Script | Description | Requires |
|--------|-------------|----------|
| `qfd_6d_heatmap.py` | 6D confusion heatmap of model predictions | matplotlib |

## Dependency Map

```
model_nuclide_topology.py        (standalone core, NUBASE parser)
    |
    +-- channel_analysis.py      (imports load_nubase, compute_geometric_state)
    +-- isomer_clock_analysis.py (imports load_nubase, compute_geometric_state)
    |       |
    |       +-- test_density_shells.py    (imports find_nubase, build_dataframe)
    |       +-- test_diameter_ceiling.py  (imports find_nubase, build_dataframe)
    |       +-- test_overflow_unified.py  (imports find_nubase, build_dataframe)
    |
    +-- test_lyapunov_clock.py   (imports model_nuclide_topology)
    +-- test_resonance_spacing.py(imports load_nubase from model_nuclide_topology)
    +-- test_tennis_racket.py    (imports model_nuclide_topology)
    +-- qfd_6d_heatmap.py        (imports model_nuclide_topology)

test_177_algebraic.py             (self-contained, no imports)
```

## Documentation

See `docs/` for the full analysis write-ups:

- **CHAPTER_NUCLIDES.md** — Main paper: complete model description and results
- **QFD_NUCLIDE_ENGINE.md** — Technical reference: all equations, validation tables
- **FROZEN_CORE_CONJECTURE.md** — Frozen core theory of nuclear density limits
- **TENNIS_RACKET_LYAPUNOV_FEB8.md** — Dzhanibekov effect in nuclear isomers
- **DECAY_MECHANISM_FEB8.md** — Physical mechanisms for each decay species
- **Soliton_Nuclide_Model.md** — Narrative overview of the soliton model

## Pre-generated Figures

The `figures/` directory contains 29 publication-quality PNGs. Running the
scripts regenerates PNGs in `scripts/` (alongside the source files). These
generated copies are excluded by `.gitignore`.
