# Data Provenance

## Source Data

### AME2020 — Atomic Mass Evaluation 2020
- **File**: `ame2020.csv`
- **Citation**: M. Wang, W.J. Huang, F.G. Kondev, G. Audi, S. Naimi,
  "The AME 2020 atomic mass evaluation (II). Tables, graphs and references,"
  Chinese Physics C 45, 030003 (2021).
- **Contents**: Atomic mass excesses (keV) for ~3,500 nuclides.
- **Used for**: Computing Q-values (energy release) and Coulomb barriers
  for alpha, beta, and other decay modes.

### NUBASE2020 — Nuclear and Decay Properties
- **Files**: `nubase_excitation_spin.csv` (extracted spin-parity and
  excitation energies), `iterative_peeling_predictions.csv` (half-lives,
  decay modes, valley deviation epsilon)
- **Citation**: F.G. Kondev, M. Wang, W.J. Huang, S. Naimi, G. Audi,
  "The NUBASE2020 evaluation of nuclear and decay properties,"
  Chinese Physics C 45, 030001 (2021).
- **Contents**: Half-lives, decay modes, spin-parity assignments, excitation
  energies for 4,948 nuclear states (ground + isomers).

## Derived Data

### clean_species_sorted.csv
- **Contents**: 4,948 nuclear states with clean species classification,
  Q-values, Coulomb barriers, spin-parity, multipolarity (lambda), and
  tracking bin assignments.
- **Produced by**: `scripts/clean_species_sort.py` operating on the three
  source files above.
- **Key columns**:
  - `A`, `Z`, `N`: Mass number, proton number, neutron number
  - `clean_species`: Decay channel (beta-, beta+, alpha, IT, IT_platypus,
    SF, proton, and isomeric variants)
  - `epsilon`: Signed Z-deviation from the valley of stability
  - `log_hl`: log10 of half-life in seconds
  - `parity`: ee/eo/oe/oo (even-even, etc.)
  - `Q_keV`: Decay Q-value in keV
  - `V_coulomb_keV`: Coulomb barrier in keV (alpha only)
  - `correct_lambda`: Electromagnetic multipolarity (IT only)
  - `tracking_bin`: stable / tracked / suspect / ephemeral / rare
