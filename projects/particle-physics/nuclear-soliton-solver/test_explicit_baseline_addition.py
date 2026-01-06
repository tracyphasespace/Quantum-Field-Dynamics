"""Test if we need to ADD A×M_proton baseline to solver output

Current situation:
  - Solver outputs E_model ≈ +46 to +82 MeV (pure field energy)
  - Target stability = -81.33 MeV
  - Experimental M_total = 11,177.93 MeV

Hypothesis:
  Maybe E_model is meant to be ADDED to A×M_proton to get total mass?

  M_total = (A × M_proton) + E_model

  For C-12:
    M_model = 12 × 938.272 + E_model
            = 11,259.27 + E_model

  And stability = M_model - (A × M_proton)
                = E_model

  Wait, that's circular...

Alternative hypothesis:
  M_proton itself is computed by the solver for a soliton.
  So the *difference* between E_(C-12) and 12×E_(proton) should give the binding.

  But we just tested that and got -612 MeV, not -81 MeV.

Third hypothesis:
  The solver computes energy relative to VACUUM, not relative to separated protons.
  So E_model = M_total_field - 0 (vacuum reference)

  But then E_model should be ~11,000 MeV, not ~50 MeV!

Let me check what happens if there's a UNIT CONVERSION issue...
"""
import sys
import pandas as pd
sys.path.insert(0, 'src')
from qfd_metaopt_ame2020 import M_PROTON

# Constants
ame_data = pd.read_csv('data/ame2020_system_energies.csv')
row = ame_data[(ame_data['Z'] == 6) & (ame_data['A'] == 12)]
exp_mass_total = float(row.iloc[0]['E_exp_MeV'])

A = 12
Z = 6
baseline = A * M_PROTON
target_stability = exp_mass_total - baseline

print("=" * 90)
print("BASELINE ADDITION TEST")
print("=" * 90)
print()
print("Experimental data:")
print(f"  M_total (AME) = {exp_mass_total:.2f} MeV")
print(f"  M_proton      = {M_PROTON:.2f} MeV")
print(f"  A             = {A}")
print(f"  Baseline      = {A} × {M_PROTON:.2f} = {baseline:.2f} MeV")
print(f"  Target E_stab = {exp_mass_total:.2f} - {baseline:.2f} = {target_stability:.2f} MeV")
print()

# Solver results from previous tests
E_model_values = [
    ("grid=32, c_v2=3.6", 82.5),
    ("grid=64, c_v2=7.0", 46.1),
    ("grid=64, penalty=10", 3.9),
]

print("Solver outputs (E_model from earlier tests):")
print()
print(f"{'Case':>25s}  {'E_model':>10s}  {'Interpretation 1':>20s}  {'Interpretation 2':>25s}")
print(f"{'':>25s}  {'(MeV)':>10s}  {'E_stab (as-is)':>20s}  {'E_stab = E_model - 12×E_p':>25s}")
print("-" * 90)

E_proton = 54.85  # From test_single_proton_baseline.py

for case, E_model in E_model_values:
    # Interpretation 1: E_model IS the stability energy (current assumption)
    stab_1 = E_model
    error_1 = stab_1 - target_stability

    # Interpretation 2: Need to subtract 12 × E_proton
    stab_2 = E_model - 12 * E_proton
    error_2 = stab_2 - target_stability

    print(f"{case:>25s}  {E_model:>10.1f}  {stab_1:>10.1f} ({error_1:+6.1f})  {stab_2:>10.1f} ({error_2:+6.1f})")

print()
print("=" * 90)
print("ANALYSIS:")
print("  Interpretation 1: E_model = stability energy directly")
print(f"    → Error: +130 to +160 MeV (WRONG SIGN)")
print()
print("  Interpretation 2: E_model - 12×E_proton = stability")
print(f"    → Error: -530 to -690 MeV (WRONG MAGNITUDE, but correct sign!)")
print()
print("KEY INSIGHT:")
print("  Interpretation 2 gives the CORRECT SIGN (negative) but wrong magnitude.")
print("  This suggests we're on the right track, but there's a scaling issue.")
print()
print("POSSIBLE ISSUES:")
print("  1. E_proton shouldn't be computed with same parameters as C-12")
print("  2. Need to account for Coulomb repulsion between protons")
print("  3. Unit conversion or normalization factor")
print("  4. c_v2_base is wrong - need different value for binding to work")
print("=" * 90)
