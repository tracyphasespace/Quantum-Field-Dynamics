#!/usr/bin/env python3
"""
Test: Tennis Racket Theorem (Intermediate Axis Theorem) Applied to Peanut Solitons

THEORY
======

The Intermediate Axis Theorem (Dzhanibekov effect / Tennis Racket Theorem):

A rigid body has three principal axes with moments of inertia I₁ ≤ I₂ ≤ I₃.
Rotation about the SMALLEST (I₁) and LARGEST (I₃) axes is STABLE.
Rotation about the INTERMEDIATE axis (I₂) is UNSTABLE — the object will
spontaneously flip, periodically exchanging rotation between I₂ and I₁.

For a peanut-shaped soliton with two lobes connected by a neck:

  Axis 1 (long):   lobe-to-lobe axis.  Smallest I (mass near axis).
  Axis 2 (medium):  perpendicular in the plane of lobe asymmetry.
  Axis 3 (short):  perpendicular to both.  Largest I (mass far from axis).

For a PERFECTLY symmetric peanut, I₂ = I₃ and there is no intermediate
axis instability.  But real solitons are NOT symmetric:
  - Proton-rich vs neutron-rich lobes (charge asymmetry)
  - Odd-N or odd-Z breaks mirror symmetry
  - The winding itself has handedness

Once the symmetry is broken (I₂ ≠ I₃), the intermediate axis rotation
is unstable.  The soliton periodically FLIPS — the spin axis swings from
perpendicular (I₂, across a lobe) to parallel (I₁, along the long/neck
axis) and back.

During each flip, the angular momentum briefly aligns with the LONG AXIS
(through the neck).  This creates:
  1. Centrifugal stress on the neck (spinning around the pinch)
  2. The winding wraps around the neck instead of around the lobes
  3. If the neck is already thin (high pf), the stress can trigger pinch-off

PREDICTION: Mode-switching isomers are solitons caught in the FLIPPED
orientation — the spin is along the long axis instead of perpendicular.
This changes the available decay channels:
  - Short-axis spin (normal): beta decay (charge conversion within a lobe)
  - Long-axis spin (flipped): alpha/SF (neck pinch-off or bifurcation)

TESTABLE CONSEQUENCES:
  1. Mode-switching isomers should have different J than ground states
  2. The switches B+↔alpha should concentrate in the peanut regime (pf > 0)
  3. Odd-A (broken symmetry) should show MORE mode switches than even-A
  4. The J values of alpha-switching isomers should be INTERMEDIATE
     (not the minimum or maximum available J for that nucleus)
  5. Pf should correlate with the spin gap |ΔJ| of mode-switching isomers

Data: NUBASE2020 (all states including isomers)
"""

import math
import os
import sys
import numpy as np
from collections import defaultdict

# Import engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_nuclide_topology as m


def main():
    # ── Load data ──
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRS = [
        os.path.join(SCRIPT_DIR, '..', 'data', 'raw'),
        os.path.join(SCRIPT_DIR, 'data'),
    ]
    NUBASE_PATH = None
    for d in DATA_DIRS:
        candidate = os.path.join(d, 'nubase2020_raw.txt')
        if os.path.exists(candidate):
            NUBASE_PATH = candidate
            break

    if not NUBASE_PATH:
        print("ERROR: nubase2020_raw.txt not found")
        return

    all_entries = m.load_nubase(NUBASE_PATH, include_isomers=True)
    nuclide_states = m.group_nuclide_states(all_entries)
    print(f"Loaded {len(all_entries)} entries, {len(nuclide_states)} (Z,A) pairs\n")

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: Symmetry breaking — odd vs even parity and mode switches
    # ═══════════════════════════════════════════════════════════════
    print("=" * 72)
    print("  TEST 1: SYMMETRY BREAKING — Odd vs Even and Mode Switches")
    print("=" * 72)
    print("""
  The intermediate axis theorem requires I₁ ≠ I₂ ≠ I₃ (three distinct
  moments).  A perfectly symmetric peanut has I₂ = I₃ → no instability.
  Odd-N or odd-Z breaks the peanut's mirror symmetry → I₂ ≠ I₃ →
  intermediate axis instability is possible.

  Prediction: odd-A nuclides should have MORE mode-switching isomers
  per capita than even-A (even-even in particular).
""")

    # Classify each (Z,A) pair
    parity_stats = {
        'ee': {'total': 0, 'has_isomers': 0, 'mode_switches': 0},
        'eo': {'total': 0, 'has_isomers': 0, 'mode_switches': 0},
        'oe': {'total': 0, 'has_isomers': 0, 'mode_switches': 0},
        'oo': {'total': 0, 'has_isomers': 0, 'mode_switches': 0},
    }

    all_switches = []  # Collect for later tests

    for (Z, A), states in nuclide_states.items():
        if A < 3:
            continue
        geo = m.compute_geometric_state(Z, A)
        parity = geo.parity

        parity_stats[parity]['total'] += 1

        if len(states) < 2:
            continue
        parity_stats[parity]['has_isomers'] += 1

        gs = states[0]
        gs_mode = m.normalize_nubase(gs['dominant_mode'])
        if gs_mode in ('unknown', 'IT'):
            continue
        gs_j = m._parse_spin_value(gs.get('jpi', ''))

        for iso in states[1:]:
            iso_mode = m.normalize_nubase(iso['dominant_mode'])
            if iso_mode in ('unknown', 'IT'):
                continue
            if iso_mode != gs_mode:
                parity_stats[parity]['mode_switches'] += 1
                iso_j = m._parse_spin_value(iso.get('jpi', ''))
                dj = abs(gs_j - iso_j) if gs_j is not None and iso_j is not None else None

                all_switches.append({
                    'Z': Z, 'A': A, 'N': A - Z,
                    'gs_mode': gs_mode, 'iso_mode': iso_mode,
                    'gs_j': gs_j, 'iso_j': iso_j,
                    'gs_jpi': gs.get('jpi', ''), 'iso_jpi': iso.get('jpi', ''),
                    'delta_j': dj,
                    'geo': geo,
                    'gs_hl': gs.get('half_life_s', np.nan),
                    'iso_hl': iso.get('half_life_s', np.nan),
                    'iso_state': iso.get('state', ''),
                    'iso_exc': iso.get('exc_energy_keV', 0),
                })

    print(f"  {'Parity':>8s}  {'Total':>6s}  {'With iso':>8s}  {'Switches':>9s}  {'Rate':>8s}")
    print(f"  {'-'*48}")
    for p in ['ee', 'eo', 'oe', 'oo']:
        s = parity_stats[p]
        rate = s['mode_switches'] / max(s['has_isomers'], 1) * 100
        print(f"  {p:>8s}  {s['total']:>6d}  {s['has_isomers']:>8d}  {s['mode_switches']:>9d}  {rate:>7.1f}%")

    # Even-A vs odd-A
    even_A_switches = sum(1 for sw in all_switches if sw['A'] % 2 == 0)
    odd_A_switches = sum(1 for sw in all_switches if sw['A'] % 2 != 0)
    even_A_pairs = parity_stats['ee']['has_isomers'] + parity_stats['oo']['has_isomers']
    odd_A_pairs = parity_stats['eo']['has_isomers'] + parity_stats['oe']['has_isomers']
    even_rate = even_A_switches / max(even_A_pairs, 1) * 100
    odd_rate = odd_A_switches / max(odd_A_pairs, 1) * 100
    print(f"\n  Even-A (ee+oo) switch rate: {even_A_switches}/{even_A_pairs} = {even_rate:.1f}%")
    print(f"  Odd-A (eo+oe) switch rate:  {odd_A_switches}/{odd_A_pairs} = {odd_rate:.1f}%")
    print(f"  Ratio odd/even: {odd_rate/max(even_rate,0.01):.2f}")

    if odd_rate > even_rate:
        print(f"\n  ✓ Odd-A has higher mode-switch rate — symmetry breaking enables flips")
    else:
        print(f"\n  ✗ Even-A has higher or equal rate — symmetry breaking test inconclusive")

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: Peanut regime concentration
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 2: PEANUT REGIME — Where Do Mode Switches Happen?")
    print("=" * 72)
    print("""
  The tennis racket effect requires an elongated body (peanut).  Spherical
  solitons (Zone 1) have nearly equal moments → no intermediate axis → no
  flip.  The effect should CONCENTRATE in the peanut regime (Zone 2+3).

  Prediction: mode switches per capita should increase with pf.
""")

    # Zone distribution of switches
    zone_switches = {1: 0, 2: 0, 3: 0}
    zone_totals = {1: 0, 2: 0, 3: 0}
    for (Z, A), states in nuclide_states.items():
        if A < 3 or len(states) < 2:
            continue
        geo = m.compute_geometric_state(Z, A)
        zone_totals[geo.zone] += 1

    for sw in all_switches:
        zone_switches[sw['geo'].zone] += 1

    print(f"  {'Zone':>6s}  {'Pairs w/iso':>11s}  {'Switches':>9s}  {'Rate':>8s}")
    print(f"  {'-'*42}")
    for z in (1, 2, 3):
        rate = zone_switches[z] / max(zone_totals[z], 1) * 100
        print(f"  {z:>6d}  {zone_totals[z]:>11d}  {zone_switches[z]:>9d}  {rate:>7.1f}%")

    # Pf binned switch rate
    pf_bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0),
               (1.0, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 3.0)]
    print(f"\n  {'pf range':>12s}  {'Pairs':>6s}  {'Switches':>9s}  {'Rate':>8s}")
    print(f"  {'-'*42}")

    for lo, hi in pf_bins:
        pairs_in_bin = 0
        switches_in_bin = 0
        for (Z, A), states in nuclide_states.items():
            if A < 3 or len(states) < 2:
                continue
            geo = m.compute_geometric_state(Z, A)
            if lo <= geo.peanut_f < hi:
                pairs_in_bin += 1
        for sw in all_switches:
            if lo <= sw['geo'].peanut_f < hi:
                switches_in_bin += 1
        rate = switches_in_bin / max(pairs_in_bin, 1) * 100
        print(f"  [{lo:.2f}, {hi:.2f})  {pairs_in_bin:>6d}  {switches_in_bin:>9d}  {rate:>7.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # TEST 3: Spin distribution — are switching J values "intermediate"?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 3: INTERMEDIATE SPIN — Are Switching J Values 'In Between'?")
    print("=" * 72)
    print("""
  The intermediate axis theorem predicts instability at the MIDDLE moment.
  If the isomer's J value is "intermediate" (not the lowest or highest J
  available for that nucleus), the flip is more likely.

  We test: for each mode-switching isomer, is the isomer's J closer to
  the median J available at that (Z,A) than non-switching isomers?
""")

    # For each (Z,A) with mode switches, collect all J values
    intermediate_count = 0
    extreme_count = 0
    tested = 0

    for sw in all_switches:
        Z, A = sw['Z'], sw['A']
        states = nuclide_states.get((Z, A), [])
        if len(states) < 3:  # Need at least gs + switching iso + one more
            continue

        all_j = []
        for st in states:
            j = m._parse_spin_value(st.get('jpi', ''))
            if j is not None:
                all_j.append(j)

        if len(all_j) < 3:
            continue

        iso_j = sw['iso_j']
        if iso_j is None:
            continue

        j_min = min(all_j)
        j_max = max(all_j)
        j_med = np.median(all_j)

        tested += 1

        # Is the switching isomer's J "intermediate"?
        # Define intermediate as not the min or max
        if j_min < iso_j < j_max:
            intermediate_count += 1
        else:
            extreme_count += 1

    if tested > 0:
        print(f"  Tested: {tested} mode-switching isomers with 3+ measured J values")
        print(f"  Intermediate J (not min/max): {intermediate_count}/{tested} ({intermediate_count/tested*100:.1f}%)")
        print(f"  Extreme J (min or max):       {extreme_count}/{tested} ({extreme_count/tested*100:.1f}%)")
        print(f"\n  If random, expected intermediate fraction: ~{(1-2/3)*100:.0f}% (for 3 states) to ~{(1-2/10)*100:.0f}% (for 10 states)")

        if intermediate_count > extreme_count:
            print(f"  ✓ Switching isomers prefer intermediate J — consistent with intermediate axis")
        else:
            print(f"  ✗ Switching isomers prefer extreme J — intermediate axis NOT confirmed")
    else:
        print(f"  Not enough data to test (need 3+ J values at same (Z,A))")

    # ═══════════════════════════════════════════════════════════════
    # TEST 4: The specific B+↔alpha switches — peanut + spin flip
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 4: B+↔ALPHA SWITCHES — The Signature Channel")
    print("=" * 72)
    print("""
  B+↔alpha is the cleanest test: both channels are available, and the
  switch from charge conversion to soliton shedding is exactly what the
  tennis racket flip predicts (short-axis → long-axis spin).

  If the spin flips to the long axis:
    - The neck experiences centrifugal stress → pinch-off → alpha
    - Beta (charge conversion within a lobe) becomes less favorable
    - The higher J of the isomer provides the angular momentum for the flip
""")

    ba_switches = [sw for sw in all_switches
                   if (sw['gs_mode'], sw['iso_mode']) in (('B+', 'alpha'), ('alpha', 'B+'))]

    if ba_switches:
        print(f"  B+↔alpha switches: {len(ba_switches)}")
        print(f"\n  {'Nuclide':>10s}  {'gs→iso':>14s}  {'gs J':>6s}  {'iso J':>6s}  {'|ΔJ|':>5s}  {'pf':>6s}  {'zone':>4s}  {'ε':>6s}")
        print(f"  {'-'*66}")

        djs = []
        pfs = []
        for sw in sorted(ba_switches, key=lambda s: s['A']):
            elem = m.ELEMENTS.get(sw['Z'], f"Z{sw['Z']}")
            name = f"{elem}-{sw['A']}"
            gs_j_str = f"{sw['gs_j']:.1f}" if sw['gs_j'] is not None else "?"
            iso_j_str = f"{sw['iso_j']:.1f}" if sw['iso_j'] is not None else "?"
            dj_str = f"{sw['delta_j']:.1f}" if sw['delta_j'] is not None else "?"
            print(f"  {name:>10s}  {sw['gs_mode']:>6s}→{sw['iso_mode']:<6s}  "
                  f"{gs_j_str:>6s}  {iso_j_str:>6s}  {dj_str:>5s}  "
                  f"{sw['geo'].peanut_f:>6.2f}  {sw['geo'].zone:>4d}  {sw['geo'].eps:>+6.2f}")
            if sw['delta_j'] is not None:
                djs.append(sw['delta_j'])
            pfs.append(sw['geo'].peanut_f)

        if djs:
            print(f"\n  |ΔJ| statistics: mean={np.mean(djs):.1f}, median={np.median(djs):.1f}, "
                  f"std={np.std(djs):.1f}")
        print(f"  pf statistics:  mean={np.mean(pfs):.2f}, median={np.median(pfs):.2f}")
        print(f"  All in peanut regime (pf > 0): {sum(1 for p in pfs if p > 0)}/{len(pfs)}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 5: J direction and mode — does high J correlate with shedding?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 5: HIGH SPIN → SHEDDING — J Correlates with Decay Channel")
    print("=" * 72)
    print("""
  The tennis racket theory predicts: higher J makes the intermediate axis
  instability STRONGER (more angular momentum to flip).  So for any (Z,A)
  with multiple states, the HIGHER J state should be more likely to shed
  (alpha/SF) while the LOWER J state should prefer beta.

  Test: across all (Z,A) pairs with multiple measured J states, does the
  shedding mode (alpha/SF) correlate with HIGHER J?
""")

    higher_j_sheds = 0
    lower_j_sheds = 0
    tested_5 = 0

    for (Z, A), states in nuclide_states.items():
        if len(states) < 2:
            continue
        geo = m.compute_geometric_state(Z, A)
        if geo.peanut_f < 0.1:  # Only test in peanut regime
            continue

        # Get states with measured J and known mode
        valid = []
        for st in states:
            j = m._parse_spin_value(st.get('jpi', ''))
            mode = m.normalize_nubase(st['dominant_mode'])
            if j is not None and mode not in ('unknown', 'IT'):
                valid.append((j, mode))

        if len(valid) < 2:
            continue

        # Sort by J
        valid.sort(key=lambda x: x[0])

        # Does the higher J state shed more?
        low_j, low_mode = valid[0]
        high_j, high_mode = valid[-1]

        if low_j == high_j:
            continue

        shed_modes = {'alpha', 'SF'}
        beta_modes = {'B-', 'B+'}

        high_sheds = high_mode in shed_modes
        low_sheds = low_mode in shed_modes
        high_beta = high_mode in beta_modes
        low_beta = low_mode in beta_modes

        if (high_sheds and low_beta) or (high_sheds and low_mode == 'stable'):
            higher_j_sheds += 1
            tested_5 += 1
        elif (low_sheds and high_beta) or (low_sheds and high_mode == 'stable'):
            lower_j_sheds += 1
            tested_5 += 1

    if tested_5 > 0:
        print(f"  Tested: {tested_5} (Z,A) pairs with J-resolved shedding vs beta states")
        print(f"  Higher J sheds: {higher_j_sheds}/{tested_5} ({higher_j_sheds/tested_5*100:.1f}%)")
        print(f"  Lower J sheds:  {lower_j_sheds}/{tested_5} ({lower_j_sheds/tested_5*100:.1f}%)")

        if higher_j_sheds > lower_j_sheds:
            print(f"\n  ✓ Higher spin preferentially SHEDS — consistent with tennis racket flip")
            print(f"    The angular momentum along the long axis drives neck pinch-off")
        else:
            print(f"\n  ✗ Lower spin sheds more — tennis racket prediction NOT confirmed")
    else:
        print(f"  Not enough data to test")

    # ═══════════════════════════════════════════════════════════════
    # TEST 6: Moment of inertia proxy — pf as aspect ratio
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 6: ASPECT RATIO — More Elongated = More Flips?")
    print("=" * 72)
    print("""
  The instability strength grows with the DIFFERENCE between moments:
    ε_flip ~ (I₃ - I₂)(I₂ - I₁) / (I₁·I₃)

  For a peanut, I₃-I₁ grows with pf (more elongated = bigger ratio).
  A sphere (pf=0) has I₁=I₂=I₃ → no instability.
  A deep peanut (pf>>1) has I₃ >> I₁ → strong instability.

  Prediction: the spin gap |ΔJ| of mode-switching isomers should
  INCREASE with pf (more elongated = bigger flip = bigger ΔJ needed).
""")

    switches_with_dj = [(sw['geo'].peanut_f, sw['delta_j'])
                        for sw in all_switches if sw['delta_j'] is not None]

    if len(switches_with_dj) >= 5:
        pf_vals = np.array([x[0] for x in switches_with_dj])
        dj_vals = np.array([x[1] for x in switches_with_dj])

        from scipy import stats as sp_stats
        r, p = sp_stats.spearmanr(pf_vals, dj_vals)
        print(f"  Data points: {len(switches_with_dj)}")
        print(f"  Spearman correlation (pf vs |ΔJ|): r = {r:+.3f}, p = {p:.4f}")

        # Bin by pf
        pf_edges = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        print(f"\n  {'pf range':>12s}  {'n':>4s}  {'median |ΔJ|':>12s}  {'mean |ΔJ|':>10s}")
        print(f"  {'-'*44}")
        for i in range(len(pf_edges) - 1):
            mask = (pf_vals >= pf_edges[i]) & (pf_vals < pf_edges[i+1])
            if mask.sum() > 0:
                med = np.median(dj_vals[mask])
                mean = np.mean(dj_vals[mask])
                print(f"  [{pf_edges[i]:.1f}, {pf_edges[i+1]:.1f})  {mask.sum():>4d}  {med:>12.1f}  {mean:>10.1f}")

        if r > 0 and p < 0.05:
            print(f"\n  ✓ |ΔJ| increases with pf — more elongated peanuts flip harder")
        elif r > 0:
            print(f"\n  ~ Positive trend but not significant (p={p:.3f})")
        else:
            print(f"\n  ✗ No positive correlation — aspect ratio doesn't predict flip size")

    # ═══════════════════════════════════════════════════════════════
    # TEST 7: The twinning signature — SF from counter-rotation
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 7: TWINNING — Does Spin Flip Trigger SF?")
    print("=" * 72)
    print("""
  The most dramatic prediction: if the two lobes COUNTER-ROTATE after
  a tennis racket flip, the winding coherence across the neck is
  destroyed → topological bifurcation (SF).

  Test: for (Z,A) pairs where ground state is alpha/B- but an isomer
  does SF, does the isomer have HIGHER J?  (Counter-rotation requires
  more angular momentum than single-axis rotation.)
""")

    sf_switches = [sw for sw in all_switches if sw['iso_mode'] == 'SF']
    from_sf = [sw for sw in all_switches if sw['gs_mode'] == 'SF']

    print(f"  Isomers that SWITCH TO SF: {len(sf_switches)}")
    if sf_switches:
        print(f"\n  {'Nuclide':>10s}  {'gs→iso':>14s}  {'gs J':>6s}  {'iso J':>6s}  {'|ΔJ|':>5s}  {'pf':>6s}  {'cf':>5s}")
        print(f"  {'-'*58}")
        for sw in sorted(sf_switches, key=lambda s: s['A']):
            elem = m.ELEMENTS.get(sw['Z'], f"Z{sw['Z']}")
            name = f"{elem}-{sw['A']}"
            gs_j_str = f"{sw['gs_j']:.1f}" if sw['gs_j'] is not None else "?"
            iso_j_str = f"{sw['iso_j']:.1f}" if sw['iso_j'] is not None else "?"
            dj_str = f"{sw['delta_j']:.1f}" if sw['delta_j'] is not None else "?"
            print(f"  {name:>10s}  {sw['gs_mode']:>6s}→{'SF':<6s}  "
                  f"{gs_j_str:>6s}  {iso_j_str:>6s}  {dj_str:>5s}  "
                  f"{sw['geo'].peanut_f:>6.2f}  {sw['geo'].core_full:>5.3f}")

        # Check J direction for SF switches
        sf_higher_j = sum(1 for sw in sf_switches
                          if sw['gs_j'] is not None and sw['iso_j'] is not None
                          and sw['iso_j'] > sw['gs_j'])
        sf_lower_j = sum(1 for sw in sf_switches
                         if sw['gs_j'] is not None and sw['iso_j'] is not None
                         and sw['iso_j'] < sw['gs_j'])
        sf_measured = sf_higher_j + sf_lower_j

        if sf_measured > 0:
            print(f"\n  SF isomer has HIGHER J than gs: {sf_higher_j}/{sf_measured}")
            print(f"  SF isomer has LOWER J than gs:  {sf_lower_j}/{sf_measured}")

    print(f"\n  Isomers that SWITCH FROM SF: {len(from_sf)}")
    if from_sf:
        for sw in sorted(from_sf, key=lambda s: s['A']):
            elem = m.ELEMENTS.get(sw['Z'], f"Z{sw['Z']}")
            name = f"{elem}-{sw['A']}"
            gs_j_str = f"{sw['gs_j']:.1f}" if sw['gs_j'] is not None else "?"
            iso_j_str = f"{sw['iso_j']:.1f}" if sw['iso_j'] is not None else "?"
            print(f"    {name:>10s}  SF→{sw['iso_mode']:<6s}  gs J={gs_j_str}  iso J={iso_j_str}  "
                  f"pf={sw['geo'].peanut_f:.2f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 8: The half-life signature — flipped state lives longer?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 8: HALF-LIFE SHIFT — Does the Flipped State Live Longer?")
    print("=" * 72)
    print("""
  A soliton spinning about the long axis (after flip) has the angular
  momentum stabilizing the neck (gyroscopic effect) during most of the
  flip cycle, but periodically destabilizing it.  The NET effect on
  half-life depends on the competition:
    - Gyroscopic stabilization → LONGER half-life (most of the cycle)
    - Periodic neck stress → SHORTER half-life (at flip moments)

  For alpha-emitting isomers: if the flip drives shedding, the isomer
  should have a SHORTER half-life than if it were doing beta like the
  ground state.  But some high-spin isomers are METASTABLE (trapped in
  the flipped orientation) — the flip cycle is so slow that the isomer
  effectively waits.  Bi-210 m1 (3.04 Myr vs 5.0 d gs) is the extreme
  case — the high spin TRAPS it in a long-lived configuration.
""")

    # Compare half-lives for mode-switching isomers
    alpha_switch_hl = []
    beta_switch_hl = []

    for sw in all_switches:
        gs_hl = sw['gs_hl']
        iso_hl = sw['iso_hl']
        if not np.isfinite(gs_hl) or gs_hl <= 0 or gs_hl > 1e30:
            continue
        if not np.isfinite(iso_hl) or iso_hl <= 0 or iso_hl > 1e30:
            continue

        log_ratio = math.log10(iso_hl / gs_hl)

        if sw['iso_mode'] in ('alpha', 'SF'):
            alpha_switch_hl.append((sw, log_ratio))
        elif sw['iso_mode'] in ('B-', 'B+'):
            beta_switch_hl.append((sw, log_ratio))

    if alpha_switch_hl:
        ratios = [x[1] for x in alpha_switch_hl]
        print(f"  Isomers that switch TO alpha/SF (n={len(alpha_switch_hl)}):")
        print(f"    log₁₀(t½_iso / t½_gs):  mean={np.mean(ratios):+.2f}, "
              f"median={np.median(ratios):+.2f}")
        print(f"    Isomer lives LONGER: {sum(1 for r in ratios if r > 0)}/{len(ratios)}")
        print(f"    Isomer lives SHORTER: {sum(1 for r in ratios if r < 0)}/{len(ratios)}")

        # Print individual cases
        print(f"\n  {'Nuclide':>10s}  {'gs→iso':>14s}  {'gs t½':>12s}  {'iso t½':>12s}  {'log ratio':>10s}  {'|ΔJ|':>5s}")
        print(f"  {'-'*68}")
        for sw, lr in sorted(alpha_switch_hl, key=lambda x: -x[1]):
            elem = m.ELEMENTS.get(sw['Z'], f"Z{sw['Z']}")
            name = f"{elem}-{sw['A']}"
            gs_hl_str = _fmt_hl(sw['gs_hl'])
            iso_hl_str = _fmt_hl(sw['iso_hl'])
            dj_str = f"{sw['delta_j']:.1f}" if sw['delta_j'] is not None else "?"
            print(f"  {name:>10s}  {sw['gs_mode']:>6s}→{sw['iso_mode']:<6s}  "
                  f"{gs_hl_str:>12s}  {iso_hl_str:>12s}  {lr:>+10.2f}  {dj_str:>5s}")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  SUMMARY — Tennis Racket Theorem Applied to Peanut Solitons")
    print("=" * 72)
    print(f"""
  The intermediate axis theorem (tennis racket / Dzhanibekov effect)
  predicts that a peanut-shaped soliton spinning about its intermediate
  axis will periodically flip, causing the spin to align with the long
  axis (through the neck).  This flip changes the available decay
  channels: short-axis spin favors beta, long-axis spin favors
  shedding (alpha/SF) by stressing the neck.

  RESULTS:
""")

    tests_passed = 0
    tests_total = 0

    # Summarize each test
    tests = [
        ("Symmetry breaking", odd_rate > even_rate if even_rate > 0 else None,
         f"Odd-A switch rate {odd_rate:.1f}% vs even-A {even_rate:.1f}%"),
        ("Peanut concentration", zone_switches[3] > zone_switches[1] if zone_switches[1] > 0 else True,
         f"Zone 3: {zone_switches[3]} switches, Zone 1: {zone_switches[1]}"),
    ]

    if tested > 0:
        tests.append(("Intermediate J", intermediate_count > extreme_count,
                       f"{intermediate_count}/{tested} intermediate"))
    if tested_5 > 0:
        tests.append(("High J → shedding", higher_j_sheds > lower_j_sheds,
                       f"{higher_j_sheds}/{tested_5} higher-J sheds"))
    if len(switches_with_dj) >= 5:
        tests.append(("|ΔJ| ~ pf correlation", r > 0,
                       f"Spearman r={r:+.3f}, p={p:.4f}"))

    for name, passed, detail in tests:
        tests_total += 1
        if passed:
            tests_passed += 1
            mark = "✓"
        elif passed is None:
            mark = "—"
        else:
            mark = "✗"
        print(f"    {mark} {name}: {detail}")

    print(f"\n  Score: {tests_passed}/{tests_total} tests consistent with tennis racket mechanism")


def _fmt_hl(hl_s):
    """Format half-life for display."""
    if not np.isfinite(hl_s) or hl_s <= 0:
        return "?"
    if hl_s > 1e30:
        return "stable"
    if hl_s > 3.1557e13:
        return f"{hl_s/3.1557e13:.2f} Myr"
    if hl_s > 3.1557e10:
        return f"{hl_s/3.1557e10:.2f} kyr"
    if hl_s > 3.1557e7:
        return f"{hl_s/3.1557e7:.2f} yr"
    if hl_s > 86400:
        return f"{hl_s/86400:.1f} d"
    if hl_s > 3600:
        return f"{hl_s/3600:.1f} hr"
    if hl_s > 60:
        return f"{hl_s/60:.1f} min"
    if hl_s > 1:
        return f"{hl_s:.2f} s"
    if hl_s > 1e-3:
        return f"{hl_s*1e3:.2f} ms"
    if hl_s > 1e-6:
        return f"{hl_s*1e6:.2f} us"
    return f"{hl_s:.2e} s"


if __name__ == '__main__':
    main()
