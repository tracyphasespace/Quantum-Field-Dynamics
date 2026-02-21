#!/usr/bin/env python3
"""
Regional Physics Study — Understanding Each Decay Mode in Its Own Territory

NOT a global fit. Each region is studied on its own terms:
  - Neutron emission: what makes a soliton shed neutrons?
  - Proton emission: what makes it shed protons?
  - IT (gamma): what makes a ground state emit photons?
  - Alpha: what opens the barrier in the peanut regime?
  - SF: what triggers topological bifurcation?
  - Beta: what drives the geodesic glide?

A tritium atom doesn't see the same forces a uranium atom does.
"""

import csv
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qfd_nuclide_predictor import compute_geometric_state, BETA, A_CRIT

PI = math.pi
E = math.e

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING — include ALL modes, including IT
# ═══════════════════════════════════════════════════════════════════

def load_all_gs():
    """Load ALL ground-state nuclides, including IT."""
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'three-layer-lagrangian', 'data', 'clean_species_sorted.csv'
    )
    MODE_MAP = {
        'beta-': 'B-', 'beta+': 'B+', 'alpha': 'alpha',
        'stable': 'stable', 'SF': 'SF',
        'proton': 'p', 'neutron': 'n', 'IT': 'IT',
    }
    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            A = int(row['A'])
            Z = int(row['Z'])
            if A < 2:
                continue
            if int(row.get('az_order', 0)) != 0:
                continue
            species = row.get('clean_species', '')
            mode = MODE_MAP.get(species)
            if mode is None:
                continue
            log_hl = None
            try:
                log_hl = float(row.get('log_hl', ''))
            except (ValueError, TypeError):
                pass
            geo = compute_geometric_state(Z, A)
            data.append({
                'A': A, 'Z': Z, 'N': A - Z,
                'mode': mode, 'log_hl': log_hl,
                'eps': geo.eps, 'pf': geo.peanut_f,
                'cf': geo.core_full, 'zone': geo.zone,
                'element': row.get('element', '??'),
            })
    return data


# ═══════════════════════════════════════════════════════════════════
# REGION 1: DRIP LINES — Neutron and Proton Emission
# ═══════════════════════════════════════════════════════════════════

def study_drip_lines(data):
    """What distinguishes n/p emitters from B-/B+ at the drip line?"""
    print("=" * 72)
    print("  REGION 1: DRIP LINES — Neutron and Proton Emission")
    print("  Where does beta decay give way to direct particle emission?")
    print("=" * 72)

    # Neutron emitters vs beta-minus (both neutron-rich, ε < 0)
    n_emitters = [d for d in data if d['mode'] == 'n']
    bm_light = [d for d in data if d['mode'] == 'B-' and d['A'] <= 30]

    print(f"\n  ── Neutron Drip Line (n={len(n_emitters)}, B- with A≤30: {len(bm_light)}) ──\n")

    # Key variables: ε, N/Z, A
    n_eps = [d['eps'] for d in n_emitters]
    n_nz = [d['N'] / d['Z'] for d in n_emitters]
    n_A = [d['A'] for d in n_emitters]

    bm_eps = [d['eps'] for d in bm_light]
    bm_nz = [d['N'] / d['Z'] for d in bm_light]
    bm_A = [d['A'] for d in bm_light]

    print(f"  {'':30s}  {'neutron':>10s}  {'B- (A≤30)':>10s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}")
    print(f"  {'ε range':30s}  {min(n_eps):+5.1f}/{max(n_eps):+5.1f}  {min(bm_eps):+5.1f}/{max(bm_eps):+5.1f}")
    print(f"  {'ε mean':30s}  {np.mean(n_eps):+10.2f}  {np.mean(bm_eps):+10.2f}")
    print(f"  {'N/Z range':30s}  {min(n_nz):5.2f}/{max(n_nz):5.2f}  {min(bm_nz):5.2f}/{max(bm_nz):5.2f}")
    print(f"  {'N/Z mean':30s}  {np.mean(n_nz):10.2f}  {np.mean(bm_nz):10.2f}")
    print(f"  {'A range':30s}  {min(n_A):5d}/{max(n_A):5d}  {min(bm_A):5d}/{max(bm_A):5d}")
    print(f"  {'Z range':30s}  {min(d['Z'] for d in n_emitters):5d}/{max(d['Z'] for d in n_emitters):5d}  {min(d['Z'] for d in bm_light):5d}/{max(d['Z'] for d in bm_light):5d}")

    # Is there a clean ε threshold?
    print(f"\n  ── Neutron drip-line threshold scan ──\n")
    print(f"  {'ε_thresh':>10s}  {'n→n':>5s}  {'n→B-':>5s}  {'B-→n':>5s}  {'B-→B-':>5s}  {'n_acc':>6s}  {'B-_acc':>6s}")
    for eps_th in [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0]:
        nn = sum(1 for d in n_emitters if d['eps'] <= eps_th)  # correctly called n
        nb = sum(1 for d in n_emitters if d['eps'] > eps_th)   # n called B-
        bn = sum(1 for d in bm_light if d['eps'] <= eps_th)     # B- called n
        bb = sum(1 for d in bm_light if d['eps'] > eps_th)     # B- correctly
        n_acc = nn / len(n_emitters) * 100 if n_emitters else 0
        b_acc = bb / len(bm_light) * 100 if bm_light else 0
        print(f"  {eps_th:+10.1f}  {nn:5d}  {nb:5d}  {bn:5d}  {bb:5d}  {n_acc:5.1f}%  {b_acc:5.1f}%")

    # Actually, ε alone won't do it — we need to consider Z too
    # Neutron emitters are Z=1-9, B- at A≤30 is Z=1-12
    # The real question: at a given Z, what N/Z triggers neutron emission?
    print(f"\n  ── Per-element drip line (neutron-rich side) ──\n")
    print(f"  {'Z':>3s}  {'El':>3s}  {'n_count':>7s}  {'max_n_A':>7s}  {'min_B-_A':>8s}  {'B-_count':>8s}  {'drip_ε':>7s}")
    for Z in range(1, 10):
        z_n = [d for d in n_emitters if d['Z'] == Z]
        z_bm = [d for d in bm_light if d['Z'] == Z]
        if not z_n and not z_bm:
            continue
        max_n_A = max(d['A'] for d in z_n) if z_n else '-'
        min_bm_A = min(d['A'] for d in z_bm) if z_bm else '-'
        drip_eps = max(d['eps'] for d in z_n) if z_n else None
        el = z_n[0]['element'] if z_n else (z_bm[0]['element'] if z_bm else '?')
        drip_str = f"{drip_eps:+7.2f}" if drip_eps is not None else "   -   "
        print(f"  {Z:3d}  {el:>3s}  {len(z_n):7d}  {str(max_n_A):>7s}  {str(min_bm_A):>8s}  {len(z_bm):8d}  {drip_str}")

    print()

    # ── Proton drip line ──
    p_emitters = [d for d in data if d['mode'] == 'p']
    bp_light = [d for d in data if d['mode'] == 'B+' and d['A'] <= 30]
    bp_heavy = [d for d in data if d['mode'] == 'B+' and d['A'] > 100]

    # Split proton emitters into light and heavy
    p_light = [d for d in p_emitters if d['A'] <= 30]
    p_heavy = [d for d in p_emitters if d['A'] > 30]

    print(f"  ── Proton Drip Line (light p={len(p_light)}, heavy p={len(p_heavy)}) ──\n")

    if p_light:
        p_eps = [d['eps'] for d in p_light]
        p_nz = [d['N'] / d['Z'] for d in p_light]
        bp_eps = [d['eps'] for d in bp_light]
        bp_nz = [d['N'] / d['Z'] for d in bp_light]

        print(f"  LIGHT (A ≤ 30):")
        print(f"  {'':30s}  {'proton':>10s}  {'B+ (A≤30)':>10s}")
        print(f"  {'-'*30}  {'-'*10}  {'-'*10}")
        print(f"  {'ε range':30s}  {min(p_eps):+5.1f}/{max(p_eps):+5.1f}  {min(bp_eps):+5.1f}/{max(bp_eps):+5.1f}")
        print(f"  {'ε mean':30s}  {np.mean(p_eps):+10.2f}  {np.mean(bp_eps):+10.2f}")
        print(f"  {'N/Z range':30s}  {min(p_nz):5.2f}/{max(p_nz):5.2f}  {min(bp_nz):5.2f}/{max(bp_nz):5.2f}")
        print(f"  {'N/Z mean':30s}  {np.mean(p_nz):10.2f}  {np.mean(bp_nz):10.2f}")

    if p_heavy:
        print(f"\n  HEAVY (A > 30):")
        ph_eps = [d['eps'] for d in p_heavy]
        # Compare heavy p emitters to B+ in the same A range
        bp_same = [d for d in data if d['mode'] == 'B+' and
                   min(d2['A'] for d2 in p_heavy) <= d['A'] <= max(d2['A'] for d2 in p_heavy)]
        bps_eps = [d['eps'] for d in bp_same]

        print(f"  {'':30s}  {'proton':>10s}  {'B+ (same A)':>10s}")
        print(f"  {'-'*30}  {'-'*10}  {'-'*11}")
        print(f"  {'ε range':30s}  {min(ph_eps):+5.1f}/{max(ph_eps):+5.1f}  {min(bps_eps):+5.1f}/{max(bps_eps):+5.1f}" if bps_eps else "")
        print(f"  {'ε mean':30s}  {np.mean(ph_eps):+10.2f}  {np.mean(bps_eps):+10.2f}" if bps_eps else "")
        print(f"  {'A range':30s}  {min(d['A'] for d in p_heavy):5d}/{max(d['A'] for d in p_heavy):5d}  {min(d['A'] for d in bp_same):5d}/{max(d['A'] for d in bp_same):5d}" if bp_same else "")

    # Proton drip threshold
    print(f"\n  ── Proton emission threshold (ε > thresh → proton?) ──\n")

    # For each A bin, what's the minimum ε that triggers proton emission?
    # Group by A ranges
    bp_all = [d for d in data if d['mode'] == 'B+']

    print(f"  {'A_range':>12s}  {'n_p':>4s}  {'n_B+':>5s}  {'min_p_ε':>8s}  {'max_B+_ε':>9s}  {'gap':>6s}")
    for a_lo, a_hi in [(2, 20), (20, 50), (50, 80), (80, 120), (120, 160), (160, 200)]:
        p_bin = [d for d in p_emitters if a_lo <= d['A'] < a_hi]
        bp_bin = [d for d in bp_all if a_lo <= d['A'] < a_hi]
        if not p_bin:
            continue
        min_p = min(d['eps'] for d in p_bin)
        max_bp = max(d['eps'] for d in bp_bin) if bp_bin else float('-inf')
        gap = min_p - max_bp if bp_bin else float('inf')
        print(f"  {a_lo:3d}-{a_hi:<3d}       {len(p_bin):4d}  {len(bp_bin):5d}  {min_p:+8.2f}  {max_bp:+9.2f}  {gap:+6.2f}")


# ═══════════════════════════════════════════════════════════════════
# REGION 2: IT GROUND STATES — Gamma Emission
# ═══════════════════════════════════════════════════════════════════

def study_IT(data):
    """What makes a ground state emit gamma instead of beta/alpha?"""
    print("\n" + "=" * 72)
    print("  REGION 2: IT GROUND STATES — Gamma Emission")
    print("  15 nuclides where the ground state decays by isomeric transition")
    print("=" * 72)

    it_nuclides = [d for d in data if d['mode'] == 'IT']

    print(f"\n  ── All 15 IT ground states ──\n")
    print(f"  {'El-A':>8s}   Z    N   N/Z      ε      pf    zone  {'log_hl':>7s}")
    for d in sorted(it_nuclides, key=lambda x: x['A']):
        hl = f"{d['log_hl']:7.2f}" if d['log_hl'] is not None else "     - "
        print(f"  {d['element']:>4s}-{d['A']:<3d}  {d['Z']:3d}  {d['N']:3d}  {d['N']/d['Z']:.2f}  {d['eps']:+6.2f}  {d['pf']:5.2f}  Z{d['zone']}  {hl}")

    # Two clusters?
    it_light = [d for d in it_nuclides if d['A'] < 100]
    it_heavy = [d for d in it_nuclides if d['A'] >= 100]

    print(f"\n  Cluster 1 (A<100): {len(it_light)} nuclides, Z={min(d['Z'] for d in it_light)}-{max(d['Z'] for d in it_light)}")
    print(f"    ε range: {min(d['eps'] for d in it_light):+.2f} to {max(d['eps'] for d in it_light):+.2f}")
    print(f"    ALL neutron-rich (ε < 0), ALL Zone 1 (pf=0)")
    print(f"    N range: {min(d['N'] for d in it_light)}-{max(d['N'] for d in it_light)}")

    print(f"\n  Cluster 2 (A≥100): {len(it_heavy)} nuclides, Z={min(d['Z'] for d in it_heavy)}-{max(d['Z'] for d in it_heavy)}")
    print(f"    ε range: {min(d['eps'] for d in it_heavy):+.2f} to {max(d['eps'] for d in it_heavy):+.2f}")
    print(f"    Mixed ε sign, Zone 2-3 (pf=0.05-1.19)")

    # What's special about these compared to their neighbors?
    # For each IT nuclide, what does the model predict for it?
    print(f"\n  ── What would the models predict for IT nuclides? ──\n")

    # Import the predictors
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from qfd_nuclide_predictor import predict_decay
    from validate_kinetic_fracture import predict_kinetic_coulomb

    PI = math.pi
    print(f"  {'El-A':>8s}   actual   v8_pred  barrier_pred    ε      pf")
    for d in sorted(it_nuclides, key=lambda x: x['A']):
        v8_mode, _ = predict_decay(d['Z'], d['A'])
        barrier = predict_kinetic_coulomb(d['Z'], d['A'], K_SHEAR=PI, k_coul_scale=4.0)
        print(f"  {d['element']:>4s}-{d['A']:<3d}  IT       {v8_mode:8s}  {barrier:12s}  {d['eps']:+6.2f}  {d['pf']:5.2f}")

    # N=50, 82 magic check — are IT nuclides near magic N?
    print(f"\n  ── Magic number proximity ──\n")
    magic_N = [2, 8, 20, 28, 50, 82, 126]
    for d in sorted(it_nuclides, key=lambda x: x['A']):
        dists = [(d['N'] - m, m) for m in magic_N]
        closest = min(dists, key=lambda x: abs(x[0]))
        print(f"  {d['element']:>4s}-{d['A']:<3d}  N={d['N']:3d}  closest magic: N={closest[1]} (Δ={closest[0]:+d})")


# ═══════════════════════════════════════════════════════════════════
# REGION 3: MODE BOUNDARIES — Where Modes Actually Compete
# ═══════════════════════════════════════════════════════════════════

def study_boundaries(data):
    """Map the actual boundaries between modes in (A, ε) space."""
    print("\n" + "=" * 72)
    print("  REGION 3: MODE BOUNDARIES — Where Modes Actually Compete")
    print("  Not a global fit — just mapping where transitions happen")
    print("=" * 72)

    # For each Z, find the ε at which the mode changes
    # This reveals the ACTUAL decision boundary

    modes = ['B-', 'B+', 'alpha', 'stable', 'SF', 'p', 'n', 'IT']

    # Group by Z
    from collections import defaultdict
    by_Z = defaultdict(list)
    for d in data:
        by_Z[d['Z']].append(d)

    # For each Z, sort by N and find mode transitions
    print(f"\n  ── Mode transitions along isotope chains ──\n")
    print(f"  Z  El   A_range   n_iso  modes_seen")

    transitions = []
    for Z in sorted(by_Z.keys()):
        chain = sorted(by_Z[Z], key=lambda x: x['A'])
        if len(chain) < 3:
            continue
        mode_seq = [d['mode'] for d in chain]
        unique_modes = []
        for m in mode_seq:
            if not unique_modes or unique_modes[-1] != m:
                unique_modes.append(m)

        # Find transition points
        for i in range(len(chain) - 1):
            if chain[i]['mode'] != chain[i+1]['mode']:
                transitions.append({
                    'Z': Z, 'el': chain[i]['element'],
                    'A1': chain[i]['A'], 'mode1': chain[i]['mode'],
                    'eps1': chain[i]['eps'],
                    'A2': chain[i+1]['A'], 'mode2': chain[i+1]['mode'],
                    'eps2': chain[i+1]['eps'],
                })

    # Count transition types
    from collections import Counter
    trans_types = Counter()
    for t in transitions:
        key = f"{t['mode1']}→{t['mode2']}"
        trans_types[key] += 1

    print(f"\n  {'Transition':>20s}  {'Count':>5s}  {'ε1_mean':>8s}  {'ε2_mean':>8s}")
    print(f"  {'-'*20}  {'-'*5}  {'-'*8}  {'-'*8}")
    for key, count in trans_types.most_common(20):
        m1, m2 = key.split('→')
        these = [t for t in transitions if t['mode1'] == m1 and t['mode2'] == m2]
        e1 = np.mean([t['eps1'] for t in these])
        e2 = np.mean([t['eps2'] for t in these])
        print(f"  {key:>20s}  {count:5d}  {e1:+8.2f}  {e2:+8.2f}")

    # The B-/alpha boundary is the critical one
    ba_trans = [t for t in transitions if
                (t['mode1'] == 'B-' and t['mode2'] == 'alpha') or
                (t['mode1'] == 'alpha' and t['mode2'] == 'B-')]

    if ba_trans:
        print(f"\n  ── B-/Alpha boundary (n={len(ba_trans)} transitions) ──\n")
        print(f"  Z  El    A   {'from':>6s}→{'to':>6s}   ε_from   ε_to     pf")
        for t in sorted(ba_trans, key=lambda x: x['Z']):
            geo = compute_geometric_state(t['Z'], (t['A1'] + t['A2']) // 2)
            print(f"  {t['Z']:2d}  {t['el']:>3s}  {t['A1']:3d}  {t['mode1']:>6s}→{t['mode2']:>6s}  {t['eps1']:+6.2f}  {t['eps2']:+6.2f}  {geo.peanut_f:5.2f}")

    # B+/alpha boundary
    bpa_trans = [t for t in transitions if
                 (t['mode1'] == 'B+' and t['mode2'] == 'alpha') or
                 (t['mode1'] == 'alpha' and t['mode2'] == 'B+')]

    if bpa_trans:
        print(f"\n  ── B+/Alpha boundary (n={len(bpa_trans)} transitions) ──\n")
        print(f"  Z  El    A   {'from':>6s}→{'to':>6s}   ε_from   ε_to     pf")
        for t in sorted(bpa_trans, key=lambda x: x['Z']):
            geo = compute_geometric_state(t['Z'], (t['A1'] + t['A2']) // 2)
            print(f"  {t['Z']:2d}  {t['el']:>3s}  {t['A1']:3d}  {t['mode1']:>6s}→{t['mode2']:>6s}  {t['eps1']:+6.2f}  {t['eps2']:+6.2f}  {geo.peanut_f:5.2f}")

    # Alpha/SF boundary
    asf_trans = [t for t in transitions if
                 (t['mode1'] == 'alpha' and t['mode2'] == 'SF') or
                 (t['mode1'] == 'SF' and t['mode2'] == 'alpha')]

    if asf_trans:
        print(f"\n  ── Alpha/SF boundary (n={len(asf_trans)} transitions) ──\n")
        print(f"  Z  El    A   {'from':>6s}→{'to':>6s}   ε_from   ε_to     pf     cf")
        for t in sorted(asf_trans, key=lambda x: x['Z']):
            geo = compute_geometric_state(t['Z'], (t['A1'] + t['A2']) // 2)
            print(f"  {t['Z']:2d}  {t['el']:>3s}  {t['A1']:3d}  {t['mode1']:>6s}→{t['mode2']:>6s}  {t['eps1']:+6.2f}  {t['eps2']:+6.2f}  {geo.peanut_f:5.2f}  {geo.core_full:5.3f}")

    # B+/proton boundary
    bp_trans = [t for t in transitions if
                (t['mode1'] == 'B+' and t['mode2'] == 'p') or
                (t['mode1'] == 'p' and t['mode2'] == 'B+')]

    if bp_trans:
        print(f"\n  ── B+/Proton boundary (n={len(bp_trans)} transitions) ──\n")
        print(f"  Z  El    A   {'from':>6s}→{'to':>6s}   ε_from   ε_to")
        for t in sorted(bp_trans, key=lambda x: x['Z'])[:20]:
            print(f"  {t['Z']:2d}  {t['el']:>3s}  {t['A1']:3d}  {t['mode1']:>6s}→{t['mode2']:>6s}  {t['eps1']:+6.2f}  {t['eps2']:+6.2f}")

    # B-/neutron boundary
    bn_trans = [t for t in transitions if
                (t['mode1'] == 'B-' and t['mode2'] == 'n') or
                (t['mode1'] == 'n' and t['mode2'] == 'B-')]

    if bn_trans:
        print(f"\n  ── B-/Neutron boundary (n={len(bn_trans)} transitions) ──\n")
        print(f"  Z  El    A   {'from':>6s}→{'to':>6s}   ε_from   ε_to")
        for t in sorted(bn_trans, key=lambda x: x['Z'])[:20]:
            print(f"  {t['Z']:2d}  {t['el']:>3s}  {t['A1']:3d}  {t['mode1']:>6s}→{t['mode2']:>6s}  {t['eps1']:+6.2f}  {t['eps2']:+6.2f}")


# ═══════════════════════════════════════════════════════════════════
# REGION 4: LIGHT NUCLEI — A ≤ 30 (its own world)
# ═══════════════════════════════════════════════════════════════════

def study_light_nuclei(data):
    """Light nuclei: where n, p, B-, B+, stable, and alpha ALL coexist."""
    print("\n" + "=" * 72)
    print("  REGION 4: LIGHT NUCLEI (A ≤ 30)")
    print("  Six modes compete in a tiny region — completely different physics")
    print("=" * 72)

    light = [d for d in data if d['A'] <= 30]

    from collections import Counter
    modes = Counter(d['mode'] for d in light)
    print(f"\n  Mode census (A ≤ 30, n={len(light)}):")
    for m, c in modes.most_common():
        print(f"    {m:8s}: {c:4d}  ({100*c/len(light):.1f}%)")

    # What does v8 predict for these?
    from qfd_nuclide_predictor import predict_decay

    correct = 0
    conf = {}
    for d in light:
        pred, _ = predict_decay(d['Z'], d['A'])
        if pred == d['mode']:
            correct += 1
        key = (d['mode'], pred)
        conf[key] = conf.get(key, 0) + 1

    print(f"\n  v8 accuracy on A ≤ 30: {correct}/{len(light)} = {100*correct/len(light):.1f}%")

    # Show confusion for n and p specifically
    print(f"\n  v8 predictions for neutron emitters:")
    for d in sorted([x for x in light if x['mode'] == 'n'], key=lambda x: x['A']):
        pred, _ = predict_decay(d['Z'], d['A'])
        mark = '✓' if pred == 'n' else '✗'
        print(f"    {d['element']:>3s}-{d['A']:<3d}  actual=n  v8={pred:6s}  {mark}  ε={d['eps']:+.2f}  N/Z={d['N']/d['Z']:.2f}")

    print(f"\n  v8 predictions for proton emitters (A ≤ 30):")
    for d in sorted([x for x in light if x['mode'] == 'p'], key=lambda x: x['A']):
        pred, _ = predict_decay(d['Z'], d['A'])
        mark = '✓' if pred == 'p' else '✗'
        print(f"    {d['element']:>3s}-{d['A']:<3d}  actual=p  v8={pred:6s}  {mark}  ε={d['eps']:+.2f}  N/Z={d['N']/d['Z']:.2f}")

    # Simple rule test: can |ε| alone separate n/p from B-/B+?
    print(f"\n  ── |ε| separation test (A ≤ 30) ──\n")

    # Collect |ε| for each mode
    for mode in ['n', 'B-', 'p', 'B+', 'stable', 'alpha']:
        mode_d = [d for d in light if d['mode'] == mode]
        if not mode_d:
            continue
        eps_vals = [abs(d['eps']) for d in mode_d]
        print(f"    {mode:8s} (n={len(mode_d):3d}): |ε| = {min(eps_vals):.2f} – {max(eps_vals):.2f}  (mean {np.mean(eps_vals):.2f})")


# ═══════════════════════════════════════════════════════════════════
# REGION 5: ALPHA TERRITORY — The peanut regime
# ═══════════════════════════════════════════════════════════════════

def study_alpha_territory(data):
    """Alpha decay: where does the barrier actually open?"""
    print("\n" + "=" * 72)
    print("  REGION 5: ALPHA TERRITORY (pf > 0, A > 137)")
    print("  Where alpha competes with B+ — the fundamental bottleneck")
    print("=" * 72)

    peanut = [d for d in data if d['pf'] > 0]

    from collections import Counter
    modes = Counter(d['mode'] for d in peanut)
    print(f"\n  Mode census (pf > 0, n={len(peanut)}):")
    for m, c in modes.most_common():
        print(f"    {m:8s}: {c:4d}  ({100*c/len(peanut):.1f}%)")

    # Alpha vs B+ in peanut regime
    alpha_d = [d for d in peanut if d['mode'] == 'alpha']
    bp_d = [d for d in peanut if d['mode'] == 'B+']

    print(f"\n  Alpha (n={len(alpha_d)}) vs B+ (n={len(bp_d)}) in peanut:")
    print(f"  {'':20s}  {'alpha':>10s}  {'B+':>10s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}")
    print(f"  {'ε mean':20s}  {np.mean([d['eps'] for d in alpha_d]):+10.2f}  {np.mean([d['eps'] for d in bp_d]):+10.2f}")
    print(f"  {'pf mean':20s}  {np.mean([d['pf'] for d in alpha_d]):10.2f}  {np.mean([d['pf'] for d in bp_d]):10.2f}")
    print(f"  {'pf range':20s}  {min(d['pf'] for d in alpha_d):.2f}-{max(d['pf'] for d in alpha_d):.2f}   {min(d['pf'] for d in bp_d):.2f}-{max(d['pf'] for d in bp_d):.2f}")

    # 2D map: what decides alpha vs B+ at a given pf?
    print(f"\n  ── Alpha vs B+ by pf bin ──\n")
    print(f"  {'pf_bin':>10s}  {'n_alpha':>7s}  {'n_B+':>5s}  {'α_frac':>7s}  {'α_ε_mean':>9s}  {'B+_ε_mean':>10s}")
    for pf_lo, pf_hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, 3.0)]:
        a_bin = [d for d in alpha_d if pf_lo <= d['pf'] < pf_hi]
        b_bin = [d for d in bp_d if pf_lo <= d['pf'] < pf_hi]
        n_a, n_b = len(a_bin), len(b_bin)
        frac = n_a / (n_a + n_b) * 100 if (n_a + n_b) > 0 else 0
        a_eps = np.mean([d['eps'] for d in a_bin]) if a_bin else float('nan')
        b_eps = np.mean([d['eps'] for d in b_bin]) if b_bin else float('nan')
        print(f"  {pf_lo:.1f}-{pf_hi:.1f}      {n_a:7d}  {n_b:5d}  {frac:6.1f}%  {a_eps:+9.2f}  {b_eps:+10.2f}")


# ═══════════════════════════════════════════════════════════════════
# REGION 6: SUPERHEAVY — SF Territory
# ═══════════════════════════════════════════════════════════════════

def study_SF_territory(data):
    """Spontaneous fission: what makes a soliton bifurcate?"""
    print("\n" + "=" * 72)
    print("  REGION 6: SUPERHEAVY — SF Territory (A > 230)")
    print("  What separates SF from alpha in the heaviest nuclei?")
    print("=" * 72)

    heavy = [d for d in data if d['A'] > 230]

    from collections import Counter
    modes = Counter(d['mode'] for d in heavy)
    print(f"\n  Mode census (A > 230, n={len(heavy)}):")
    for m, c in modes.most_common():
        print(f"    {m:8s}: {c:4d}  ({100*c/len(heavy):.1f}%)")

    sf_d = [d for d in heavy if d['mode'] == 'SF']
    alpha_d = [d for d in heavy if d['mode'] == 'alpha']

    if sf_d and alpha_d:
        print(f"\n  SF (n={len(sf_d)}) vs Alpha (n={len(alpha_d)}) in superheavy:")
        print(f"  {'':20s}  {'SF':>10s}  {'alpha':>10s}")
        print(f"  {'-'*20}  {'-'*10}  {'-'*10}")
        print(f"  {'Z mean':20s}  {np.mean([d['Z'] for d in sf_d]):10.1f}  {np.mean([d['Z'] for d in alpha_d]):10.1f}")
        print(f"  {'N mean':20s}  {np.mean([d['N'] for d in sf_d]):10.1f}  {np.mean([d['N'] for d in alpha_d]):10.1f}")
        print(f"  {'N/Z mean':20s}  {np.mean([d['N']/d['Z'] for d in sf_d]):10.3f}  {np.mean([d['N']/d['Z'] for d in alpha_d]):10.3f}")
        print(f"  {'ε mean':20s}  {np.mean([d['eps'] for d in sf_d]):+10.2f}  {np.mean([d['eps'] for d in alpha_d]):+10.2f}")
        print(f"  {'pf mean':20s}  {np.mean([d['pf'] for d in sf_d]):10.2f}  {np.mean([d['pf'] for d in alpha_d]):10.2f}")
        print(f"  {'cf mean':20s}  {np.mean([d['cf'] for d in sf_d]):10.3f}  {np.mean([d['cf'] for d in alpha_d]):10.3f}")

        # N=177 proximity
        print(f"\n  ── N=177 ceiling proximity ──\n")
        print(f"  {'':20s}  {'SF':>10s}  {'alpha':>10s}")
        n177_sf = [d['N'] for d in sf_d]
        n177_a = [d['N'] for d in alpha_d]
        print(f"  {'N max':20s}  {max(n177_sf):10d}  {max(n177_a):10d}")
        print(f"  {'N > 155':20s}  {sum(1 for n in n177_sf if n > 155):10d}  {sum(1 for n in n177_a if n > 155):10d}")
        print(f"  {'N > 165':20s}  {sum(1 for n in n177_sf if n > 165):10d}  {sum(1 for n in n177_a if n > 165):10d}")

        # Even-odd effect
        print(f"\n  ── Even/odd Z effect ──\n")
        sf_even = [d for d in sf_d if d['Z'] % 2 == 0]
        sf_odd = [d for d in sf_d if d['Z'] % 2 == 1]
        a_even = [d for d in alpha_d if d['Z'] % 2 == 0]
        a_odd = [d for d in alpha_d if d['Z'] % 2 == 1]
        print(f"  Even-Z: SF={len(sf_even)}, alpha={len(a_even)} → SF frac = {len(sf_even)/(len(sf_even)+len(a_even))*100:.1f}%")
        print(f"  Odd-Z:  SF={len(sf_odd)}, alpha={len(a_odd)} → SF frac = {len(sf_odd)/(len(sf_odd)+len(a_odd))*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 72)
    print("  REGIONAL PHYSICS STUDY")
    print("  Understanding each decay mode in its own territory")
    print("=" * 72)

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    data = load_all_gs()

    from collections import Counter
    modes = Counter(d['mode'] for d in data)
    print(f"\n  Loaded {len(data)} ground-state nuclides (ALL modes, including IT)")
    for m, c in modes.most_common():
        print(f"    {m:8s}: {c:4d}")

    study_drip_lines(data)
    study_IT(data)
    study_boundaries(data)
    study_light_nuclei(data)
    study_alpha_territory(data)
    study_SF_territory(data)

    print(f"\n  Done.")
