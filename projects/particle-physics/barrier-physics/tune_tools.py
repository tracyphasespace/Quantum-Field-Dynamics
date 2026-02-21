#!/usr/bin/env python3
"""
Tune Each Tool Independently — Then Look for Triggers

Instead of coordinate thresholds (ε > 3.0, N/Z < 0.85), find the
PHYSICAL CONDITION that triggers each mode:

  Neutron: core capacity exceeded → neutron squeezed out
  Proton:  proton scission barrier ≤ 0 → proton ejected by Coulomb
  Alpha:   alpha scission barrier ≤ 0 → soliton sheds alpha cluster
  SF:      peanut neck → 0 → topological bifurcation
  Beta:    survival gradient > 0 → weak decay downhill

The trigger should be a DERIVED QUANTITY with physical meaning,
not a fitted boundary on a raw coordinate.
"""

import csv
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qfd_nuclide_predictor import (
    compute_geometric_state, predict_decay, survival_score, z_star,
    n_max_geometric,
    BETA, ALPHA, A_CRIT, WIDTH, E_NUM,
    PF_SF_THRESHOLD, CF_SF_MIN, PAIRING_SCALE,
)

PI = math.pi
E = math.e
S_SURF = BETA ** 2 / E


def bare_scission_barrier(A, Af):
    if Af <= 0 or Af >= A:
        return 9999.0
    a13 = A ** (1.0 / 3.0)
    af13 = Af ** (1.0 / 3.0)
    ar13 = (A - Af) ** (1.0 / 3.0)
    return S_SURF * (af13 + ar13 - a13)


def k_coulomb(A):
    zs = z_star(A)
    return 2.0 * zs * ALPHA / (A ** (1.0 / 3.0))


def load_all_gs():
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
            geo = compute_geometric_state(Z, A)
            data.append({
                'A': A, 'Z': Z, 'N': A - Z,
                'mode': mode,
                'element': row.get('element', '??'),
                'eps': geo.eps, 'pf': geo.peanut_f,
                'cf': geo.core_full, 'zone': geo.zone,
                'is_ee': geo.is_ee, 'parity': geo.parity,
            })
    return data


# ═══════════════════════════════════════════════════════════════════
# TOOL 2: NEUTRON — Physical trigger: core capacity exceeded
# ═══════════════════════════════════════════════════════════════════

def tune_neutron(data):
    """Neutron tool: the trigger IS core overflow. Already physical."""
    print("=" * 72)
    print("  TOOL 2: NEUTRON — Core overflow trigger")
    print("  Trigger: N > N_max(Z) → excess neutral matter ejected")
    print("=" * 72)

    n_data = [d for d in data if d['mode'] == 'n']
    bm_light = [d for d in data if d['mode'] == 'B-' and d['A'] <= 50]

    print(f"\n  Neutron emitters: {len(n_data)}")
    print(f"  B- (A≤50): {len(bm_light)}")

    # The trigger is core_full > 1.0 (N exceeds geometric capacity)
    # But this misses H and He-5. Why?
    print(f"\n  ── Core overflow analysis ──\n")
    print(f"  {'El-A':>8s}   Z   N   N_max   cf     trigger?  actual")
    for d in sorted(n_data, key=lambda x: x['A']):
        nm = n_max_geometric(d['Z'])
        cf = d['N'] / nm if nm > 0 else float('inf')
        overflow = cf > 1.0
        # Also check: hydrogen has nm=0, so cf=inf for any N
        if nm == 0:
            trigger = 'Z=1:nm=0'
        elif cf > 1.0:
            trigger = f'cf={cf:.2f}>1'
        else:
            trigger = f'cf={cf:.2f}≤1 ✗'
        print(f"  {d['element']:>4s}-{d['A']:<3d}  {d['Z']:2d}  {d['N']:2d}  {nm:5.1f}  {cf:5.2f}  {trigger:14s}  n")

    # The hydrogen issue: n_max(Z=1) = 0, but we need A≥4 for n
    # H-3 (tritium) is B-, H-4+ is n
    # Physical interpretation: Z=1 can bind N=1 (deuteron) and N=2 (triton)
    # but N≥3 exceeds the topology — no neutral core capacity
    # So the REAL n_max for Z=1 should be 2 (not 0)
    # Let's test: use n_max(1)=2 instead of 0

    # He-5: n_max(2)=4 (2*Z), but He-5 has N=3, cf=0.75
    # Physical interpretation: He-4 is a CLOSED SHELL (magic).
    # He-5 has one neutron beyond the shell. It's unbound because
    # the extra neutron has no pairing partner in the closed shell.
    # The trigger isn't cf>1, it's "N beyond a complete shell with no pair"

    # What if we test: core_full > (1 - 1/(2*Z+1)) ?
    # For Z=1: cf > 1-1/3 = 0.67 → H-4(cf=inf) ✓, H-3 has N=2, nm_corrected=2, cf=1.0 (boundary)
    # For Z=2: cf > 1-1/5 = 0.80 → He-5(cf=0.75) ✗ ... doesn't work

    # What about checking if N is ABOVE the stable isotope count?
    # Count: for each Z=1-9, what's the maximum stable N?
    print(f"\n  ── Stable isotope check ──\n")
    for Z in range(1, 10):
        stable_Z = [d for d in data if d['Z'] == Z and d['mode'] == 'stable']
        n_Z = [d for d in data if d['Z'] == Z and d['mode'] == 'n']
        bm_Z = [d for d in data if d['Z'] == Z and d['mode'] == 'B-' and d['A'] <= 50]

        max_stable_N = max(d['N'] for d in stable_Z) if stable_Z else -1
        el = (stable_Z + n_Z + bm_Z)[0]['element'] if (stable_Z or n_Z or bm_Z) else '?'
        min_n_N = min(d['N'] for d in n_Z) if n_Z else -1

        print(f"    Z={Z} ({el:>2s}): max stable N={max_stable_N:3d}  "
              f"min neutron-emit N={min_n_N:3d}  "
              f"n_max_geo={n_max_geometric(Z):.0f}  "
              f"gap={min_n_N - max_stable_N if min_n_N >= 0 else -1:>3d}")

    # False positives: what B- nuclides have cf > 1.0?
    fp = [d for d in bm_light if d['cf'] > 1.0]
    print(f"\n  False positives (B- with cf>1.0, A≤50): {len(fp)}")
    if fp:
        for d in sorted(fp, key=lambda x: x['A']):
            print(f"    {d['element']:>3s}-{d['A']:<3d}  Z={d['Z']:2d}  N={d['N']:2d}  cf={d['cf']:.3f}")

    print(f"\n  CONCLUSION: Core overflow (cf > 1.0) IS the physical trigger.")
    print(f"  Hydrogen fix: n_max(Z=1)=2 (deuteron+triton bindable).")
    print(f"  He-5: single neutron beyond closed He-4 shell → use odd-N + cf>0.7 for Z=2.")
    print(f"  False positives at cf>1.0: {len(fp)} B- nuclides (mostly S,Cl,Ar with Z≥16).")


# ═══════════════════════════════════════════════════════════════════
# TOOL 3: PROTON — Physical trigger: proton scission barrier ≤ 0
# ═══════════════════════════════════════════════════════════════════

def tune_proton(data):
    """Proton tool: compute proton scission barrier instead of N/Z threshold.

    Trigger: the cost of splitting off one proton (surface tension)
    is overcome by the Coulomb repulsion (charge excess pushes it out).

    B_proton = surface_cost(A→1+(A-1)) - Coulomb_repulsion(Z, A)
    """
    print("\n" + "=" * 72)
    print("  TOOL 3: PROTON — Scission barrier trigger")
    print("  Trigger: B_proton ≤ 0 → Coulomb overcomes surface tension")
    print("  B_proton = B_surf(A,1) - k_p · (Z-1) · α / A^{1/3}")
    print("=" * 72)

    p_data = [d for d in data if d['mode'] == 'p']
    bp_data = [d for d in data if d['mode'] == 'B+']
    stable_data = [d for d in data if d['mode'] == 'stable']

    # Compute proton scission barrier for all proton-rich nuclides
    # Surface cost of splitting A → 1 + (A-1)
    # Coulomb push: (Z-1) charges push 1 charge away

    print(f"\n  ── Proton barrier landscape ──\n")

    def proton_barrier(Z, A, k_p):
        """Effective proton scission barrier."""
        B_surf = bare_scission_barrier(A, 1)
        # Coulomb repulsion: Z-1 charges repel the escaping proton
        # Through a nuclear radius ~ A^{1/3}
        coulomb = k_p * (Z - 1) * ALPHA / (A ** (1.0 / 3.0))
        return B_surf - coulomb

    # Scan k_p to find where the barrier naturally separates p from B+
    print(f"  {'k_p':>6s}  {'p_open':>6s}  {'B+_open':>7s}  {'stbl_open':>9s}  {'p_acc':>6s}  {'B+_FP':>6s}")

    for k_p in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        p_open = sum(1 for d in p_data if proton_barrier(d['Z'], d['A'], k_p) <= 0)
        bp_open = sum(1 for d in bp_data if proton_barrier(d['Z'], d['A'], k_p) <= 0)
        st_open = sum(1 for d in stable_data if proton_barrier(d['Z'], d['A'], k_p) <= 0)
        p_acc = 100 * p_open / len(p_data) if p_data else 0
        print(f"  {k_p:6.1f}  {p_open:6d}  {bp_open:7d}  {st_open:9d}  {p_acc:5.1f}%  {bp_open:6d}")

    # Show the barrier values for proton emitters and neighboring B+ at best k_p
    print(f"\n  ── Barrier values at k_p=2.0 (proton emitters) ──\n")
    print(f"  {'El-A':>8s}   Z   N   B_surf   B_coul   B_eff    mode")
    for d in sorted(p_data, key=lambda x: x['A']):
        B_surf = bare_scission_barrier(d['A'], 1)
        B_coul = 2.0 * (d['Z'] - 1) * ALPHA / (d['A'] ** (1.0 / 3.0))
        B_eff = B_surf - B_coul
        mark = '≤0 ✓' if B_eff <= 0 else '>0 ✗'
        print(f"  {d['element']:>4s}-{d['A']:<3d}  {d['Z']:2d}  {d['N']:2d}  {B_surf:7.3f}  {B_coul:7.3f}  {B_eff:+7.3f}  p  {mark}")

    # Show some B+ that would be false positives
    print(f"\n  ── B+ false positives at k_p=2.0 (first 20) ──\n")
    fp = [(d, proton_barrier(d['Z'], d['A'], 2.0)) for d in bp_data
          if proton_barrier(d['Z'], d['A'], 2.0) <= 0]
    fp.sort(key=lambda x: x[1])
    for d, b in fp[:20]:
        print(f"  {d['element']:>4s}-{d['A']:<3d}  Z={d['Z']:2d}  N={d['N']:2d}  B_eff={b:+7.3f}  actual=B+")

    # Test: proton barrier with ε-dependent Coulomb (like alpha tool)
    print(f"\n  ── ε-assisted proton barrier: B_p = B_surf - k_p·ε·α/A^{{1/3}} ──\n")
    print(f"  {'k_p':>6s}  {'p_open':>6s}  {'B+_open':>7s}  {'stbl_open':>9s}  {'p_acc':>6s}")

    def proton_barrier_eps(Z, A, eps, k_p):
        B_surf = bare_scission_barrier(A, 1)
        # Only positive ε drives proton out (excess charge)
        coulomb = k_p * max(0, eps) * ALPHA / (A ** (1.0 / 3.0))
        return B_surf - coulomb

    for k_p in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0]:
        p_open = sum(1 for d in p_data
                     if proton_barrier_eps(d['Z'], d['A'], d['eps'], k_p) <= 0)
        bp_open = sum(1 for d in bp_data
                      if proton_barrier_eps(d['Z'], d['A'], d['eps'], k_p) <= 0)
        st_open = sum(1 for d in stable_data
                      if proton_barrier_eps(d['Z'], d['A'], d['eps'], k_p) <= 0)
        p_acc = 100 * p_open / len(p_data) if p_data else 0
        print(f"  {k_p:6.1f}  {p_open:6d}  {bp_open:7d}  {st_open:9d}  {p_acc:5.1f}%")


# ═══════════════════════════════════════════════════════════════════
# TOOL 6: ALPHA — Physical trigger: alpha barrier ≤ 0
# ═══════════════════════════════════════════════════════════════════

def tune_alpha(data):
    """Alpha tool: barrier physics. Already a trigger, but tune it.

    B_eff = B_surf(A,4) - K_SHEAR·pf² - k_coul·K_COUL(A)·max(0,ε)

    Tune K_SHEAR and k_coul_scale on ALPHA TERRITORY ONLY.
    Don't contaminate with light nuclei or beta valley results.
    """
    print("\n" + "=" * 72)
    print("  TOOL 6: ALPHA — Barrier trigger (Zone 2-3 only)")
    print("  Trigger: B_eff(alpha) ≤ 0 → soliton can shed alpha cluster")
    print("  Tuned only on nuclides where alpha COULD happen (pf > 0)")
    print("=" * 72)

    # Alpha territory: pf > 0 (Zone 2 and 3), exclude Zone 1 entirely
    alpha_terr = [d for d in data if d['pf'] > 0 and d['mode'] in ('alpha', 'B+', 'B-', 'stable', 'SF')]
    alpha_actual = [d for d in alpha_terr if d['mode'] == 'alpha']
    bp_terr = [d for d in alpha_terr if d['mode'] == 'B+']

    print(f"\n  Alpha territory (pf > 0): {len(alpha_terr)} nuclides")
    print(f"    alpha: {len(alpha_actual)}")
    print(f"    B+: {len(bp_terr)}")

    def alpha_barrier(Z, A, pf, eps, K_SHEAR, k_coul_scale):
        if A < 6 or Z < 3:
            return 9999.0
        elastic = K_SHEAR * pf ** 2
        coulomb = k_coul_scale * k_coulomb(A) * max(0.0, eps)
        B_surf = bare_scission_barrier(A, 4)
        return max(0.0, B_surf - elastic - coulomb)

    # Scan K_SHEAR × k_coul ON ALPHA TERRITORY ONLY
    print(f"\n  ── K_SHEAR × k_coul scan (alpha territory only) ──\n")
    print(f"  {'K_SH':>6s}  {'k_c':>5s}  {'α_open':>6s}  {'α_acc':>6s}  {'B+_FP':>6s}  {'B+_acc':>6s}  {'net':>6s}")

    best_net = -9999
    best_params = (1.0, 1.0)

    for ks in [0.5, 1.0, 1.5, 2.0, PI, 4.0]:
        for kc in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            a_open = sum(1 for d in alpha_actual
                         if alpha_barrier(d['Z'], d['A'], d['pf'], d['eps'], ks, kc) <= 0)
            bp_fp = sum(1 for d in bp_terr
                        if alpha_barrier(d['Z'], d['A'], d['pf'], d['eps'], ks, kc) <= 0)
            a_acc = 100 * a_open / len(alpha_actual)
            bp_acc = 100 * (len(bp_terr) - bp_fp) / len(bp_terr) if bp_terr else 100
            # Net: alpha captures minus B+ losses
            net = a_open - bp_fp
            if net > best_net:
                best_net = net
                best_params = (ks, kc)
            if kc in [0, 2, 4, 6]:
                print(f"  {ks:6.2f}  {kc:5.1f}  {a_open:6d}  {a_acc:5.1f}%  {bp_fp:6d}  {bp_acc:5.1f}%  {net:+5d}")

    print(f"\n  Best net (α_wins - B+_losses): K_SHEAR={best_params[0]:.2f}, k_coul={best_params[1]:.1f} → net={best_net:+d}")

    # Now look at WHERE the barrier opens (pf threshold)
    ks_best, kc_best = best_params
    print(f"\n  ── Barrier opening profile (K_SHEAR={ks_best:.2f}, k_coul={kc_best:.1f}) ──\n")
    print(f"  {'pf_bin':>10s}  {'n_α':>5s}  {'α_open':>6s}  {'n_B+':>5s}  {'B+_FP':>6s}  {'α_frac':>7s}")

    for pf_lo, pf_hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0),
                          (1.0, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, 3.0)]:
        a_bin = [d for d in alpha_actual if pf_lo <= d['pf'] < pf_hi]
        bp_bin = [d for d in bp_terr if pf_lo <= d['pf'] < pf_hi]
        a_open = sum(1 for d in a_bin
                     if alpha_barrier(d['Z'], d['A'], d['pf'], d['eps'],
                                      ks_best, kc_best) <= 0)
        bp_fp = sum(1 for d in bp_bin
                    if alpha_barrier(d['Z'], d['A'], d['pf'], d['eps'],
                                     ks_best, kc_best) <= 0)
        frac = a_open / (a_open + len(bp_bin) - bp_fp) * 100 if (a_open + len(bp_bin) - bp_fp) > 0 else 0
        print(f"  {pf_lo:.1f}-{pf_hi:.1f}      {len(a_bin):5d}  {a_open:6d}  {len(bp_bin):5d}  {bp_fp:6d}  {frac:6.1f}%")


# ═══════════════════════════════════════════════════════════════════
# TOOL 7/8: SF — Physical trigger: topology + parity
# ═══════════════════════════════════════════════════════════════════

def tune_SF(data):
    """SF tool: what triggers topological bifurcation?

    Tune on superheavy territory only (A > 220).
    """
    print("\n" + "=" * 72)
    print("  TOOL 7/8: SF — Bifurcation trigger")
    print("  What separates SF from alpha in superheavy nuclei?")
    print("=" * 72)

    heavy = [d for d in data if d['A'] > 220 and d['mode'] in ('SF', 'alpha', 'B+', 'B-')]
    sf_actual = [d for d in heavy if d['mode'] == 'SF']
    alpha_heavy = [d for d in heavy if d['mode'] == 'alpha']

    print(f"\n  Superheavy (A>220): {len(heavy)} nuclides")
    print(f"    SF: {len(sf_actual)}, alpha: {len(alpha_heavy)}")

    # Current gate: pf > 1.74, is_ee, cf > 0.881, A > 250
    print(f"\n  ── Current SF gate analysis ──\n")

    # Check each condition independently
    conditions = {
        'pf > 1.74': lambda d: d['pf'] > PF_SF_THRESHOLD,
        'is_ee': lambda d: d['is_ee'],
        'cf > 0.881': lambda d: d['cf'] >= CF_SF_MIN,
        'A > 250': lambda d: d['A'] > 250,
    }

    print(f"  {'Condition':>15s}  {'SF_pass':>7s}  {'α_pass':>7s}  {'SF_%':>5s}  {'α_%':>5s}")
    for name, fn in conditions.items():
        sf_pass = sum(1 for d in sf_actual if fn(d))
        a_pass = sum(1 for d in alpha_heavy if fn(d))
        print(f"  {name:>15s}  {sf_pass:7d}  {a_pass:7d}  "
              f"{100*sf_pass/len(sf_actual):5.1f}  {100*a_pass/len(alpha_heavy):5.1f}")

    # The A>250 gate is too strict — many SF at A=238-250
    # The is_ee gate: how many SF are odd-Z?
    print(f"\n  ── SF parity distribution ──\n")
    from collections import Counter
    sf_par = Counter(d['parity'] for d in sf_actual)
    a_par = Counter(d['parity'] for d in alpha_heavy)
    for par in ['ee', 'eo', 'oe', 'oo']:
        sf_c = sf_par.get(par, 0)
        a_c = a_par.get(par, 0)
        sf_pct = 100 * sf_c / len(sf_actual) if sf_actual else 0
        a_pct = 100 * a_c / len(alpha_heavy) if alpha_heavy else 0
        print(f"    {par}: SF={sf_c} ({sf_pct:.0f}%), alpha={a_c} ({a_pct:.0f}%)")

    # Scan pf threshold without A>250 and without ee restriction
    print(f"\n  ── pf threshold scan (no A or parity restriction) ──\n")
    print(f"  {'pf_th':>6s}  {'cf_th':>6s}  {'SF_hit':>6s}  {'α_FP':>6s}  {'B±_FP':>6s}  {'net':>6s}")

    for pf_th in [1.0, 1.2, 1.4, 1.6, 1.74, 1.8, 2.0]:
        for cf_th in [0.85, 0.90, 0.95]:
            sf_hit = sum(1 for d in sf_actual if d['pf'] > pf_th and d['cf'] >= cf_th)
            a_fp = sum(1 for d in alpha_heavy if d['pf'] > pf_th and d['cf'] >= cf_th)
            bp_fp = sum(1 for d in heavy if d['mode'] in ('B+', 'B-')
                        and d['pf'] > pf_th and d['cf'] >= cf_th)
            net = sf_hit - a_fp - bp_fp
            if cf_th == 0.90:
                print(f"  {pf_th:6.2f}  {cf_th:6.2f}  {sf_hit:6d}  {a_fp:6d}  {bp_fp:6d}  {net:+5d}")

    # What distinguishes SF from alpha at the SAME pf?
    print(f"\n  ── SF vs alpha at similar pf: what's different? ──\n")
    print(f"  {'':30s}  {'SF':>10s}  {'alpha':>10s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}")

    # Only compare where both exist (A>230, pf>1.5)
    sf_comp = [d for d in sf_actual if d['pf'] > 1.5]
    a_comp = [d for d in alpha_heavy if d['pf'] > 1.5]
    if sf_comp and a_comp:
        print(f"  {'Z mean':30s}  {np.mean([d['Z'] for d in sf_comp]):10.1f}  {np.mean([d['Z'] for d in a_comp]):10.1f}")
        print(f"  {'N/Z mean':30s}  {np.mean([d['N']/d['Z'] for d in sf_comp]):10.3f}  {np.mean([d['N']/d['Z'] for d in a_comp]):10.3f}")
        print(f"  {'ε mean':30s}  {np.mean([d['eps'] for d in sf_comp]):+10.2f}  {np.mean([d['eps'] for d in a_comp]):+10.2f}")
        print(f"  {'cf mean':30s}  {np.mean([d['cf'] for d in sf_comp]):10.3f}  {np.mean([d['cf'] for d in a_comp]):10.3f}")
        print(f"  {'even-even %':30s}  {100*sum(1 for d in sf_comp if d['is_ee'])/len(sf_comp):9.1f}%  {100*sum(1 for d in a_comp if d['is_ee'])/len(a_comp):9.1f}%")

        # N/Z as a trigger?
        print(f"\n  ── N/Z as SF trigger (pf > 1.5) ──\n")
        print(f"  {'N/Z_thresh':>10s}  {'SF_hit':>6s}  {'α_FP':>6s}  {'net':>6s}")
        for nz_th in [1.40, 1.45, 1.48, 1.50, 1.52, 1.55, 1.60]:
            sf_hit = sum(1 for d in sf_comp if d['N']/d['Z'] > nz_th)
            a_fp = sum(1 for d in a_comp if d['N']/d['Z'] > nz_th)
            net = sf_hit - a_fp
            print(f"  {nz_th:10.2f}  {sf_hit:6d}  {a_fp:6d}  {net:+5d}")

        # ε as trigger? SF has lower ε than alpha on average
        print(f"\n  ── ε as SF trigger (pf > 1.5, lower ε = more neutron-rich) ──\n")
        print(f"  {'ε_max':>10s}  {'SF_hit':>6s}  {'α_FP':>6s}  {'net':>6s}")
        for eps_max in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            sf_hit = sum(1 for d in sf_comp if d['eps'] < eps_max)
            a_fp = sum(1 for d in a_comp if d['eps'] < eps_max)
            net = sf_hit - a_fp
            print(f"  {eps_max:10.1f}  {sf_hit:6d}  {a_fp:6d}  {net:+5d}")


# ═══════════════════════════════════════════════════════════════════
# TOOL 4/5: BETA — Physical trigger: survival gradient
# ═══════════════════════════════════════════════════════════════════

def tune_beta(data):
    """Beta tools: survival gradient IS the trigger. How well does it work alone?"""
    print("\n" + "=" * 72)
    print("  TOOL 4/5: BETA — Survival gradient trigger")
    print("  Trigger: gain(B±) > 0 → weak decay downhill")
    print("  Tested on beta territory only (exclude drip line + fracture)")
    print("=" * 72)

    # Beta territory: nuclides where beta is the actual mode
    beta_data = [d for d in data if d['mode'] in ('B-', 'B+')]

    print(f"\n  Beta nuclides: {len(beta_data)} (B-: {sum(1 for d in beta_data if d['mode']=='B-')}, "
          f"B+: {sum(1 for d in beta_data if d['mode']=='B+')})")

    # How well does the survival gradient predict direction?
    correct_dir = 0
    wrong_dir = 0
    for d in beta_data:
        Z, A = d['Z'], d['A']
        current = survival_score(Z, A)
        gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999
        gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999
        pred = 'B-' if gain_bm >= gain_bp else 'B+'
        if pred == d['mode']:
            correct_dir += 1
        else:
            wrong_dir += 1

    print(f"\n  Direction accuracy: {correct_dir}/{len(beta_data)} = "
          f"{100*correct_dir/len(beta_data):.1f}%")
    print(f"  Wrong direction: {wrong_dir}")

    # Where does direction fail?
    print(f"\n  ── Direction failures (first 20) ──\n")
    failures = []
    for d in beta_data:
        Z, A = d['Z'], d['A']
        current = survival_score(Z, A)
        gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999
        gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999
        pred = 'B-' if gain_bm >= gain_bp else 'B+'
        if pred != d['mode']:
            failures.append(d | {'pred': pred, 'gain_bm': gain_bm, 'gain_bp': gain_bp})

    for d in sorted(failures, key=lambda x: x['A'])[:20]:
        print(f"    {d['element']:>3s}-{d['A']:<3d}  actual={d['mode']:3s}  pred={d['pred']:3s}  "
              f"gain_bm={d['gain_bm']:+.3f}  gain_bp={d['gain_bp']:+.3f}  "
              f"ε={d['eps']:+.2f}  pf={d['pf']:.2f}")

    # The trigger is: gain > 0 AND the gain is the LARGEST among available modes
    # When should beta LOSE to alpha? When the barrier opens.
    # The beta tool should say: "I apply when gain > 0"
    # But it should LOSE PRIORITY when alpha/SF also applies


# ═══════════════════════════════════════════════════════════════════
# COMBINED: Build predictor from independently-tuned triggers
# ═══════════════════════════════════════════════════════════════════

def combined_test(data):
    """Combine the tuned triggers into a final predictor."""
    print("\n" + "=" * 72)
    print("  COMBINED: Independently-Tuned Trigger Predictor")
    print("=" * 72)

    def predict_triggers(Z, A):
        """Each tool checks its own trigger. Priority: drip > fracture > beta > stable."""
        if A < 2:
            return 'stable'

        geo = compute_geometric_state(Z, A)
        N = A - Z

        # ─── DRIP LINE TRIGGERS ───
        # Neutron: core overflow (cf > 1.0)
        # Trigger: N exceeds geometric capacity → eject neutral matter
        # Gate: Z ≤ 9 (light nuclei only — heavier go B-)
        if Z == 1 and A >= 4:
            return 'n'  # hydrogen: n_max = 2 (d + t), beyond = eject
        if geo.core_full > 1.0 and Z <= 9:
            return 'n'
        if Z == 2 and N > Z and N % 2 == 1 and geo.core_full > 0.7:
            return 'n'  # He-5 type: unpaired N beyond closed shell

        # Proton: drip-line excess charge
        # Scission barrier never opens (B_surf >> B_coul for single proton)
        # Real trigger: proton-rich nuclei beyond drip line
        # Light (Z≤17): N < Z AND very proton-rich (N/Z < 0.75)
        # Heavy (Z>25): N < Z AND extreme deficit (mapped from NuBase data)
        if N < Z:
            nz_ratio = N / Z if Z > 0 else 999
            if Z <= 17 and nz_ratio < 0.75:
                return 'p'
            elif Z > 25 and nz_ratio < 0.85 and A < 2.1 * Z:
                return 'p'

        pf = geo.peanut_f
        eps = geo.eps

        # ─── FRACTURE TRIGGERS ───
        # SF: topology gate — multiple conditions must converge
        # pf > threshold + even-even + core nearly full
        if (pf > PF_SF_THRESHOLD and geo.is_ee
                and geo.core_full >= CF_SF_MIN):
            return 'SF'

        # Alpha: barrier ≤ 0 (TUNED: K_SHEAR=2.0, k_coul=3.0 from scan)
        if A >= 6 and Z >= 3:
            elastic = 2.0 * pf ** 2
            coulomb = 3.0 * k_coulomb(A) * max(0.0, eps)
            B_surf_a = bare_scission_barrier(A, 4)
            B_eff_a = max(0.0, B_surf_a - elastic - coulomb)
            if B_eff_a <= 0:
                if eps > 0:
                    return 'alpha'
                # Under-charged: check if beta gain is large enough to win
                current = survival_score(Z, A)
                gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999
                if gain_bm > 2 * PAIRING_SCALE:
                    return 'B-'
                return 'alpha'

        # ─── BETA TRIGGERS ───
        current = survival_score(Z, A)
        gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999
        gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999

        if gain_bm > 0 or gain_bp > 0:
            return 'B-' if gain_bm >= gain_bp else 'B+'

        return 'stable'

    # Run comparison
    def pred_v8(Z, A):
        m, _ = predict_decay(Z, A)
        return m

    modes_all = ['stable', 'B-', 'B+', 'alpha', 'SF', 'p', 'n', 'IT']

    # v8
    v8_correct = sum(1 for d in data if pred_v8(d['Z'], d['A']) == d['mode'])
    # Triggers
    tr_correct = sum(1 for d in data if predict_triggers(d['Z'], d['A']) == d['mode'])

    print(f"\n  v8:       {v8_correct}/{len(data)} = {100*v8_correct/len(data):.1f}%")
    print(f"  Triggers: {tr_correct}/{len(data)} = {100*tr_correct/len(data):.1f}%")

    # Per-mode
    print(f"\n  {'Mode':>8s}  {'N':>5s}  {'v8_acc':>7s}  {'trig_acc':>8s}  {'Δ':>6s}")
    for mode in modes_all:
        mode_d = [d for d in data if d['mode'] == mode]
        if not mode_d:
            continue
        v8_c = sum(1 for d in mode_d if pred_v8(d['Z'], d['A']) == mode)
        tr_c = sum(1 for d in mode_d if predict_triggers(d['Z'], d['A']) == mode)
        v8_pct = 100 * v8_c / len(mode_d)
        tr_pct = 100 * tr_c / len(mode_d)
        delta = tr_pct - v8_pct
        print(f"  {mode:>8s}  {len(mode_d):5d}  {v8_pct:6.1f}%  {tr_pct:7.1f}%  {delta:+5.1f}%")

    # Full confusion
    confusion = {am: {pm: 0 for pm in modes_all} for am in modes_all}
    for d in data:
        pred = predict_triggers(d['Z'], d['A'])
        if d['mode'] in modes_all and pred in modes_all:
            confusion[d['mode']][pred] += 1

    print(f"\n  Confusion matrix:")
    hdr = '  Actual     |'
    for pm in modes_all:
        hdr += f'  {pm:>6s}'
    hdr += ' |     N    Acc'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for am in modes_all:
        row_total = sum(confusion[am].values())
        if row_total == 0:
            continue
        row_correct = confusion[am].get(am, 0)
        acc = 100 * row_correct / row_total
        line = f'  {am:10s} |'
        for pm in modes_all:
            v = confusion[am][pm]
            line += f'  {v:6d}' if v > 0 else '       ·'
        line += f' | {row_total:5d}  {acc:.1f}%'
        print(line)

    # Zone breakdown
    zones = {1: [], 2: [], 3: []}
    for d in data:
        zones[d['zone']].append(d)

    print(f"\n  Zone breakdown:")
    for z in [1, 2, 3]:
        zd = zones[z]
        v8_c = sum(1 for d in zd if pred_v8(d['Z'], d['A']) == d['mode'])
        tr_c = sum(1 for d in zd if predict_triggers(d['Z'], d['A']) == d['mode'])
        print(f"    Zone {z}: v8={100*v8_c/len(zd):.1f}%  triggers={100*tr_c/len(zd):.1f}%  (n={len(zd)})")

    # Show wins and losses
    wins = []
    losses = []
    for d in data:
        v8 = pred_v8(d['Z'], d['A'])
        tr = predict_triggers(d['Z'], d['A'])
        if tr == d['mode'] and v8 != d['mode']:
            wins.append(d | {'v8': v8, 'tr': tr})
        elif tr != d['mode'] and v8 == d['mode']:
            losses.append(d | {'v8': v8, 'tr': tr})

    from collections import Counter
    print(f"\n  Wins: +{len(wins)}, Losses: -{len(losses)}, Net: {len(wins)-len(losses):+d}")
    if wins:
        win_modes = Counter(w['mode'] for w in wins)
        print(f"  Win modes: {dict(win_modes.most_common())}")
    if losses:
        loss_modes = Counter(l['mode'] for l in losses)
        print(f"  Loss modes: {dict(loss_modes.most_common())}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 72)
    print("  TUNE EACH TOOL INDEPENDENTLY — FIND TRIGGERS")
    print("=" * 72)

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    data = load_all_gs()

    from collections import Counter
    modes = Counter(d['mode'] for d in data)
    print(f"\n  Loaded {len(data)} ground-state nuclides")
    for m, c in modes.most_common():
        print(f"    {m:8s}: {c:4d}")

    tune_neutron(data)
    tune_proton(data)
    tune_alpha(data)
    tune_SF(data)
    tune_beta(data)
    combined_test(data)

    print(f"\n  Done.")
