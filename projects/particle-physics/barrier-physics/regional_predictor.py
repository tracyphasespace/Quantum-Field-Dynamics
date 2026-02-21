#!/usr/bin/env python3
"""
Regional Decay Predictor — 8 Independent Physics Tools

Architecture: each tool answers two questions independently:
  1. Does my physics APPLY to this nuclide? (yes/no gate)
  2. If yes, how STRONG is the drive? (scalar)

The tools:
  1. STABLE    — valley floor, no channels open
  2. NEUTRON   — neutral core overflow (drip line)
  3. PROTON    — charged shell overflow (drip line)
  4. BETA+     — weak decay, proton-rich side
  5. BETA-     — weak decay, neutron-rich side
  6. ALPHA     — soliton shedding (barrier open, no SF)
  7. SF        — topological bifurcation (no alpha competition)
  8. ALPHA+SF  — competition zone (both channels open)

A tritium atom doesn't see the same forces a uranium atom does.
The neutron tool doesn't know about pf or barriers.
The SF tool doesn't know about N/Z ratios.
Each tool uses only the physics relevant to its territory.
"""

import csv
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from qfd_nuclide_predictor import (
    compute_geometric_state, predict_decay, survival_score, z_star,
    BETA, ALPHA, A_CRIT, WIDTH, E_NUM,
    PF_ALPHA_POSSIBLE, PF_PEANUT_ONLY, PF_DEEP_PEANUT,
    PF_SF_THRESHOLD, CF_SF_MIN, PAIRING_SCALE,
    n_max_geometric,
)

PI = math.pi
E = math.e
S_SURF = BETA ** 2 / E


# ═══════════════════════════════════════════════════════════════════
# SHARED GEOMETRY (used by multiple tools)
# ═══════════════════════════════════════════════════════════════════

def bare_scission_barrier(A, Af):
    a13 = A ** (1.0 / 3.0)
    af13 = Af ** (1.0 / 3.0)
    ar13 = (A - Af) ** (1.0 / 3.0) if A > Af else 0.0
    return S_SURF * (af13 + ar13 - a13)


def k_coulomb(A):
    zs = z_star(A)
    return 2.0 * zs * ALPHA / (A ** (1.0 / 3.0))


# ═══════════════════════════════════════════════════════════════════
# TOOL 1: STABLE — Valley floor, no open channels
# ═══════════════════════════════════════════════════════════════════

class StableTool:
    """Stability = default. Applies when nothing else claims the nuclide.

    The stable tool doesn't compete — it's what's left when all other
    tools say 'no'. A nuclide near the valley floor with no drip-line
    issues and no open fracture channels is stable.
    """
    mode = 'stable'

    def applies(self, Z, A, geo):
        # Stable is the default — always "available" as a fallback
        return True

    def strength(self, Z, A, geo):
        # Inverse of displacement: closer to valley → more stable
        return 1.0 / (1.0 + abs(geo.eps))


# ═══════════════════════════════════════════════════════════════════
# TOOL 2: NEUTRON — Neutral core overflow
# ═══════════════════════════════════════════════════════════════════

class NeutronTool:
    """Neutron emission: the soliton's neutral core can't hold.

    Physics: for light nuclei, there's a maximum N the topology can
    bind. Beyond this, the excess neutral matter is ejected directly
    (faster than beta decay can convert it).

    Territory: A < 50, Z < 15. Doesn't exist in heavy nuclei.
    Metrics: core_full, N/Z, Z (NOT pf, NOT barriers, NOT survival score)
    """
    mode = 'n'

    def applies(self, Z, A, geo):
        # Gate: only light, neutron-rich nuclei
        if A > 50 or Z > 15:
            return False
        N = A - Z
        if N <= Z:
            return False  # not neutron-rich

        # Hydrogen special case: beyond tritium, can't hold
        if Z == 1 and A >= 4:
            return True

        # General: core overflow
        if geo.core_full > 1.0:
            return True

        # He-5 type: single neutron beyond closed shell
        # He-4 (alpha) is maximally stable; He-5 is immediately unbound
        if Z == 2 and N > Z and N % 2 == 1 and geo.core_full > 0.7:
            return True

        return False

    def strength(self, Z, A, geo):
        # How far past the drip line
        if Z == 1:
            return 10.0  # hydrogen n-emitters are instantaneous
        return max(0, geo.core_full - 0.7)


# ═══════════════════════════════════════════════════════════════════
# TOOL 3: PROTON — Charged shell overflow
# ═══════════════════════════════════════════════════════════════════

class ProtonTool:
    """Proton emission: the soliton's charged shell can't hold.

    Physics: when the proton excess is so extreme that the weak
    interaction (B+) is too slow — the Coulomb repulsion directly
    ejects a proton from the surface.

    Two territories with different metrics:
      Light (A < 30): N/Z is the key — too few neutrons to "glue"
      Heavy (A > 30): extreme ε + under-filled core

    Does NOT use: pf, barriers, survival score
    """
    mode = 'p'

    def __init__(self, nz_light=0.85, eps_heavy=3.0, cf_heavy=0.55):
        self.nz_light = nz_light
        self.eps_heavy = eps_heavy
        self.cf_heavy = cf_heavy

    def applies(self, Z, A, geo):
        N = A - Z
        if N >= Z:
            return False  # not proton-rich enough

        # Light territory: N/Z below threshold
        if A <= 30 and Z <= 20:
            nz = N / Z if Z > 0 else 999
            if nz < self.nz_light:
                return True

        # Heavy territory: extreme ε + under-filled core
        if A < 200 and geo.eps > self.eps_heavy and geo.core_full < self.cf_heavy:
            return True

        return False

    def strength(self, Z, A, geo):
        N = A - Z
        nz = N / Z if Z > 0 else 999
        if A <= 30:
            return (1.0 - nz)  # further from N=Z → stronger drive
        return geo.eps / 10.0  # extreme ε → stronger drive


# ═══════════════════════════════════════════════════════════════════
# TOOL 4: BETA+ — Weak decay, proton-rich side
# ═══════════════════════════════════════════════════════════════════

class BetaPlusTool:
    """Beta-plus / EC: geodesic glide toward the valley center.

    Physics: the soliton has excess charge (ε > 0) and can reduce it
    via the weak interaction. The survival score gradient points
    toward fewer protons.

    Metrics: ε (sign), survival score gradient
    Does NOT use: pf, barriers, core_full
    """
    mode = 'B+'

    def applies(self, Z, A, geo):
        if Z < 2:
            return False
        # Must have a downhill gradient toward B+
        current = survival_score(Z, A)
        gain = survival_score(Z - 1, A) - current
        return gain > 0

    def strength(self, Z, A, geo):
        current = survival_score(Z, A)
        gain = survival_score(Z - 1, A) - current
        return max(0, gain)


# ═══════════════════════════════════════════════════════════════════
# TOOL 5: BETA- — Weak decay, neutron-rich side
# ═══════════════════════════════════════════════════════════════════

class BetaMinusTool:
    """Beta-minus: geodesic glide toward the valley center.

    Physics: the soliton has excess neutral matter (ε < 0) and can
    increase its charge via the weak interaction.

    Metrics: ε (sign), survival score gradient
    Does NOT use: pf, barriers, core_full
    """
    mode = 'B-'

    def applies(self, Z, A, geo):
        if Z + 1 > A:
            return False
        current = survival_score(Z, A)
        gain = survival_score(Z + 1, A) - current
        return gain > 0

    def strength(self, Z, A, geo):
        current = survival_score(Z, A)
        gain = survival_score(Z + 1, A) - current
        return max(0, gain)


# ═══════════════════════════════════════════════════════════════════
# TOOL 6: ALPHA-ONLY — Soliton shedding, no SF competition
# ═══════════════════════════════════════════════════════════════════

class AlphaOnlyTool:
    """Alpha decay without SF competition.

    Physics: the peanut deformation opens a scission barrier,
    allowing the soliton to shed an alpha particle. But the nucleus
    is not heavy enough or deformed enough for SF.

    Territory: Zone 2-3, A < ~250 or pf < SF threshold
    Metrics: pf, ε, Coulomb barrier (NOT core_full, NOT N/Z)
    """
    mode = 'alpha'

    def __init__(self, K_SHEAR=PI, k_coul_scale=4.0):
        self.K_SHEAR = K_SHEAR
        self.k_coul_scale = k_coul_scale

    def _sf_possible(self, geo):
        """Is SF even in the picture for this nuclide?"""
        return (geo.peanut_f > PF_SF_THRESHOLD
                and geo.core_full >= CF_SF_MIN
                and geo.A > 230)

    def _barrier(self, Z, A, geo):
        """Compute effective alpha barrier."""
        if A < 6 or Z < 3:
            return 9999.0
        pf = geo.peanut_f
        eps = geo.eps
        elastic = self.K_SHEAR * pf ** 2
        coulomb = self.k_coul_scale * k_coulomb(A) * max(0.0, eps)
        B_surf = bare_scission_barrier(A, 4)
        return max(0.0, B_surf - elastic - coulomb)

    def applies(self, Z, A, geo):
        # Not applicable if SF is in the picture
        if self._sf_possible(geo):
            return False
        # Alpha barrier must be open
        return self._barrier(Z, A, geo) <= 0.0

    def strength(self, Z, A, geo):
        # How far below the barrier (more negative = stronger drive)
        B_eff = self._barrier(Z, A, geo)
        return max(0, -B_eff)


# ═══════════════════════════════════════════════════════════════════
# TOOL 7: SF-ONLY — Topological bifurcation, no alpha competition
# ═══════════════════════════════════════════════════════════════════

class SFOnlyTool:
    """Spontaneous fission without alpha competition.

    Physics: the peanut is so deeply deformed that the neck snaps.
    The soliton is even-even (paired configuration allows symmetric
    bifurcation). The core is near saturation.

    Territory: A > 250, extreme pf, even-even
    Metrics: pf, core_full, parity (NOT ε, NOT barriers, NOT N/Z)
    """
    mode = 'SF'

    def __init__(self, K_SHEAR=PI, k_coul_scale=4.0):
        self.K_SHEAR = K_SHEAR
        self.k_coul_scale = k_coul_scale

    def _alpha_barrier(self, Z, A, geo):
        if A < 6 or Z < 3:
            return 9999.0
        pf = geo.peanut_f
        eps = geo.eps
        elastic = self.K_SHEAR * pf ** 2
        coulomb = self.k_coul_scale * k_coulomb(A) * max(0.0, eps)
        B_surf = bare_scission_barrier(A, 4)
        return max(0.0, B_surf - elastic - coulomb)

    def _alpha_open(self, Z, A, geo):
        return self._alpha_barrier(Z, A, geo) <= 0.0

    def applies(self, Z, A, geo):
        # Must meet SF topology conditions
        if not (geo.peanut_f > PF_SF_THRESHOLD
                and geo.is_ee
                and geo.core_full >= CF_SF_MIN):
            return False
        # SF-ONLY: alpha barrier must still be closed
        if self._alpha_open(Z, A, geo):
            return False
        return True

    def strength(self, Z, A, geo):
        return geo.peanut_f - PF_SF_THRESHOLD


# ═══════════════════════════════════════════════════════════════════
# TOOL 8: ALPHA+SF — Competition zone
# ═══════════════════════════════════════════════════════════════════

class AlphaSFTool:
    """Alpha and SF both physically possible — resolve the competition.

    Physics: in the superheavy region (Z~98-114), both channels are
    open simultaneously. The decision depends on:
      - Even-even strongly favors SF (pairing allows symmetric split)
      - Higher pf favors SF (deeper deformation)
      - Higher ε favors alpha (Coulomb push helps shedding)

    Territory: A > 230, SF conditions met, alpha barrier open
    This tool returns EITHER 'alpha' or 'SF' depending on competition.
    """
    mode = 'alpha'  # default, overridden by resolve()

    def __init__(self, K_SHEAR=PI, k_coul_scale=4.0):
        self.K_SHEAR = K_SHEAR
        self.k_coul_scale = k_coul_scale

    def _alpha_barrier(self, Z, A, geo):
        if A < 6 or Z < 3:
            return 9999.0
        pf = geo.peanut_f
        eps = geo.eps
        elastic = self.K_SHEAR * pf ** 2
        coulomb = self.k_coul_scale * k_coulomb(A) * max(0.0, eps)
        B_surf = bare_scission_barrier(A, 4)
        return max(0.0, B_surf - elastic - coulomb)

    def applies(self, Z, A, geo):
        # Both SF conditions AND alpha barrier open
        sf_ok = (geo.peanut_f > PF_SF_THRESHOLD
                 and geo.is_ee
                 and geo.core_full >= CF_SF_MIN)
        alpha_ok = self._alpha_barrier(Z, A, geo) <= 0.0
        return sf_ok and alpha_ok

    def resolve(self, Z, A, geo):
        """Decide alpha vs SF in the competition zone."""
        # SF wins when deformation is extreme and even-even
        # Use v8's A > 250 gate for now
        if A > 250:
            return 'SF'
        return 'alpha'

    def strength(self, Z, A, geo):
        return geo.peanut_f  # proxy


# ═══════════════════════════════════════════════════════════════════
# DISPATCHER — Run all tools, pick the winner
# ═══════════════════════════════════════════════════════════════════

def make_predictor(tools=None, **kwargs):
    """Build a predict function from a set of tools."""

    if tools is None:
        nz_light = kwargs.get('nz_light', 0.85)
        eps_heavy = kwargs.get('eps_heavy', 3.0)
        cf_heavy = kwargs.get('cf_heavy', 0.55)
        K_SHEAR = kwargs.get('K_SHEAR', PI)
        k_coul_scale = kwargs.get('k_coul_scale', 4.0)

        tools = [
            NeutronTool(),
            ProtonTool(nz_light=nz_light, eps_heavy=eps_heavy, cf_heavy=cf_heavy),
            AlphaSFTool(K_SHEAR=K_SHEAR, k_coul_scale=k_coul_scale),
            SFOnlyTool(K_SHEAR=K_SHEAR, k_coul_scale=k_coul_scale),
            AlphaOnlyTool(K_SHEAR=K_SHEAR, k_coul_scale=k_coul_scale),
            BetaPlusTool(),
            BetaMinusTool(),
            StableTool(),
        ]

    def predict(Z, A):
        if A < 2:
            return 'stable'
        geo = compute_geometric_state(Z, A)

        # Run each tool in priority order
        # Drip-line tools first (they gate out before anything else)
        # Then fracture tools (alpha/SF)
        # Then beta (weak process, slowest)
        # Stable is the fallback

        for tool in tools:
            if tool.applies(Z, A, geo):
                if isinstance(tool, AlphaSFTool):
                    return tool.resolve(Z, A, geo)
                if isinstance(tool, StableTool):
                    return 'stable'  # only reached if nothing else applied
                return tool.mode

        return 'stable'

    return predict


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

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
            data.append({
                'A': A, 'Z': Z, 'N': A - Z,
                'mode': mode,
                'element': row.get('element', '??'),
            })
    return data


# ═══════════════════════════════════════════════════════════════════
# COMPARISON ENGINE
# ═══════════════════════════════════════════════════════════════════

def run_comparison(data, predictor_fn, label, show_confusion=True):
    modes_all = ['stable', 'B-', 'B+', 'alpha', 'SF', 'p', 'n', 'IT']
    confusion = {am: {pm: 0 for pm in modes_all} for am in modes_all}
    correct = 0
    total = 0

    for d in data:
        Z, A = d['Z'], d['A']
        actual = d['mode']
        pred = predictor_fn(Z, A)
        if actual not in modes_all:
            continue
        total += 1
        if pred == actual:
            correct += 1
        if actual in modes_all and pred in modes_all:
            confusion[actual][pred] += 1

    if show_confusion:
        print(f"\n  ════ {label} ════")
        print(f"  Mode accuracy:  {correct}/{total} = {100*correct/total:.1f}%")

        hdr = '  Actual     |'
        for pm in modes_all:
            hdr += f'  {pm:>6s}'
        hdr += ' |     N    Acc'
        print(f"\n{hdr}")
        print('  ' + '-' * (len(hdr) - 2))
        for am in modes_all:
            row_total = sum(confusion[am].values())
            if row_total == 0:
                continue
            row_correct = confusion[am].get(am, 0)
            acc = 100 * row_correct / row_total if row_total > 0 else 0
            line = f'  {am:10s} |'
            for pm in modes_all:
                v = confusion[am][pm]
                line += f'  {v:6d}' if v > 0 else '       ·'
            line += f' | {row_total:5d}  {acc:.1f}%'
            print(line)

    mode_stats = {}
    for am in modes_all:
        row_total = sum(confusion[am].values())
        if row_total == 0:
            continue
        mode_stats[am] = {
            'correct': confusion[am].get(am, 0),
            'total': row_total,
            'acc': 100 * confusion[am].get(am, 0) / row_total,
        }

    return {
        'correct': correct, 'total': total,
        'acc': 100 * correct / total if total > 0 else 0,
        'mode_stats': mode_stats,
    }


# ═══════════════════════════════════════════════════════════════════
# TOOL DIAGNOSTIC — Which tools fire for each nuclide?
# ═══════════════════════════════════════════════════════════════════

def tool_diagnostic(data):
    """Show which tools fire for each nuclide — understand coverage."""
    print("=" * 72)
    print("  TOOL DIAGNOSTIC — Which tools apply to each nuclide?")
    print("=" * 72)

    tools = [
        ('neutron', NeutronTool()),
        ('proton', ProtonTool()),
        ('alpha+SF', AlphaSFTool()),
        ('SF-only', SFOnlyTool()),
        ('alpha-only', AlphaOnlyTool()),
        ('B+', BetaPlusTool()),
        ('B-', BetaMinusTool()),
        ('stable', StableTool()),
    ]

    from collections import Counter

    # Count how many tools apply per nuclide
    tool_counts = Counter()
    tool_by_mode = {m: Counter() for m in ['stable', 'B-', 'B+', 'alpha', 'SF', 'p', 'n', 'IT']}

    for d in data:
        geo = compute_geometric_state(d['Z'], d['A'])
        active = []
        for name, tool in tools:
            if name == 'stable':
                continue  # always applies
            if tool.applies(d['Z'], d['A'], geo):
                active.append(name)
        tool_counts[len(active)] += 1
        for name in active:
            tool_by_mode[d['mode']][name] += 1

    print(f"\n  Tools active per nuclide:")
    for n, count in sorted(tool_counts.items()):
        print(f"    {n} tools: {count} nuclides ({100*count/len(data):.1f}%)")

    print(f"\n  Tool activation by actual mode:")
    print(f"  {'Mode':>8s}  {'N':>5s}  {'neutron':>7s}  {'proton':>7s}  {'α+SF':>7s}  "
          f"{'SF-only':>7s}  {'α-only':>7s}  {'B+':>7s}  {'B-':>7s}")
    for mode in ['n', 'p', 'stable', 'B-', 'B+', 'alpha', 'SF', 'IT']:
        mode_data = [d for d in data if d['mode'] == mode]
        if not mode_data:
            continue
        n_total = len(mode_data)
        counts = tool_by_mode[mode]
        print(f"  {mode:>8s}  {n_total:5d}  "
              f"{counts.get('neutron', 0):7d}  "
              f"{counts.get('proton', 0):7d}  "
              f"{counts.get('alpha+SF', 0):7d}  "
              f"{counts.get('SF-only', 0):7d}  "
              f"{counts.get('alpha-only', 0):7d}  "
              f"{counts.get('B+', 0):7d}  "
              f"{counts.get('B-', 0):7d}")

    # Show nuclides where WRONG tool fires
    predict = make_predictor()
    print(f"\n  ── Nuclides where no non-beta tool fires but mode is n/p/alpha/SF ──\n")
    for d in data:
        if d['mode'] not in ('n', 'p', 'alpha', 'SF'):
            continue
        geo = compute_geometric_state(d['Z'], d['A'])
        active = []
        for name, tool in tools:
            if name in ('B+', 'B-', 'stable'):
                continue
            if tool.applies(d['Z'], d['A'], geo):
                active.append(name)
        if not active:
            pred = predict(d['Z'], d['A'])
            print(f"    {d['element']:>3s}-{d['A']:<3d}  actual={d['mode']:6s}  pred={pred:6s}  "
                  f"ε={geo.eps:+.2f} pf={geo.peanut_f:.2f} cf={geo.core_full:.3f}  "
                  f"NO fracture/drip tool fires")


# ═══════════════════════════════════════════════════════════════════
# PARAMETER SCAN — Find best thresholds for each tool
# ═══════════════════════════════════════════════════════════════════

def parameter_scan(data):
    """Scan tool parameters independently."""
    print("\n" + "=" * 72)
    print("  PARAMETER SCAN — Each tool's thresholds")
    print("=" * 72)

    # Baseline
    def pred_v8(Z, A):
        m, _ = predict_decay(Z, A)
        return m

    v8_stats = run_comparison(data, pred_v8, "v8 baseline")

    # Scan proton N/Z threshold
    print(f"\n  ── Proton tool: N/Z threshold scan ──\n")
    print(f"  {'nz_th':>6s}  {'total':>6s}  {'p_acc':>6s}  {'B+_acc':>6s}  {'stbl':>6s}  {'vs_v8':>6s}")
    for nz_th in [0.0, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 1.00]:
        pred = make_predictor(nz_light=nz_th)
        stats = run_comparison(data, pred, f"nz={nz_th}", show_confusion=False)
        ms = stats['mode_stats']
        delta = stats['acc'] - v8_stats['acc']
        p_a = ms.get('p', {}).get('acc', 0)
        bp_a = ms.get('B+', {}).get('acc', 0)
        s_a = ms.get('stable', {}).get('acc', 0)
        print(f"  {nz_th:6.2f}  {stats['acc']:5.1f}%  {p_a:5.1f}%  {bp_a:5.1f}%  {s_a:5.1f}%  {delta:+5.1f}%")

    # Scan heavy proton ε threshold
    print(f"\n  ── Proton tool: heavy ε threshold scan ──\n")
    print(f"  {'eps_th':>6s}  {'total':>6s}  {'p_acc':>6s}  {'B+_acc':>6s}  {'vs_v8':>6s}")
    for eps_th in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        pred = make_predictor(nz_light=0.85, eps_heavy=eps_th)
        stats = run_comparison(data, pred, f"eps={eps_th}", show_confusion=False)
        ms = stats['mode_stats']
        delta = stats['acc'] - v8_stats['acc']
        p_a = ms.get('p', {}).get('acc', 0)
        bp_a = ms.get('B+', {}).get('acc', 0)
        print(f"  {eps_th:6.1f}  {stats['acc']:5.1f}%  {p_a:5.1f}%  {bp_a:5.1f}%  {delta:+5.1f}%")

    # Scan alpha K_SHEAR
    print(f"\n  ── Alpha tool: K_SHEAR scan ──\n")
    print(f"  {'K_SH':>6s}  {'total':>6s}  {'α_acc':>6s}  {'B+_acc':>6s}  {'SF_acc':>6s}  {'vs_v8':>6s}")
    for ks in [1.0, 2.0, PI, 4.0, 5.0, 2*PI]:
        pred = make_predictor(nz_light=0.85, K_SHEAR=ks, k_coul_scale=4.0)
        stats = run_comparison(data, pred, f"KS={ks:.2f}", show_confusion=False)
        ms = stats['mode_stats']
        delta = stats['acc'] - v8_stats['acc']
        a_a = ms.get('alpha', {}).get('acc', 0)
        bp_a = ms.get('B+', {}).get('acc', 0)
        sf_a = ms.get('SF', {}).get('acc', 0)
        print(f"  {ks:6.2f}  {stats['acc']:5.1f}%  {a_a:5.1f}%  {bp_a:5.1f}%  {sf_a:5.1f}%  {delta:+5.1f}%")

    # Scan Coulomb scale
    print(f"\n  ── Alpha tool: k_coul_scale scan ──\n")
    print(f"  {'k_c':>6s}  {'total':>6s}  {'α_acc':>6s}  {'B+_acc':>6s}  {'stbl':>6s}  {'vs_v8':>6s}")
    for kc in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
        pred = make_predictor(nz_light=0.85, K_SHEAR=PI, k_coul_scale=kc)
        stats = run_comparison(data, pred, f"kc={kc}", show_confusion=False)
        ms = stats['mode_stats']
        delta = stats['acc'] - v8_stats['acc']
        a_a = ms.get('alpha', {}).get('acc', 0)
        bp_a = ms.get('B+', {}).get('acc', 0)
        s_a = ms.get('stable', {}).get('acc', 0)
        print(f"  {kc:6.1f}  {stats['acc']:5.1f}%  {a_a:5.1f}%  {bp_a:5.1f}%  {s_a:5.1f}%  {delta:+5.1f}%")


# ═══════════════════════════════════════════════════════════════════
# WINS/LOSSES
# ═══════════════════════════════════════════════════════════════════

def wins_losses(data):
    """Show exactly what changes vs v8."""
    print("\n" + "=" * 72)
    print("  WINS AND LOSSES vs v8")
    print("=" * 72)

    def pred_v8(Z, A):
        m, _ = predict_decay(Z, A)
        return m

    pred_regional = make_predictor(nz_light=0.85)

    wins = []
    losses = []
    for d in data:
        Z, A = d['Z'], d['A']
        actual = d['mode']
        v8 = pred_v8(Z, A)
        reg = pred_regional(Z, A)
        if reg == actual and v8 != actual:
            wins.append(d | {'v8': v8, 'reg': reg})
        elif reg != actual and v8 == actual:
            losses.append(d | {'v8': v8, 'reg': reg})

    print(f"\n  Net: +{len(wins)} wins, -{len(losses)} losses = {len(wins)-len(losses):+d}")

    from collections import Counter

    if wins:
        win_modes = Counter(w['mode'] for w in wins)
        print(f"\n  WINS ({len(wins)}):")
        for mode, count in win_modes.most_common():
            print(f"    {mode}: +{count}")
            for w in sorted([x for x in wins if x['mode'] == mode], key=lambda x: x['A']):
                geo = compute_geometric_state(w['Z'], w['A'])
                print(f"      {w['element']:>3s}-{w['A']:<3d}  v8={w['v8']:6s} → {w['reg']:6s}  "
                      f"ε={geo.eps:+.2f} pf={geo.peanut_f:.2f} cf={geo.core_full:.3f}")

    if losses:
        loss_modes = Counter(l['mode'] for l in losses)
        print(f"\n  LOSSES ({len(losses)}):")
        for mode, count in loss_modes.most_common():
            print(f"    {mode}: -{count}")
            for l in sorted([x for x in losses if x['mode'] == mode],
                           key=lambda x: x['A'])[:15]:
                geo = compute_geometric_state(l['Z'], l['A'])
                print(f"      {l['element']:>3s}-{l['A']:<3d}  v8={l['v8']:6s} → {l['reg']:6s}  "
                      f"actual={l['mode']:6s}  ε={geo.eps:+.2f} pf={geo.peanut_f:.2f}")


# ═══════════════════════════════════════════════════════════════════
# GRAND SUMMARY
# ═══════════════════════════════════════════════════════════════════

def grand_summary(data):
    print("\n" + "=" * 72)
    print("  GRAND SUMMARY — 8-TOOL REGIONAL PREDICTOR")
    print("=" * 72)

    def pred_v8(Z, A):
        m, _ = predict_decay(Z, A)
        return m

    models = {
        'v8 gradient': (pred_v8, 0),
        'Regional (8-tool)': (make_predictor(nz_light=0.85), 0),
    }

    print(f"\n  {'Model':25s}  {'Total':>6s}  {'n':>6s}  {'p':>6s}  {'stbl':>6s}  "
          f"{'B-':>6s}  {'B+':>6s}  {'α':>6s}  {'SF':>6s}  {'IT':>6s}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  "
          f"{'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    for name, (pred, _) in models.items():
        stats = run_comparison(data, pred, name, show_confusion=False)
        ms = stats['mode_stats']
        def ga(m):
            return f"{ms[m]['acc']:5.1f}%" if m in ms else "  -  "
        print(f"  {name:25s}  {stats['acc']:5.1f}%  {ga('n')}  {ga('p')}  {ga('stable')}  "
              f"{ga('B-')}  {ga('B+')}  {ga('alpha')}  {ga('SF')}  {ga('IT')}")

    # Full confusion for best model
    pred_best = make_predictor(nz_light=0.85)
    run_comparison(data, pred_best, "Regional (8-tool) — Full confusion")

    # Zone breakdown
    zones = {1: [], 2: [], 3: []}
    for d in data:
        geo = compute_geometric_state(d['Z'], d['A'])
        zones[geo.zone].append(d)

    print(f"\n  ── Zone breakdown ──\n")
    print(f"  {'Model':25s}  {'Zone1':>6s}  {'Zone2':>6s}  {'Zone3':>6s}")
    for name, (pred, _) in models.items():
        z_acc = {}
        for z in [1, 2, 3]:
            zd = zones[z]
            correct = sum(1 for d in zd if pred(d['Z'], d['A']) == d['mode'])
            z_acc[z] = 100 * correct / len(zd) if zd else 0
        print(f"  {name:25s}  {z_acc[1]:5.1f}%  {z_acc[2]:5.1f}%  {z_acc[3]:5.1f}%")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 72)
    print("  8-TOOL REGIONAL DECAY PREDICTOR")
    print("  Each tool has its own physics, its own territory")
    print("=" * 72)

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    data = load_all_gs()

    from collections import Counter
    modes = Counter(d['mode'] for d in data)
    print(f"\n  Loaded {len(data)} ground-state nuclides (ALL modes, including IT)")
    for m, c in modes.most_common():
        print(f"    {m:8s}: {c:4d}")

    tool_diagnostic(data)
    parameter_scan(data)
    wins_losses(data)
    grand_summary(data)

    print(f"\n  Done.")
