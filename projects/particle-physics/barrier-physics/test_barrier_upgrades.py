#!/usr/bin/env python3
"""
Barrier Physics Upgrades — Three Experiments
==============================================

Starting from the additive Coulomb model (81.3%, 1 fitted param),
test three physics upgrades identified from the research_decay_kinetic
models:

  Experiment 1: Gamow soft barrier — exp(-k√B) replaces step function
  Experiment 2: Asymmetric SF — frozen core fragment replaces A/2
  Experiment 3: Daughter channel pull — 2-body strain in channel space

Each experiment is tested independently and then combined.

Depends on: ../qfd_nuclide_predictor.py (v8 constants and functions)
            ../validate_kinetic_fracture.py (barrier functions, data loader)
"""
import math
import sys
import os

# Import from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from qfd_nuclide_predictor import (
    ALPHA, PI, E_NUM, BETA, S_SURF, R_REG, C_HEAVY, C_LIGHT,
    BETA_LIGHT, A_CRIT, WIDTH, OMEGA, AMP, PHI, A_ALPHA_ONSET,
    N_MAX_ABSOLUTE, CORE_SLOPE, K_COH, K_DEN, PAIRING_SCALE,
    PF_ALPHA_POSSIBLE, PF_DEEP_PEANUT, PF_SF_THRESHOLD, CF_SF_MIN,
    z_star, z0_backbone, compute_geometric_state, survival_score,
    predict_decay, element_name,
)
from validate_kinetic_fracture import (
    bare_scission_barrier, k_coulomb, valley_slope,
    alpha_alignment_penalty, pairing_change_cost,
    predict_kinetic_coulomb, predict_decay_wrapper,
    load_data, run_comparison, print_results, print_per_mode_delta,
    FIT_B, z_fitB, assign_channel, triaxiality,
)
from collections import Counter


# ═══════════════════════════════════════════════════════════════════════
# BASELINE: ADDITIVE COULOMB (81.3%)
# B_eff = max(0, B_surf - π·pf² - 4·K_COUL·max(0,ε))
# ═══════════════════════════════════════════════════════════════════════

BEST_K_SHEAR = PI
BEST_K_COUL_SCALE = 4.0


def predict_additive_coulomb(Z, A):
    """Additive Coulomb baseline — the model to beat."""
    return predict_kinetic_coulomb(Z, A, K_SHEAR=BEST_K_SHEAR,
                                   k_coul_scale=BEST_K_COUL_SCALE,
                                   k_align=0.0)


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: GAMOW SOFT BARRIER
# Replace step function (B_eff ≤ 0 → alpha) with soft crossover:
#   P_alpha = exp(-k_exp · √max(0, B_eff))
# Alpha wins when P_alpha > threshold (competing with beta drive)
# ═══════════════════════════════════════════════════════════════════════

def predict_gamow(Z, A, K_SHEAR=PI, k_coul_scale=4.0, k_exp=1.0):
    """Gamow soft barrier: exponential penetration probability.

    Instead of alpha = available when B_eff ≤ 0 (step function),
    alpha probability = exp(-k_exp · √max(0, B_eff)).

    When B_eff < 0: probability = 1 (barrier fully open, same as before)
    When B_eff = 0: probability = 1 (threshold, same as before)
    When B_eff > 0: probability decays exponentially (NEW — soft leakage)

    Alpha competes with beta: alpha wins when
      P_alpha · alpha_drive > beta_drive
    """
    if A < 1 or Z < 0 or A < Z:
        return 'unknown'
    if Z <= 1:
        return 'stable' if A <= 2 else 'B-'

    geo = compute_geometric_state(Z, A)
    pf = geo.peanut_f
    eps = geo.eps

    # Special modes (same as v8)
    if geo.core_full > 1.0 and A < 50:
        return 'n'
    if geo.core_full < 0.55 and eps > 3.0 and A < 120:
        return 'p'

    # Three-term barrier (same as additive Coulomb)
    elastic = K_SHEAR * pf ** 2
    coulomb = k_coul_scale * k_coulomb(A) * max(0.0, eps)

    if A >= 6 and Z >= 3:
        B_surf_alpha = bare_scission_barrier(A, 4)
        B_eff_alpha = B_surf_alpha - elastic - coulomb  # signed, not clipped
    else:
        B_eff_alpha = 9999.0

    # SF gate (same as additive Coulomb — v8 topology gate)
    sf_available = (pf > PF_SF_THRESHOLD and geo.is_ee
                    and geo.core_full >= CF_SF_MIN)
    if sf_available:
        return 'SF'

    # Gamow penetration probability
    if B_eff_alpha <= 0:
        P_alpha = 1.0  # barrier fully open
    else:
        P_alpha = math.exp(-k_exp * math.sqrt(B_eff_alpha))

    # Beta gradients
    current = survival_score(Z, A)
    gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999.0
    gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999.0
    best_gain = max(gain_bm, gain_bp)
    best_beta = 'B-' if gain_bm >= gain_bp else 'B+'

    # Alpha drive: how much the nuclide WANTS to alpha-decay
    # Over-charged → strong drive; near-valley → weak drive
    alpha_drive = max(0.0, eps)  # only over-charged nuclides are driven

    # Effective alpha action = P_alpha × alpha_drive
    alpha_action = P_alpha * alpha_drive

    # Competition: alpha vs beta
    # Alpha needs both probability (Gamow) AND motivation (eps > 0)
    if alpha_action > 0:
        # Alpha is mechanically possible and driven
        if P_alpha > 0.5:
            # Barrier is mostly open — same logic as additive Coulomb
            if eps > 0:
                return 'alpha'
            if geo.is_ee and best_gain < 2.0 * PAIRING_SCALE:
                return 'alpha'
            if best_gain > 0:
                return best_beta
            if abs(eps) < 0.5:
                return 'stable'
            return 'alpha'
        else:
            # Barrier is partially open — soft leakage regime
            # Alpha wins only if drive is strong enough to overcome
            # the reduced probability
            if alpha_action > best_gain and eps > 0:
                return 'alpha'
            # ee parent with pairing cost for beta
            if (geo.is_ee and alpha_action > 0.1
                    and best_gain < 2.0 * PAIRING_SCALE):
                return 'alpha'

    # Alpha not competitive → beta
    if best_gain > 0:
        return best_beta
    return 'stable'


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: ASYMMETRIC SF
# Replace B_surf(A, A/2) with B_surf(A, Af_heavy) where Af_heavy
# comes from the frozen core conjecture.
#
# Physics: when A > ~230, one lobe fills to N_max = 2πβ³ ≈ 177.
# The heavy fragment is near A ≈ 140 (doubly-magic Ba/Ce region).
# The surface barrier for asymmetric fission is DIFFERENT from
# symmetric fission.
# ═══════════════════════════════════════════════════════════════════════

def sf_heavy_fragment(A, Z):
    """Compute the heavy fragment mass for asymmetric fission.

    From the frozen core conjecture:
    - Heavy fragment has N_heavy ≈ min(N_parent, N_MAX) neutrons
    - Z_heavy ≈ Z*(A_heavy) (on the valley floor)
    - A_heavy = Z_heavy + N_heavy

    For very heavy nuclides (A > 250), the heavy fragment clusters
    near A ≈ 140 (tin/barium region — geometric resonance).

    For lighter fissioners (230 < A < 250), the split is less
    asymmetric: Af ≈ 0.58 · A.
    """
    if A > 250:
        # Deep actinides: heavy fragment near doubly-magic region
        # A_heavy ≈ 140 (near Ba-140 / Ce-140)
        return 140
    elif A > 230:
        # Transition: interpolate between 0.58·A and 140
        f = (A - 230) / 20.0  # 0 at A=230, 1 at A=250
        return int(round((1 - f) * 0.58 * A + f * 140))
    else:
        # Lighter fissioners: moderately asymmetric
        return int(round(0.58 * A))


def predict_asymmetric_sf(Z, A, K_SHEAR=PI, k_coul_scale=4.0):
    """Additive Coulomb with asymmetric SF barrier.

    Same as additive Coulomb but SF barrier uses the physically
    motivated heavy fragment mass instead of A/2.
    """
    if A < 1 or Z < 0 or A < Z:
        return 'unknown'
    if Z <= 1:
        return 'stable' if A <= 2 else 'B-'

    geo = compute_geometric_state(Z, A)
    pf = geo.peanut_f
    eps = geo.eps

    # Special modes
    if geo.core_full > 1.0 and A < 50:
        return 'n'
    if geo.core_full < 0.55 and eps > 3.0 and A < 120:
        return 'p'

    # Alpha barrier (same as additive Coulomb)
    elastic = K_SHEAR * pf ** 2
    coulomb = k_coul_scale * k_coulomb(A) * max(0.0, eps)

    if A >= 6 and Z >= 3:
        B_surf_alpha = bare_scission_barrier(A, 4)
        B_eff_alpha = max(0.0, B_surf_alpha - elastic - coulomb)
    else:
        B_eff_alpha = 9999.0

    # SF barrier — ASYMMETRIC fragment
    if A >= 200:
        Af_heavy = sf_heavy_fragment(A, Z)
        B_surf_sf = bare_scission_barrier(A, Af_heavy)
        B_eff_sf = max(0.0, B_surf_sf - elastic - coulomb)
    else:
        B_eff_sf = 9999.0

    # Beta gradients
    current = survival_score(Z, A)
    gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999.0
    gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999.0
    best_gain = max(gain_bm, gain_bp)
    best_beta = 'B-' if gain_bm >= gain_bp else 'B+'

    # SF: barrier-based (not just topology gate)
    sf_available = (B_eff_sf <= 0.0 and geo.is_ee
                    and geo.core_full >= CF_SF_MIN)
    if sf_available:
        return 'SF'

    # Alpha
    alpha_available = (B_eff_alpha <= 0.0)
    if alpha_available:
        if eps > 0:
            return 'alpha'
        if geo.is_ee and best_gain < 2.0 * PAIRING_SCALE:
            return 'alpha'
        if best_gain > 0:
            return best_beta
        if abs(eps) < 0.5:
            return 'stable'
        return 'alpha'

    if best_gain > 0:
        return best_beta
    return 'stable'


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: DAUGHTER CHANNEL PULL
# Alpha decay: (Z, A) → (Z-2, A-4) + He-4
# The daughter (Z-2, A-4) lands in a specific channel.
# If the daughter's strain |ε_d| is LOW (near valley center),
# the transition is FAVORED (daughter is stable).
# If |ε_d| is HIGH, transition is DISFAVORED (daughter stressed).
#
# This adds a "pull" term: low daughter strain → lower effective barrier.
# ═══════════════════════════════════════════════════════════════════════

def daughter_strain(Z_parent, A_parent, mode):
    """Compute the daughter's valley strain for a given decay mode.

    Returns (Z_daughter, A_daughter, eps_daughter).
    """
    if mode == 'alpha':
        Z_d = Z_parent - 2
        A_d = A_parent - 4
    elif mode == 'B-':
        Z_d = Z_parent + 1
        A_d = A_parent
    elif mode == 'B+':
        Z_d = Z_parent - 1
        A_d = A_parent
    else:
        return Z_parent, A_parent, 0.0

    if Z_d < 1 or A_d < Z_d or A_d < 1:
        return Z_d, A_d, 99.0
    eps_d = Z_d - z_star(A_d)
    return Z_d, A_d, eps_d


def predict_daughter_pull(Z, A, K_SHEAR=PI, k_coul_scale=4.0,
                          k_pull=1.0):
    """Additive Coulomb with daughter strain as barrier modifier.

    The daughter's valley strain modifies the effective barrier:
      B_eff = B_surf - elastic - coulomb + k_pull · |ε_daughter|

    When the daughter is near the valley (|ε_d| small): barrier
    is reduced → transition favored.
    When the daughter is far from valley: barrier increased →
    transition disfavored.

    Also modifies beta: the beta daughter's strain affects the
    gradient calculation. Favors transitions that land close to
    the valley center.
    """
    if A < 1 or Z < 0 or A < Z:
        return 'unknown'
    if Z <= 1:
        return 'stable' if A <= 2 else 'B-'

    geo = compute_geometric_state(Z, A)
    pf = geo.peanut_f
    eps = geo.eps

    # Special modes
    if geo.core_full > 1.0 and A < 50:
        return 'n'
    if geo.core_full < 0.55 and eps > 3.0 and A < 120:
        return 'p'

    # Alpha barrier with daughter pull
    elastic = K_SHEAR * pf ** 2
    coulomb = k_coul_scale * k_coulomb(A) * max(0.0, eps)

    if A >= 6 and Z >= 3:
        B_surf_alpha = bare_scission_barrier(A, 4)
        _, _, eps_alpha_d = daughter_strain(Z, A, 'alpha')
        # Daughter near valley → pull reduces barrier
        # Daughter far from valley → increases barrier
        pull_alpha = k_pull * abs(eps_alpha_d)
        B_eff_alpha = max(0.0, B_surf_alpha + pull_alpha
                         - elastic - coulomb)
    else:
        B_eff_alpha = 9999.0

    # SF gate (same as additive Coulomb)
    sf_available = (pf > PF_SF_THRESHOLD and geo.is_ee
                    and geo.core_full >= CF_SF_MIN)
    if sf_available:
        return 'SF'

    # Beta gradients with daughter strain bonus
    current = survival_score(Z, A)
    gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999.0
    gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999.0

    # Daughter strain for beta modes
    _, _, eps_bm_d = daughter_strain(Z, A, 'B-')
    _, _, eps_bp_d = daughter_strain(Z, A, 'B+')

    # Beta modes that land closer to valley get a boost
    # (smaller |eps_d| → bigger boost)
    if gain_bm > -9000:
        gain_bm += k_pull * max(0.0, abs(eps) - abs(eps_bm_d))
    if gain_bp > -9000:
        gain_bp += k_pull * max(0.0, abs(eps) - abs(eps_bp_d))

    best_gain = max(gain_bm, gain_bp)
    best_beta = 'B-' if gain_bm >= gain_bp else 'B+'

    # Mode selection
    alpha_available = (B_eff_alpha <= 0.0)
    if alpha_available:
        if eps > 0:
            return 'alpha'
        if geo.is_ee and best_gain < 2.0 * PAIRING_SCALE:
            return 'alpha'
        if best_gain > 0:
            return best_beta
        if abs(eps) < 0.5:
            return 'stable'
        return 'alpha'

    if best_gain > 0:
        return best_beta
    return 'stable'


# ═══════════════════════════════════════════════════════════════════════
# COMBINED MODEL: ALL THREE UPGRADES
# ═══════════════════════════════════════════════════════════════════════

def predict_combined(Z, A, K_SHEAR=PI, k_coul_scale=4.0,
                     k_exp=1.0, k_pull=0.5):
    """Combined: Gamow + asymmetric SF + daughter pull."""
    if A < 1 or Z < 0 or A < Z:
        return 'unknown'
    if Z <= 1:
        return 'stable' if A <= 2 else 'B-'

    geo = compute_geometric_state(Z, A)
    pf = geo.peanut_f
    eps = geo.eps

    if geo.core_full > 1.0 and A < 50:
        return 'n'
    if geo.core_full < 0.55 and eps > 3.0 and A < 120:
        return 'p'

    # Alpha barrier with daughter pull
    elastic = K_SHEAR * pf ** 2
    coulomb = k_coul_scale * k_coulomb(A) * max(0.0, eps)

    if A >= 6 and Z >= 3:
        B_surf_alpha = bare_scission_barrier(A, 4)
        _, _, eps_alpha_d = daughter_strain(Z, A, 'alpha')
        pull_alpha = k_pull * abs(eps_alpha_d)
        B_eff_alpha = B_surf_alpha + pull_alpha - elastic - coulomb
    else:
        B_eff_alpha = 9999.0

    # Asymmetric SF barrier
    if A >= 200:
        Af_heavy = sf_heavy_fragment(A, Z)
        B_surf_sf = bare_scission_barrier(A, Af_heavy)
        B_eff_sf = max(0.0, B_surf_sf - elastic - coulomb)
    else:
        B_eff_sf = 9999.0

    # SF: barrier-based with asymmetric fragment
    sf_available = (B_eff_sf <= 0.0 and geo.is_ee
                    and geo.core_full >= CF_SF_MIN)
    if sf_available:
        return 'SF'

    # Gamow penetration for alpha
    if B_eff_alpha <= 0:
        P_alpha = 1.0
    else:
        P_alpha = math.exp(-k_exp * math.sqrt(B_eff_alpha))

    # Beta gradients with daughter strain
    current = survival_score(Z, A)
    gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999.0
    gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999.0

    _, _, eps_bm_d = daughter_strain(Z, A, 'B-')
    _, _, eps_bp_d = daughter_strain(Z, A, 'B+')

    if gain_bm > -9000:
        gain_bm += k_pull * max(0.0, abs(eps) - abs(eps_bm_d))
    if gain_bp > -9000:
        gain_bp += k_pull * max(0.0, abs(eps) - abs(eps_bp_d))

    best_gain = max(gain_bm, gain_bp)
    best_beta = 'B-' if gain_bm >= gain_bp else 'B+'

    alpha_drive = max(0.0, eps)
    alpha_action = P_alpha * alpha_drive

    if alpha_action > 0:
        if P_alpha > 0.5:
            if eps > 0:
                return 'alpha'
            if geo.is_ee and best_gain < 2.0 * PAIRING_SCALE:
                return 'alpha'
            if best_gain > 0:
                return best_beta
            if abs(eps) < 0.5:
                return 'stable'
            return 'alpha'
        else:
            if alpha_action > best_gain and eps > 0:
                return 'alpha'
            if (geo.is_ee and alpha_action > 0.1
                    and best_gain < 2.0 * PAIRING_SCALE):
                return 'alpha'

    if best_gain > 0:
        return best_beta
    return 'stable'


# ═══════════════════════════════════════════════════════════════════════
# ZONE-RESOLVED HELPER
# ═══════════════════════════════════════════════════════════════════════

def zone_table(data, models, stats_v8):
    """Print zone-resolved comparison table for a list of models."""
    # Pre-compute zone membership
    zone_data = {1: [], 2: [], 3: []}
    for d in data:
        geo = compute_geometric_state(d['Z'], d['A'])
        zone_data[geo.zone].append(d)

    print(f"\n  {'Model':<28s} {'Zone1':>7s} {'Zone2':>7s} {'Zone3':>7s} "
          f"{'Total':>7s} {'vs v8':>7s}")
    print(f"  {'-'*70}")

    for label, pred_fn, st_total in models:
        z_accs = {}
        for z in [1, 2, 3]:
            st_z = run_comparison(zone_data[z], pred_fn, f"{label} Z{z}")
            z_accs[z] = 100 * st_z['accuracy']
        tot = 100 * st_total['accuracy']
        vs_v8 = 100 * (st_total['accuracy'] - stats_v8['accuracy'])
        print(f"  {label:<28s} {z_accs[1]:6.1f}% {z_accs[2]:6.1f}% "
              f"{z_accs[3]:6.1f}% {tot:6.1f}% {vs_v8:+6.1f}%")


def per_mode_table(stats):
    """Print per-mode accuracy from a stats dict."""
    modes = ['stable', 'B-', 'B+', 'alpha', 'SF', 'p', 'n']
    for m in modes:
        row = stats['confusion'].get(m, {})
        n = sum(row.values())
        if n == 0:
            continue
        acc = 100 * row.get(m, 0) / n
        print(f"    {m:<10s} {row.get(m,0):4d}/{n:<4d} = {acc:5.1f}%")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 72)
    print("  BARRIER PHYSICS UPGRADES — THREE EXPERIMENTS")
    print("  Starting point: Additive Coulomb (81.3%, 1 fitted param)")
    print("=" * 72)

    data = load_data()
    print(f"\n  Loaded {len(data)} ground-state nuclides")
    mode_counts = Counter(d['mode'] for d in data)
    for m in sorted(mode_counts):
        print(f"    {m:<10s} {mode_counts[m]:5d}")

    # ── Baselines ──
    stats_v8 = run_comparison(data, predict_decay_wrapper,
                               "v8 GRADIENT (baseline)")
    stats_coul = run_comparison(data, predict_additive_coulomb,
                                 "ADDITIVE COULOMB (baseline)")
    print_results(stats_v8)
    print_results(stats_coul)

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: GAMOW SOFT BARRIER
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  EXPERIMENT 1: GAMOW SOFT BARRIER")
    print(f"  P_alpha = exp(-k_exp · √max(0, B_eff))")
    print(f"  Replaces step function with exponential tunneling")
    print(f"{'='*72}")

    # Scan k_exp
    k_exp_values = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]

    print(f"\n  {'k_exp':>6s} {'Mode%':>7s} {'β-dir%':>7s} "
          f"{'α%':>6s} {'B+%':>6s} {'stbl%':>6s} {'SF%':>5s}")
    print(f"  {'-'*52}")

    best_gamow_acc = 0
    best_k_exp = 0

    for ke in k_exp_values:
        def pred_fn(Z, A, _ke=ke):
            return predict_gamow(Z, A, k_exp=_ke)
        st = run_comparison(data, pred_fn, f"k_exp={ke}")

        alpha_row = st['confusion'].get('alpha', {})
        alpha_n = sum(alpha_row.values())
        alpha_acc = 100 * alpha_row.get('alpha', 0) / alpha_n if alpha_n else 0
        bp_row = st['confusion'].get('B+', {})
        bp_n = sum(bp_row.values())
        bp_acc = 100 * bp_row.get('B+', 0) / bp_n if bp_n else 0
        stbl_row = st['confusion'].get('stable', {})
        stbl_n = sum(stbl_row.values())
        stbl_acc = 100 * stbl_row.get('stable', 0) / stbl_n if stbl_n else 0
        sf_row = st['confusion'].get('SF', {})
        sf_n = sum(sf_row.values())
        sf_acc = 100 * sf_row.get('SF', 0) / sf_n if sf_n else 0

        marker = ''
        if st['accuracy'] > best_gamow_acc:
            best_gamow_acc = st['accuracy']
            best_k_exp = ke
            marker = ' ◄'

        print(f"  {ke:6.1f} {100*st['accuracy']:6.1f}% "
              f"{100*st['beta_dir']:6.1f}% {alpha_acc:5.1f}% "
              f"{bp_acc:5.1f}% {stbl_acc:5.1f}% {sf_acc:4.1f}%{marker}")

    print(f"\n  BEST: k_exp={best_k_exp} → {100*best_gamow_acc:.1f}%")

    # Best Gamow model
    def best_gamow_pred(Z, A):
        return predict_gamow(Z, A, k_exp=best_k_exp)
    stats_gamow = run_comparison(data, best_gamow_pred,
                                  f"GAMOW (k_exp={best_k_exp})")
    print_results(stats_gamow)
    print(f"\n  ── Per-mode: Additive Coulomb → Gamow ──")
    print_per_mode_delta(stats_coul, stats_gamow)

    # Wins/losses vs additive Coulomb
    gamow_wins = []
    gamow_losses = []
    for d in data:
        actual = d['mode']
        p_c = predict_additive_coulomb(d['Z'], d['A'])
        p_g = best_gamow_pred(d['Z'], d['A'])
        if p_c != p_g:
            geo = compute_geometric_state(d['Z'], d['A'])
            entry = (d['Z'], d['A'], element_name(d['Z']), actual,
                     p_c, p_g, geo.eps, geo.peanut_f, geo.parity)
            if p_g == actual and p_c != actual:
                gamow_wins.append(entry)
            elif p_c == actual and p_g != actual:
                gamow_losses.append(entry)

    print(f"\n  Gamow WINS ({len(gamow_wins)}):")
    for Z_d, A_d, el, act, c, g, eps, pf, par in gamow_wins[:15]:
        print(f"    {el}-{A_d:3d} (Z={Z_d:3d}) actual={act:6s} "
              f"Coul={c:6s} Gamow={g:6s} ε={eps:+.2f} pf={pf:.2f} {par}")
    print(f"  Gamow LOSSES ({len(gamow_losses)}):")
    for Z_d, A_d, el, act, c, g, eps, pf, par in gamow_losses[:15]:
        print(f"    {el}-{A_d:3d} (Z={Z_d:3d}) actual={act:6s} "
              f"Coul={c:6s} Gamow={g:6s} ε={eps:+.2f} pf={pf:.2f} {par}")
    print(f"  Net: +{len(gamow_wins)} wins, -{len(gamow_losses)} losses")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: ASYMMETRIC SF
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  EXPERIMENT 2: ASYMMETRIC SF FRAGMENTS")
    print(f"  Heavy fragment from frozen core: Af ≈ 140 (A>250)")
    print(f"  B_surf(A, Af) replaces B_surf(A, A/2)")
    print(f"{'='*72}")

    # Show the asymmetric barrier landscape
    print(f"\n  SF Barrier Landscape (asymmetric vs symmetric):")
    print(f"  {'A':>5s} {'Af_sym':>7s} {'Af_asym':>8s} {'B_sym':>7s} "
          f"{'B_asym':>8s} {'ratio':>7s}")
    print(f"  {'-'*50}")
    for A_show in [220, 230, 240, 250, 258, 260, 270, 280, 295]:
        zs = z_star(A_show)
        Z_show = round(zs)
        Af_sym = A_show // 2
        Af_asym = sf_heavy_fragment(A_show, Z_show)
        B_sym = bare_scission_barrier(A_show, Af_sym)
        B_asym = bare_scission_barrier(A_show, Af_asym)
        ratio = B_asym / B_sym if B_sym > 0 else 0
        print(f"  {A_show:5d} {Af_sym:7d} {Af_asym:8d} {B_sym:7.3f} "
              f"{B_asym:8.3f} {ratio:7.3f}")

    # Run asymmetric SF model
    stats_asym = run_comparison(data, predict_asymmetric_sf,
                                 "ASYMMETRIC SF")
    print_results(stats_asym)

    print(f"\n  ── Per-mode: Additive Coulomb → Asymmetric SF ──")
    print_per_mode_delta(stats_coul, stats_asym)

    # SF diagnostic
    sf_nuclides = [d for d in data if d['mode'] == 'SF']
    print(f"\n  SF Nuclide Detail (asymmetric vs symmetric):")
    print(f"  {'El-A':>8s} {'Z':>4s} {'actual':>7s} "
          f"{'Coul':>6s} {'Asym':>6s} {'Af':>4s}")
    print(f"  {'-'*45}")
    for d in sorted(sf_nuclides, key=lambda x: (x['A'], x['Z'])):
        Z_d, A_d = d['Z'], d['A']
        p_c = predict_additive_coulomb(Z_d, A_d)
        p_a = predict_asymmetric_sf(Z_d, A_d)
        Af = sf_heavy_fragment(A_d, Z_d)
        mc = '✓' if p_c == 'SF' else '✗'
        ma = '✓' if p_a == 'SF' else '✗'
        print(f"  {element_name(Z_d)}-{A_d:3d} {Z_d:4d} {'SF':>7s} "
              f"{p_c:>5s} {mc} {p_a:>4s} {ma} {Af:4d}")

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: DAUGHTER CHANNEL PULL
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  EXPERIMENT 3: DAUGHTER CHANNEL PULL")
    print(f"  Alpha daughter strain modifies barrier:")
    print(f"  B_eff += k_pull · |ε_daughter|")
    print(f"  Beta daughter strain modifies gradient boost")
    print(f"{'='*72}")

    # Scan k_pull
    k_pull_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]

    print(f"\n  {'k_pull':>7s} {'Mode%':>7s} {'α%':>6s} {'B+%':>6s} "
          f"{'B-%':>6s} {'stbl%':>6s}")
    print(f"  {'-'*42}")

    best_pull_acc = 0
    best_k_pull = 0

    for kp in k_pull_values:
        def pred_fn(Z, A, _kp=kp):
            return predict_daughter_pull(Z, A, k_pull=_kp)
        st = run_comparison(data, pred_fn, f"k_pull={kp}")

        alpha_row = st['confusion'].get('alpha', {})
        alpha_n = sum(alpha_row.values())
        alpha_acc = 100 * alpha_row.get('alpha', 0) / alpha_n if alpha_n else 0
        bp_row = st['confusion'].get('B+', {})
        bp_n = sum(bp_row.values())
        bp_acc = 100 * bp_row.get('B+', 0) / bp_n if bp_n else 0
        bm_row = st['confusion'].get('B-', {})
        bm_n = sum(bm_row.values())
        bm_acc = 100 * bm_row.get('B-', 0) / bm_n if bm_n else 0
        stbl_row = st['confusion'].get('stable', {})
        stbl_n = sum(stbl_row.values())
        stbl_acc = 100 * stbl_row.get('stable', 0) / stbl_n if stbl_n else 0

        marker = ''
        if st['accuracy'] > best_pull_acc:
            best_pull_acc = st['accuracy']
            best_k_pull = kp
            marker = ' ◄'

        print(f"  {kp:7.2f} {100*st['accuracy']:6.1f}% "
              f"{alpha_acc:5.1f}% {bp_acc:5.1f}% {bm_acc:5.1f}% "
              f"{stbl_acc:5.1f}%{marker}")

    print(f"\n  BEST: k_pull={best_k_pull} → {100*best_pull_acc:.1f}%")

    def best_pull_pred(Z, A):
        return predict_daughter_pull(Z, A, k_pull=best_k_pull)
    stats_pull = run_comparison(data, best_pull_pred,
                                 f"DAUGHTER PULL (k_pull={best_k_pull})")
    print_results(stats_pull)
    print(f"\n  ── Per-mode: Additive Coulomb → Daughter Pull ──")
    print_per_mode_delta(stats_coul, stats_pull)

    # ══════════════════════════════════════════════════════════════════
    # COMBINED: ALL THREE UPGRADES
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  COMBINED MODEL: GAMOW + ASYMMETRIC SF + DAUGHTER PULL")
    print(f"  k_exp={best_k_exp}, k_pull={best_k_pull}")
    print(f"{'='*72}")

    # Scan k_exp × k_pull jointly
    ke_short = [0.5, 1.0, 2.0, 3.0, 5.0]
    kp_short = [0.0, 0.1, 0.3, 0.5, 1.0]

    best_combo_acc = 0
    best_combo_ke = 0
    best_combo_kp = 0

    print(f"\n  {'k_exp':>6s}", end='')
    for kp in kp_short:
        print(f" {'kp='+str(kp):>7s}", end='')
    print()
    print(f"  {'-'*6}", end='')
    for _ in kp_short:
        print(f" {'-'*7}", end='')
    print()

    for ke in ke_short:
        print(f"  {ke:6.1f}", end='')
        for kp in kp_short:
            def pred_fn(Z, A, _ke=ke, _kp=kp):
                return predict_combined(Z, A, k_exp=_ke, k_pull=_kp)
            st = run_comparison(data, pred_fn, f"ke={ke},kp={kp}")
            acc = 100 * st['accuracy']
            print(f" {acc:6.1f}%", end='')
            if st['accuracy'] > best_combo_acc:
                best_combo_acc = st['accuracy']
                best_combo_ke = ke
                best_combo_kp = kp
        print()

    print(f"\n  BEST COMBINED: k_exp={best_combo_ke}, "
          f"k_pull={best_combo_kp} → {100*best_combo_acc:.1f}%")

    def best_combo_pred(Z, A):
        return predict_combined(Z, A, k_exp=best_combo_ke,
                                k_pull=best_combo_kp)
    stats_combo = run_comparison(data, best_combo_pred,
                                  f"COMBINED (ke={best_combo_ke}, "
                                  f"kp={best_combo_kp})")
    print_results(stats_combo)

    # ══════════════════════════════════════════════════════════════════
    # GRAND COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  GRAND COMPARISON — ALL MODELS")
    print(f"{'='*72}")

    all_models = [
        ("v8 gradient", stats_v8),
        ("Additive Coulomb", stats_coul),
        (f"Gamow (k_exp={best_k_exp})", stats_gamow),
        ("Asymmetric SF", stats_asym),
        (f"Daughter pull (k={best_k_pull})", stats_pull),
        (f"Combined", stats_combo),
    ]

    print(f"\n  {'Model':<28s} {'Mode%':>6s} {'β-dir':>6s} "
          f"{'α%':>6s} {'SF%':>5s} {'B+%':>5s} {'vs v8':>7s} {'Params':>6s}")
    print(f"  {'-'*78}")

    param_counts = [0, 1, 2, 1, 2, 3]
    for (name, st), npar in zip(all_models, param_counts):
        alpha_row = st['confusion'].get('alpha', {})
        alpha_n = sum(alpha_row.values())
        alpha_acc = 100 * alpha_row.get('alpha', 0) / alpha_n if alpha_n else 0
        sf_row = st['confusion'].get('SF', {})
        sf_n = sum(sf_row.values())
        sf_acc = 100 * sf_row.get('SF', 0) / sf_n if sf_n else 0
        bp_row = st['confusion'].get('B+', {})
        bp_n = sum(bp_row.values())
        bp_acc = 100 * bp_row.get('B+', 0) / bp_n if bp_n else 0
        delta = st['accuracy'] - stats_v8['accuracy']
        print(f"  {name:<28s} {100*st['accuracy']:5.1f}% "
              f"{100*st['beta_dir']:5.1f}% {alpha_acc:5.1f}% "
              f"{sf_acc:4.1f}% {bp_acc:4.1f}% {100*delta:+6.1f}% "
              f"{npar:5d}")

    # Zone-resolved comparison
    print(f"\n  ── Zone-Resolved ──")
    zone_models = [
        ("v8 gradient", predict_decay_wrapper, stats_v8),
        ("Additive Coulomb", predict_additive_coulomb, stats_coul),
        (f"Gamow (ke={best_k_exp})", best_gamow_pred, stats_gamow),
        ("Asymmetric SF", predict_asymmetric_sf, stats_asym),
        (f"Daughter pull (k={best_k_pull})", best_pull_pred, stats_pull),
        ("Combined", best_combo_pred, stats_combo),
    ]
    zone_table(data, zone_models, stats_v8)

    # Per-mode delta: v8 → best model
    best_model = max(all_models, key=lambda x: x[1]['accuracy'])
    print(f"\n  ── Per-mode: v8 → {best_model[0]} ──")
    print_per_mode_delta(stats_v8, best_model[1])

    print(f"\n  ── Per-mode: Additive Coulomb → {best_model[0]} ──")
    print_per_mode_delta(stats_coul, best_model[1])

    print(f"\n  Done.")
