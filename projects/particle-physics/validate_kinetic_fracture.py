#!/usr/bin/env python3
"""
Validate: Kinetic Fracture Model vs v8 Gradient Predictor
==========================================================

Tests whether the Topological Cleavage Barrier (scission barrier +
Dzhanibekov discount + valley alignment) improves decay mode prediction
over the v8 zone-separated gradient predictor.

Physics (Tracy, 2026-02-21):
  - Beta = phase slip (barrier ~ 0, except pairing cost for ee parents)
  - Alpha/SF = topological rupture (barrier = surface energy of new surface)
  - Dzhanibekov effect: pf^2 elastic energy cancels scission barrier
  - Valley alignment: alpha vector (slope 0.5) vs valley tangent

Imports constants and functions from qfd_nuclide_predictor.py.
"""
import math
import csv
import sys
import os

# Import everything from the predictor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qfd_nuclide_predictor import (
    ALPHA, PI, E_NUM, BETA, S_SURF, R_REG, C_HEAVY, C_LIGHT,
    BETA_LIGHT, A_CRIT, WIDTH, OMEGA, AMP, PHI, A_ALPHA_ONSET,
    N_MAX_ABSOLUTE, CORE_SLOPE, K_COH, K_DEN, PAIRING_SCALE,
    PF_ALPHA_POSSIBLE, PF_DEEP_PEANUT, PF_SF_THRESHOLD, CF_SF_MIN,
    z_star, z0_backbone, compute_geometric_state, survival_score,
    predict_decay, element_name,
)


# ═══════════════════════════════════════════════════════════════════════
# KINETIC FRACTURE MODEL
# ═══════════════════════════════════════════════════════════════════════

def bare_scission_barrier(A_parent, A_fragment):
    """Surface energy cost of tearing the soliton into two pieces.

    B_surf = S_SURF * [(A-Af)^{2/3} + Af^{2/3} - A^{2/3}]

    This is the geometric cost of creating new vacuum-exposed surface
    when the density field pinches off a fragment.
    """
    A_rem = A_parent - A_fragment
    if A_rem < 1 or A_fragment < 1:
        return 9999.0
    return S_SURF * (A_rem ** (2.0/3) + A_fragment ** (2.0/3)
                     - A_parent ** (2.0/3))


def valley_slope(A):
    """Local slope dZ*/dA of the stability valley."""
    if A < 2:
        return 0.5
    return z_star(A + 0.5) - z_star(A - 0.5)


def alpha_alignment_penalty(A):
    """Misalignment between alpha vector (slope=0.5) and valley tangent.

    Alpha decay vector: (Delta_A, Delta_Z) = (-4, -2), slope = 0.5
    Valley tangent slope: dZ*/dA

    When these match (light nuclei, slope~0.5), alpha slides along the
    valley floor.  When they diverge (heavy nuclei, slope~0.38), alpha
    scrapes against the valley walls.
    """
    slope = valley_slope(A)
    return abs(0.5 - slope)


def pairing_change_cost(Z, A, mode):
    """Cost of pairing change for a given decay mode.

    Beta: changes parity of BOTH Z and N.
      ee -> oo: cost = 2/beta  (lose bonus, gain penalty)
      oo -> ee: cost = -2/beta (gain bonus, lose penalty -- HELPS)
      eo -> oe or oe -> eo: cost = 0

    Alpha: (Z-2, N-2) preserves parity of both Z and N.
      Always 0.
    """
    N = A - Z
    z_even = (Z % 2 == 0)
    n_even = (N % 2 == 0)

    if mode in ('alpha', 'SF'):
        return 0.0  # preserves parity

    # Beta modes change both Z and N parity
    if z_even and n_even:      # ee -> oo
        return 2.0 * PAIRING_SCALE
    elif not z_even and not n_even:  # oo -> ee
        return -2.0 * PAIRING_SCALE
    return 0.0  # eo->oe or oe->eo


def predict_kinetic(Z, A, K_SHEAR, k_align=0.0):
    """Kinetic fracture model: scission barrier + Dzhanibekov discount.

    Algorithm:
    1. Compute scission barriers for alpha/SF
    2. Apply Dzhanibekov discount: B_eff = max(0, B_surf - K_SHEAR * pf^2)
    3. Alpha available when B_eff = 0 (barrier fully erased)
    4. When alpha is available:
       - Over-charged (eps > 0): alpha wins (removes excess charge)
       - Under-charged ee: alpha may win (beta has pairing cost 2/beta)
       - Under-charged non-ee: beta wins (zero-cost path exists)
    5. When alpha blocked: pure beta gradient (same as v8)

    K_SHEAR: elastic shear constant (from peanut deformation)
    k_align: weight for valley alignment penalty
    """
    if A < 1 or Z < 0 or A < Z:
        return 'unknown'
    if Z <= 1:
        if A <= 2:
            return 'stable'
        return 'B-'

    geo = compute_geometric_state(Z, A)
    pf = geo.peanut_f
    eps = geo.eps

    # ── Special modes (same as v8) ──
    if geo.core_full > 1.0 and A < 50:
        return 'n'
    if geo.core_full < 0.55 and eps > 3.0 and A < 120:
        return 'p'

    # ── Scission barriers ──
    elastic = K_SHEAR * pf ** 2

    # Alpha barrier
    if A >= 6 and Z >= 3:
        B_surf_alpha = bare_scission_barrier(A, 4)
        align_cost = k_align * alpha_alignment_penalty(A) if k_align > 0 else 0.0
        B_eff_alpha = max(0.0, B_surf_alpha + align_cost - elastic)
    else:
        B_eff_alpha = 9999.0

    # SF barrier
    if A >= 200:
        A_half = A // 2
        B_surf_sf = bare_scission_barrier(A, A_half)
        B_eff_sf = max(0.0, B_surf_sf - elastic)
    else:
        B_eff_sf = 9999.0

    # ── Beta gradients (same as v8) ──
    current = survival_score(Z, A)
    gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999.0
    gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999.0

    best_gain = max(gain_bm, gain_bp)
    best_beta = 'B-' if gain_bm >= gain_bp else 'B+'

    # ── Mode selection via barrier physics ──
    alpha_available = (B_eff_alpha <= 0.0)
    sf_available = (B_eff_sf <= 0.0 and A > 250 and geo.is_ee
                    and geo.core_full >= CF_SF_MIN)

    if sf_available:
        return 'SF'

    if alpha_available:
        # Alpha is mechanically free (barrier erased by Dzhanibekov)
        if eps > 0:
            # Over-charged: shedding charge is doubly favorable
            return 'alpha'
        # Under-charged ee: beta has pairing cost 2/beta
        if geo.is_ee and best_gain < 2.0 * PAIRING_SCALE:
            return 'alpha'
        # Under-charged non-ee or strong beta gradient: beta wins
        if best_gain > 0:
            return best_beta
        # Near valley, no favorable gradient
        if abs(eps) < 0.5:
            return 'stable'
        return 'alpha'

    # Alpha NOT available (barrier too high) → pure beta decision
    if best_gain > 0:
        return best_beta

    return 'stable'


# ═══════════════════════════════════════════════════════════════════════
# FIT B — 32-CHANNEL ASSIGNMENT (from qfd_32ch_decay_pull.py)
# 15 constants fitted to AME2020 — used ONLY for channel geometry
# ═══════════════════════════════════════════════════════════════════════

FIT_B = dict(
    a0=-10.498311, b0=-1.955412, g0=+0.160312,
    a1= +1.442776, b1=+0.910368, g1=-0.111113,
    a2= +0.999855, b2=-0.234427, g2=+0.027793,
    a3= +0.249419, b3=+0.018786, g3=-0.002310,
    D0= -3.526233, D1=-5.377797, D2=+5.313587,
)


def z_fitB(A, ell, m):
    """Fit B valley center for channel (ell, m)."""
    p = FIT_B
    deltas = [p['D0'], p['D1'], p['D2'], 0.0]
    A13 = A ** (1.0 / 3); A23 = A ** (2.0 / 3)
    return ((p['a0'] + p['b0'] * m + p['g0'] * m * m + deltas[ell])
            + (p['a1'] + p['b1'] * m + p['g1'] * m * m) * A13
            + (p['a2'] + p['b2'] * m + p['g2'] * m * m) * A23
            + (p['a3'] + p['b3'] * m + p['g3'] * m * m) * A)


def assign_channel(A, Z):
    """Assign (ell, m, parity) channel for a single nuclide.

    Searches all 16 (ell, m) combinations, picks the one whose
    Fit B prediction is closest to Z, then snaps parity.
    """
    best_key = None
    best_r = 1e6
    for ell in range(4):
        for m in range(-ell, ell + 1):
            z_c = z_fitB(A, ell, m)
            r = abs(z_c - Z)
            if r < best_r:
                best_r = r
                ze = int(2 * round(z_c / 2))
                zo = int(2 * math.floor(z_c / 2) + 1)
                par = 0 if abs(ze - Z) < abs(zo - Z) else 1
                best_key = (ell, m, par)
    return best_key


# ═══════════════════════════════════════════════════════════════════════
# TRIAXIALITY — Dzhanibekov modulation
# ═══════════════════════════════════════════════════════════════════════

def triaxiality(ell, m):
    """T in [0,1]: 0 = axially symmetric, 1 = maximally triaxial.

    The Dzhanibekov instability requires I1 < I2 < I3 (three distinct
    principal moments).  Axially symmetric shapes (m=0) have I1=I2,
    so NO intermediate-axis tumbling.  T = |m|/ell measures the degree
    of triaxial deformation in the channel geometry.
    """
    if ell == 0:
        return 0.0  # s-wave: spherical, no preferred axis
    return abs(m) / ell


def predict_kinetic_triax(Z, A, K_SHEAR, f_triax_fn, k_align=0.0):
    """Kinetic fracture model with triaxiality-dependent Dzhanibekov discount.

    Same as predict_kinetic(), but elastic energy is modulated by
    f_triax(T) where T = triaxiality of the assigned channel.

    When T=0 (axially symmetric, m=0): barrier stays high.
    When T=1 (maximally triaxial): full Dzhanibekov discount.
    """
    if A < 1 or Z < 0 or A < Z:
        return 'unknown'
    if Z <= 1:
        if A <= 2:
            return 'stable'
        return 'B-'

    geo = compute_geometric_state(Z, A)
    pf = geo.peanut_f
    eps = geo.eps

    # ── Special modes (same as v8) ──
    if geo.core_full > 1.0 and A < 50:
        return 'n'
    if geo.core_full < 0.55 and eps > 3.0 and A < 120:
        return 'p'

    # ── Channel assignment and triaxiality ──
    key = assign_channel(A, Z)
    T = triaxiality(key[0], key[1]) if key is not None else 0.0
    f_T = f_triax_fn(T)

    # ── Scission barriers with triaxiality modulation ──
    elastic = K_SHEAR * pf ** 2 * f_T

    # Alpha barrier
    if A >= 6 and Z >= 3:
        B_surf_alpha = bare_scission_barrier(A, 4)
        align_cost = k_align * alpha_alignment_penalty(A) if k_align > 0 else 0.0
        B_eff_alpha = max(0.0, B_surf_alpha + align_cost - elastic)
    else:
        B_eff_alpha = 9999.0

    # SF barrier
    if A >= 200:
        A_half = A // 2
        B_surf_sf = bare_scission_barrier(A, A_half)
        B_eff_sf = max(0.0, B_surf_sf - elastic)
    else:
        B_eff_sf = 9999.0

    # ── Beta gradients (same as v8) ──
    current = survival_score(Z, A)
    gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999.0
    gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999.0

    best_gain = max(gain_bm, gain_bp)
    best_beta = 'B-' if gain_bm >= gain_bp else 'B+'

    # ── Mode selection via barrier physics ──
    alpha_available = (B_eff_alpha <= 0.0)
    sf_available = (B_eff_sf <= 0.0 and A > 250 and geo.is_ee
                    and geo.core_full >= CF_SF_MIN)

    if sf_available:
        return 'SF'

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

    # Alpha NOT available → pure beta decision
    if best_gain > 0:
        return best_beta

    return 'stable'


# ═══════════════════════════════════════════════════════════════════════
# COULOMB-ASSISTED SCISSION — Three-term barrier
# See scission.md for full derivation
# ═══════════════════════════════════════════════════════════════════════

def k_coulomb(A):
    """Coulomb stress coefficient: excess electromagnetic self-energy
    per unit charge displacement ε.

    K_COUL(A) = 2 · Z*(A) · α / A^{1/3}

    This is the differential Coulomb energy released when an alpha
    separates from a soliton with charge excess ε relative to the
    valley floor.  Zero free parameters.
    """
    zs = z_star(A)
    return 2.0 * zs * ALPHA / (A ** (1.0 / 3))


def predict_kinetic_coulomb(Z, A, K_SHEAR, k_coul_scale=1.0, k_align=0.0):
    """Three-term barrier: scission + Dzhanibekov + Coulomb-assisted.

    B_eff = max(0, B_surf + align - K_SHEAR·pf² - k_coul_scale·K_COUL(A)·max(0, ε))

    k_coul_scale: multiplier on the geometric K_COUL (1.0 = pure theory).
    """
    if A < 1 or Z < 0 or A < Z:
        return 'unknown'
    if Z <= 1:
        if A <= 2:
            return 'stable'
        return 'B-'

    geo = compute_geometric_state(Z, A)
    pf = geo.peanut_f
    eps = geo.eps

    # ── Special modes (same as v8) ──
    if geo.core_full > 1.0 and A < 50:
        return 'n'
    if geo.core_full < 0.55 and eps > 3.0 and A < 120:
        return 'p'

    # ── Three-term barrier ──
    elastic = K_SHEAR * pf ** 2
    coulomb = k_coul_scale * k_coulomb(A) * max(0.0, eps)

    # Alpha barrier
    if A >= 6 and Z >= 3:
        B_surf_alpha = bare_scission_barrier(A, 4)
        align_cost = k_align * alpha_alignment_penalty(A) if k_align > 0 else 0.0
        B_eff_alpha = max(0.0, B_surf_alpha + align_cost - elastic - coulomb)
    else:
        B_eff_alpha = 9999.0

    # SF barrier (Coulomb also applies — high-Z fissioners are overcharged)
    if A >= 200:
        A_half = A // 2
        B_surf_sf = bare_scission_barrier(A, A_half)
        B_eff_sf = max(0.0, B_surf_sf - elastic - coulomb)
    else:
        B_eff_sf = 9999.0

    # ── Beta gradients (same as v8) ──
    current = survival_score(Z, A)
    gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999.0
    gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999.0

    best_gain = max(gain_bm, gain_bp)
    best_beta = 'B-' if gain_bm >= gain_bp else 'B+'

    # ── Mode selection ──
    # SF: v8 shape gate (deep peanut → topological bifurcation)
    # SF is shape-driven, not Coulomb-driven.  The soliton splits when
    # the peanut neck thins to zero — that's a geometric inevitability
    # at very large pf, independent of charge excess.
    sf_available = (pf > PF_SF_THRESHOLD and geo.is_ee
                    and geo.core_full >= CF_SF_MIN)

    if sf_available:
        return 'SF'

    # Alpha: Coulomb barrier physics
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

    # Alpha NOT available → pure beta decision
    if best_gain > 0:
        return best_beta

    return 'stable'


# ═══════════════════════════════════════════════════════════════════════
# DZHANIBEKOV-COUPLED COULOMB — pf² gates Coulomb repulsion
# The tumbling exposes the Coulomb repulsion; without tumbling,
# surface tension fully compensates Coulomb.  See scission.md §7-8.
# ═══════════════════════════════════════════════════════════════════════

def k_coulomb_alpha(A, Z):
    """Coulomb repulsion between alpha fragment and daughter.

    For alpha: Z_α=2, Z_d=Z-2 → E_C = 2(Z-2)α/A^{1/3}
    The TOTAL Coulomb, not just the excess.  The tumbling exposes the
    full proton-proton repulsion between the separating fragments.
    """
    return 2.0 * max(0, Z - 2) * ALPHA / (A ** (1.0 / 3))


def k_coulomb_sf(A, Z):
    """Coulomb repulsion between two symmetric fission fragments.

    For SF: Z₁≈Z/2, Z₂≈Z/2 → E_C = (Z/2)²·α/A^{1/3}
    Quadratically larger than alpha — explains why SF exists at all.
    """
    z_half = Z / 2.0
    return z_half * z_half * ALPHA / (A ** (1.0 / 3))


def electron_screening(Z, n_inner=10):
    """Fraction of Coulomb repulsion surviving after electron screening.

    f_unscreen = 1 - n_inner/Z

    n_inner: number of inner-shell electrons with significant nuclear
    penetration (K + L shells ≈ 10).  These partially neutralize the
    charge field at the scission point.

    Returns value in (0, 1].  For Z >> n_inner, screening is small.
    """
    if Z <= n_inner:
        return 0.5  # floor: at least half the Coulomb survives
    return 1.0 - n_inner / Z


def predict_kinetic_coupled(Z, A, K_ELASTIC, k_coul_scale=1.0,
                            screen=False, n_inner=10):
    """Dzhanibekov-coupled barrier: tumbling gates both elastic and Coulomb.

    B_eff = max(0, B_surf - pf²·(K_ELASTIC + k_cs·K_C_frag(A,Z)·f_unscreen))

    The key physics: without tumbling (pf=0), the Coulomb repulsion is
    fully compensated by surface tension.  Tumbling thins the neck,
    uncompensating the Coulomb repulsion between the lobes.
    """
    if A < 1 or Z < 0 or A < Z:
        return 'unknown'
    if Z <= 1:
        if A <= 2:
            return 'stable'
        return 'B-'

    geo = compute_geometric_state(Z, A)
    pf = geo.peanut_f
    eps = geo.eps

    # ── Special modes (same as v8) ──
    if geo.core_full > 1.0 and A < 50:
        return 'n'
    if geo.core_full < 0.55 and eps > 3.0 and A < 120:
        return 'p'

    # ── Screening ──
    f_unscreen = electron_screening(Z, n_inner) if screen else 1.0

    # ── Alpha barrier: pf²·(elastic + Coulomb) ──
    if A >= 6 and Z >= 3:
        B_surf_alpha = bare_scission_barrier(A, 4)
        kc_alpha = k_coul_scale * k_coulomb_alpha(A, Z) * max(0.0, eps)
        discount = pf ** 2 * (K_ELASTIC + kc_alpha * f_unscreen)
        B_eff_alpha = max(0.0, B_surf_alpha - discount)
    else:
        B_eff_alpha = 9999.0

    # ── SF barrier: pf²·(elastic + Coulomb_SF) ──
    if A >= 200:
        A_half = A // 2
        B_surf_sf = bare_scission_barrier(A, A_half)
        kc_sf = k_coul_scale * k_coulomb_sf(A, Z) * max(0.0, eps)
        discount_sf = pf ** 2 * (K_ELASTIC + kc_sf * f_unscreen)
        B_eff_sf = max(0.0, B_surf_sf - discount_sf)
    else:
        B_eff_sf = 9999.0

    # ── Beta gradients (same as v8) ──
    current = survival_score(Z, A)
    gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999.0
    gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999.0

    best_gain = max(gain_bm, gain_bp)
    best_beta = 'B-' if gain_bm >= gain_bp else 'B+'

    # ── Mode selection via barrier physics ──
    alpha_available = (B_eff_alpha <= 0.0)
    sf_available = (B_eff_sf <= 0.0 and A > 250 and geo.is_ee
                    and geo.core_full >= CF_SF_MIN)

    if sf_available:
        return 'SF'

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

    # Alpha NOT available → pure beta decision
    if best_gain > 0:
        return best_beta

    return 'stable'


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data():
    """Load NuBase2020 ground-state nuclides."""
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'three-layer-lagrangian', 'data', 'clean_species_sorted.csv'
    )
    if not os.path.exists(csv_path):
        print(f"  ERROR: CSV not found at {csv_path}")
        return []

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
                continue  # GS only
            species = row.get('clean_species', '')
            mode = MODE_MAP.get(species)
            if mode is None or mode == 'IT':
                continue  # skip IT (spin physics, not topology)
            log_hl = None
            try:
                log_hl = float(row.get('log_hl', ''))
            except (ValueError, TypeError):
                pass
            data.append({
                'A': A, 'Z': Z, 'mode': mode, 'log_hl': log_hl,
                'species': species,
            })
    return data


# ═══════════════════════════════════════════════════════════════════════
# COMPARISON ENGINE
# ═══════════════════════════════════════════════════════════════════════

def run_comparison(data, predictor_fn, label):
    """Run a predictor on all data and return accuracy statistics."""
    modes_all = ['stable', 'B-', 'B+', 'alpha', 'SF', 'p', 'n']
    confusion = {am: {pm: 0 for pm in modes_all} for am in modes_all}
    correct = 0
    total = 0
    beta_correct = 0
    beta_total = 0

    for d in data:
        Z, A = d['Z'], d['A']
        actual = d['mode']
        pred = predictor_fn(Z, A)

        if actual not in confusion:
            continue
        if pred not in confusion[actual]:
            confusion[actual][pred] = 0
        confusion[actual][pred] += 1
        total += 1
        if actual == pred:
            correct += 1

        # Beta direction
        if actual in ('B-', 'B+') and pred in ('B-', 'B+', 'alpha', 'SF'):
            beta_total += 1
            if actual == 'B-' and pred in ('B-',):
                beta_correct += 1
            elif actual == 'B+' and pred in ('B+', 'alpha'):
                # alpha from B+ side is directionally correct (over-charged)
                beta_correct += 1

    return {
        'label': label,
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
        'beta_correct': beta_correct,
        'beta_total': beta_total,
        'beta_dir': beta_correct / beta_total if beta_total > 0 else 0,
        'confusion': confusion,
    }


def print_results(stats):
    """Print comparison results with confusion matrix."""
    print(f"\n  ════ {stats['label']} ════")
    print(f"  Mode accuracy:  {stats['correct']}/{stats['total']} "
          f"= {100*stats['accuracy']:.1f}%")
    if stats['beta_total'] > 0:
        print(f"  β-direction:    {stats['beta_correct']}/{stats['beta_total']} "
              f"= {100*stats['beta_dir']:.1f}%")

    modes_show = ['stable', 'B-', 'B+', 'alpha', 'SF', 'p', 'n']
    print(f"\n  {'Actual':<10s} |", end='')
    for pm in modes_show:
        print(f" {pm:>7s}", end='')
    print(f" | {'N':>5s} {'Acc':>6s}")
    print(f"  {'-'*78}")
    for am in modes_show:
        row = stats['confusion'].get(am, {})
        row_total = sum(row.values())
        if row_total == 0:
            continue
        cor = row.get(am, 0)
        print(f"  {am:<10s} |", end='')
        for pm in modes_show:
            v = row.get(pm, 0)
            print(f" {v:7d}" if v > 0 else "       ·", end='')
        print(f" | {row_total:5d} {100*cor/row_total:5.1f}%")


def print_per_mode_delta(stats_v8, stats_kin):
    """Print per-mode accuracy changes."""
    modes = ['stable', 'B-', 'B+', 'alpha', 'SF', 'p', 'n']
    print(f"\n  {'Mode':<10s} | {'v8':>6s} | {'Kinetic':>7s} | {'Delta':>7s}")
    print(f"  {'-'*45}")
    for m in modes:
        row_v8 = stats_v8['confusion'].get(m, {})
        row_kin = stats_kin['confusion'].get(m, {})
        n_v8 = sum(row_v8.values())
        n_kin = sum(row_kin.values())
        if n_v8 == 0:
            continue
        acc_v8 = 100 * row_v8.get(m, 0) / n_v8
        acc_kin = 100 * row_kin.get(m, 0) / n_kin if n_kin > 0 else 0
        delta = acc_kin - acc_v8
        marker = '▲' if delta > 1 else ('▼' if delta < -1 else ' ')
        print(f"  {m:<10s} | {acc_v8:5.1f}% | {acc_kin:6.1f}% | {delta:+6.1f}% {marker}")


# ═══════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: BARRIER LANDSCAPE
# ═══════════════════════════════════════════════════════════════════════

def print_barrier_landscape(K_SHEAR, k_align=0.0):
    """Show how the alpha barrier varies with A and pf."""
    print(f"\n  ── Alpha Barrier Landscape (K_SHEAR={K_SHEAR:.4f}) ──")
    print(f"  {'A':>5s} {'pf':>6s} {'B_surf':>7s} {'elastic':>8s} "
          f"{'align':>6s} {'B_eff':>7s} {'pf_crit':>8s}")
    print(f"  {'-'*55}")
    for A in [100, 120, 140, 160, 180, 200, 220, 240, 260, 280]:
        pf = max(0.0, (A - A_CRIT) / WIDTH)
        B_surf = bare_scission_barrier(A, 4)
        elastic = K_SHEAR * pf ** 2
        align = k_align * alpha_alignment_penalty(A) if k_align > 0 else 0.0
        B_eff = max(0.0, B_surf + align - elastic)
        pf_crit = math.sqrt((B_surf + align) / K_SHEAR) if K_SHEAR > 0 else 999
        print(f"  {A:5d} {pf:6.2f} {B_surf:7.3f} {elastic:8.3f} "
              f"{align:6.3f} {B_eff:7.3f} {pf_crit:8.3f}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def predict_decay_wrapper(Z, A):
    """Wrapper around v8 predict_decay to return just the mode string."""
    mode, _ = predict_decay(Z, A)
    return mode


if __name__ == '__main__':
    print("=" * 72)
    print("  KINETIC FRACTURE MODEL — VALIDATION")
    print("  Scission barrier + Dzhanibekov discount + valley alignment")
    print("=" * 72)

    # Load data
    data = load_data()
    print(f"\n  Loaded {len(data)} ground-state nuclides")
    from collections import Counter
    mode_counts = Counter(d['mode'] for d in data)
    for m in sorted(mode_counts):
        print(f"    {m:<10s} {mode_counts[m]:5d}")

    # ── Run v8 baseline ──
    stats_v8 = run_comparison(data, predict_decay_wrapper, "v8 GRADIENT (baseline)")
    print_results(stats_v8)

    # ── Scan K_SHEAR values ──
    print(f"\n{'='*72}")
    print(f"  K_SHEAR SCAN — finding optimal elastic constant")
    print(f"{'='*72}")

    # Candidate geometric constants
    candidates = [
        ("β/2",           BETA / 2),
        ("1",             1.0),
        ("β",             BETA),
        ("S_SURF=β²/e",   S_SURF),
        ("π",             PI),
        ("e",             E_NUM),
        ("2S_SURF",       2 * S_SURF),
        ("β²",            BETA ** 2),
        ("πβ/e",          PI * BETA / E_NUM),
        ("3S_SURF",       3 * S_SURF),
        ("2β²/e",         2 * BETA**2 / E_NUM),
        ("4S_SURF",       4 * S_SURF),
    ]

    best_acc = 0
    best_label = ""
    best_kshear = 0

    print(f"\n  {'K_SHEAR':<20s} {'Value':>8s} {'Mode%':>7s} {'β-dir%':>7s} "
          f"{'α-acc%':>7s} {'stbl%':>6s}")
    print(f"  {'-'*62}")

    for label, ks in candidates:
        def pred_fn(Z, A, _ks=ks):
            return predict_kinetic(Z, A, K_SHEAR=_ks, k_align=0.0)
        st = run_comparison(data, pred_fn, label)
        # Per-mode alpha accuracy
        alpha_row = st['confusion'].get('alpha', {})
        alpha_n = sum(alpha_row.values())
        alpha_acc = 100 * alpha_row.get('alpha', 0) / alpha_n if alpha_n > 0 else 0
        # Stable accuracy
        stbl_row = st['confusion'].get('stable', {})
        stbl_n = sum(stbl_row.values())
        stbl_acc = 100 * stbl_row.get('stable', 0) / stbl_n if stbl_n > 0 else 0

        print(f"  {label:<20s} {ks:8.4f} {100*st['accuracy']:6.1f}% "
              f"{100*st['beta_dir']:6.1f}% {alpha_acc:6.1f}% {stbl_acc:5.1f}%")

        if st['accuracy'] > best_acc:
            best_acc = st['accuracy']
            best_label = label
            best_kshear = ks

    print(f"\n  BEST: K_SHEAR = {best_label} = {best_kshear:.4f} "
          f"→ {100*best_acc:.1f}%")

    # ── Also scan with alignment penalty ──
    print(f"\n{'='*72}")
    print(f"  ALIGNMENT SCAN — K_SHEAR={best_label}, varying k_align")
    print(f"{'='*72}")

    align_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, BETA, S_SURF]
    align_labels = ['0', '0.5', '1.0', '2.0', '5.0', '10.0', 'β', 'S_SURF']

    best_align_acc = 0
    best_align_val = 0

    print(f"\n  {'k_align':<10s} {'Mode%':>7s} {'β-dir%':>7s} {'α-acc%':>7s}")
    print(f"  {'-'*38}")
    for lbl, ka in zip(align_labels, align_values):
        def pred_fn(Z, A, _ks=best_kshear, _ka=ka):
            return predict_kinetic(Z, A, K_SHEAR=_ks, k_align=_ka)
        st = run_comparison(data, pred_fn, lbl)
        alpha_row = st['confusion'].get('alpha', {})
        alpha_n = sum(alpha_row.values())
        alpha_acc = 100 * alpha_row.get('alpha', 0) / alpha_n if alpha_n > 0 else 0
        print(f"  {lbl:<10s} {100*st['accuracy']:6.1f}% "
              f"{100*st['beta_dir']:6.1f}% {alpha_acc:6.1f}%")
        if st['accuracy'] > best_align_acc:
            best_align_acc = st['accuracy']
            best_align_val = ka

    # ── Full comparison: v8 vs best kinetic ──
    print(f"\n{'='*72}")
    print(f"  FULL COMPARISON: v8 vs KINETIC FRACTURE")
    print(f"  K_SHEAR={best_label}, k_align={best_align_val:.2f}")
    print(f"{'='*72}")

    def best_pred(Z, A):
        return predict_kinetic(Z, A, K_SHEAR=best_kshear, k_align=best_align_val)

    stats_kin = run_comparison(data, best_pred, "KINETIC FRACTURE (best)")
    print_results(stats_v8)
    print_results(stats_kin)
    print_per_mode_delta(stats_v8, stats_kin)

    # ── Barrier landscape ──
    print_barrier_landscape(best_kshear, best_align_val)

    # ── Cases where kinetic wins/loses vs v8 ──
    print(f"\n  ── Sample disagreements (kinetic vs v8) ──")
    wins = []
    losses = []
    for d in data:
        Z, A = d['Z'], d['A']
        actual = d['mode']
        pred_v8, _ = predict_decay(Z, A)
        pred_kin = best_pred(Z, A)
        if pred_v8 != pred_kin:
            geo = compute_geometric_state(Z, A)
            entry = (Z, A, element_name(Z), actual, pred_v8, pred_kin,
                     geo.eps, geo.peanut_f, geo.parity)
            if pred_kin == actual and pred_v8 != actual:
                wins.append(entry)
            elif pred_v8 == actual and pred_kin != actual:
                losses.append(entry)

    print(f"\n  Kinetic WINS ({len(wins)} nuclides where kinetic is right, v8 wrong):")
    for i, (Z, A, el, act, v8, kin, eps, pf, par) in enumerate(wins[:15]):
        print(f"    {el}-{A:3d} (Z={Z:3d}) actual={act:6s} v8={v8:6s} "
              f"kin={kin:6s} ε={eps:+.2f} pf={pf:.2f} {par}")

    print(f"\n  Kinetic LOSSES ({len(losses)} nuclides where v8 is right, kinetic wrong):")
    for i, (Z, A, el, act, v8, kin, eps, pf, par) in enumerate(losses[:15]):
        print(f"    {el}-{A:3d} (Z={Z:3d}) actual={act:6s} v8={v8:6s} "
              f"kin={kin:6s} ε={eps:+.2f} pf={pf:.2f} {par}")

    print(f"\n  Net: +{len(wins)} wins, -{len(losses)} losses = "
          f"{'IMPROVEMENT' if len(wins) > len(losses) else 'REGRESSION'}")

    # ── Summary ──
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    delta = stats_kin['accuracy'] - stats_v8['accuracy']
    print(f"  v8 gradient:      {100*stats_v8['accuracy']:.1f}% mode, "
          f"{100*stats_v8['beta_dir']:.1f}% β-dir")
    print(f"  Kinetic fracture: {100*stats_kin['accuracy']:.1f}% mode, "
          f"{100*stats_kin['beta_dir']:.1f}% β-dir")
    print(f"  Delta:            {100*delta:+.1f}%")
    print(f"\n  Constants used:")
    print(f"    K_SHEAR = {best_label} = {best_kshear:.6f}")
    print(f"    k_align = {best_align_val:.4f}")
    print(f"    B_parity = 2/β = {2*PAIRING_SCALE:.6f} (ee pairing cost)")
    print(f"    B_surf = β²/e * surface_term (zero free params)")
    print(f"\n  Empirical thresholds REPLACED by barrier physics:")
    print(f"    PF_ALPHA_POSSIBLE = {PF_ALPHA_POSSIBLE} → B_eff(α) = 0 crossover")
    print(f"    PF_DEEP_PEANUT   = {PF_DEEP_PEANUT} → B_eff(α) = 0 + ee pairing gate")
    print(f"    PF_SF_THRESHOLD  = {PF_SF_THRESHOLD} → B_eff(SF) = 0 crossover")

    # ══════════════════════════════════════════════════════════════════
    # TRIAXIALITY-DEPENDENT DZHANIBEKOV BARRIER
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  TRIAXIALITY-DEPENDENT DZHANIBEKOV BARRIER")
    print(f"  T = |m|/ell: 0 = axially symmetric, 1 = maximally triaxial")
    print(f"  elastic = K_SHEAR * pf^2 * f(T)")
    print(f"{'='*72}")

    # ── Define f(T) candidates ──
    f_triax_candidates = {
        'linear':    lambda T: T,
        'quadratic': lambda T: T ** 2,
        'binary':    lambda T: 1.0 if T > 0 else 0.0,
        'sqrt':      lambda T: math.sqrt(T) if T > 0 else 0.0,
    }

    # ── Scan K_SHEAR × f(T) for each formulation ──
    triax_results = []

    for f_name, f_fn in f_triax_candidates.items():
        print(f"\n  ── f(T) = {f_name} ──")
        print(f"  {'K_SHEAR':<20s} {'Value':>8s} {'Mode%':>7s} {'β-dir%':>7s} "
              f"{'α-acc%':>7s} {'stbl%':>6s}")
        print(f"  {'-'*62}")

        best_triax_acc = 0
        best_triax_label = ""
        best_triax_ks = 0

        for label, ks in candidates:
            def pred_fn(Z, A, _ks=ks, _f=f_fn):
                return predict_kinetic_triax(Z, A, K_SHEAR=_ks,
                                             f_triax_fn=_f, k_align=0.0)
            st = run_comparison(data, pred_fn, label)
            alpha_row = st['confusion'].get('alpha', {})
            alpha_n = sum(alpha_row.values())
            alpha_acc = 100 * alpha_row.get('alpha', 0) / alpha_n if alpha_n > 0 else 0
            stbl_row = st['confusion'].get('stable', {})
            stbl_n = sum(stbl_row.values())
            stbl_acc = 100 * stbl_row.get('stable', 0) / stbl_n if stbl_n > 0 else 0

            print(f"  {label:<20s} {ks:8.4f} {100*st['accuracy']:6.1f}% "
                  f"{100*st['beta_dir']:6.1f}% {alpha_acc:6.1f}% {stbl_acc:5.1f}%")

            if st['accuracy'] > best_triax_acc:
                best_triax_acc = st['accuracy']
                best_triax_label = label
                best_triax_ks = ks

        print(f"\n  BEST ({f_name}): K_SHEAR = {best_triax_label} = "
              f"{best_triax_ks:.4f} → {100*best_triax_acc:.1f}%")
        triax_results.append((f_name, f_fn, best_triax_label, best_triax_ks,
                              best_triax_acc))

    # ── Best triaxiality formulation ──
    triax_results.sort(key=lambda x: -x[4])
    best_f_name, best_f_fn, best_f_label, best_f_ks, best_f_acc = triax_results[0]

    print(f"\n{'='*72}")
    print(f"  TRIAXIALITY SCAN SUMMARY")
    print(f"{'='*72}")
    print(f"\n  {'f(T)':<12s} {'K_SHEAR':<12s} {'Mode%':>7s}")
    print(f"  {'-'*35}")
    for f_name, _, f_label, f_ks, f_acc in triax_results:
        marker = ' ◄' if f_name == best_f_name else ''
        print(f"  {f_name:<12s} {f_label:<12s} {100*f_acc:6.1f}%{marker}")

    # ── Diagnostic: Channel landscape of kinetic losses ──
    print(f"\n{'='*72}")
    print(f"  DIAGNOSTIC: CHANNEL ASSIGNMENTS FOR KINETIC LOSSES")
    print(f"  (nuclides where flat kinetic is wrong but v8 is right)")
    print(f"{'='*72}")

    loss_channels = []
    for d in data:
        Z_d, A_d = d['Z'], d['A']
        actual = d['mode']
        pred_v8, _ = predict_decay(Z_d, A_d)
        pred_kin = predict_kinetic(Z_d, A_d, K_SHEAR=best_kshear, k_align=0.0)
        if pred_v8 == actual and pred_kin != actual:
            key = assign_channel(A_d, Z_d)
            T = triaxiality(key[0], key[1]) if key is not None else 0.0
            geo = compute_geometric_state(Z_d, A_d)
            loss_channels.append({
                'Z': Z_d, 'A': A_d, 'el': element_name(Z_d),
                'actual': actual, 'v8': pred_v8, 'kin': pred_kin,
                'key': key, 'T': T,
                'eps': geo.eps, 'pf': geo.peanut_f, 'par': geo.parity,
            })

    # Sort by T to see if losses cluster at low T
    loss_channels.sort(key=lambda x: x['T'])

    print(f"\n  Total kinetic losses: {len(loss_channels)}")

    # Count by T value
    from collections import Counter
    t_counts = Counter(f"{lc['T']:.2f}" for lc in loss_channels)
    print(f"\n  Losses by T value:")
    for t_val, cnt in sorted(t_counts.items()):
        bar = '█' * cnt
        print(f"    T={t_val}: {cnt:3d} {bar}")

    # Count losses by actual mode
    mode_counts_loss = Counter(lc['actual'] for lc in loss_channels)
    print(f"\n  Losses by actual mode:")
    for mode, cnt in mode_counts_loss.most_common():
        print(f"    {mode:<10s} {cnt:3d}")

    # Count alpha losses specifically
    alpha_losses = [lc for lc in loss_channels if lc['actual'] == 'alpha']
    print(f"\n  Alpha losses: {len(alpha_losses)}")
    if alpha_losses:
        t_alpha = Counter(f"{lc['T']:.2f}" for lc in alpha_losses)
        print(f"  Alpha losses by T:")
        for t_val, cnt in sorted(t_alpha.items()):
            print(f"    T={t_val}: {cnt:3d}")

    # Print detailed table
    print(f"\n  {'El-A':>8s} {'Z':>4s} {'actual':>7s} {'v8':>7s} {'kin':>7s} "
          f"{'(l,m,p)':>10s} {'T':>5s} {'ε':>6s} {'pf':>6s}")
    print(f"  {'-'*68}")
    for lc in loss_channels[:40]:
        key_str = f"({lc['key'][0]},{lc['key'][1]:+d},{lc['key'][2]})" if lc['key'] else "None"
        print(f"  {lc['el']}-{lc['A']:3d} {lc['Z']:4d} {lc['actual']:>7s} "
              f"{lc['v8']:>7s} {lc['kin']:>7s} {key_str:>10s} {lc['T']:5.2f} "
              f"{lc['eps']:+5.2f} {lc['pf']:5.2f}")
    if len(loss_channels) > 40:
        print(f"  ... ({len(loss_channels) - 40} more)")

    # ── Full comparison: v8 vs flat kinetic vs triaxial kinetic ──
    print(f"\n{'='*72}")
    print(f"  FULL COMPARISON: v8 vs FLAT KINETIC vs TRIAXIAL KINETIC")
    print(f"  Triaxial: f(T) = {best_f_name}, K_SHEAR = {best_f_label}")
    print(f"{'='*72}")

    def best_triax_pred(Z, A):
        return predict_kinetic_triax(Z, A, K_SHEAR=best_f_ks,
                                     f_triax_fn=best_f_fn, k_align=0.0)

    stats_triax = run_comparison(data, best_triax_pred,
                                 f"TRIAXIAL ({best_f_name}, {best_f_label})")

    print_results(stats_v8)
    print_results(stats_kin)
    print_results(stats_triax)

    # Per-mode delta: v8 vs triaxial
    print(f"\n  ── Per-mode delta: v8 → TRIAXIAL ──")
    print_per_mode_delta(stats_v8, stats_triax)

    # Per-mode delta: flat kinetic vs triaxial
    print(f"\n  ── Per-mode delta: FLAT KINETIC → TRIAXIAL ──")
    print_per_mode_delta(stats_kin, stats_triax)

    # ── Wins/losses: triaxial vs flat kinetic ──
    print(f"\n  ── Triaxial vs Flat Kinetic disagreements ──")
    triax_wins = []
    triax_losses = []
    for d in data:
        Z_d, A_d = d['Z'], d['A']
        actual = d['mode']
        pred_flat = predict_kinetic(Z_d, A_d, K_SHEAR=best_kshear, k_align=0.0)
        pred_triax = best_triax_pred(Z_d, A_d)
        if pred_flat != pred_triax:
            key = assign_channel(A_d, Z_d)
            T = triaxiality(key[0], key[1]) if key is not None else 0.0
            geo = compute_geometric_state(Z_d, A_d)
            entry = (Z_d, A_d, element_name(Z_d), actual, pred_flat, pred_triax,
                     geo.eps, geo.peanut_f, T, key)
            if pred_triax == actual and pred_flat != actual:
                triax_wins.append(entry)
            elif pred_flat == actual and pred_triax != actual:
                triax_losses.append(entry)

    print(f"\n  Triaxial WINS ({len(triax_wins)} nuclides where triaxial is right, "
          f"flat wrong):")
    for i, (Z_d, A_d, el, act, flat, tri, eps, pf, T, key) in enumerate(triax_wins[:20]):
        key_str = f"({key[0]},{key[1]:+d},{key[2]})" if key else "None"
        print(f"    {el}-{A_d:3d} (Z={Z_d:3d}) actual={act:6s} flat={flat:6s} "
              f"triax={tri:6s} T={T:.2f} {key_str} pf={pf:.2f}")

    print(f"\n  Triaxial LOSSES ({len(triax_losses)} nuclides where flat is right, "
          f"triaxial wrong):")
    for i, (Z_d, A_d, el, act, flat, tri, eps, pf, T, key) in enumerate(triax_losses[:20]):
        key_str = f"({key[0]},{key[1]:+d},{key[2]})" if key else "None"
        print(f"    {el}-{A_d:3d} (Z={Z_d:3d}) actual={act:6s} flat={flat:6s} "
              f"triax={tri:6s} T={T:.2f} {key_str} pf={pf:.2f}")

    print(f"\n  Net: +{len(triax_wins)} wins, -{len(triax_losses)} losses = "
          f"{'IMPROVEMENT' if len(triax_wins) > len(triax_losses) else 'REGRESSION'}")

    # ── Final summary ──
    print(f"\n{'='*72}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*72}")
    delta_flat = stats_kin['accuracy'] - stats_v8['accuracy']
    delta_triax = stats_triax['accuracy'] - stats_v8['accuracy']
    delta_triax_flat = stats_triax['accuracy'] - stats_kin['accuracy']

    print(f"  v8 gradient:       {100*stats_v8['accuracy']:.1f}% mode, "
          f"{100*stats_v8['beta_dir']:.1f}% β-dir")
    print(f"  Flat kinetic:      {100*stats_kin['accuracy']:.1f}% mode, "
          f"{100*stats_kin['beta_dir']:.1f}% β-dir  ({100*delta_flat:+.1f}% vs v8)")
    print(f"  Triaxial kinetic:  {100*stats_triax['accuracy']:.1f}% mode, "
          f"{100*stats_triax['beta_dir']:.1f}% β-dir  ({100*delta_triax:+.1f}% vs v8)")
    print(f"  Triax vs flat:     {100*delta_triax_flat:+.1f}%")
    print(f"\n  Best triaxiality: f(T) = {best_f_name}, K_SHEAR = {best_f_label}")
    print(f"  Channel constants: 15 FIT B (fitted to AME2020)")
    print(f"  New free parameters: 0 (T = |m|/ell is pure channel geometry)")

    # ══════════════════════════════════════════════════════════════════
    # COULOMB-ASSISTED SCISSION — Three-Term Barrier
    # B_eff = B_surf + align - K_SHEAR·pf² - K_COUL·max(0,ε)
    # See scission.md for derivation
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  COULOMB-ASSISTED SCISSION — Three-Term Barrier")
    print(f"  B_eff = B_surf + align - K_SHEAR·pf² - k·K_COUL(A)·max(0,ε)")
    print(f"  K_COUL(A) = 2·Z*(A)·α/A^(1/3)  (zero free parameters)")
    print(f"{'='*72}")

    # Show K_COUL(A) across the chart
    print(f"\n  K_COUL(A) landscape:")
    print(f"  {'A':>5s} {'Z*(A)':>7s} {'K_COUL':>8s} {'pf':>6s}")
    print(f"  {'-'*30}")
    for A_show in [100, 140, 180, 196, 200, 210, 220, 240, 260]:
        kc = k_coulomb(A_show)
        zs = z_star(A_show)
        geo_show = compute_geometric_state(round(zs), A_show)
        print(f"  {A_show:5d} {zs:7.2f} {kc:8.5f} {geo_show.peanut_f:6.2f}")

    # ── Scan k_coul_scale with best K_SHEAR from flat model ──
    print(f"\n  ── K_SHEAR = {best_label} = {best_kshear:.4f} ──")

    # Alignment scan: try a few k_align values with Coulomb
    align_opts = [0.0, 5.0, 10.0]

    for ka in align_opts:
        print(f"\n  k_align = {ka:.1f}:")
        print(f"  {'k_coul_scale':<14s} {'Mode%':>7s} {'β-dir%':>7s} "
              f"{'α-acc%':>7s} {'stbl%':>6s} {'B+%':>6s} {'SF%':>5s}")
        print(f"  {'-'*62}")

        coul_scales = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        best_coul_acc = 0
        best_coul_scale = 0

        for cs in coul_scales:
            def pred_fn(Z, A, _ks=best_kshear, _cs=cs, _ka=ka):
                return predict_kinetic_coulomb(Z, A, K_SHEAR=_ks,
                                               k_coul_scale=_cs, k_align=_ka)
            st = run_comparison(data, pred_fn, f"cs={cs}")
            alpha_row = st['confusion'].get('alpha', {})
            alpha_n = sum(alpha_row.values())
            alpha_acc = 100 * alpha_row.get('alpha', 0) / alpha_n if alpha_n > 0 else 0
            stbl_row = st['confusion'].get('stable', {})
            stbl_n = sum(stbl_row.values())
            stbl_acc = 100 * stbl_row.get('stable', 0) / stbl_n if stbl_n > 0 else 0
            bp_row = st['confusion'].get('B+', {})
            bp_n = sum(bp_row.values())
            bp_acc = 100 * bp_row.get('B+', 0) / bp_n if bp_n > 0 else 0
            sf_row = st['confusion'].get('SF', {})
            sf_n = sum(sf_row.values())
            sf_acc = 100 * sf_row.get('SF', 0) / sf_n if sf_n > 0 else 0

            marker = ''
            if st['accuracy'] > best_coul_acc:
                best_coul_acc = st['accuracy']
                best_coul_scale = cs
                best_coul_ka = ka

            print(f"  {cs:<14.1f} {100*st['accuracy']:6.1f}% "
                  f"{100*st['beta_dir']:6.1f}% {alpha_acc:6.1f}% "
                  f"{stbl_acc:5.1f}% {bp_acc:5.1f}% {sf_acc:4.1f}%")

        print(f"  BEST: k_coul_scale={best_coul_scale:.1f} → {100*best_coul_acc:.1f}%")

    # ── Also scan K_SHEAR × k_coul_scale jointly ──
    print(f"\n{'='*72}")
    print(f"  JOINT SCAN: K_SHEAR × k_coul_scale (k_align=0)")
    print(f"{'='*72}")

    best_joint_acc = 0
    best_joint_ks_label = ""
    best_joint_ks = 0
    best_joint_cs = 0

    # Narrower K_SHEAR candidates focused around the optimum
    ks_candidates_joint = [
        ("S_SURF",   S_SURF),
        ("π",        PI),
        ("e",        E_NUM),
        ("2S_SURF",  2 * S_SURF),
        ("β²",       BETA ** 2),
        ("πβ/e",     PI * BETA / E_NUM),
        ("3S_SURF",  3 * S_SURF),
        ("2β²/e",    2 * BETA**2 / E_NUM),
        ("4S_SURF",  4 * S_SURF),
    ]
    cs_candidates_joint = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

    print(f"\n  {'K_SHEAR':<12s}", end='')
    for cs in cs_candidates_joint:
        print(f" {'cs='+str(cs):>7s}", end='')
    print()
    print(f"  {'-'*12}", end='')
    for _ in cs_candidates_joint:
        print(f" {'-'*7}", end='')
    print()

    for ks_label, ks_val in ks_candidates_joint:
        print(f"  {ks_label:<12s}", end='')
        for cs in cs_candidates_joint:
            def pred_fn(Z, A, _ks=ks_val, _cs=cs):
                return predict_kinetic_coulomb(Z, A, K_SHEAR=_ks,
                                               k_coul_scale=_cs, k_align=0.0)
            st = run_comparison(data, pred_fn, f"{ks_label},cs={cs}")
            acc = 100 * st['accuracy']
            print(f" {acc:6.1f}%", end='')
            if st['accuracy'] > best_joint_acc:
                best_joint_acc = st['accuracy']
                best_joint_ks_label = ks_label
                best_joint_ks = ks_val
                best_joint_cs = cs
        print()

    print(f"\n  BEST JOINT: K_SHEAR={best_joint_ks_label}={best_joint_ks:.4f}, "
          f"k_coul_scale={best_joint_cs:.1f} → {100*best_joint_acc:.1f}%")

    # ── Full comparison: v8 vs flat vs Coulomb ──
    print(f"\n{'='*72}")
    print(f"  FULL COMPARISON: v8 vs FLAT KINETIC vs COULOMB-ASSISTED")
    print(f"  Coulomb: K_SHEAR={best_joint_ks_label}, k_coul_scale={best_joint_cs}")
    print(f"{'='*72}")

    def best_coul_pred(Z, A):
        return predict_kinetic_coulomb(Z, A, K_SHEAR=best_joint_ks,
                                       k_coul_scale=best_joint_cs, k_align=0.0)

    stats_coul = run_comparison(data, best_coul_pred,
                                f"COULOMB ({best_joint_ks_label}, cs={best_joint_cs})")

    print_results(stats_v8)
    print_results(stats_kin)
    print_results(stats_coul)

    # Per-mode deltas
    print(f"\n  ── Per-mode delta: v8 → COULOMB ──")
    print_per_mode_delta(stats_v8, stats_coul)

    print(f"\n  ── Per-mode delta: FLAT KINETIC → COULOMB ──")
    print_per_mode_delta(stats_kin, stats_coul)

    # ── Wins/losses: Coulomb vs flat kinetic ──
    print(f"\n  ── Coulomb vs Flat Kinetic disagreements ──")
    coul_wins = []
    coul_losses = []
    for d in data:
        Z_d, A_d = d['Z'], d['A']
        actual = d['mode']
        pred_flat = predict_kinetic(Z_d, A_d, K_SHEAR=best_kshear, k_align=0.0)
        pred_coul = best_coul_pred(Z_d, A_d)
        if pred_flat != pred_coul:
            geo = compute_geometric_state(Z_d, A_d)
            entry = (Z_d, A_d, element_name(Z_d), actual, pred_flat, pred_coul,
                     geo.eps, geo.peanut_f, geo.parity)
            if pred_coul == actual and pred_flat != actual:
                coul_wins.append(entry)
            elif pred_flat == actual and pred_coul != actual:
                coul_losses.append(entry)

    print(f"\n  Coulomb WINS ({len(coul_wins)} nuclides where Coulomb is right, "
          f"flat wrong):")
    for i, (Z_d, A_d, el, act, flat, coul, eps, pf, par) in enumerate(coul_wins[:25]):
        print(f"    {el}-{A_d:3d} (Z={Z_d:3d}) actual={act:6s} flat={flat:6s} "
              f"coul={coul:6s} ε={eps:+.2f} pf={pf:.2f} {par}")

    print(f"\n  Coulomb LOSSES ({len(coul_losses)} nuclides where flat is right, "
          f"Coulomb wrong):")
    for i, (Z_d, A_d, el, act, flat, coul, eps, pf, par) in enumerate(coul_losses[:25]):
        print(f"    {el}-{A_d:3d} (Z={Z_d:3d}) actual={act:6s} flat={flat:6s} "
              f"coul={coul:6s} ε={eps:+.2f} pf={pf:.2f} {par}")

    print(f"\n  Net: +{len(coul_wins)} wins, -{len(coul_losses)} losses = "
          f"{'IMPROVEMENT' if len(coul_wins) > len(coul_losses) else 'REGRESSION'}")

    # ── A=196 diagnostic: does the boundary land correctly? ──
    print(f"\n  ── A=196 Boundary Diagnostic ──")
    print(f"  {'El':>4s} {'Z':>3s} {'ε':>7s} {'pf':>6s} {'B_surf':>7s} "
          f"{'elastic':>8s} {'coulomb':>8s} {'B_eff':>7s} {'α?':>4s} "
          f"{'pred':>6s} {'actual':>7s}")
    print(f"  {'-'*80}")
    for Z_d in range(78, 92):
        A_d = 196
        geo = compute_geometric_state(Z_d, A_d)
        eps_d = geo.eps
        pf_d = geo.peanut_f
        B_surf_d = bare_scission_barrier(A_d, 4)
        elastic_d = best_joint_ks * pf_d ** 2
        coulomb_d = best_joint_cs * k_coulomb(A_d) * max(0.0, eps_d)
        B_eff_d = max(0.0, B_surf_d - elastic_d - coulomb_d)
        alpha_avail = B_eff_d <= 0.0
        pred_coul_d = best_coul_pred(Z_d, A_d)
        actual_d = next((d['mode'] for d in data
                         if d['Z'] == Z_d and d['A'] == A_d), '?')
        marker = ' ✓' if pred_coul_d == actual_d else ' ✗'
        el = element_name(Z_d)
        print(f"  {el:>4s} {Z_d:3d} {eps_d:+7.2f} {pf_d:6.2f} {B_surf_d:7.3f} "
              f"{elastic_d:8.3f} {coulomb_d:8.3f} {B_eff_d:7.3f} "
              f"{'YES' if alpha_avail else 'no':>4s} {pred_coul_d:>6s} "
              f"{actual_d:>7s}{marker}")

    # ── Final summary ──
    print(f"\n{'='*72}")
    print(f"  GRAND SUMMARY")
    print(f"{'='*72}")
    delta_flat_v8 = stats_kin['accuracy'] - stats_v8['accuracy']
    delta_coul_v8 = stats_coul['accuracy'] - stats_v8['accuracy']
    delta_coul_flat = stats_coul['accuracy'] - stats_kin['accuracy']

    print(f"  v8 gradient:       {100*stats_v8['accuracy']:.1f}% mode, "
          f"{100*stats_v8['beta_dir']:.1f}% β-dir")
    print(f"  Flat kinetic:      {100*stats_kin['accuracy']:.1f}% mode, "
          f"{100*stats_kin['beta_dir']:.1f}% β-dir  ({100*delta_flat_v8:+.1f}% vs v8)")
    print(f"  Coulomb-assisted:  {100*stats_coul['accuracy']:.1f}% mode, "
          f"{100*stats_coul['beta_dir']:.1f}% β-dir  ({100*delta_coul_v8:+.1f}% vs v8)")
    print(f"  Coulomb vs flat:   {100*delta_coul_flat:+.1f}%")
    print(f"\n  Barrier: B_eff = B_surf - {best_joint_ks_label}·pf² "
          f"- {best_joint_cs:.1f}·K_COUL(A)·max(0,ε)")
    print(f"  K_COUL(A) = 2·Z*(A)·α/A^{{1/3}}")
    print(f"  Free parameters from Coulomb term: "
          f"{'0 (k_coul_scale=1.0 = pure theory)' if best_joint_cs == 1.0 else f'{best_joint_cs:.1f} (geometric search needed)'}")

    # ══════════════════════════════════════════════════════════════════
    # DZHANIBEKOV-COUPLED COULOMB — pf² gates BOTH elastic and Coulomb
    # B_eff = B_surf - pf²·(K_ELASTIC + K_COUL_frag·max(0,ε)·f_unscreen)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  DZHANIBEKOV-COUPLED COULOMB SCISSION")
    print(f"  B_eff = B_surf - pf²·(K_ELASTIC + k·K_C(A,Z)·ε·f_unscreen)")
    print(f"  Tumbling gates Coulomb; electrons screen it")
    print(f"{'='*72}")

    # ── Joint scan: K_ELASTIC × k_coul_scale, no screening ──
    print(f"\n  ── COUPLED (no screening) ──")

    ke_candidates = [
        ("1",       1.0),
        ("β/2",     BETA / 2),
        ("S_SURF",  S_SURF),
        ("π",       PI),
        ("e",       E_NUM),
        ("β",       BETA),
        ("2S_SURF", 2 * S_SURF),
    ]
    cs_coupled = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    best_coupled_acc = 0
    best_coupled_ke_label = ""
    best_coupled_ke = 0
    best_coupled_cs = 0

    print(f"\n  {'K_ELASTIC':<12s}", end='')
    for cs in cs_coupled:
        print(f" {'cs='+str(cs):>7s}", end='')
    print()
    print(f"  {'-'*12}", end='')
    for _ in cs_coupled:
        print(f" {'-'*7}", end='')
    print()

    for ke_label, ke_val in ke_candidates:
        print(f"  {ke_label:<12s}", end='')
        for cs in cs_coupled:
            def pred_fn(Z, A, _ke=ke_val, _cs=cs):
                return predict_kinetic_coupled(Z, A, K_ELASTIC=_ke,
                                               k_coul_scale=_cs, screen=False)
            st = run_comparison(data, pred_fn, f"{ke_label},cs={cs}")
            acc = 100 * st['accuracy']
            print(f" {acc:6.1f}%", end='')
            if st['accuracy'] > best_coupled_acc:
                best_coupled_acc = st['accuracy']
                best_coupled_ke_label = ke_label
                best_coupled_ke = ke_val
                best_coupled_cs = cs
        print()

    print(f"\n  BEST COUPLED (no screen): K_ELASTIC={best_coupled_ke_label}"
          f"={best_coupled_ke:.4f}, k_coul={best_coupled_cs} "
          f"→ {100*best_coupled_acc:.1f}%")

    # ── Same scan WITH screening ──
    print(f"\n  ── COUPLED + SCREENED (n_inner=10) ──")

    best_screened_acc = 0
    best_screened_ke_label = ""
    best_screened_ke = 0
    best_screened_cs = 0

    print(f"\n  {'K_ELASTIC':<12s}", end='')
    for cs in cs_coupled:
        print(f" {'cs='+str(cs):>7s}", end='')
    print()
    print(f"  {'-'*12}", end='')
    for _ in cs_coupled:
        print(f" {'-'*7}", end='')
    print()

    for ke_label, ke_val in ke_candidates:
        print(f"  {ke_label:<12s}", end='')
        for cs in cs_coupled:
            def pred_fn(Z, A, _ke=ke_val, _cs=cs):
                return predict_kinetic_coupled(Z, A, K_ELASTIC=_ke,
                                               k_coul_scale=_cs,
                                               screen=True, n_inner=10)
            st = run_comparison(data, pred_fn, f"{ke_label},cs={cs}")
            acc = 100 * st['accuracy']
            print(f" {acc:6.1f}%", end='')
            if st['accuracy'] > best_screened_acc:
                best_screened_acc = st['accuracy']
                best_screened_ke_label = ke_label
                best_screened_ke = ke_val
                best_screened_cs = cs
        print()

    print(f"\n  BEST COUPLED (screened): K_ELASTIC={best_screened_ke_label}"
          f"={best_screened_ke:.4f}, k_coul={best_screened_cs} "
          f"→ {100*best_screened_acc:.1f}%")

    # ── Full comparison across all models ──
    # Pick the best coupled model (screened or not)
    if best_screened_acc > best_coupled_acc:
        final_ke = best_screened_ke
        final_ke_label = best_screened_ke_label
        final_cs = best_screened_cs
        final_screen = True
        final_label = f"COUPLED+SCREENED ({final_ke_label}, cs={final_cs})"
    else:
        final_ke = best_coupled_ke
        final_ke_label = best_coupled_ke_label
        final_cs = best_coupled_cs
        final_screen = False
        final_label = f"COUPLED ({final_ke_label}, cs={final_cs})"

    def final_pred(Z, A):
        return predict_kinetic_coupled(Z, A, K_ELASTIC=final_ke,
                                       k_coul_scale=final_cs,
                                       screen=final_screen, n_inner=10)

    stats_coupled = run_comparison(data, final_pred, final_label)

    print(f"\n{'='*72}")
    print(f"  ALL MODELS COMPARISON")
    print(f"{'='*72}")
    print_results(stats_v8)
    print_results(stats_coul)
    print_results(stats_coupled)

    print(f"\n  ── Per-mode delta: ADDITIVE COULOMB → COUPLED ──")
    print_per_mode_delta(stats_coul, stats_coupled)

    # ── A=196 diagnostic for coupled model ──
    print(f"\n  ── A=196 Boundary: COUPLED model ──")
    print(f"  {'El':>4s} {'Z':>3s} {'ε':>7s} {'pf':>6s} {'B_surf':>7s} "
          f"{'pf²·K_E':>8s} {'pf²·K_C':>8s} {'f_scr':>6s} {'B_eff':>7s} "
          f"{'α?':>4s} {'pred':>6s} {'actual':>7s}")
    print(f"  {'-'*90}")
    for Z_d in range(78, 92):
        A_d = 196
        geo = compute_geometric_state(Z_d, A_d)
        eps_d = geo.eps
        pf_d = geo.peanut_f
        B_surf_d = bare_scission_barrier(A_d, 4)
        f_unscr = electron_screening(Z_d, 10) if final_screen else 1.0
        kc_a = final_cs * k_coulomb_alpha(A_d, Z_d) * max(0.0, eps_d)
        disc = pf_d ** 2 * (final_ke + kc_a * f_unscr)
        B_eff_d = max(0.0, B_surf_d - disc)
        alpha_avail = B_eff_d <= 0.0
        pred_d = final_pred(Z_d, A_d)
        actual_d = next((d['mode'] for d in data
                         if d['Z'] == Z_d and d['A'] == A_d), '?')
        marker = ' ✓' if pred_d == actual_d else ' ✗'
        el = element_name(Z_d)
        print(f"  {el:>4s} {Z_d:3d} {eps_d:+7.2f} {pf_d:6.2f} {B_surf_d:7.3f} "
              f"{pf_d**2*final_ke:8.3f} {pf_d**2*kc_a*f_unscr:8.3f} "
              f"{f_unscr:6.3f} {B_eff_d:7.3f} "
              f"{'YES' if alpha_avail else 'no':>4s} {pred_d:>6s} "
              f"{actual_d:>7s}{marker}")

    # ══════════════════════════════════════════════════════════════════
    # PERTURBATION SPECTRUM MODEL — Alpha vs SF rate competition
    # Small ΔE → alpha (frequent); Large ΔE → SF (rare)
    # Alpha wins when ε is large (strong driving force).
    # SF wins at deep peanut when ε is low (alpha too slow).
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  PERTURBATION SPECTRUM MODEL")
    print(f"  Small ΔE → alpha (frequent);  Large ΔE → SF (rare)")
    print(f"  Alpha priority; SF wins only at deep peanut + low ε")
    print(f"{'='*72}")

    def predict_perturbation(Z, A, K_SHEAR, k_coul_scale=1.0, eps_sf_crit=3.0):
        """Perturbation-spectrum barrier model.

        Two perturbation types:
          Small ΔE → alpha (barrier-based, frequent)
          Large ΔE → SF (shape-based, rare)

        Alpha has priority (small perturbations more frequent).
        SF wins only when:
          (a) pf > PF_SF_THRESHOLD (topology at bifurcation), AND
          (b) ε < eps_sf_crit (alpha driving force too weak to dominate)
          (c) even-even and core full (same as v8)
        """
        if A < 1 or Z < 0 or A < Z:
            return 'unknown'
        if Z <= 1:
            return 'stable' if A <= 2 else 'B-'

        geo = compute_geometric_state(Z, A)
        pf = geo.peanut_f
        eps = geo.eps

        # ── Special modes (same as v8) ──
        if geo.core_full > 1.0 and A < 50:
            return 'n'
        if geo.core_full < 0.55 and eps > 3.0 and A < 120:
            return 'p'

        # ── Alpha barrier (three-term: surface - elastic - Coulomb) ──
        elastic = K_SHEAR * pf ** 2
        coulomb = k_coul_scale * k_coulomb(A) * max(0.0, eps)

        if A >= 6 and Z >= 3:
            B_surf_alpha = bare_scission_barrier(A, 4)
            B_eff_alpha = max(0.0, B_surf_alpha - elastic - coulomb)
        else:
            B_eff_alpha = 9999.0

        alpha_available = (B_eff_alpha <= 0.0)

        # ── SF: topology at bifurcation + weak alpha driving force ──
        # SF is spontaneous at very deep peanut (neck → 0).
        # But alpha, being more frequent, wins UNLESS ε is too low
        # for the Coulomb driving force to push alpha fast enough.
        sf_topology = (pf > PF_SF_THRESHOLD and geo.is_ee
                       and geo.core_full >= CF_SF_MIN)
        alpha_driving_weak = (eps < eps_sf_crit)
        sf_wins = sf_topology and alpha_driving_weak

        # ── Beta gradients (same as v8) ──
        current = survival_score(Z, A)
        gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999.0
        gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999.0

        best_gain = max(gain_bm, gain_bp)
        best_beta = 'B-' if gain_bm >= gain_bp else 'B+'

        # ── Mode selection: perturbation spectrum priority ──
        # SF spontaneous (rare, large ΔE) wins only when alpha is slow
        if sf_wins:
            return 'SF'

        # Alpha (frequent, small ΔE) wins when barrier is open
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

        # Both blocked → beta
        if best_gain > 0:
            return best_beta

        return 'stable'

    # ── Scan eps_sf_crit values ──
    eps_crit_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

    print(f"\n  ── K_SHEAR=π, k_coul_scale=4.0, scanning eps_sf_crit ──")
    print(f"  {'ε_crit':<8s} {'Mode%':>7s} {'β-dir%':>7s} "
          f"{'α-acc%':>7s} {'SF%':>5s} {'stbl%':>6s}")
    print(f"  {'-'*48}")

    best_perturb_acc = 0
    best_eps_crit = 0

    for ec in eps_crit_values:
        def pred_fn(Z, A, _ec=ec):
            return predict_perturbation(Z, A, K_SHEAR=PI,
                                         k_coul_scale=4.0,
                                         eps_sf_crit=_ec)
        st = run_comparison(data, pred_fn, f"ec={ec}")
        alpha_row = st['confusion'].get('alpha', {})
        alpha_n = sum(alpha_row.values())
        alpha_acc = 100 * alpha_row.get('alpha', 0) / alpha_n if alpha_n > 0 else 0
        sf_row = st['confusion'].get('SF', {})
        sf_n = sum(sf_row.values())
        sf_acc = 100 * sf_row.get('SF', 0) / sf_n if sf_n > 0 else 0
        stbl_row = st['confusion'].get('stable', {})
        stbl_n = sum(stbl_row.values())
        stbl_acc = 100 * stbl_row.get('stable', 0) / stbl_n if stbl_n > 0 else 0

        marker = ''
        if st['accuracy'] > best_perturb_acc:
            best_perturb_acc = st['accuracy']
            best_eps_crit = ec
            marker = ' ◄'

        print(f"  {ec:<8.1f} {100*st['accuracy']:6.1f}% "
              f"{100*st['beta_dir']:6.1f}% {alpha_acc:6.1f}% "
              f"{sf_acc:4.1f}% {stbl_acc:5.1f}%{marker}")

    print(f"\n  BEST: eps_sf_crit={best_eps_crit:.1f} → {100*best_perturb_acc:.1f}%")

    # ── Joint scan: K_SHEAR × k_coul_scale × eps_sf_crit ──
    print(f"\n  ── Joint scan: K_SHEAR × cs, eps_sf_crit={best_eps_crit:.1f} ──")

    best_perturb_joint_acc = 0
    best_perturb_ks_label = ""
    best_perturb_ks = 0
    best_perturb_cs = 0

    ks_short = [("S_SURF", S_SURF), ("π", PI), ("e", E_NUM),
                ("2S_SURF", 2*S_SURF), ("β²", BETA**2)]
    cs_short = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

    print(f"\n  {'K_SHEAR':<12s}", end='')
    for cs in cs_short:
        print(f" {'cs='+str(cs):>7s}", end='')
    print()
    print(f"  {'-'*12}", end='')
    for _ in cs_short:
        print(f" {'-'*7}", end='')
    print()

    for ks_label, ks_val in ks_short:
        print(f"  {ks_label:<12s}", end='')
        for cs in cs_short:
            def pred_fn(Z, A, _ks=ks_val, _cs=cs, _ec=best_eps_crit):
                return predict_perturbation(Z, A, K_SHEAR=_ks,
                                             k_coul_scale=_cs,
                                             eps_sf_crit=_ec)
            st = run_comparison(data, pred_fn, f"{ks_label},cs={cs}")
            acc = 100 * st['accuracy']
            print(f" {acc:6.1f}%", end='')
            if st['accuracy'] > best_perturb_joint_acc:
                best_perturb_joint_acc = st['accuracy']
                best_perturb_ks_label = ks_label
                best_perturb_ks = ks_val
                best_perturb_cs = cs
        print()

    print(f"\n  BEST JOINT: K_SHEAR={best_perturb_ks_label}={best_perturb_ks:.4f}, "
          f"cs={best_perturb_cs:.1f}, eps_crit={best_eps_crit:.1f} "
          f"→ {100*best_perturb_joint_acc:.1f}%")

    # ── Full comparison: all models ──
    def best_perturb_pred(Z, A):
        return predict_perturbation(Z, A, K_SHEAR=best_perturb_ks,
                                     k_coul_scale=best_perturb_cs,
                                     eps_sf_crit=best_eps_crit)

    stats_perturb = run_comparison(data, best_perturb_pred,
                                    f"PERTURBATION ({best_perturb_ks_label}, "
                                    f"cs={best_perturb_cs}, ec={best_eps_crit})")

    print_results(stats_perturb)

    print(f"\n  ── Per-mode delta: v8 → PERTURBATION ──")
    print_per_mode_delta(stats_v8, stats_perturb)

    print(f"\n  ── Per-mode delta: ADDITIVE COULOMB → PERTURBATION ──")
    print_per_mode_delta(stats_coul, stats_perturb)

    # ── SF diagnostic: what do the SF nuclides look like? ──
    print(f"\n  ── SF Nuclide Diagnostic ──")
    print(f"  {'El-A':>8s} {'Z':>4s} {'ε':>6s} {'pf':>6s} {'ee?':>4s} "
          f"{'actual':>7s} {'perturb':>8s} {'addCoul':>8s} {'v8':>7s}")
    print(f"  {'-'*70}")
    sf_nuclides = [d for d in data if d['mode'] == 'SF']
    for d in sorted(sf_nuclides, key=lambda x: (x['A'], x['Z'])):
        Z_d, A_d = d['Z'], d['A']
        geo = compute_geometric_state(Z_d, A_d)
        pred_p = best_perturb_pred(Z_d, A_d)
        pred_c = best_coul_pred(Z_d, A_d)
        pred_v8_d, _ = predict_decay(Z_d, A_d)
        ee = 'ee' if geo.is_ee else '  '
        m_p = '✓' if pred_p == 'SF' else '✗'
        m_c = '✓' if pred_c == 'SF' else '✗'
        m_v = '✓' if pred_v8_d == 'SF' else '✗'
        print(f"  {element_name(Z_d)}-{A_d:3d} {Z_d:4d} {geo.eps:+5.2f} "
              f"{geo.peanut_f:5.2f} {ee:>4s} {'SF':>7s} "
              f"{pred_p:>6s} {m_p} {pred_c:>6s} {m_c} {pred_v8_d:>6s} {m_v}")

    # ── Grand final summary ──
    print(f"\n{'='*72}")
    print(f"  GRAND FINAL SUMMARY")
    print(f"{'='*72}")
    models = [
        ("v8 gradient",        stats_v8),
        ("Flat kinetic",       stats_kin),
        ("Additive Coulomb",   stats_coul),
        ("Coupled Coulomb",    stats_coupled),
        ("Perturbation",       stats_perturb),
    ]
    for name, st in models:
        alpha_row = st['confusion'].get('alpha', {})
        alpha_n = sum(alpha_row.values())
        alpha_acc = 100 * alpha_row.get('alpha', 0) / alpha_n if alpha_n > 0 else 0
        sf_row = st['confusion'].get('SF', {})
        sf_n = sum(sf_row.values())
        sf_acc = 100 * sf_row.get('SF', 0) / sf_n if sf_n > 0 else 0
        delta = st['accuracy'] - stats_v8['accuracy']
        print(f"  {name:<22s} {100*st['accuracy']:5.1f}% mode  "
              f"{100*st['beta_dir']:5.1f}% β-dir  "
              f"{alpha_acc:5.1f}% α  {sf_acc:4.1f}% SF  "
              f"({100*delta:+.1f}%)")

    print(f"\n  Best coupled: K_ELASTIC={final_ke_label}, "
          f"k_coul_scale={final_cs}, screen={'yes (n=10)' if final_screen else 'no'}")
    print(f"  Barrier: B_eff = B_surf - pf²·(K_ELASTIC + k·K_C(A,Z)·ε"
          f"{'·f_screen)' if final_screen else ')'}")

    # ══════════════════════════════════════════════════════════════════
    # MULTIPLICATIVE DZHANIBEKOV BARRIER (from channel geometry)
    # barrier_eff = base_barrier * (1 - dzh_coupling * T * clip(pf,0,1))
    # T = triaxiality from (ell, m) channel; ell<2 or m=0 → no discount
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  MULTIPLICATIVE DZHANIBEKOV BARRIER + PERTURBATION SPECTRUM")
    print(f"  Dynamic barrier: B_eff = B_surf · (1 - dzh · T · clip(pf))")
    print(f"  Alpha priority; SF wins at deep peanut + low ε")
    print(f"{'='*72}")

    def get_dzhanibekov_barrier(ell, m, base_barrier, pf, dzh_coupling=0.35):
        """Multiplicative Dzhanibekov barrier from channel geometry.

        Requires:
          - ell >= 2 (quadrupole or higher — spheres/dipoles can't tumble)
          - m != 0 (axially symmetric shapes have I1=I2, no intermediate axis)

        The instability reduces the barrier multiplicatively:
          B_eff = base_barrier * (1 - dzh_coupling * T * clip(pf, 0, 1))

        where T = |m|/ell is the triaxiality (0=axial, 1=maximal).
        """
        if ell < 2 or m == 0:
            return base_barrier
        T = abs(m) / ell
        effective_pf = max(0.0, min(pf, 1.0))
        drop = dzh_coupling * T * effective_pf
        return base_barrier * (1.0 - drop)

    def predict_mult_dzh(Z, A, dzh_coupling=0.35, k_coul_scale=4.0,
                         eps_sf_crit=3.0):
        """Multiplicative Dzhanibekov barrier + Coulomb + perturbation spectrum.

        Alpha barrier:
          B_eff_alpha = B_surf(A,4) * (1 - dzh*T*clip(pf)) - K_COUL(A)*max(0,ε)
          (Coulomb remains additive — it's a separate force, not shape-dependent)

        SF: perturbation spectrum — wins only when:
          (a) deep peanut (v8 gate)
          (b) weak alpha driving force (ε < eps_sf_crit)
          (c) even-even and core full
        """
        if A < 1 or Z < 0 or A < Z:
            return 'unknown'
        if Z <= 1:
            return 'stable' if A <= 2 else 'B-'

        geo = compute_geometric_state(Z, A)
        pf = geo.peanut_f
        eps = geo.eps

        # ── Special modes (same as v8) ──
        if geo.core_full > 1.0 and A < 50:
            return 'n'
        if geo.core_full < 0.55 and eps > 3.0 and A < 120:
            return 'p'

        # ── Channel assignment for Dzhanibekov geometry ──
        key = assign_channel(A, Z)
        ell, m_ch = (key[0], key[1]) if key is not None else (0, 0)

        # ── Alpha barrier: multiplicative Dzhanibekov + additive Coulomb ──
        if A >= 6 and Z >= 3:
            B_surf_alpha = bare_scission_barrier(A, 4)
            B_dzh_alpha = get_dzhanibekov_barrier(
                ell, m_ch, B_surf_alpha, pf, dzh_coupling)
            coulomb = k_coul_scale * k_coulomb(A) * max(0.0, eps)
            B_eff_alpha = max(0.0, B_dzh_alpha - coulomb)
        else:
            B_eff_alpha = 9999.0

        alpha_available = (B_eff_alpha <= 0.0)

        # ── SF: perturbation spectrum + topology gate ──
        sf_topology = (pf > PF_SF_THRESHOLD and geo.is_ee
                       and geo.core_full >= CF_SF_MIN)
        alpha_driving_weak = (eps < eps_sf_crit)
        sf_wins = sf_topology and alpha_driving_weak

        # ── Beta gradients (same as v8) ──
        current = survival_score(Z, A)
        gain_bm = survival_score(Z + 1, A) - current if Z + 1 <= A else -9999.0
        gain_bp = survival_score(Z - 1, A) - current if Z >= 1 else -9999.0
        best_gain = max(gain_bm, gain_bp)
        best_beta = 'B-' if gain_bm >= gain_bp else 'B+'

        # ── Mode selection ──
        if sf_wins:
            return 'SF'
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

    # ── Scan dzh_coupling × k_coul_scale × eps_sf_crit ──
    # First: scan dzh_coupling with best Coulomb settings from additive model
    print(f"\n  ── Scan dzh_coupling (cs=4.0, eps_crit=best from perturbation) ──")
    dzh_values = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"  {'dzh':>6s} {'Mode%':>7s} {'β-dir%':>7s} "
          f"{'α-acc%':>7s} {'SF%':>5s} {'stbl%':>6s} {'B+%':>5s}")
    print(f"  {'-'*55}")

    best_dzh_acc = 0
    best_dzh = 0.0

    for dzh in dzh_values:
        def pred_fn(Z, A, _d=dzh, _ec=best_eps_crit):
            return predict_mult_dzh(Z, A, dzh_coupling=_d,
                                     k_coul_scale=4.0, eps_sf_crit=_ec)
        st = run_comparison(data, pred_fn, f"dzh={dzh}")
        alpha_row = st['confusion'].get('alpha', {})
        alpha_n = sum(alpha_row.values())
        alpha_acc = 100 * alpha_row.get('alpha', 0) / alpha_n if alpha_n > 0 else 0
        sf_row = st['confusion'].get('SF', {})
        sf_n = sum(sf_row.values())
        sf_acc = 100 * sf_row.get('SF', 0) / sf_n if sf_n > 0 else 0
        stbl_row = st['confusion'].get('stable', {})
        stbl_n = sum(stbl_row.values())
        stbl_acc = 100 * stbl_row.get('stable', 0) / stbl_n if stbl_n > 0 else 0
        bp_row = st['confusion'].get('B+', {})
        bp_n = sum(bp_row.values())
        bp_acc = 100 * bp_row.get('B+', 0) / bp_n if bp_n > 0 else 0

        marker = ''
        if st['accuracy'] > best_dzh_acc:
            best_dzh_acc = st['accuracy']
            best_dzh = dzh
            marker = ' ◄'

        print(f"  {dzh:6.2f} {100*st['accuracy']:6.1f}% "
              f"{100*st['beta_dir']:6.1f}% {alpha_acc:6.1f}% "
              f"{sf_acc:4.1f}% {stbl_acc:5.1f}% {bp_acc:4.1f}%{marker}")

    print(f"\n  BEST: dzh_coupling={best_dzh:.2f} → {100*best_dzh_acc:.1f}%")

    # ── Now scan eps_sf_crit with best dzh ──
    print(f"\n  ── Scan eps_sf_crit (dzh={best_dzh:.2f}, cs=4.0) ──")
    print(f"  {'ε_crit':<8s} {'Mode%':>7s} {'α-acc%':>7s} {'SF%':>5s}")
    print(f"  {'-'*35}")

    best_dzh_ec_acc = 0
    best_dzh_ec = 0

    for ec in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
        def pred_fn(Z, A, _d=best_dzh, _ec=ec):
            return predict_mult_dzh(Z, A, dzh_coupling=_d,
                                     k_coul_scale=4.0, eps_sf_crit=_ec)
        st = run_comparison(data, pred_fn, f"ec={ec}")
        alpha_row = st['confusion'].get('alpha', {})
        alpha_n = sum(alpha_row.values())
        alpha_acc = 100 * alpha_row.get('alpha', 0) / alpha_n if alpha_n > 0 else 0
        sf_row = st['confusion'].get('SF', {})
        sf_n = sum(sf_row.values())
        sf_acc = 100 * sf_row.get('SF', 0) / sf_n if sf_n > 0 else 0

        marker = ''
        if st['accuracy'] > best_dzh_ec_acc:
            best_dzh_ec_acc = st['accuracy']
            best_dzh_ec = ec
            marker = ' ◄'

        print(f"  {ec:<8.1f} {100*st['accuracy']:6.1f}% "
              f"{alpha_acc:6.1f}% {sf_acc:4.1f}%{marker}")

    print(f"\n  BEST: eps_sf_crit={best_dzh_ec:.1f}, dzh={best_dzh:.2f} "
          f"→ {100*best_dzh_ec_acc:.1f}%")

    # ── Joint scan: dzh × cs (with best eps_sf_crit) ──
    print(f"\n  ── Joint scan: dzh × cs (eps_crit={best_dzh_ec:.1f}) ──")

    dzh_short = [0.0, 0.2, 0.35, 0.5, 0.7, 1.0]
    cs_dzh = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

    best_mdzh_acc = 0
    best_mdzh_dzh = 0
    best_mdzh_cs = 0

    print(f"\n  {'dzh':>6s}", end='')
    for cs in cs_dzh:
        print(f" {'cs='+str(cs):>7s}", end='')
    print()
    print(f"  {'-'*6}", end='')
    for _ in cs_dzh:
        print(f" {'-'*7}", end='')
    print()

    for dzh in dzh_short:
        print(f"  {dzh:6.2f}", end='')
        for cs in cs_dzh:
            def pred_fn(Z, A, _d=dzh, _cs=cs, _ec=best_dzh_ec):
                return predict_mult_dzh(Z, A, dzh_coupling=_d,
                                         k_coul_scale=_cs, eps_sf_crit=_ec)
            st = run_comparison(data, pred_fn, f"dzh={dzh},cs={cs}")
            acc = 100 * st['accuracy']
            print(f" {acc:6.1f}%", end='')
            if st['accuracy'] > best_mdzh_acc:
                best_mdzh_acc = st['accuracy']
                best_mdzh_dzh = dzh
                best_mdzh_cs = cs
        print()

    print(f"\n  BEST JOINT: dzh={best_mdzh_dzh:.2f}, cs={best_mdzh_cs:.1f}, "
          f"ec={best_dzh_ec:.1f} → {100*best_mdzh_acc:.1f}%")

    # ── Full comparison with multiplicative model ──
    def best_mdzh_pred(Z, A):
        return predict_mult_dzh(Z, A, dzh_coupling=best_mdzh_dzh,
                                 k_coul_scale=best_mdzh_cs,
                                 eps_sf_crit=best_dzh_ec)

    stats_mdzh = run_comparison(data, best_mdzh_pred,
                                 f"MULT-DZH (dzh={best_mdzh_dzh:.2f}, "
                                 f"cs={best_mdzh_cs}, ec={best_dzh_ec})")
    print_results(stats_mdzh)

    print(f"\n  ── Per-mode delta: v8 → MULT-DZH ──")
    print_per_mode_delta(stats_v8, stats_mdzh)

    # ── Channel diagnostic: which channels benefit from Dzhanibekov? ──
    print(f"\n  ── Channel Diagnostic: Mult-DZH vs Additive Coulomb ──")
    mdzh_wins = []
    mdzh_losses = []
    for d in data:
        Z_d, A_d = d['Z'], d['A']
        actual = d['mode']
        pred_ac = best_coul_pred(Z_d, A_d)
        pred_md = best_mdzh_pred(Z_d, A_d)
        if pred_ac != pred_md:
            key = assign_channel(A_d, Z_d)
            T = triaxiality(key[0], key[1]) if key is not None else 0.0
            geo = compute_geometric_state(Z_d, A_d)
            entry = (Z_d, A_d, element_name(Z_d), actual, pred_ac, pred_md,
                     geo.eps, geo.peanut_f, T, key)
            if pred_md == actual and pred_ac != actual:
                mdzh_wins.append(entry)
            elif pred_ac == actual and pred_md != actual:
                mdzh_losses.append(entry)

    print(f"\n  MULT-DZH WINS ({len(mdzh_wins)}):")
    for Z_d, A_d, el, act, ac, md, eps, pf, T, key in mdzh_wins[:20]:
        key_str = f"({key[0]},{key[1]:+d},{key[2]})" if key else "None"
        print(f"    {el}-{A_d:3d} (Z={Z_d:3d}) actual={act:6s} addCoul={ac:6s} "
              f"mDzh={md:6s} T={T:.2f} {key_str} pf={pf:.2f} ε={eps:+.2f}")

    print(f"\n  MULT-DZH LOSSES ({len(mdzh_losses)}):")
    for Z_d, A_d, el, act, ac, md, eps, pf, T, key in mdzh_losses[:20]:
        key_str = f"({key[0]},{key[1]:+d},{key[2]})" if key else "None"
        print(f"    {el}-{A_d:3d} (Z={Z_d:3d}) actual={act:6s} addCoul={ac:6s} "
              f"mDzh={md:6s} T={T:.2f} {key_str} pf={pf:.2f} ε={eps:+.2f}")

    # ══════════════════════════════════════════════════════════════════
    # HARMONIC MODE DIAGNOSTIC
    # N = A (mass number = topological winding number)
    # Odd A → forced asymmetric fission → already captured by is_ee gate
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  HARMONIC MODE DIAGNOSTIC")
    print(f"  N_topo = A (mass number = winding number)")
    print(f"  Odd A → forced asymmetric fission (Lean theorem)")
    print(f"{'='*72}")

    # Verify: all SF nuclides have even A and is_ee
    sf_data = [d for d in data if d['mode'] == 'SF']
    odd_a_sf = [d for d in sf_data if d['A'] % 2 == 1]
    even_a_sf = [d for d in sf_data if d['A'] % 2 == 0]
    print(f"\n  SF nuclides: {len(sf_data)} total")
    print(f"    Even A: {len(even_a_sf)} ({100*len(even_a_sf)/len(sf_data):.0f}%)")
    print(f"    Odd A:  {len(odd_a_sf)} ({100*len(odd_a_sf)/len(sf_data):.0f}%)")
    if odd_a_sf:
        print(f"    Odd-A SF:")
        for d in odd_a_sf:
            print(f"      {element_name(d['Z'])}-{d['A']} Z={d['Z']}")
    else:
        print(f"    → ALL SF nuclides have even A (consistent with asymmetry lock)")
        print(f"    → is_ee gate already captures this topology")

    # ── Peanut asymmetry diagnostic ──
    print(f"\n  ── Peanut Asymmetry: (β/2)·(A-2Z)²/A ──")
    print(f"  {'El-A':>8s} {'Z':>4s} {'ε':>6s} {'pf':>6s} {'N-Z':>5s} "
          f"{'(N-Z)²/A':>9s} {'E_pea':>7s} {'actual':>7s} {'ee?':>4s}")
    print(f"  {'-'*66}")
    for d in sorted(sf_data, key=lambda x: x['A']):
        Z_d, A_d = d['Z'], d['A']
        geo = compute_geometric_state(Z_d, A_d)
        N_d = A_d - Z_d
        nz = N_d - Z_d  # neutron excess
        nz2_A = nz**2 / A_d
        E_pea = (BETA / 2) * nz2_A
        ee = 'ee' if geo.is_ee else '  '
        print(f"  {element_name(Z_d)}-{A_d:3d} {Z_d:4d} {geo.eps:+5.2f} "
              f"{geo.peanut_f:5.2f} {nz:5d} {nz2_A:9.2f} {E_pea:7.2f} "
              f"{'SF':>7s} {ee:>4s}")

    # Also show alpha emitters near the SF boundary
    alpha_heavy = [d for d in data if d['mode'] == 'alpha' and d['A'] >= 230]
    print(f"\n  Heavy alpha emitters (A≥230):")
    for d in sorted(alpha_heavy, key=lambda x: x['A'])[:15]:
        Z_d, A_d = d['Z'], d['A']
        geo = compute_geometric_state(Z_d, A_d)
        N_d = A_d - Z_d
        nz = N_d - Z_d
        nz2_A = nz**2 / A_d
        E_pea = (BETA / 2) * nz2_A
        ee = 'ee' if geo.is_ee else '  '
        print(f"  {element_name(Z_d)}-{A_d:3d} {Z_d:4d} {geo.eps:+5.2f} "
              f"{geo.peanut_f:5.2f} {nz:5d} {nz2_A:9.2f} {E_pea:7.2f} "
              f"{'alpha':>7s} {ee:>4s}")

    # ══════════════════════════════════════════════════════════════════
    # GRAND FINAL SUMMARY (ALL MODELS)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print(f"  GRAND FINAL SUMMARY — ALL MODELS")
    print(f"{'='*72}")
    all_models = [
        ("v8 gradient",        stats_v8),
        ("Flat kinetic",       stats_kin),
        ("Additive Coulomb",   stats_coul),
        ("Coupled Coulomb",    stats_coupled),
        ("Perturbation",       stats_perturb),
        ("Mult-Dzhanibekov",   stats_mdzh),
    ]
    for name, st in all_models:
        alpha_row = st['confusion'].get('alpha', {})
        alpha_n = sum(alpha_row.values())
        alpha_acc = 100 * alpha_row.get('alpha', 0) / alpha_n if alpha_n > 0 else 0
        sf_row = st['confusion'].get('SF', {})
        sf_n = sum(sf_row.values())
        sf_acc = 100 * sf_row.get('SF', 0) / sf_n if sf_n > 0 else 0
        delta = st['accuracy'] - stats_v8['accuracy']
        print(f"  {name:<22s} {100*st['accuracy']:5.1f}% mode  "
              f"{100*st['beta_dir']:5.1f}% β-dir  "
              f"{alpha_acc:5.1f}% α  {sf_acc:4.1f}% SF  "
              f"({100*delta:+.1f}%)")

    print(f"\n  Done.")
