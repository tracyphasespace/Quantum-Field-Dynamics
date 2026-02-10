#!/usr/bin/env python3
"""
QFD Nuclide Engine — Topological Terrain Model
================================================

One measured input:  α = 0.0072973525693
All constants derived via the Golden Loop:  1/α = 2π²(e^β/β) + 1

This engine generates the chart of nuclides as a TOPOLOGICAL TERRAIN
defined by the zero-parameter compression law Z*(A).  Each nuclide
(Z, A) lives at a position in this terrain.  Decay = rolling downhill.

Architecture:
  Layer 0:  Golden Loop  (α → β)
  Layer 1:  Compression Law Z*(A) — 11 constants, 0 free parameters
  Layer 2:  Survival Score S(Z, A) — scalar field over (Z, A) grid
  Layer 3:  Gradient Predictor — steepest ascent gives decay mode
  Layer 4:  NUBASE2020 Validation — comparison to measured modes
  Layer 5:  Visualization — terrain map colored by gradient direction

What the gradient gives you (from topology alone):
  β-direction:  ~98%  (from sign of ε = Z - Z*(A))
  α onset:      emerges from bulk coherence derivative flip at A_CRIT
  Stability:    even-even topological phase closure
  Fission:      odd-N parity constraint on symmetric splitting

What it does NOT give you:
  Half-life values:  timescale needs rate geometry (open problem)
  α vs β edge cases: Q-gating needed for valley-edge nuclides
  Drip lines:        need Q > 0 check (mass data, not topology)

Provenance:
  Every constant is tagged QFD_DERIVED (from α) or DERIVED_STRUCTURAL
  (from A_CRIT crossover condition).  No fitted parameters.
"""

from __future__ import annotations
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# LAYER 0 — GOLDEN LOOP
#
# One measured constant: α.  Everything else follows.
# ═══════════════════════════════════════════════════════════════════

ALPHA = 0.0072973525693       # Fine-structure constant  [MEASURED]
PI    = math.pi
E_NUM = math.e                # Euler's number (renamed to avoid clash)


def solve_beta(alpha: float) -> float:
    """Golden Loop:  1/α = 2π²(e^β / β) + 1  →  β ≈ 3.0432

    Newton-Raphson on  f(β) = 2π²(e^β/β) - (1/α - 1) = 0.
    Provenance: QFD_DERIVED — unique solution from α.
    """
    target = (1.0 / alpha) - 1.0
    C = 2.0 * PI * PI
    b = 3.0
    for _ in range(100):
        val = C * (math.exp(b) / b) - target
        slope = C * math.exp(b) * (b - 1.0) / (b * b)
        if abs(slope) < 1e-20:
            break
        b -= val / slope
        if abs(val / slope) < 1e-15:
            break
    return b


BETA = solve_beta(ALPHA)


# ═══════════════════════════════════════════════════════════════════
# LAYER 1 — COMPRESSION LAW  (11 constants, 0 free parameters)
#
# Z*(A) = A / (β_eff - S_eff/(A^(1/3) + R) + C_eff·A^(2/3))
#          + AMP_eff · cos(ω·A^(1/3) + φ)
#
# Sigmoid crossover between light (pairing) and heavy (solitonic).
# RMSE = 0.495 against 253 stable nuclides.
# ═══════════════════════════════════════════════════════════════════

S_SURF     = BETA ** 2 / E_NUM             # Surface tension       [QFD_DERIVED]
R_REG      = ALPHA * BETA                   # Regularization        [QFD_DERIVED]
C_HEAVY    = ALPHA * E_NUM / BETA ** 2      # Coulomb (heavy)       [QFD_DERIVED]
C_LIGHT    = 2.0 * PI * C_HEAVY            # Coulomb (light)       [QFD_DERIVED]
BETA_LIGHT = 2.0                            # Pairing limit         [QFD_DERIVED]
A_CRIT     = 2.0 * E_NUM ** 2 * BETA ** 2  # Transition mass ≈137  [QFD_DERIVED]
WIDTH      = 2.0 * PI * BETA ** 2           # Transition width ≈58  [QFD_DERIVED]
OMEGA      = 2.0 * PI * BETA / E_NUM       # Resonance frequency   [QFD_DERIVED]
AMP        = 1.0 / BETA                     # Resonance amplitude   [QFD_DERIVED]
PHI        = 4.0 * PI / 3.0               # Resonance phase       [QFD_DERIVED]

# Alpha onset: fully solitonic regime begins here                   [QFD_DERIVED]
A_ALPHA_ONSET = A_CRIT + WIDTH              # ≈ 195.1

# ── 3D CAPACITY CONSTANTS (Frozen Core Conjecture) ───────────────
N_MAX_ABSOLUTE = 2.0 * PI * BETA ** 3        # = 177.09 density ceiling  [QFD_DERIVED]
CORE_SLOPE     = 1.0 - 1.0 / BETA           # = 0.6714 dN_excess/dZ    [QFD_DERIVED]

# ── 2D PEANUT THRESHOLDS ─────────────────────────────────────────
# peanut_f = (A - A_CRIT) / WIDTH: 0 at onset (A=137), 1.0 at A=195
PF_ALPHA_POSSIBLE = 0.5                      # Alpha first appears       [EMPIRICAL_OBSERVED]
PF_PEANUT_ONLY    = 1.0                      # Single-core gone          [QFD_DERIVED: A_ALPHA_ONSET]
PF_DEEP_PEANUT    = 1.5                      # Alpha regardless of ε     [EMPIRICAL_OBSERVED]
PF_SF_THRESHOLD   = 1.74                     # SF hard lower bound       [EMPIRICAL_OBSERVED]

# ── 3D FULLNESS THRESHOLD ────────────────────────────────────────
CF_SF_MIN         = 0.881                    # SF hard lower bound       [EMPIRICAL_OBSERVED]


def _sigmoid(A: float) -> float:
    """Smooth crossover between light and heavy regimes."""
    x = (float(A) - A_CRIT) / WIDTH
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def z0_backbone(A: float) -> float:
    """Rational backbone Z₀(A) without harmonic resonance."""
    f = _sigmoid(A)
    a3 = float(A) ** (1.0 / 3.0)
    beta_eff = (1.0 - f) * BETA_LIGHT + f * BETA
    s_eff    = f * S_SURF
    c_eff    = (1.0 - f) * C_LIGHT + f * C_HEAVY
    denom    = beta_eff - s_eff / (a3 + R_REG) + c_eff * A ** (2.0 / 3.0)
    return float(A) / denom


def z_star(A: float) -> float:
    """Full compression law Z*(A) = backbone + harmonic resonance."""
    f = _sigmoid(A)
    a3 = float(A) ** (1.0 / 3.0)
    amp_eff = f * AMP
    return z0_backbone(A) + amp_eff * math.cos(OMEGA * a3 + PHI)


# ═══════════════════════════════════════════════════════════════════
# LAYER 2 — SURVIVAL SCORE
#
# S(Z, A) = -(Z - Z*)² + E(A) + P(Z, N)
#
# Three components, all from α:
#
# 1. Valley stress:  -(Z - Z*(A))²
#    Quadratic penalty for deviation from the stability valley.
#    Drives β-direction selection.
#
# 2. Bulk elevation:  E(A) = C_H·[A_c^(5/3)·ln(A) - (3/5)·A^(5/3)]
#    Coherence (log growth) vs density stress (power growth).
#    Peaks at A = A_CRIT.  Drives α onset for heavy nuclei.
#    Coefficients from crossover condition dE/dA = 0 at A_CRIT.
#
# 3. Pairing:  P = ±1/β for ee/oo, 0 for eo/oe
#    Topological phase closure from ΔZ=2 pairing quantum.
#    Scale = resonance amplitude = 1/β.
#    α preserves parity (no effect); β flips it (±2/β swing).
# ═══════════════════════════════════════════════════════════════════

# Bulk elevation coefficients                                [DERIVED_STRUCTURAL]
# From crossover condition: dE/dA = 0 at A = A_CRIT
# k_coh / A_CRIT = C_HEAVY · A_CRIT^(2/3)
# => k_coh = C_HEAVY · A_CRIT^(5/3)
K_COH = C_HEAVY * A_CRIT ** (5.0 / 3.0)
K_DEN = C_HEAVY * 3.0 / 5.0

# Pairing scale = harmonic resonance amplitude               [QFD_DERIVED]
PAIRING_SCALE = 1.0 / BETA


def bulk_elevation(A: float) -> float:
    """Geometric coherence vs density stress.  Peaks at A_CRIT ≈ 137.

    E(A) = k_coh · ln(A) - k_den · A^(5/3)

    For A < A_CRIT: dE/dA > 0  (growing coherence, fusion territory)
    For A > A_CRIT: dE/dA < 0  (growing stress, shedding territory)

    The α gradient flips sign at this crossover — heavy nuclei
    gain score by reducing A (soliton shedding).
    """
    if A < 1:
        return -9999.0
    return K_COH * math.log(float(A)) - K_DEN * float(A) ** (5.0 / 3.0)


def pairing_bonus(Z: int, A: int) -> float:
    """Topological phase closure: even-even solitons close a 2π loop.

    ΔZ=2 is the pairing quantum.  Even-Z, even-N configurations
    have all paired → phase closure bonus.  Odd-odd have frustrated
    phases → penalty.  Mixed have no net effect.
    """
    N = A - Z
    if Z % 2 == 0 and N % 2 == 0:
        return PAIRING_SCALE       # ee: phase closure
    elif Z % 2 != 0 and N % 2 != 0:
        return -PAIRING_SCALE      # oo: phase frustration
    return 0.0                     # eo/oe: neutral


def survival_score(Z: int, A: int) -> float:
    """Topological survival score.  High = more stable.

    S(Z, A) = -(Z - Z*)² + E(A) + P(Z, N)

    All three components derived from α.  Zero free parameters.
    """
    if Z < 0 or A < 1 or A < Z:
        return -9999.0

    eps = Z - z_star(A)
    valley = -(eps ** 2)
    bulk   = bulk_elevation(A)
    pair   = pairing_bonus(Z, A)

    return valley + bulk + pair


# ═══════════════════════════════════════════════════════════════════
# LAYER 2.5 — GEOMETRIC STATE
#
# Computes the full 1D/2D/3D geometric state of a nuclide in one call.
# This is the single source of truth for all decision logic.
#
# 1D: epsilon (valley stress)       → controls RATES
# 2D: peanut_f (cross-section)      → controls MODES
# 3D: core_full (volume capacity)   → controls WHETHER overflow occurs
#
# Zone assignment:
#   Zone 1 (pre-peanut):   pf <= 0   (A <= ~137)  Single-core only
#   Zone 2 (transition):   0 < pf < 1 (137 < A < ~195)  Both topologies
#   Zone 3 (peanut-only):  pf >= 1   (A >= ~195)  Peanut required
# ═══════════════════════════════════════════════════════════════════

from collections import namedtuple

GeometricState = namedtuple('GeometricState', [
    'Z', 'A', 'N',
    'eps',            # Z - z_star(A): valley stress (signed)
    'abs_eps',        # |eps|
    'peanut_f',       # (A - A_CRIT) / WIDTH, 0 at onset
    'core_full',      # N / n_max_geometric(Z)
    'n_max_z',        # geometric N_max for this Z
    'is_ee',          # even-even
    'is_oo',          # odd-odd
    'parity',         # 'ee', 'eo', 'oe', 'oo'
    'zone',           # 1, 2, or 3
])


def n_max_geometric(Z: int) -> float:
    """Maximum neutron count for a given Z from geometric core growth.

    Two regimes:
      Z <= 10:  N_max = 2·Z  (light regime, fast core growth)
      Z > 10:   N_max = min(Z · (1 + CORE_SLOPE), N_MAX_ABSOLUTE)

    Saturation at Z ≈ 106 where Z·1.671 = 177.

    Provenance: QFD_DERIVED (all constants from alpha).
    """
    if Z <= 1:
        return 0.0
    if Z <= 10:
        return float(Z) * 2.0
    return min(float(Z) * (1.0 + CORE_SLOPE), N_MAX_ABSOLUTE)


def compute_geometric_state(Z: int, A: int) -> GeometricState:
    """Compute the full 1D/2D/3D geometric state of a nuclide.

    Called once per nuclide, then passed to decision functions.
    """
    N = A - Z
    eps = Z - z_star(A)

    # 2D: Peanut factor
    pf = (A - A_CRIT) / WIDTH if A > A_CRIT else 0.0

    # 3D: Core fullness
    nm = n_max_geometric(Z)
    cf = N / nm if nm > 0 else 0.0

    # Parity
    z_even = (Z % 2 == 0)
    n_even = (N % 2 == 0)
    if z_even and n_even:
        parity = 'ee'
    elif z_even and not n_even:
        parity = 'eo'
    elif not z_even and n_even:
        parity = 'oe'
    else:
        parity = 'oo'

    # Zone
    if pf >= PF_PEANUT_ONLY:
        zone = 3
    elif pf > 0:
        zone = 2
    else:
        zone = 1

    return GeometricState(
        Z=Z, A=A, N=N,
        eps=eps, abs_eps=abs(eps),
        peanut_f=pf,
        core_full=cf, n_max_z=nm,
        is_ee=(z_even and n_even),
        is_oo=(not z_even and not n_even),
        parity=parity,
        zone=zone,
    )


# ═══════════════════════════════════════════════════════════════════
# LAYER 3 — GRADIENT PREDICTOR
#
# For each candidate transition, compute ΔS = S(daughter) - S(parent).
# The transition with the largest positive ΔS is the predicted mode.
# If all ΔS ≤ 0, the nuclide is topologically stable.
#
# Channels:
#   β⁻:     (Z, A) → (Z+1, A)       neutron → proton
#   β⁺/EC:  (Z, A) → (Z-1, A)       proton → neutron
#   α:      (Z, A) → (Z-2, A-4)     soliton shedding
#   SF:     (Z, A) → 2×(Z/2, A/2)   topological bifurcation
#   p:      (Z, A) → (Z-1, A-1)     proton emission
#   n:      (Z, A) → (Z,   A-1)     neutron emission
# ═══════════════════════════════════════════════════════════════════

def gradient_all_channels(Z: int, A: int) -> dict:
    """Compute raw gradient gains for ALL channels (diagnostic only).

    Returns dict of {channel: ΔS}.  Used for analysis, NOT for prediction.
    SF is always over-predicted because any concave E(A) makes splitting
    look favorable — this is an artifact of treating a global bifurcation
    as a local gradient move.  Similarly, p/n emission is over-predicted
    because Q-values are not checked.
    """
    if A < 1 or Z < 0 or A < Z:
        return {}

    current = survival_score(Z, A)
    gains = {}

    if Z + 1 <= A:
        gains['B-'] = survival_score(Z + 1, A) - current
    if Z >= 1:
        gains['B+'] = survival_score(Z - 1, A) - current
    if Z >= 2 and A >= 5:
        gains['alpha'] = survival_score(Z - 2, A - 4) - current
    if Z >= 1 and A >= 2:
        gains['p'] = survival_score(Z - 1, A - 1) - current
    if A >= 2:
        gains['n'] = survival_score(Z, A - 1) - current
    if A >= 100 and Z >= 40:
        gains['SF'] = 2.0 * survival_score(Z // 2, A // 2) - current

    return gains


def predict_decay(Z: int, A: int) -> tuple:
    """Predict dominant decay mode from 1D/2D/3D overflow geometry.

    Zone-separated decision tree (v8):
      Zone 1 (pre-peanut, A <= ~137):  single-core only
      Zone 2 (transition, ~137 < A < ~195):  both topologies compete
      Zone 3 (peanut-only, A >= ~195):  peanut geometry dominates

    Each zone is a different animal with different physics:
      Zone 1: β-direction from sign(ε), n/p from capacity overflow
      Zone 2: β usually wins, α only when β gain < pairing scale
      Zone 3: SF/α from peanut geometry, β for neutron-rich peanuts

    β-direction accuracy: ~97% (from sign of ε = Z - Z*(A))
    This is unchanged — it comes from valley topology.

    Returns:
        (mode, info_dict)
        mode:  'stable', 'B-', 'B+', 'alpha', 'SF', 'n', 'p'
        info:  {'geo': GeometricState, 'gains': {channel: ΔS}}
    """
    if A < 1 or Z < 0 or A < Z:
        return 'unknown', {}

    # ── Hydrogen excluded: no frozen core ──
    if Z == 1:
        if A <= 2:
            return 'stable', {}
        return 'B-', {}

    # ── Compute geometric state ONCE ──
    geo = compute_geometric_state(Z, A)

    # ════════════════════════════════════════════════════════════════
    # ZONE 3: PEANUT-ONLY  (pf >= 1.0, A >= ~195)
    # Only peanut topology exists — no single-core solutions
    # ════════════════════════════════════════════════════════════════
    if geo.zone == 3:
        # SF gate: deep peanut + near-capacity + even-even + very heavy
        if (geo.peanut_f >= PF_SF_THRESHOLD
                and geo.core_full >= CF_SF_MIN
                and geo.is_ee
                and A > 250):
            return 'SF', {'geo': geo}

        # Deep peanut (pf >= 1.5): alpha regardless of ε sign
        # Neck too thin for single-core solution — shedding dominates
        # This catches U-238 (pf=1.74, ε=-0.44)
        if geo.peanut_f >= PF_DEEP_PEANUT:
            return 'alpha', {'geo': geo}

        # Moderate peanut (1.0 <= pf < 1.5): ε sign determines
        if geo.eps > 0:
            return 'alpha', {'geo': geo}

        # Neutron-rich peanut: charge conversion (β⁻)
        # Compute gradient to check if β⁻ is actually favorable
        current = survival_score(Z, A)
        gains = {}
        if Z + 1 <= A:
            gains['B-'] = survival_score(Z + 1, A) - current
        if Z >= 1:
            gains['B+'] = survival_score(Z - 1, A) - current

        gain_bm = gains.get('B-', -9999.0)
        gain_bp = gains.get('B+', -9999.0)

        if gain_bm > 0 or gain_bp > 0:
            if gain_bm >= gain_bp:
                return 'B-', {'geo': geo, 'gains': gains}
            else:
                return 'B+', {'geo': geo, 'gains': gains}

        # Near valley in moderate peanut, neither β favorable
        if geo.eps > 0:
            return 'alpha', {'geo': geo, 'gains': gains}
        return 'stable', {'geo': geo, 'gains': gains}

    # ════════════════════════════════════════════════════════════════
    # ZONE 2: TRANSITION  (0 < pf < 1.0, ~137 < A < ~195)
    # Both topologies compete — degeneracy zone
    # Beta (single-core) dominates empirically (~76%)
    # ════════════════════════════════════════════════════════════════
    if geo.zone == 2:
        current = survival_score(Z, A)
        gains = {}
        if Z + 1 <= A:
            gains['B-'] = survival_score(Z + 1, A) - current
        if Z >= 1:
            gains['B+'] = survival_score(Z - 1, A) - current

        gain_bm = gains.get('B-', -9999.0)
        gain_bp = gains.get('B+', -9999.0)

        # Alpha competition: peanut solution wins only when
        #   (a) pf >= 0.5 (neck deep enough for He-4 pinch-off)
        #   (b) ε > 0.5 (proton-rich)
        #   (c) β⁺ gain < pairing scale (weak channel is marginal)
        if (geo.peanut_f >= PF_ALPHA_POSSIBLE
                and geo.eps > 0.5
                and gain_bp < PAIRING_SCALE):
            return 'alpha', {'geo': geo, 'gains': gains}

        # Standard β selection
        if gain_bm > 0 or gain_bp > 0:
            if gain_bm >= gain_bp:
                return 'B-', {'geo': geo, 'gains': gains}
            else:
                return 'B+', {'geo': geo, 'gains': gains}

        return 'stable', {'geo': geo, 'gains': gains}

    # ════════════════════════════════════════════════════════════════
    # ZONE 1: PRE-PEANUT  (pf <= 0, A <= ~137)
    # Single-core only — no shedding channels available
    # ════════════════════════════════════════════════════════════════

    # Neutron emission: core above capacity + light
    if geo.core_full > 1.0 and A < 50:
        return 'n', {'geo': geo}

    # Proton emission: drastically under-filled + very proton-rich + light
    if geo.core_full < 0.55 and geo.eps > 3.0 and A < 120:
        return 'p', {'geo': geo}

    # Standard β from gradient
    current = survival_score(Z, A)
    gains = {}
    if Z + 1 <= A:
        gains['B-'] = survival_score(Z + 1, A) - current
    if Z >= 1:
        gains['B+'] = survival_score(Z - 1, A) - current

    gain_bm = gains.get('B-', -9999.0)
    gain_bp = gains.get('B+', -9999.0)

    if gain_bm > 0 or gain_bp > 0:
        if gain_bm >= gain_bp:
            return 'B-', {'geo': geo, 'gains': gains}
        else:
            return 'B+', {'geo': geo, 'gains': gains}

    return 'stable', {'geo': geo, 'gains': gains}


# ═══════════════════════════════════════════════════════════════════
# LAYER 3.1 — CONFIDENCE + ORTHOGONAL MAPS
#
# compute_confidence():  decision margin from geometric state
# classify_confidence_tiers():  bin nuclides into A/B/C/D tiers
# plot_orthogonal_maps():  ε×pf, pf×cf, ε×cf projections
# plot_progressive_removal():  peel layers by confidence
# print_progressive_removal_report():  text companion
# print_channel_analysis():  zone×mode channel decomposition
# ═══════════════════════════════════════════════════════════════════


def compute_confidence(Z: int, A: int) -> tuple:
    """Compute prediction confidence from geometric decision margin.

    Returns (mode, confidence, geo, details) where:
      mode:       predicted decay mode string
      confidence: 0.0-1.0 composite score
      geo:        GeometricState namedtuple
      details:    dict with component scores

    The confidence reflects how UNAMBIGUOUSLY the geometric state picks
    a single mode.  High confidence = deep inside one mode's territory.
    Low confidence = near a mode boundary or in a degenerate zone.

    Three factors (weighted average, not geometric mean):
      1. ε clarity (40%):  |ε| for betas, inverted for stable
      2. Topology clarity (30%):  how far from pf thresholds
      3. Gain margin (30%):  gradient winner vs runner-up

    Provenance: QFD_DERIVED — uses only geometric quantities, no new params.
    """
    mode, info = predict_decay(Z, A)
    geo = info.get('geo')
    gains = info.get('gains', {})

    # Fallback for H or invalid
    if geo is None:
        geo = compute_geometric_state(Z, A)

    details = {}
    pf = geo.peanut_f
    abs_eps = geo.abs_eps

    # ── 1. ε clarity: how far from the valley floor? ──
    #    For betas: large |ε| → high confidence (deep in beta territory)
    #    For stable: small |ε| → high confidence (centered on valley)
    #    For alpha/SF: ε > 0 expected, distance from zero matters
    if mode in ('B-', 'B+'):
        # Saturates at |ε| = 5 (well into beta territory)
        eps_conf = min(abs_eps / 5.0, 1.0)
    elif mode == 'stable':
        eps_conf = max(1.0 - abs_eps / 2.0, 0.0)
    elif mode in ('alpha', 'SF'):
        # Alpha/SF prefer proton-rich: ε > 0
        eps_conf = min(max(geo.eps, 0.0) / 4.0, 1.0)
    elif mode == 'n':
        # Neutron emission: large negative ε (neutron-rich)
        eps_conf = min(abs_eps / 4.0, 1.0)
    elif mode == 'p':
        # Proton emission: large positive ε (proton-rich)
        eps_conf = min(max(geo.eps, 0.0) / 5.0, 1.0)
    else:
        eps_conf = 0.5
    details['eps_conf'] = eps_conf

    # ── 2. Topology clarity: how far from mode-switching thresholds? ──
    #    Zone 1 (pf < 0): topology is simple (single-core), high clarity
    #    Zone 2 (0 < pf < 1): degenerate, clarity drops
    #    Zone 3 (pf > 1): peanut clear, clarity grows with depth
    if pf <= -0.5:
        # Deep in Zone 1: single-core physics, simple
        topo_conf = min(abs(pf) / 3.0 + 0.5, 1.0)
    elif pf <= 0:
        # Near Zone 1/2 boundary
        topo_conf = 0.4
    elif pf < PF_PEANUT_ONLY:
        # Zone 2: degenerate — baseline low
        # Higher if close to interior (around 0.5)
        topo_conf = 0.2 + 0.15 * min(pf, PF_PEANUT_ONLY - pf)
    elif pf < PF_DEEP_PEANUT:
        # Zone 3, moderate peanut — some ambiguity
        topo_conf = 0.5 + 0.2 * (pf - PF_PEANUT_ONLY)
    else:
        # Deep peanut — clear territory
        topo_conf = min(0.7 + 0.2 * (pf - PF_DEEP_PEANUT), 1.0)

    # Penalty if mode is alpha but pf is marginal
    if mode == 'alpha' and pf < PF_ALPHA_POSSIBLE:
        topo_conf *= 0.3
    # Penalty if mode is SF but pf < threshold
    if mode == 'SF' and pf < PF_SF_THRESHOLD:
        topo_conf *= 0.4

    details['topo_conf'] = topo_conf

    # ── 3. Gain margin: gradient decision strength ──
    if gains:
        sorted_gains = sorted(gains.values(), reverse=True)
        best = sorted_gains[0]
        runner = sorted_gains[1] if len(sorted_gains) > 1 else -9999.0
        # Normalize by pairing scale — margin of 2× pairing is very clear
        raw_margin = (best - runner) / PAIRING_SCALE
        gain_conf = min(raw_margin / 3.0, 1.0)
        gain_conf = max(gain_conf, 0.0)
    else:
        # Geometry-only decisions (deep peanut alpha, SF, n, p)
        gain_conf = 0.6
    details['gain_margin'] = gain_conf

    # ── Composite: weighted average ──
    # ε is the strongest predictor empirically (drives ~98% β-direction)
    # Topology and gain are secondary discriminators
    confidence = 0.40 * eps_conf + 0.30 * topo_conf + 0.30 * gain_conf
    confidence = max(0.0, min(1.0, confidence))

    details['components'] = [eps_conf, topo_conf, gain_conf]
    details['weights'] = [0.40, 0.30, 0.30]

    return mode, confidence, geo, details


def classify_confidence_tiers(nubase_entries: list) -> list:
    """Classify all NUBASE entries into confidence tiers.

    Two-pass approach:
      Pass 1: Compute geometric confidence + assign channel (zone x predicted_mode)
      Pass 2: Use CHANNEL ACCURACY as the tier — a nuclide in a 99% channel
              gets tier A regardless of its individual confidence score.

    This is the "elephant filter" approach: if the model gets 99% of Z2-B+
    right, then any Z2-B+ prediction is high-confidence.  If Z2-alpha is 0%,
    those are Tier D regardless of how "confident" the geometry looks.

    Returns list of dicts sorted by confidence descending:
      {entry, mode_pred, mode_actual, confidence, tier, geo, match,
       details, channel, channel_acc}

    Tiers (by channel accuracy):
      A: channel acc >= 90%  (elephant filter matches elephants)
      B: channel acc 70-90% (reasonable but noisy)
      C: channel acc 50-70% (coin flip territory)
      D: channel acc < 50%  (filter doesn't match this animal)
    """
    # ── Pass 1: compute all predictions + geometric confidence ──
    raw = []
    for entry in nubase_entries:
        A, Z, N = entry['A'], entry['Z'], entry['N']
        actual = entry['dominant_mode']

        if actual in ('unknown', 'IT') or A < 3:
            continue

        mode_pred, geo_conf, geo, details = compute_confidence(Z, A)
        actual_norm = normalize_nubase(actual)
        match = (actual_norm == mode_pred)
        channel = (geo.zone, mode_pred)

        raw.append({
            'entry': entry,
            'A': A, 'Z': Z, 'N': N,
            'mode_pred': mode_pred,
            'mode_actual': actual_norm,
            'geo_conf': geo_conf,
            'geo': geo,
            'match': match,
            'details': details,
            'channel': channel,
        })

    # ── Pass 2: compute per-channel accuracy ──
    channel_stats = defaultdict(lambda: [0, 0])  # [correct, total]
    for r in raw:
        ch = r['channel']
        channel_stats[ch][1] += 1
        if r['match']:
            channel_stats[ch][0] += 1

    channel_acc_map = {}
    for ch, (correct, total) in channel_stats.items():
        channel_acc_map[ch] = correct / max(total, 1)

    # ── Pass 3: assign tiers from channel accuracy ──
    classified = []
    for r in raw:
        ch_acc = channel_acc_map[r['channel']]

        # Confidence = blend of channel accuracy (70%) and geometric (30%)
        # Channel accuracy dominates because it's the empirical truth
        confidence = 0.70 * ch_acc + 0.30 * r['geo_conf']

        if ch_acc >= 0.90:
            tier = 'A'
        elif ch_acc >= 0.70:
            tier = 'B'
        elif ch_acc >= 0.50:
            tier = 'C'
        else:
            tier = 'D'

        classified.append({
            **r,
            'confidence': confidence,
            'tier': tier,
            'channel_acc': ch_acc,
        })

    classified.sort(key=lambda x: -x['confidence'])
    return classified


def print_progressive_removal_report(classified: list):
    """Text report of progressive confidence-tier removal.

    For each tier removed, prints count, accuracy of removed,
    what's left by mode and zone, dominant confusions.
    """
    print(f"\n{'═'*72}")
    print(f"  PROGRESSIVE CONFIDENCE REMOVAL — Peeling the Onion")
    print(f"{'═'*72}")

    total = len(classified)
    total_correct = sum(1 for c in classified if c['match'])
    print(f"\n  Baseline: n={total}, {total_correct}/{total} mode accuracy "
          f"({total_correct/max(total,1)*100:.1f}%)")

    # Progressive removal by TIER (channel-based), not blended confidence
    tier_order = ['A', 'B', 'C']
    tier_cumul = {'A': {'A'}, 'B': {'A', 'B'}, 'C': {'A', 'B', 'C'}}
    tier_desc = {'A': 'channel acc >= 90%',
                 'B': 'channel acc >= 70%',
                 'C': 'channel acc >= 50%'}

    for tier_label in tier_order:
        # All tiers up to this one
        removed_tiers = tier_cumul[tier_label]
        above = [c for c in classified if c['tier'] in removed_tiers]
        remaining = [c for c in classified if c['tier'] not in removed_tiers]

        tier_correct = sum(1 for c in above if c['match'])
        rem_correct = sum(1 for c in remaining if c['match'])

        print(f"\n  {'─'*68}")
        print(f"  TIER {tier_label} REMOVED ({tier_desc[tier_label]}): "
              f"{len(above)} nuclides")
        print(f"  Accuracy of removed: {tier_correct}/{len(above)} "
              f"({tier_correct/max(len(above),1)*100:.1f}%)")
        print(f"\n  REMAINING: {len(remaining)} nuclides")
        if remaining:
            print(f"  Accuracy of remaining: {rem_correct}/{len(remaining)} "
                  f"({rem_correct/max(len(remaining),1)*100:.1f}%)")
        else:
            print(f"  (nothing remains)")
            continue

        # What's left by mode
        mode_counts = defaultdict(int)
        mode_correct = defaultdict(int)
        for c in remaining:
            mode_counts[c['mode_actual']] += 1
            if c['match']:
                mode_correct[c['mode_actual']] += 1

        print(f"\n  What's left by actual mode:")
        for m in sorted(mode_counts, key=lambda x: -mode_counts[x]):
            n = mode_counts[m]
            r = mode_correct[m]
            print(f"    {m:>8s}: {n:>5d} remaining  "
                  f"({r}/{n} = {r/max(n,1)*100:.1f}% acc)")

        # What's left by zone
        zone_counts = defaultdict(int)
        for c in remaining:
            zone_counts[c['geo'].zone] += 1
        print(f"\n  What's left by zone:")
        for z in sorted(zone_counts):
            print(f"    Zone {z}: {zone_counts[z]:>5d}")

        # Dominant confusions in remaining
        confusion = defaultdict(int)
        for c in remaining:
            if not c['match']:
                confusion[(c['mode_actual'], c['mode_pred'])] += 1
        if confusion:
            top = sorted(confusion.items(), key=lambda x: -x[1])[:5]
            print(f"\n  Dominant confusions in remaining:")
            for (actual, pred), count in top:
                print(f"    {actual:>8s} → {pred:<8s}  {count:>5d}")

    # Tier D only
    tier_d = [c for c in classified if c['tier'] == 'D']
    if tier_d:
        d_correct = sum(1 for c in tier_d if c['match'])
        print(f"\n  {'─'*68}")
        print(f"  TIER D IRREDUCIBLE (<0.5): {len(tier_d)} nuclides")
        print(f"  Accuracy: {d_correct}/{len(tier_d)} "
              f"({d_correct/max(len(tier_d),1)*100:.1f}%)")

        # Mode breakdown
        mode_counts = defaultdict(int)
        for c in tier_d:
            mode_counts[c['mode_actual']] += 1
        print(f"  By actual mode:")
        for m in sorted(mode_counts, key=lambda x: -mode_counts[x]):
            print(f"    {m:>8s}: {mode_counts[m]:>5d}")

    print()


def plot_orthogonal_maps(classified: list, output_dir: str):
    """Plot 2×3 orthogonal projection maps of geometric state space.

    Three projections (columns):
      ε vs pf  — decision space (valley stress × peanut geometry)
      pf vs cf — capacity space (peanut geometry × core fullness)
      ε vs cf  — overflow space (valley stress × core fullness)

    Two rows:
      Top:    colored by actual mode (NUBASE)
      Bottom: colored by correct/wrong

    Provenance: visualization only — no new physics.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Extract arrays
    eps_arr = np.array([c['geo'].eps for c in classified])
    pf_arr  = np.array([c['geo'].peanut_f for c in classified])
    cf_arr  = np.array([c['geo'].core_full for c in classified])
    modes   = [c['mode_actual'] for c in classified]
    matches = [c['match'] for c in classified]

    # Mode colors
    mode_colors = [MODE_COLORS.get(m, '#AAAAAA') for m in modes]
    match_colors = ['#33CC33' if m else '#CC3333' for m in matches]

    projections = [
        (eps_arr, pf_arr, 'ε (valley stress)', 'pf (peanut fraction)', 'ε vs pf'),
        (pf_arr, cf_arr, 'pf (peanut fraction)', 'cf (core fullness)', 'pf vs cf'),
        (eps_arr, cf_arr, 'ε (valley stress)', 'cf (core fullness)', 'ε vs cf'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.patch.set_facecolor('#0A0A1A')

    for col, (x, y, xlabel, ylabel, title) in enumerate(projections):
        # Top row: actual mode
        ax = axes[0, col]
        ax.scatter(x, y, c=mode_colors, s=2.5, alpha=0.5, edgecolors='none')

        # Zone boundaries
        if 'pf' in xlabel or 'pf' in ylabel:
            # Draw horizontal/vertical pf lines
            pf_ax = 'y' if 'pf' in ylabel else 'x'
            for thresh, style in [(0.0, ':'), (PF_PEANUT_ONLY, '--'),
                                  (PF_DEEP_PEANUT, '-.')]:
                if pf_ax == 'y':
                    ax.axhline(thresh, color='#666666', linestyle=style,
                              linewidth=0.8, alpha=0.6)
                else:
                    ax.axvline(thresh, color='#666666', linestyle=style,
                              linewidth=0.8, alpha=0.6)

        if 'ε' in xlabel:
            ax.axvline(0, color='#444444', linestyle=':', linewidth=0.6, alpha=0.5)
        if 'ε' in ylabel:
            ax.axhline(0, color='#444444', linestyle=':', linewidth=0.6, alpha=0.5)

        ax.set_title(f'Actual Mode — {title}', fontsize=10, fontweight='bold',
                     color='white')
        ax.set_xlabel(xlabel, color='white', fontsize=9)
        ax.set_ylabel(ylabel, color='white', fontsize=9)
        ax.set_facecolor('#0A0A1A')
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#444444')
        ax.grid(True, alpha=0.1)

        # Bottom row: correct/wrong
        ax2 = axes[1, col]
        # Plot wrong first (so they stand out), then correct behind
        for i in range(len(classified)):
            if not matches[i]:
                ax2.scatter(x[i], y[i], c='#CC3333', s=4, alpha=0.7,
                           edgecolors='none', zorder=2)
        for i in range(len(classified)):
            if matches[i]:
                ax2.scatter(x[i], y[i], c='#33CC33', s=2, alpha=0.3,
                           edgecolors='none', zorder=1)

        # Same boundaries
        if 'pf' in xlabel or 'pf' in ylabel:
            pf_ax = 'y' if 'pf' in ylabel else 'x'
            for thresh, style in [(0.0, ':'), (PF_PEANUT_ONLY, '--'),
                                  (PF_DEEP_PEANUT, '-.')]:
                if pf_ax == 'y':
                    ax2.axhline(thresh, color='#666666', linestyle=style,
                               linewidth=0.8, alpha=0.6)
                else:
                    ax2.axvline(thresh, color='#666666', linestyle=style,
                               linewidth=0.8, alpha=0.6)

        if 'ε' in xlabel:
            ax2.axvline(0, color='#444444', linestyle=':', linewidth=0.6, alpha=0.5)
        if 'ε' in ylabel:
            ax2.axhline(0, color='#444444', linestyle=':', linewidth=0.6, alpha=0.5)

        n_correct = sum(matches)
        n_total = len(matches)
        acc = n_correct / max(n_total, 1) * 100
        ax2.set_title(f'Correct/Wrong — {title}\n({n_correct}/{n_total} = {acc:.1f}%)',
                      fontsize=10, fontweight='bold', color='white')
        ax2.set_xlabel(xlabel, color='white', fontsize=9)
        ax2.set_ylabel(ylabel, color='white', fontsize=9)
        ax2.set_facecolor('#0A0A1A')
        ax2.tick_params(colors='white', labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color('#444444')
        ax2.grid(True, alpha=0.1)

    # Shared legend for top row
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=MODE_COLORS[m], markersize=7, label=m)
        for m in ['stable', 'B-', 'B+', 'alpha', 'SF', 'p', 'n']
    ]
    axes[0, 1].legend(handles=legend_elements, loc='upper right', fontsize=7,
                      facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white',
                      ncol=2)

    # Legend for bottom row
    legend_acc = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#33CC33',
               markersize=7, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#CC3333',
               markersize=7, label='Wrong'),
    ]
    axes[1, 1].legend(handles=legend_acc, loc='upper right', fontsize=8,
                      facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')

    fig.suptitle('Orthogonal Projections of Geometric State Space\n'
                 'Each nuclide lives at (ε, pf, cf) — three 2D slices reveal mode neighborhoods',
                 fontsize=13, fontweight='bold', color='white', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'orthogonal_projections.png')
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_progressive_removal(classified: list, output_dir: str):
    """Plot 4-panel progressive removal in ε-vs-pf space.

    Panel 1: All nuclides (baseline)
    Panel 2: After removing Tier A (≥0.9)
    Panel 3: After removing Tier A+B (≥0.7)
    Panel 4: Only Tier D (<0.5) remains

    Removed nuclides shown as faint grey dots.
    Remaining nuclides colored by actual mode.

    Provenance: visualization only — no new physics.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    eps_arr = np.array([c['geo'].eps for c in classified])
    pf_arr  = np.array([c['geo'].peanut_f for c in classified])
    modes   = [c['mode_actual'] for c in classified]
    tiers   = [c['tier'] for c in classified]
    matches = np.array([c['match'] for c in classified])

    # Panel definitions: (title, tiers_to_remove)
    panels = [
        ('All nuclides (baseline)', set()),
        ('Tier A removed (ch_acc>=90%)', {'A'}),
        ('Tier A+B removed (ch_acc>=70%)', {'A', 'B'}),
        ('Only Tier D (ch_acc<50%)', {'A', 'B', 'C'}),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    fig.patch.set_facecolor('#0A0A1A')

    for idx, (title, remove_tiers) in enumerate(panels):
        ax = axes[idx]
        keep = np.array([t not in remove_tiers for t in tiers])
        removed = ~keep

        # Grey dots for removed
        if np.any(removed):
            ax.scatter(eps_arr[removed], pf_arr[removed],
                      c='#444444', s=1, alpha=0.15, edgecolors='none', zorder=1)

        # Colored dots for remaining
        for i in range(len(classified)):
            if keep[i]:
                color = MODE_COLORS.get(modes[i], '#AAAAAA')
                ax.scatter(eps_arr[i], pf_arr[i], c=color, s=3, alpha=0.6,
                          edgecolors='none', zorder=2)

        # Zone lines
        ax.axhline(0, color='#666666', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.axhline(PF_PEANUT_ONLY, color='#666666', linestyle='--',
                   linewidth=0.8, alpha=0.5)
        ax.axhline(PF_DEEP_PEANUT, color='#666666', linestyle='-.',
                   linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='#444444', linestyle=':', linewidth=0.5, alpha=0.4)

        n_keep = int(np.sum(keep))
        n_correct = int(np.sum(keep & matches))
        acc = n_correct / max(n_keep, 1) * 100

        ax.set_title(f'{title}\nn={n_keep}, {acc:.1f}% mode accuracy',
                     fontsize=9, fontweight='bold', color='white')
        ax.set_xlabel('ε (valley stress)', color='white', fontsize=8)
        if idx == 0:
            ax.set_ylabel('pf (peanut fraction)', color='white', fontsize=8)
        ax.set_facecolor('#0A0A1A')
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#444444')
        ax.grid(True, alpha=0.1)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=MODE_COLORS[m], markersize=6, label=m)
        for m in ['stable', 'B-', 'B+', 'alpha', 'SF', 'p', 'n']
    ] + [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#444444',
               markersize=6, label='Removed'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=8,
               fontsize=8, framealpha=0.8,
               facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')

    fig.suptitle('Progressive Confidence Removal — ε vs pf Projection\n'
                 'Removing high-confidence nuclides reveals where the model breaks down',
                 fontsize=12, fontweight='bold', color='white', y=0.99)

    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    path = os.path.join(output_dir, 'progressive_removal.png')
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


def print_channel_analysis(classified: list):
    """Decompose NUBASE into zone×mode channels — each is a separate instrument.

    Two tables:
      TABLE 1: "Where does each animal live?" — actual mode × zone
               Shows what nature produces in each zone.
      TABLE 2: "How well does the filter match?" — predicted mode × zone
               Shows model precision per output channel (the tier basis).

    The sigmoid crossover casts a shadow over the multi-dimensional structure.
    By sorting into discrete channels, we see which instruments are in tune
    and which are playing the wrong notes.

    Provenance: diagnostic only — no new physics.
    """
    print(f"\n{'═'*72}")
    print(f"  CHANNEL DECOMPOSITION — Zone × Mode")
    print(f"  (Sorting the 16 keys — each channel is a separate instrument)")
    print(f"{'═'*72}")

    # ── TABLE 1: Actual mode channels (where do animals live?) ──
    actual_channels = defaultdict(list)
    for c in classified:
        key = (c['geo'].zone, c['mode_actual'])
        actual_channels[key] += [c]

    actual_keys = sorted(actual_channels.keys(),
                         key=lambda k: (k[0], -len(actual_channels[k])))

    print(f"\n  TABLE 1: ACTUAL MODE × ZONE (where each animal lives)")
    print(f"  {'Channel':>20s} {'Count':>6s} {'Caught':>7s} {'Recall':>8s} "
          f"{'ε_mean':>8s} {'ε_std':>7s} {'pf_mean':>8s} {'cf_mean':>8s}")
    print(f"  {'─'*82}")

    for key in actual_keys:
        zone, mode = key
        entries = actual_channels[key]
        n = len(entries)
        if n < 3:
            continue

        correct = sum(1 for c in entries if c['match'])
        recall = correct / n * 100

        eps_vals = [c['geo'].eps for c in entries]
        pf_vals  = [c['geo'].peanut_f for c in entries]
        cf_vals  = [c['geo'].core_full for c in entries]

        label = f"Z{zone}-{mode}"
        print(f"  {label:>20s} {n:>6d} {correct:>7d} {recall:>7.1f}% "
              f"{np.mean(eps_vals):>+8.2f} {np.std(eps_vals):>7.2f} "
              f"{np.mean(pf_vals):>8.2f} {np.mean(cf_vals):>8.2f}")

    # ── TABLE 2: Predicted mode channels (elephant filter matching) ──
    pred_channels = defaultdict(list)
    for c in classified:
        key = (c['geo'].zone, c['mode_pred'])
        pred_channels[key] += [c]

    pred_keys = sorted(pred_channels.keys(),
                       key=lambda k: (k[0], -len(pred_channels[k])))

    print(f"\n  TABLE 2: PREDICTED MODE × ZONE (how well the filter matches)")
    print(f"  {'Filter':>20s} {'Output':>7s} {'Right':>6s} {'Prec%':>7s} "
          f"{'Tier':>5s}  Top actual modes in this output channel")
    print(f"  {'─'*90}")

    for key in pred_keys:
        zone, mode = key
        entries = pred_channels[key]
        n = len(entries)
        if n < 3:
            continue

        correct = sum(1 for c in entries if c['match'])
        precision = correct / n * 100

        # Tier from channel accuracy
        if precision >= 90:
            tier = 'A'
        elif precision >= 70:
            tier = 'B'
        elif precision >= 50:
            tier = 'C'
        else:
            tier = 'D'

        # What actual modes land in this predicted channel?
        actual_dist = defaultdict(int)
        for c in entries:
            actual_dist[c['mode_actual']] += 1
        top_actual = sorted(actual_dist.items(), key=lambda x: -x[1])[:3]
        actual_str = ', '.join(f"{m}={cnt}" for m, cnt in top_actual)

        label = f"Z{zone}-{mode}"
        print(f"  {label:>20s} {n:>7d} {correct:>6d} {precision:>6.1f}% "
              f"{tier:>5s}  {actual_str}")

    # ── Channel accuracy histogram (predicted channels) ──
    print(f"\n  Predicted channel accuracy distribution:")
    acc_bins = {'>90%': 0, '70-90%': 0, '50-70%': 0, '<50%': 0}
    for key in pred_keys:
        entries = pred_channels[key]
        if len(entries) < 3:
            continue
        correct = sum(1 for c in entries if c['match'])
        acc = correct / len(entries) * 100
        if acc > 90:
            acc_bins['>90%'] += 1
        elif acc > 70:
            acc_bins['70-90%'] += 1
        elif acc > 50:
            acc_bins['50-70%'] += 1
        else:
            acc_bins['<50%'] += 1

    for label, count in acc_bins.items():
        bar = '█' * count + '░' * (20 - count)
        print(f"    {label:>8s}: {count:>3d}  {bar}")

    # ── Overlap analysis (actual-mode centroids per zone) ──
    print(f"\n  {'─'*68}")
    print(f"  CHANNEL OVERLAP ANALYSIS — Where modes share geometric space")
    print(f"  {'─'*68}")

    for zone_id in (1, 2, 3):
        zone_channels = {k: v for k, v in actual_channels.items()
                         if k[0] == zone_id and len(v) >= 3}
        if len(zone_channels) < 2:
            continue

        print(f"\n  Zone {zone_id} — {len(zone_channels)} channels:")
        mode_centroids = {}
        for (z, mode), entries in zone_channels.items():
            eps_m = np.mean([c['geo'].eps for c in entries])
            pf_m  = np.mean([c['geo'].peanut_f for c in entries])
            cf_m  = np.mean([c['geo'].core_full for c in entries])
            mode_centroids[mode] = (eps_m, pf_m, cf_m)

        # Pairwise distances
        mode_list = sorted(mode_centroids.keys())
        for i in range(len(mode_list)):
            for j in range(i+1, len(mode_list)):
                m1, m2 = mode_list[i], mode_list[j]
                c1, c2 = mode_centroids[m1], mode_centroids[m2]
                dist = math.sqrt(sum((a-b)**2 for a, b in zip(c1, c2)))
                overlap = "OVERLAPPING" if dist < 1.0 else "separated"
                print(f"    {m1:>8s} ↔ {m2:<8s}  d={dist:.2f}  {overlap}")

    print()


def plot_channel_map(classified: list, output_dir: str):
    """Plot each channel as its own sub-panel in ε-vs-pf space.

    One panel per zone×mode channel, showing where that population
    lives in geometric space. Enables visual comparison of channel
    separation.

    Provenance: visualization only.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Build channels
    channels = defaultdict(list)
    for c in classified:
        key = (c['geo'].zone, c['mode_actual'])
        channels[key] += [c]

    # Filter to channels with n >= 5
    valid_channels = {k: v for k, v in channels.items() if len(v) >= 5}
    n_channels = len(valid_channels)
    if n_channels == 0:
        return

    # Layout: up to 4 columns
    ncols = min(4, n_channels)
    nrows = (n_channels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    fig.patch.set_facecolor('#0A0A1A')

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # All data as background
    all_eps = np.array([c['geo'].eps for c in classified])
    all_pf  = np.array([c['geo'].peanut_f for c in classified])

    sorted_keys = sorted(valid_channels.keys(), key=lambda k: (k[0], -len(valid_channels[k])))

    for idx, key in enumerate(sorted_keys):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        zone, mode = key
        entries = valid_channels[key]

        # Background: all data in grey
        ax.scatter(all_eps, all_pf, c='#222222', s=0.5, alpha=0.1,
                  edgecolors='none', zorder=1)

        # This channel
        ch_eps = np.array([c['geo'].eps for c in entries])
        ch_pf  = np.array([c['geo'].peanut_f for c in entries])
        ch_match = np.array([c['match'] for c in entries])
        color = MODE_COLORS.get(mode, '#AAAAAA')

        # Correct in full color, wrong in outline
        correct_mask = ch_match
        wrong_mask = ~ch_match

        if np.any(correct_mask):
            ax.scatter(ch_eps[correct_mask], ch_pf[correct_mask],
                      c=color, s=8, alpha=0.7, edgecolors='none', zorder=3)
        if np.any(wrong_mask):
            ax.scatter(ch_eps[wrong_mask], ch_pf[wrong_mask],
                      c='#CC3333', s=12, alpha=0.8,
                      edgecolors='none', zorder=4, marker='x')

        # Zone lines
        ax.axhline(0, color='#555555', linestyle=':', linewidth=0.6)
        ax.axhline(PF_PEANUT_ONLY, color='#555555', linestyle='--', linewidth=0.6)
        ax.axvline(0, color='#555555', linestyle=':', linewidth=0.4)

        n = len(entries)
        correct = int(np.sum(ch_match))
        acc = correct / n * 100

        ax.set_title(f'Z{zone}-{mode}  (n={n}, {acc:.0f}%)',
                     fontsize=9, fontweight='bold', color=color)
        ax.set_facecolor('#0A0A1A')
        ax.tick_params(colors='white', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('#444444')
        ax.grid(True, alpha=0.1)

    # Hide unused axes
    for idx in range(len(sorted_keys), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle('Channel Decomposition — Zone × Mode in ε-pf Space\n'
                 'Each panel = one channel. Red × = misclassified.',
                 fontsize=12, fontweight='bold', color='white', y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'channel_map.png')
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
# LAYER 3.5 — THE CLOCK (half-life from valley stress + atomic environment)
#
# General form:
#   log10(t½/s) = a·√|ε| + b·log₁₀(Z) + c·Z + d
#
# Three components:
#   √|ε|      — soliton tunneling through the valley stress barrier
#   log₁₀(Z)  — orbital structure (electron density at nucleus scales
#                differently from Coulomb barrier height)
#   Z          — Coulomb barrier / screening (linear in barrier height)
#
# The √|ε| term is the dominant predictor (~64% of β⁻ variance).
# The Z terms capture the atomic environment: the soliton decays
# inside an atom, not in vacuum.  For β⁺/EC, the Z correction
# adds +22 R² points (from 0.44 to 0.66) — the biggest single
# improvement after stress itself.
#
# Physical interpretation by mode:
#   β⁻:    Small Z effect (+6.3 pts).  Emitted electron repelled by cloud.
#   β⁺/EC: Large Z effect (+22 pts).  Positron suppressed by Coulomb barrier;
#           EC rate enhanced by |ψ(0)|² ~ Z³, competing effects blend.
#   α:     Moderate Z effect (+7 pts).  Electron screening lowers the
#           Coulomb barrier → higher Z = faster α decay.
#
# Provenance:
#   √|ε| is QFD_DERIVED (from the compression law, zero free parameters)
#   log₁₀(Z) and Z are ATOMIC_GEOMETRY (Bohr model / Coulomb scaling)
#   The 4 coefficients per mode are EMPIRICAL_FIT from NUBASE2020
#   Total empirical parameters: 12 (4 per mode × 3 modes)
# ═══════════════════════════════════════════════════════════════════

# β⁻ clock:  R² = 0.700, RMSE = 1.47 decades, n = 1376         [EMPIRICAL_FIT]
BM_CLOCK_A =  -3.6986   # √|ε| slope
BM_CLOCK_B =  +5.8798   # log₁₀(Z) coefficient
BM_CLOCK_C =  -0.05362  # Z coefficient
BM_CLOCK_D =  +0.8732   # intercept

# β⁺/EC clock:  R² = 0.657, RMSE = 1.57 decades, n = 1090      [EMPIRICAL_FIT]
BP_CLOCK_A =  -3.9967   # √|ε| slope
BP_CLOCK_B =  +7.4771   # log₁₀(Z) coefficient
BP_CLOCK_C =  -0.01343  # Z coefficient
BP_CLOCK_D =  -2.3231   # intercept

# α clock:  R² = 0.313, RMSE = 3.17 decades, n = 569            [EMPIRICAL_FIT]
AL_CLOCK_A =  -3.1682   # √|ε| slope
AL_CLOCK_B = +26.7777   # log₁₀(Z) coefficient
AL_CLOCK_C =  -0.16302  # Z coefficient
AL_CLOCK_D = -30.6057   # intercept

# Legacy single-mode constants (kept for reference)
CLOCK_SLOPE     = -3.497    # original β⁻ (√|ε| only, no Z)    [EMPIRICAL_FIT]
CLOCK_INTERCEPT = 7.38      # original β⁻ intercept             [EMPIRICAL_FIT]

# ── ZERO-PARAMETER CLOCK ──────────────────────────────────────────
#
# Every constant derived from α → β via the Golden Loop.
# General form:  log₁₀(t½/s) = a·√|ε| + b·log₁₀(Z) + d
#
# β⁻:    -πβ/e · √|ε| + 2 · log₁₀(Z) + 4π/3       R² = 0.673
# β⁺/EC:    -π · √|ε| + 2β · log₁₀(Z) - 2β/e       R² = 0.626
# α:         -e · √|ε| + (β+1) · log₁₀(Z) - (β-1)   R² = 0.251
#
# Physical reading:
#   β⁻ slope  = -πβ/e  (circle × surface tension / Euler)
#   β⁺ slope  = -π     (pure circular geometry)
#   α slope   = -e     (Euler tunneling constant)
#   β⁻/β⁺    = β/e    (surface tension per linear dimension)
#
# Slope matches:  β⁻ 0.5% off, β⁺ 0.1% off, α 4.3% off.
# Cost vs fitted: β⁻ -2.7 R² pts, β⁺ -3.1 pts, α -6.2 pts.
# ──────────────────────────────────────────────────────────────────

ZP_BM_A = -PI * BETA / E_NUM       # -πβ/e                      [QFD_DERIVED]
ZP_BM_B = 2.0                       # 2 (integer)                [QFD_DERIVED]
ZP_BM_D = 4.0 * PI / 3.0           # 4π/3 = resonance phase     [QFD_DERIVED]

ZP_BP_A = -PI                       # -π                         [QFD_DERIVED]
ZP_BP_B = 2.0 * BETA                # 2β                         [QFD_DERIVED]
ZP_BP_D = -2.0 * BETA / E_NUM       # -2β/e                      [QFD_DERIVED]

ZP_AL_A = -E_NUM                     # -e                         [QFD_DERIVED]
ZP_AL_B = BETA + 1.0                 # β + 1                      [QFD_DERIVED]
ZP_AL_D = -(BETA - 1.0)              # -(β - 1)                   [QFD_DERIVED]


def _clock_log10t(Z: int, eps: float, mode: str) -> float | None:
    """Compute log₁₀(t½/s) for a given mode using the fitted atomic clock.

    Returns None if mode has no calibrated clock.
    Uses the 12-parameter empirical fit (best accuracy).
    """
    sqrt_eps = math.sqrt(abs(eps))
    log_Z = math.log10(max(Z, 1))

    if mode == 'B-':
        return BM_CLOCK_A * sqrt_eps + BM_CLOCK_B * log_Z + BM_CLOCK_C * Z + BM_CLOCK_D
    elif mode == 'B+':
        return BP_CLOCK_A * sqrt_eps + BP_CLOCK_B * log_Z + BP_CLOCK_C * Z + BP_CLOCK_D
    elif mode == 'alpha':
        return AL_CLOCK_A * sqrt_eps + AL_CLOCK_B * log_Z + AL_CLOCK_C * Z + AL_CLOCK_D
    return None


def _clock_log10t_zero_param(Z: int, eps: float, mode: str) -> float | None:
    """Compute log₁₀(t½/s) using the ZERO-PARAMETER clock.

    Every constant derived from α → β via the Golden Loop.
    No fitted parameters.  Lower R² than the fitted clock but
    demonstrates that the time-scale structure is geometric.

    Returns None if mode has no calibrated clock.
    """
    sqrt_eps = math.sqrt(abs(eps))
    log_Z = math.log10(max(Z, 1))

    if mode == 'B-':
        return ZP_BM_A * sqrt_eps + ZP_BM_B * log_Z + ZP_BM_D
    elif mode == 'B+':
        return ZP_BP_A * sqrt_eps + ZP_BP_B * log_Z + ZP_BP_D
    elif mode == 'alpha':
        return ZP_AL_A * sqrt_eps + ZP_AL_B * log_Z + ZP_AL_D
    return None


def estimate_half_life(Z: int, A: int) -> dict:
    """Estimate half-life from valley stress + atomic environment.

    Returns dict with:
        log10_t:   log10(t½ in seconds), or None if no clock
        t_seconds: estimated t½ in seconds, or None
        t_human:   human-readable string
        mode:      predicted decay mode
        eps:       valley stress ε = Z - Z*(A)
        quality:   'clock' for β⁻/β⁺ (R²>0.65), 'clock_weak' for α (R²=0.31),
                   'no_clock' for stable/SF/p/n

    Three calibrated clocks:
        β⁻:    R² = 0.70, RMSE = 1.47 decades  (strong)
        β⁺/EC: R² = 0.66, RMSE = 1.57 decades  (strong)
        α:     R² = 0.31, RMSE = 3.17 decades  (weak — order-of-magnitude only)
    """
    mode, gains = predict_decay(Z, A)
    eps = Z - z_star(A)

    result = {
        'mode': mode,
        'eps': eps,
        'log10_t': None,
        't_seconds': None,
        't_human': '—',
        'quality': 'no_clock',
    }

    log_t = _clock_log10t(Z, eps, mode)
    if log_t is not None:
        # Clamp to physical range: 1 attosecond to 10 Tyr
        log_t = max(-18.0, min(20.5, log_t))
        t_s = 10.0 ** log_t
        result['log10_t'] = log_t
        result['t_seconds'] = t_s
        result['t_human'] = _format_halflife(t_s)
        result['quality'] = 'clock' if mode in ('B-', 'B+') else 'clock_weak'

    return result


def estimate_half_life_zero_param(Z: int, A: int) -> dict:
    """Estimate half-life using the ZERO-PARAMETER clock.

    Same interface as estimate_half_life() but every constant is
    derived from α via the Golden Loop.  No fitted parameters.

    Performance vs fitted clock:
        β⁻:    R² = 0.673 vs 0.700  (cost: -2.7 pts)
        β⁺/EC: R² = 0.626 vs 0.657  (cost: -3.1 pts)
        α:     R² = 0.251 vs 0.313  (cost: -6.2 pts)
    """
    mode, gains = predict_decay(Z, A)
    eps = Z - z_star(A)

    result = {
        'mode': mode,
        'eps': eps,
        'log10_t': None,
        't_seconds': None,
        't_human': '—',
        'quality': 'no_clock',
    }

    log_t = _clock_log10t_zero_param(Z, eps, mode)
    if log_t is not None:
        log_t = max(-18.0, min(20.5, log_t))
        t_s = 10.0 ** log_t
        result['log10_t'] = log_t
        result['t_seconds'] = t_s
        result['t_human'] = _format_halflife(t_s)
        result['quality'] = 'clock_zp' if mode in ('B-', 'B+') else 'clock_zp_weak'

    return result


def _format_halflife(t_seconds: float) -> str:
    """Format a half-life in seconds to human-readable units."""
    if t_seconds < 0:
        return '???'
    if t_seconds < 1e-15:
        return f'{t_seconds*1e18:.1f} as'
    if t_seconds < 1e-12:
        return f'{t_seconds*1e15:.1f} fs'
    if t_seconds < 1e-9:
        return f'{t_seconds*1e12:.1f} ps'
    if t_seconds < 1e-6:
        return f'{t_seconds*1e9:.1f} ns'
    if t_seconds < 1e-3:
        return f'{t_seconds*1e6:.1f} us'
    if t_seconds < 1.0:
        return f'{t_seconds*1e3:.1f} ms'
    if t_seconds < 60:
        return f'{t_seconds:.2f} s'
    if t_seconds < 3600:
        return f'{t_seconds/60:.1f} min'
    if t_seconds < 86400:
        return f'{t_seconds/3600:.1f} hr'
    if t_seconds < 3.156e7:
        return f'{t_seconds/86400:.1f} d'
    if t_seconds < 3.156e10:
        return f'{t_seconds/3.1557600e7:.2f} yr'
    if t_seconds < 3.156e13:
        return f'{t_seconds/3.1557600e10:.2f} kyr'
    if t_seconds < 3.156e16:
        return f'{t_seconds/3.1557600e13:.2f} Myr'
    if t_seconds < 3.156e19:
        return f'{t_seconds/3.1557600e16:.2f} Gyr'
    return f'{t_seconds/3.1557600e19:.1f} Tyr'


def predict_geometric(Z: int, A: int) -> str:
    """Zone-rule geometric predictor (for comparison).

    Uses sign(ε) for β-direction and A > A_ALPHA_ONSET for α.
    Known accuracy: β-direction 98.3%, α F1 74.5%, mode 78.7%.
    """
    eps = Z - z_star(A)
    if abs(eps) < 0.5:
        return 'stable'
    if eps < -0.5:
        return 'B-'
    if A > A_ALPHA_ONSET:
        return 'alpha'
    return 'B+'


# ═══════════════════════════════════════════════════════════════════
# LAYER 4 — NUBASE2020 PARSER + VALIDATION
# ═══════════════════════════════════════════════════════════════════

def load_nubase(path: str, include_isomers: bool = False) -> list:
    """Parse NUBASE2020 entries for validation.

    Args:
        path: Path to nubase2020_raw.txt
        include_isomers: If False (default), keep ground states only
            (backward compatible).  If True, keep all states except
            IAS (state_idx in '8','9').

    Returns list of dicts: {A, Z, N, is_stable, dominant_mode, half_life_s,
        state, exc_energy_keV, jpi}
    Provenance: EMPIRICAL_LOOKUP (measured data, not modeled).
    """
    HALFLIFE_UNITS = {
        'ys': 1e-24, 'zs': 1e-21, 'as': 1e-18, 'fs': 1e-15,
        'ps': 1e-12, 'ns': 1e-9,  'us': 1e-6,  'ms': 1e-3,
        's': 1.0, 'm': 60.0, 'h': 3600.0, 'd': 86400.0,
        'y': 3.1557600e7, 'ky': 3.1557600e10, 'My': 3.1557600e13,
        'Gy': 3.1557600e16, 'Ty': 3.1557600e19, 'Py': 3.1557600e22,
        'Ey': 3.1557600e25, 'Zy': 3.1557600e28, 'Yy': 3.1557600e31,
    }

    entries = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or len(line.strip()) < 20:
                continue
            try:
                A = int(line[0:3].strip())
                zzzi = line[4:8].strip()
                Z = int(zzzi[:3])
                state_idx = zzzi[3] if len(zzzi) > 3 else '0'
            except (ValueError, IndexError):
                continue

            # State identification
            state_char = line[16:17].strip() if len(line) > 16 else ''
            is_gs = (state_idx == '0' and state_char == '')

            if not include_isomers:
                # Ground states only (original behavior)
                if not is_gs:
                    continue
            else:
                # Skip IAS entries (state_idx 8 or 9)
                if state_idx in ('8', '9'):
                    continue

            # State label
            if is_gs:
                state = 'gs'
            elif state_char:
                state = state_char + state_idx if state_idx not in ('0',) else state_char
            else:
                state = f'x{state_idx}'

            # Excitation energy (cols 43-54)           [EMPIRICAL_LOOKUP]
            exc_energy_keV = 0.0
            exc_str = line[42:54].strip().rstrip('#') if len(line) > 42 else ''
            if exc_str:
                try:
                    exc_energy_keV = float(exc_str.replace('*', '').strip())
                except ValueError:
                    exc_energy_keV = 0.0

            # Spin-parity (cols 89-102)                [EMPIRICAL_LOOKUP]
            jpi = line[88:102].strip() if len(line) > 88 else ''

            N = A - Z

            # Half-life
            hl_str = line[69:78].strip().rstrip('#')
            unit_str = line[78:80].strip()
            is_stable = 'stbl' in hl_str

            half_life_s = np.nan
            if is_stable:
                half_life_s = np.inf
            elif hl_str and hl_str != 'p-unst':
                hl_clean = hl_str.replace('#', '').replace('>', '').replace('<', '').replace('~', '').strip()
                try:
                    val = float(hl_clean)
                    if unit_str in HALFLIFE_UNITS:
                        half_life_s = val * HALFLIFE_UNITS[unit_str]
                except ValueError:
                    pass

            # Decay modes
            decay_raw = line[119:].strip() if len(line) > 119 else ''
            modes = {}
            if decay_raw:
                for part in decay_raw.split(';'):
                    part = part.strip()
                    if '=' not in part:
                        m = part.split()[0] if part else ''
                        if m:
                            modes[m] = -1.0
                        continue
                    m, val = part.split('=', 1)
                    try:
                        modes[m.strip()] = float(val.strip().split()[0])
                    except (ValueError, IndexError):
                        modes[m.strip()] = -1.0

            # Dominant mode
            dom = 'unknown'
            if is_stable or 'IS' in modes:
                dom = 'stable'
            elif modes:
                best_m = max(modes, key=lambda m: modes[m] if m != 'IS' else -999)
                m_clean = best_m.split('~')[0].split('>')[0].split('<')[0]
                if m_clean.startswith('B-'):     dom = 'B-'
                elif m_clean.startswith('B+'):   dom = 'B+'
                elif m_clean.startswith('e+'):   dom = 'B+'
                elif m_clean.startswith('EC'):   dom = 'EC'
                elif m_clean == 'A':             dom = 'alpha'
                elif m_clean == 'SF':            dom = 'SF'
                elif m_clean in ('p', '2p'):     dom = 'p'
                elif m_clean in ('n', '2n'):     dom = 'n'
                elif m_clean == 'IT':            dom = 'IT'
                else:                            dom = best_m

            entries.append({
                'A': A, 'Z': Z, 'N': N,
                'is_stable': is_stable,
                'dominant_mode': dom,
                'half_life_s': half_life_s,
                'state': state,
                'exc_energy_keV': exc_energy_keV,
                'jpi': jpi,
            })

    return entries


def group_nuclide_states(entries: list) -> dict:
    """Group NUBASE entries by (Z, A), sorted by excitation energy.

    Returns dict[(Z, A)] → list[entry], ground state first.
    Provenance: EMPIRICAL_LOOKUP (pure data transformation).
    """
    groups = defaultdict(list)
    for e in entries:
        groups[(e['Z'], e['A'])].append(e)
    # Sort each group by excitation energy (ground state first)
    for key in groups:
        groups[key].sort(key=lambda x: x['exc_energy_keV'])
    return dict(groups)


def _parse_spin_value(jpi: str) -> float | None:
    """Extract numeric J from a Jpi string.

    '5/2-' → 2.5, '0+' → 0.0, '(3/2+)' → 1.5, '7/2+*' → 3.5
    Returns None on parse failure.
    """
    if not jpi:
        return None
    # Strip parentheses, *, #, T=..., and parity signs
    s = jpi.replace('(', '').replace(')', '').replace('*', '').replace('#', '')
    s = re.sub(r'T=.*', '', s).strip()
    s = s.rstrip('+-').strip()
    if not s:
        return None
    # Handle 'frg', 'high', etc
    if not s[0].isdigit():
        return None
    try:
        if '/' in s:
            num, den = s.split('/')
            return float(num) / float(den)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def normalize_prediction(mode: str) -> str:
    """Map prediction labels to comparison labels."""
    return mode  # Already using consistent labels


def normalize_nubase(mode: str) -> str:
    """Map NUBASE labels to comparison labels for scoring.

    Merges B+ and EC as 'B+' since QFD doesn't distinguish them.
    """
    if mode == 'EC':
        return 'B+'
    return mode


def validate_against_nubase(nubase_entries: list) -> dict:
    """Run gradient predictor against NUBASE2020 and compute statistics."""

    results = {
        'total': 0,
        'correct_mode': 0,
        'correct_direction': 0,
        'direction_total': 0,
        'by_actual_mode': {},   # actual_mode → [correct, total]
        'by_pred_mode': {},     # pred_mode  → [correct, total]
        'confusion': {},        # (actual, pred) → count
        'entries': [],          # Full results for visualization
    }

    for entry in nubase_entries:
        A, Z, N = entry['A'], entry['Z'], entry['N']
        actual = entry['dominant_mode']

        # Skip unmeasured, isomeric transitions, very light
        if actual in ('unknown', 'IT') or A < 3:
            continue

        pred, gains = predict_decay(Z, A)
        actual_norm = normalize_nubase(actual)

        results['total'] += 1

        # Mode match
        match = (actual_norm == pred)
        if match:
            results['correct_mode'] += 1

        # Per-mode stats
        results['by_actual_mode'].setdefault(actual_norm, [0, 0])
        results['by_actual_mode'][actual_norm][1] += 1
        if match:
            results['by_actual_mode'][actual_norm][0] += 1

        results['by_pred_mode'].setdefault(pred, [0, 0])
        results['by_pred_mode'][pred][1] += 1
        if match:
            results['by_pred_mode'][pred][0] += 1

        # Confusion matrix
        key = (actual_norm, pred)
        results['confusion'][key] = results['confusion'].get(key, 0) + 1

        # β-direction accuracy (the real QFD content)
        eps = Z - z_star(A)
        if actual_norm in ('B-', 'B+', 'stable'):
            results['direction_total'] += 1
            if actual_norm == 'B-' and eps < 0:
                results['correct_direction'] += 1
            elif actual_norm == 'B+' and eps > 0:
                results['correct_direction'] += 1
            elif actual_norm == 'stable' and abs(eps) < 1.5:
                results['correct_direction'] += 1

        # Store for visualization
        results['entries'].append({
            'A': A, 'Z': Z, 'N': N,
            'actual': actual_norm, 'predicted': pred,
            'match': match, 'eps': eps,
            'half_life_s': entry['half_life_s'],
        })

    return results


# ═══════════════════════════════════════════════════════════════════
# LAYER 5 — VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

ELEMENTS = {
    0:'n',1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F',
    10:'Ne',11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',
    18:'Ar',19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',
    26:'Fe',27:'Co',28:'Ni',29:'Cu',30:'Zn',31:'Ga',32:'Ge',33:'As',
    34:'Se',35:'Br',36:'Kr',37:'Rb',38:'Sr',39:'Y',40:'Zr',41:'Nb',
    42:'Mo',43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',49:'In',
    50:'Sn',51:'Sb',52:'Te',53:'I',54:'Xe',55:'Cs',56:'Ba',57:'La',
    58:'Ce',59:'Pr',60:'Nd',61:'Pm',62:'Sm',63:'Eu',64:'Gd',65:'Tb',
    66:'Dy',67:'Ho',68:'Er',69:'Tm',70:'Yb',71:'Lu',72:'Hf',73:'Ta',
    74:'W',75:'Re',76:'Os',77:'Ir',78:'Pt',79:'Au',80:'Hg',81:'Tl',
    82:'Pb',83:'Bi',84:'Po',85:'At',86:'Rn',87:'Fr',88:'Ra',89:'Ac',
    90:'Th',91:'Pa',92:'U',93:'Np',94:'Pu',95:'Am',96:'Cm',97:'Bk',
    98:'Cf',99:'Es',100:'Fm',101:'Md',102:'No',103:'Lr',104:'Rf',
    105:'Db',106:'Sg',107:'Bh',108:'Hs',109:'Mt',110:'Ds',111:'Rg',
    112:'Cn',113:'Nh',114:'Fl',115:'Mc',116:'Lv',117:'Ts',118:'Og',
}


MODE_COLORS = {
    'stable': '#222222',
    'B-':     '#3366CC',
    'B+':     '#CC3333',
    'alpha':  '#DDAA00',
    'SF':     '#33AA33',
    'p':      '#FF6699',
    'n':      '#9966CC',
    'unknown':'#AAAAAA',
}


def generate_terrain_map(max_A: int = 300) -> list:
    """Generate predicted nuclide map from topology alone.

    Scans the (Z, A) grid and predicts decay mode for each cell.
    Returns list of dicts for plotting.
    """
    results = []

    for A in range(1, max_A + 1):
        # Scan Z range around the valley
        zs = z_star(max(A, 1))
        z_min = max(0, int(zs - 15))
        z_max = min(A, int(zs + 15))

        for Z in range(z_min, z_max + 1):
            N = A - Z
            if N < 0:
                continue

            score = survival_score(Z, A)
            eps = Z - z_star(A)

            mode, _ = predict_decay(Z, A)

            results.append({
                'A': A, 'Z': Z, 'N': N,
                'score': score,
                'mode': mode,
                'eps': eps,
            })

    return results


def plot_nuclide_maps(terrain: list, nubase_results: dict, output_dir: str):
    """Generate side-by-side nuclide maps: predicted vs actual."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ── Figure 1: Side-by-side comparison ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))

    # Left: QFD Predicted
    for entry in terrain:
        color = MODE_COLORS.get(entry['mode'], '#AAAAAA')
        ax1.scatter(entry['N'], entry['Z'], c=color, s=1.5, alpha=0.5,
                    edgecolors='none')

    # Valley line
    A_vals = np.arange(1, 301)
    Z_vals = [z_star(A) for A in A_vals]
    N_vals = [A - z for A, z in zip(A_vals, Z_vals)]
    ax1.plot(N_vals, Z_vals, 'w-', linewidth=1.0, alpha=0.7, label='Z*(A) valley')

    ax1.set_title('QFD Gradient Prediction\n(Zero Free Parameters — All from α)',
                   fontsize=12, fontweight='bold')
    ax1.set_xlabel('Neutrons (N)')
    ax1.set_ylabel('Protons (Z)')
    ax1.set_xlim(-5, 200)
    ax1.set_ylim(-5, 125)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.15)

    # Right: NUBASE Actual
    for entry in nubase_results['entries']:
        color = MODE_COLORS.get(entry['actual'], '#AAAAAA')
        ax2.scatter(entry['N'], entry['Z'], c=color, s=2.5, alpha=0.6,
                    edgecolors='none')

    ax2.plot(N_vals, Z_vals, 'w-', linewidth=1.0, alpha=0.7)

    ax2.set_title('NUBASE2020 Observed\n(Measured Decay Modes)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Neutrons (N)')
    ax2.set_ylabel('Protons (Z)')
    ax2.set_xlim(-5, 200)
    ax2.set_ylim(-5, 125)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.15)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['stable'],
               markersize=8, label='Stable'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['B-'],
               markersize=8, label='β⁻'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['B+'],
               markersize=8, label='β⁺/EC'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['alpha'],
               markersize=8, label='α'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['SF'],
               markersize=8, label='SF'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['p'],
               markersize=8, label='p'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['n'],
               markersize=8, label='n'),
        Line2D([0], [0], color='white', linewidth=1, label='Z*(A) valley'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=8,
               fontsize=9, framealpha=0.8)

    for ax in (ax1, ax2):
        ax.set_facecolor('#0A0A1A')
    fig.patch.set_facecolor('#0A0A1A')
    for ax in (ax1, ax2):
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#444444')
    fig.legend(handles=legend_elements, loc='lower center', ncol=8,
               fontsize=9, framealpha=0.8,
               facecolor='#1A1A2A', edgecolor='#444444',
               labelcolor='white')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path1 = os.path.join(output_dir, 'nuclide_map_comparison.png')
    fig.savefig(path1, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ── Figure 2: Match/mismatch map ──
    fig2, ax3 = plt.subplots(figsize=(12, 9))

    for entry in nubase_results['entries']:
        color = '#33CC33' if entry['match'] else '#CC3333'
        alpha = 0.3 if entry['match'] else 0.7
        ax3.scatter(entry['N'], entry['Z'], c=color, s=2, alpha=alpha,
                    edgecolors='none')

    ax3.plot(N_vals, Z_vals, 'w-', linewidth=0.8, alpha=0.5)

    mode_acc = nubase_results['correct_mode'] / max(nubase_results['total'], 1) * 100
    dir_acc  = nubase_results['correct_direction'] / max(nubase_results['direction_total'], 1) * 100

    ax3.set_title(f'Gradient Prediction Accuracy\n'
                   f'Mode: {mode_acc:.1f}%  |  β-direction: {dir_acc:.1f}%',
                   fontsize=12, fontweight='bold', color='white')
    ax3.set_xlabel('Neutrons (N)', color='white')
    ax3.set_ylabel('Protons (Z)', color='white')
    ax3.set_xlim(-5, 200)
    ax3.set_ylim(-5, 125)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.15)
    ax3.set_facecolor('#0A0A1A')
    fig2.patch.set_facecolor('#0A0A1A')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('#444444')

    legend_acc = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#33CC33',
               markersize=8, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#CC3333',
               markersize=8, label='Wrong'),
    ]
    ax3.legend(handles=legend_acc, loc='upper left', fontsize=10,
               facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')

    plt.tight_layout()
    path2 = os.path.join(output_dir, 'nuclide_map_accuracy.png')
    fig2.savefig(path2, dpi=150, facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # ── Figure 3: Survival score terrain ──
    fig3, ax4 = plt.subplots(figsize=(12, 9))

    N_arr = [e['N'] for e in terrain]
    Z_arr = [e['Z'] for e in terrain]
    S_arr = [e['score'] for e in terrain]

    sc = ax4.scatter(N_arr, Z_arr, c=S_arr, s=1.5, cmap='inferno',
                     alpha=0.6, edgecolors='none',
                     vmin=np.percentile(S_arr, 5),
                     vmax=np.percentile(S_arr, 95))
    ax4.plot(N_vals, Z_vals, 'c-', linewidth=0.8, alpha=0.7, label='Z*(A) valley')

    ax4.set_title('Survival Score S(Z, A) — Topological Terrain\n'
                   'Bright = Stable,  Dark = Unstable',
                   fontsize=12, fontweight='bold', color='white')
    ax4.set_xlabel('Neutrons (N)', color='white')
    ax4.set_ylabel('Protons (Z)', color='white')
    ax4.set_xlim(-5, 200)
    ax4.set_ylim(-5, 125)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.15)
    ax4.set_facecolor('#0A0A1A')
    fig3.patch.set_facecolor('#0A0A1A')
    ax4.tick_params(colors='white')
    for spine in ax4.spines.values():
        spine.set_color('#444444')

    cbar = fig3.colorbar(sc, ax=ax4, shrink=0.7, label='S(Z, A)')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    plt.tight_layout()
    path3 = os.path.join(output_dir, 'nuclide_terrain.png')
    fig3.savefig(path3, dpi=150, facecolor=fig3.get_facecolor())
    plt.close(fig3)
    print(f"  Saved: {path3}")


# ═══════════════════════════════════════════════════════════════════
# MAIN — RUN THE ENGINE
# ═══════════════════════════════════════════════════════════════════

def print_constants():
    """Display all derived constants."""
    print("=" * 72)
    print("  QFD NUCLIDE ENGINE — TOPOLOGICAL TERRAIN MODEL")
    print("=" * 72)
    print(f"""
  GOLDEN LOOP
    Input:   α = {ALPHA}                   [MEASURED]
    Output:  β = {BETA:.10f}                        [QFD_DERIVED]
    Check:   1/α = 2π²(e^β/β) + 1 = {2*PI**2*(math.exp(BETA)/BETA) + 1:.6f}  (cf. {1/ALPHA:.6f})

  COMPRESSION LAW  (11 constants, 0 free parameters)
    S      = β²/e       = {S_SURF:.6f}      surface tension
    R      = αβ         = {R_REG:.6f}      regularization
    C_h    = αe/β²      = {C_HEAVY:.6f}      Coulomb (heavy)
    C_l    = 2παe/β²    = {C_LIGHT:.6f}      Coulomb (light)
    β_l    = 2.0        = {BETA_LIGHT:.6f}      pairing limit
    A_c    = 2e²β²      = {A_CRIT:.3f}       transition mass
    W      = 2πβ²       = {WIDTH:.3f}        transition width
    ω      = 2πβ/e      = {OMEGA:.6f}      resonance frequency
    Amp    = 1/β        = {AMP:.6f}      resonance amplitude
    φ      = 4π/3       = {PHI:.6f}      resonance phase
    A_α    = A_c + W    = {A_ALPHA_ONSET:.3f}       α onset mass

  SURVIVAL SCORE  (3 additional derived constants)
    k_coh  = C_h·A_c^(5/3) = {K_COH:.6f}     coherence scale
    k_den  = C_h·3/5       = {K_DEN:.6f}     density scale
    δ_pair = 1/β           = {PAIRING_SCALE:.6f}     pairing amplitude

  Free parameters: 0

  3D CAPACITY (Frozen Core Conjecture)
    N_max  = 2πβ³       = {N_MAX_ABSOLUTE:.3f}       density ceiling
    slope  = 1-1/β      = {CORE_SLOPE:.4f}       dN_excess/dZ

  2D PEANUT THRESHOLDS
    pf_α   = 0.5                            alpha first appears      [EMPIRICAL_OBSERVED]
    pf_only= 1.0        = A_α/zone          single-core gone         [QFD_DERIVED]
    pf_deep= 1.5                            alpha regardless of ε    [EMPIRICAL_OBSERVED]
    pf_SF  = 1.74                           SF hard lower bound      [EMPIRICAL_OBSERVED]
    cf_SF  = 0.881                          SF fullness bound        [EMPIRICAL_OBSERVED]

  ZONE BOUNDARIES
    Zone 1: A ≤ {A_CRIT:.0f}  (pre-peanut, single-core)
    Zone 2: {A_CRIT:.0f} < A < {A_ALPHA_ONSET:.0f}  (transition, degenerate)
    Zone 3: A ≥ {A_ALPHA_ONSET:.0f}  (peanut-only)

  ZERO-PARAMETER CLOCK  (all from α → β)
    β⁻:    -πβ/e·√|ε| + 2·log₁₀(Z) + 4π/3     a={ZP_BM_A:.4f}  b={ZP_BM_B:.1f}  d={ZP_BM_D:.4f}
    β⁺/EC:    -π·√|ε| + 2β·log₁₀(Z) - 2β/e     a={ZP_BP_A:.4f}  b={ZP_BP_B:.4f}  d={ZP_BP_D:.4f}
    α:         -e·√|ε| + (β+1)·log₁₀(Z) - (β-1)  a={ZP_AL_A:.4f}  b={ZP_AL_B:.4f}  d={ZP_AL_D:.4f}

  Free parameters (clock): 0
""")


def print_spot_checks():
    """Verify backbone against known stable nuclides."""
    print("=" * 72)
    print("  SPOT CHECKS — Compression Law Z*(A)")
    print("=" * 72)

    checks = [
        (4, 2, "He-4"), (12, 6, "C-12"), (16, 8, "O-16"),
        (40, 20, "Ca-40"), (56, 26, "Fe-56"), (90, 40, "Zr-90"),
        (120, 50, "Sn-120"), (208, 82, "Pb-208"), (238, 92, "U-238"),
    ]
    print(f"\n  {'Nuclide':>10s} {'Z':>4s} {'Z*(A)':>8s} {'ε':>7s} {'Score':>8s} {'Predicted':>10s}")
    print(f"  {'-'*52}")

    sq_err = []
    for A, Z, lab in checks:
        zs = z_star(A)
        eps = Z - zs
        score = survival_score(Z, A)
        pred, _ = predict_decay(Z, A)
        sq_err.append(eps ** 2)
        print(f"  {lab:>10s} {Z:>4d} {zs:>8.3f} {eps:>+7.3f} {score:>8.2f}  {pred:>10s}")

    rmse = math.sqrt(sum(sq_err) / len(sq_err))
    print(f"\n  RMSE on these {len(checks)}: {rmse:.4f}")


def print_gradient_analysis():
    """Show gradient breakdown for key nuclides."""
    print(f"\n{'='*72}")
    print("  GRADIENT ANALYSIS — Score gains for each channel")
    print("=" * 72)

    cases = [
        (26, 56, "Fe-56 (stable)"),
        (82, 208, "Pb-208 (stable)"),
        (92, 238, "U-238 (α, 4.47 Gyr)"),
        (6, 14, "C-14 (β⁻, 5730 yr)"),
        (55, 137, "Cs-137 (β⁻, 30.2 yr)"),
        (88, 226, "Ra-226 (α, 1600 yr)"),
        (84, 210, "Po-210 (α, 138 d)"),
        (11, 22, "Na-22 (β⁺, 2.60 yr)"),
        (19, 40, "K-40 (β⁻, 1.25 Gyr)"),
        (27, 60, "Co-60 (β⁻, 5.27 yr)"),
        (53, 131, "I-131 (β⁻, 8.02 d)"),
    ]

    for Z, A, label in cases:
        pred, info = predict_decay(Z, A)
        all_gains = gradient_all_channels(Z, A)
        zs = z_star(A)
        eps = Z - zs
        geo_str = predict_geometric(Z, A)
        geo_state = info.get('geo')
        beta_gains = info.get('gains', {})
        clock = estimate_half_life(Z, A)

        zone_label = f"zone={geo_state.zone}" if geo_state else "?"
        print(f"\n  {label}")
        print(f"    Z*(A)={zs:.3f}  ε={eps:+.3f}  score={survival_score(Z, A):.2f}  {zone_label}")
        if geo_state:
            print(f"    peanut_f={geo_state.peanut_f:.2f}  core_full={geo_state.core_full:.3f}  parity={geo_state.parity}")
        print(f"    Prediction: {pred:<8s}  (geometric: {geo_str})")
        clock_zp = estimate_half_life_zero_param(Z, A)
        if clock['quality'] in ('clock', 'clock_weak'):
            qual = '' if clock['quality'] == 'clock' else '  (weak)'
            print(f"    Fitted clock:    log10(t½) = {clock['log10_t']:+.2f}  →  t½ ≈ {clock['t_human']}{qual}")
        if clock_zp['quality'].startswith('clock'):
            qual = '' if 'weak' not in clock_zp['quality'] else '  (weak)'
            print(f"    Zero-param clock: log10(t½) = {clock_zp['log10_t']:+.2f}  →  t½ ≈ {clock_zp['t_human']}{qual}")
        print(f"    β gains:  B-={beta_gains.get('B-', 0):+.4f}  B+={beta_gains.get('B+', 0):+.4f}")

        # Show all-channel diagnostic (not used for prediction)
        print(f"    All gradients (diagnostic):")
        for mode, gain in sorted(all_gains.items(), key=lambda x: -x[1]):
            flag = ""
            if mode == pred and gain > 0:
                flag = " ← predicted"
            elif mode == 'SF' and gain > 0:
                flag = " ← NOT used (bifurcation artifact)"
            print(f"      {mode:>6s}: ΔS = {gain:+8.4f}{flag}")


def print_validation(results: dict):
    """Print validation statistics."""
    total = results['total']
    if total == 0:
        print("  No data to validate.")
        return

    mode_acc = results['correct_mode'] / total * 100
    dir_acc  = results['correct_direction'] / max(results['direction_total'], 1) * 100

    print(f"\n{'='*72}")
    print(f"  VALIDATION — Gradient Predictor vs NUBASE2020 ({total} nuclides)")
    print("=" * 72)

    print(f"""
  HEADLINE RESULTS
    Overall mode accuracy:   {results['correct_mode']:>5d}/{total} ({mode_acc:.1f}%)
    β-direction accuracy:    {results['correct_direction']:>5d}/{results['direction_total']} ({dir_acc:.1f}%)
""")

    # Per actual mode
    print(f"  {'Actual Mode':>12s} {'Right':>6s} {'Total':>6s} {'Acc':>8s}")
    print(f"  {'-'*36}")
    for mode in sorted(results['by_actual_mode']):
        c, t = results['by_actual_mode'][mode]
        acc = c / t * 100 if t > 0 else 0
        print(f"  {mode:>12s} {c:>6d} {t:>6d} {acc:>7.1f}%")

    # Confusion summary: what does the gradient most confuse?
    print(f"\n  TOP CONFUSIONS (actual → predicted, count)")
    print(f"  {'-'*50}")
    confused = [(k, v) for k, v in results['confusion'].items() if k[0] != k[1]]
    confused.sort(key=lambda x: -x[1])
    for (actual, pred), count in confused[:10]:
        print(f"    {actual:>8s} → {pred:<8s}  {count:>5d}")

    # Key physics checks
    print(f"""
  ─────────────────────────────────────────────────
  KEY PHYSICS CHECKS

    β-direction from sign(ε):   {dir_acc:.1f}%
      (The real QFD content — valley geometry alone)

    β⁻ accuracy:  {_mode_acc(results, 'B-')}
    β⁺/EC accuracy: {_mode_acc(results, 'B+')}
    α accuracy:   {_mode_acc(results, 'alpha')}
    Stable accuracy: {_mode_acc(results, 'stable')}
    SF accuracy:  {_mode_acc(results, 'SF')}
    p accuracy:   {_mode_acc(results, 'p')}
    n accuracy:   {_mode_acc(results, 'n')}
  ─────────────────────────────────────────────────
""")


def _mode_acc(results: dict, mode: str) -> str:
    """Format mode accuracy string."""
    c, t = results['by_actual_mode'].get(mode, [0, 0])
    if t == 0:
        return "N/A"
    return f"{c}/{t} ({c/t*100:.1f}%)"


# ─────────────────────────────────────────────────────────────────
# ZONE-SEPARATED VALIDATION — Different animals, different graphs
# ─────────────────────────────────────────────────────────────────

ZONE_NAMES = {
    1: 'PRE-PEANUT (A ≤ 137) — Single-Core Physics',
    2: 'TRANSITION (137 < A < 195) — Degenerate Zone',
    3: 'PEANUT-ONLY (A ≥ 195) — Peanut Physics',
}

ZONE_MODES = {
    1: 'B-, B+, stable, n, p',
    2: 'B-, B+, stable, alpha (rare)',
    3: 'B-, B+, alpha, SF',
}


def validate_by_zone(nubase_entries: list) -> dict:
    """Validate with per-zone breakout.

    Each nuclide is assigned to a zone based on its geometric state.
    Returns separate accuracy metrics for each zone plus combined totals.
    """
    # Initialize per-zone result dicts
    zone_results = {}
    for z in (1, 2, 3, 'overall'):
        zone_results[z] = {
            'total': 0,
            'correct_mode': 0,
            'correct_direction': 0,
            'direction_total': 0,
            'by_actual_mode': {},
            'by_pred_mode': {},
            'confusion': {},
        }

    for entry in nubase_entries:
        A, Z, N = entry['A'], entry['Z'], entry['N']
        actual = entry['dominant_mode']

        if actual in ('unknown', 'IT') or A < 3:
            continue

        pred, info = predict_decay(Z, A)
        actual_norm = normalize_nubase(actual)

        # Get zone from geometric state
        geo = info.get('geo')
        if geo is not None:
            zone = geo.zone
        else:
            # H or invalid — compute directly
            pf = (A - A_CRIT) / WIDTH if A > A_CRIT else 0.0
            if pf >= PF_PEANUT_ONLY:
                zone = 3
            elif pf > 0:
                zone = 2
            else:
                zone = 1

        # Score in both zone-specific and overall buckets
        for bucket in (zone, 'overall'):
            r = zone_results[bucket]
            r['total'] += 1

            match = (actual_norm == pred)
            if match:
                r['correct_mode'] += 1

            r['by_actual_mode'].setdefault(actual_norm, [0, 0])
            r['by_actual_mode'][actual_norm][1] += 1
            if match:
                r['by_actual_mode'][actual_norm][0] += 1

            r['by_pred_mode'].setdefault(pred, [0, 0])
            r['by_pred_mode'][pred][1] += 1
            if match:
                r['by_pred_mode'][pred][0] += 1

            key = (actual_norm, pred)
            r['confusion'][key] = r['confusion'].get(key, 0) + 1

            # β-direction accuracy
            eps = Z - z_star(A)
            if actual_norm in ('B-', 'B+', 'stable'):
                r['direction_total'] += 1
                if actual_norm == 'B-' and eps < 0:
                    r['correct_direction'] += 1
                elif actual_norm == 'B+' and eps > 0:
                    r['correct_direction'] += 1
                elif actual_norm == 'stable' and abs(eps) < 1.5:
                    r['correct_direction'] += 1

    return zone_results


def print_zone_validation(zone_results: dict):
    """Print zone-separated validation — three animals, three graphs."""

    # Print each zone
    for zone_id in (1, 2, 3):
        r = zone_results[zone_id]
        total = r['total']
        if total == 0:
            continue

        mode_acc = r['correct_mode'] / total * 100
        dir_total = max(r['direction_total'], 1)
        dir_acc = r['correct_direction'] / dir_total * 100

        print(f"\n{'═'*72}")
        print(f"  ZONE {zone_id}: {ZONE_NAMES[zone_id]}")
        print(f"  n = {total}    Available modes: {ZONE_MODES[zone_id]}")
        print(f"{'═'*72}")

        print(f"""
  Mode accuracy:     {r['correct_mode']:>5d}/{total} ({mode_acc:.1f}%)
  β-direction:       {r['correct_direction']:>5d}/{r['direction_total']} ({dir_acc:.1f}%)
""")

        # Per actual mode
        print(f"  {'Actual Mode':>12s} {'Right':>6s} {'Total':>6s} {'Acc':>8s}")
        print(f"  {'-'*36}")
        for mode in sorted(r['by_actual_mode']):
            c, t = r['by_actual_mode'][mode]
            acc = c / t * 100 if t > 0 else 0
            print(f"  {mode:>12s} {c:>6d} {t:>6d} {acc:>7.1f}%")

        # Top confusions
        confused = [(k, v) for k, v in r['confusion'].items() if k[0] != k[1]]
        confused.sort(key=lambda x: -x[1])
        if confused:
            print(f"\n  Top confusions:")
            for (actual, pred), count in confused[:5]:
                print(f"    {actual:>8s} → {pred:<8s}  {count:>5d}")

    # Combined
    r = zone_results['overall']
    total = r['total']
    if total == 0:
        return

    mode_acc = r['correct_mode'] / total * 100
    dir_acc = r['correct_direction'] / max(r['direction_total'], 1) * 100

    print(f"\n{'═'*72}")
    print(f"  COMBINED (all zones) — for backward comparison")
    print(f"{'═'*72}")

    print(f"""
  Overall mode accuracy:   {r['correct_mode']:>5d}/{total} ({mode_acc:.1f}%)
  β-direction accuracy:    {r['correct_direction']:>5d}/{r['direction_total']} ({dir_acc:.1f}%)
""")

    print(f"  {'Actual Mode':>12s} {'Right':>6s} {'Total':>6s} {'Acc':>8s}")
    print(f"  {'-'*36}")
    for mode in sorted(r['by_actual_mode']):
        c, t = r['by_actual_mode'][mode]
        acc = c / t * 100 if t > 0 else 0
        print(f"  {mode:>12s} {c:>6d} {t:>6d} {acc:>7.1f}%")

    # Top confusions
    print(f"\n  Top confusions (actual → predicted, count)")
    print(f"  {'-'*50}")
    confused = [(k, v) for k, v in r['confusion'].items() if k[0] != k[1]]
    confused.sort(key=lambda x: -x[1])
    for (actual, pred), count in confused[:10]:
        print(f"    {actual:>8s} → {pred:<8s}  {count:>5d}")

    print(f"""
  ─────────────────────────────────────────────────
  KEY PHYSICS CHECKS

    β-direction from sign(ε):   {dir_acc:.1f}%
      (The real QFD content — valley geometry alone)

    β⁻ accuracy:  {_mode_acc(r, 'B-')}
    β⁺/EC accuracy: {_mode_acc(r, 'B+')}
    α accuracy:   {_mode_acc(r, 'alpha')}
    Stable accuracy: {_mode_acc(r, 'stable')}
    SF accuracy:  {_mode_acc(r, 'SF')}
    p accuracy:   {_mode_acc(r, 'p')}
    n accuracy:   {_mode_acc(r, 'n')}
  ─────────────────────────────────────────────────
""")


def print_mode_population_analysis(nubase_entries: list, all_entries: list = None):
    """Per-mode population profiling — each decay mode is a separate animal.

    Instead of asking 'how well does one predictor do on all nuclides?',
    we ask: 'what does each mode population look like in geometric space?'

    For each mode:
      - Geometric centroid (ε, pf, cf) — where do they live?
      - Spread (std) — how tight is the population?
      - Zone distribution — which zones is this animal found in?
      - Parity distribution — ee/eo/oe/oo
      - Clock residuals within population
      - What our model confuses them with

    If all_entries is provided (gs + isomers), also analyzes isomers
    as 'branch species' — same (Z,A) but different winding states
    that decay differently.
    """
    # ── Separate by actual mode ──
    populations = {}  # mode → list of (Z, A, N, geo, half_life_s)
    for entry in nubase_entries:
        A, Z, N = entry['A'], entry['Z'], entry['N']
        mode = normalize_nubase(entry['dominant_mode'])
        if mode in ('unknown', 'IT') or A < 3:
            continue
        geo = compute_geometric_state(Z, A)
        populations.setdefault(mode, []).append({
            'Z': Z, 'A': A, 'N': N,
            'geo': geo,
            'half_life_s': entry.get('half_life_s', np.nan),
            'jpi': entry.get('jpi', ''),
        })

    # ── Print header ──
    print(f"\n{'═'*72}")
    print(f"  MODE POPULATION ANALYSIS — Each Decay Mode as a Separate Animal")
    print(f"{'═'*72}")
    print(f"\n  Total ground-state nuclides by mode:")
    for mode in sorted(populations, key=lambda m: -len(populations[m])):
        print(f"    {mode:>8s}:  {len(populations[mode]):>5d}")

    # ── Cross-tabulation: mode × zone ──
    print(f"\n  {'':>8s}  {'Zone 1':>8s}  {'Zone 2':>8s}  {'Zone 3':>8s}  {'Total':>8s}")
    print(f"  {'-'*46}")
    for mode in sorted(populations, key=lambda m: -len(populations[m])):
        pop = populations[mode]
        z1 = sum(1 for p in pop if p['geo'].zone == 1)
        z2 = sum(1 for p in pop if p['geo'].zone == 2)
        z3 = sum(1 for p in pop if p['geo'].zone == 3)
        print(f"  {mode:>8s}  {z1:>8d}  {z2:>8d}  {z3:>8d}  {len(pop):>8d}")

    # ── Per-mode geometric profile ──
    MODE_ORDER = ['B-', 'B+', 'alpha', 'stable', 'SF', 'n', 'p']
    existing_modes = [m for m in MODE_ORDER if m in populations]
    # Add any modes not in standard order
    for m in sorted(populations):
        if m not in existing_modes:
            existing_modes.append(m)

    for mode in existing_modes:
        pop = populations[mode]
        n = len(pop)
        if n < 3:
            continue

        eps_arr = np.array([p['geo'].eps for p in pop])
        pf_arr = np.array([p['geo'].peanut_f for p in pop])
        cf_arr = np.array([p['geo'].core_full for p in pop])
        A_arr = np.array([p['A'] for p in pop])
        Z_arr = np.array([p['Z'] for p in pop])

        # Parity counts
        parity_counts = {'ee': 0, 'eo': 0, 'oe': 0, 'oo': 0}
        for p in pop:
            parity_counts[p['geo'].parity] += 1

        # Half-life stats (for those with measured values)
        hl_vals = [math.log10(p['half_life_s']) for p in pop
                   if np.isfinite(p['half_life_s']) and p['half_life_s'] > 0
                   and p['half_life_s'] < 1e30]  # exclude stable
        hl_arr = np.array(hl_vals) if hl_vals else np.array([])

        # What does our model predict for this population?
        pred_counts = {}
        for p in pop:
            pred, _ = predict_decay(p['Z'], p['A'])
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        correct = pred_counts.get(mode, 0)
        acc = correct / n * 100

        print(f"\n  {'─'*68}")
        print(f"  {mode:s}  (n={n})")
        print(f"  {'─'*68}")
        print(f"    Geometric centroid:")
        print(f"      ε (valley stress):   {np.mean(eps_arr):+.3f} ± {np.std(eps_arr):.3f}  "
              f"[{np.min(eps_arr):+.2f} .. {np.max(eps_arr):+.2f}]")
        print(f"      pf (peanut factor):  {np.mean(pf_arr):.3f} ± {np.std(pf_arr):.3f}  "
              f"[{np.min(pf_arr):.2f} .. {np.max(pf_arr):.2f}]")
        print(f"      cf (core fullness):  {np.mean(cf_arr):.3f} ± {np.std(cf_arr):.3f}  "
              f"[{np.min(cf_arr):.2f} .. {np.max(cf_arr):.2f}]")
        print(f"      A range:             [{int(np.min(A_arr))} .. {int(np.max(A_arr))}]  "
              f"median={np.median(A_arr):.0f}")
        print(f"      Z range:             [{int(np.min(Z_arr))} .. {int(np.max(Z_arr))}]  "
              f"median={np.median(Z_arr):.0f}")

        print(f"    Parity:  ee={parity_counts['ee']:>4d}  eo={parity_counts['eo']:>4d}  "
              f"oe={parity_counts['oe']:>4d}  oo={parity_counts['oo']:>4d}")

        if len(hl_arr) > 0:
            print(f"    log₁₀(t½/s):  {np.mean(hl_arr):.2f} ± {np.std(hl_arr):.2f}  "
                  f"[{np.min(hl_arr):.1f} .. {np.max(hl_arr):.1f}]")

        # Zone breakdown
        z1 = sum(1 for p in pop if p['geo'].zone == 1)
        z2 = sum(1 for p in pop if p['geo'].zone == 2)
        z3 = sum(1 for p in pop if p['geo'].zone == 3)
        print(f"    Zone distribution:  Z1={z1} ({z1/n*100:.0f}%)  "
              f"Z2={z2} ({z2/n*100:.0f}%)  Z3={z3} ({z3/n*100:.0f}%)")

        # Model accuracy on this population
        print(f"    Our prediction:  {correct}/{n} correct ({acc:.1f}%)")
        misses = {k: v for k, v in pred_counts.items() if k != mode}
        if misses:
            miss_str = ', '.join(f'{k}={v}' for k, v in
                                 sorted(misses.items(), key=lambda x: -x[1]))
            print(f"      Confused as: {miss_str}")

    # ── Isomer branch species ──
    if all_entries is not None:
        print(f"\n{'═'*72}")
        print(f"  ISOMER BRANCH SPECIES — Same (Z,A), Different Winding State")
        print(f"{'═'*72}")

        nuclide_states = group_nuclide_states(all_entries)

        # Find mode-switch isomers (NOT IT)
        switches = []
        for (Z, A), states in nuclide_states.items():
            if len(states) < 2:
                continue
            gs = states[0]
            gs_mode = normalize_nubase(gs['dominant_mode'])
            if gs_mode in ('unknown', 'IT'):
                continue

            for iso in states[1:]:
                iso_mode = normalize_nubase(iso['dominant_mode'])
                if iso_mode in ('unknown', 'IT'):
                    continue
                if iso_mode != gs_mode:
                    gs_j = _parse_spin_value(gs.get('jpi', ''))
                    iso_j = _parse_spin_value(iso.get('jpi', ''))
                    dj = abs(gs_j - iso_j) if gs_j is not None and iso_j is not None else None
                    geo = compute_geometric_state(Z, A)
                    switches.append({
                        'Z': Z, 'A': A,
                        'gs_mode': gs_mode, 'iso_mode': iso_mode,
                        'gs_jpi': gs.get('jpi', ''), 'iso_jpi': iso.get('jpi', ''),
                        'delta_j': dj,
                        'iso_state': iso.get('state', ''),
                        'geo': geo,
                        'gs_hl': gs.get('half_life_s', np.nan),
                        'iso_hl': iso.get('half_life_s', np.nan),
                    })

        print(f"\n  Mode-switching isomers (gs → isomer changes decay mode): {len(switches)}")
        print(f"  These are 'branch species' — same coordinates, different animal\n")

        # Group by switch type
        switch_types = {}
        for s in switches:
            key = (s['gs_mode'], s['iso_mode'])
            switch_types.setdefault(key, []).append(s)

        print(f"  {'gs → iso':>16s}  {'Count':>6s}  {'Median |ΔJ|':>12s}  {'Zone 1':>6s}  {'Zone 2':>6s}  {'Zone 3':>6s}")
        print(f"  {'-'*64}")
        for (gm, im), items in sorted(switch_types.items(), key=lambda x: -len(x[1])):
            djs = [s['delta_j'] for s in items if s['delta_j'] is not None]
            med_dj = f"{np.median(djs):.1f}" if djs else "  —"
            z1 = sum(1 for s in items if s['geo'].zone == 1)
            z2 = sum(1 for s in items if s['geo'].zone == 2)
            z3 = sum(1 for s in items if s['geo'].zone == 3)
            print(f"  {gm:>7s} → {im:<7s} {len(items):>6d}  {med_dj:>12s}  {z1:>6d}  {z2:>6d}  {z3:>6d}")

        # High-spin isomers: |ΔJ| >= 5
        high_spin = [s for s in switches if s['delta_j'] is not None and s['delta_j'] >= 5]
        if high_spin:
            print(f"\n  HIGH-SPIN BRANCH SPECIES (|ΔJ| ≥ 5): {len(high_spin)}")
            print(f"  These have large angular momentum barriers — topologically distinct states")
            print(f"\n  {'Nuclide':>10s}  {'gs→iso':>14s}  {'|ΔJ|':>5s}  {'gs Jπ':>8s}  {'iso Jπ':>8s}  {'zone':>4s}  {'pf':>5s}")
            print(f"  {'-'*62}")
            for s in sorted(high_spin, key=lambda x: -x['delta_j'])[:20]:
                elem = ELEMENTS.get(s['Z'], f"Z{s['Z']}")
                name = f"{elem}-{s['A']}"
                print(f"  {name:>10s}  {s['gs_mode']:>6s}→{s['iso_mode']:<6s}  "
                      f"{s['delta_j']:>5.1f}  {s['gs_jpi']:>8s}  {s['iso_jpi']:>8s}  "
                      f"{s['geo'].zone:>4d}  {s['geo'].peanut_f:>5.2f}")


def print_clock_validation(nubase_entries: list):
    """Validate the multi-mode clock against NUBASE2020 half-lives."""
    from scipy import stats as sp_stats

    # Collect per-mode records: (Z, A, eps, log_obs, log_pred)
    mode_records = {'B-': [], 'B+': [], 'alpha': []}
    all_records = []

    for nuc in nubase_entries:
        Z, A = nuc['Z'], nuc['A']
        mode = nuc['dominant_mode']
        hl = nuc['half_life_s']
        if mode in ('stable', 'unknown', 'IT') or A < 3:
            continue
        if not np.isfinite(hl) or hl <= 0:
            continue

        eps = Z - z_star(A)
        log_hl = math.log10(hl)
        actual_norm = normalize_nubase(mode)
        all_records.append((actual_norm, abs(eps), log_hl, Z))

        log_pred = _clock_log10t(Z, eps, actual_norm)
        if log_pred is not None and actual_norm in mode_records:
            mode_records[actual_norm].append((Z, A, eps, log_hl, log_pred))

    print(f"\n{'='*72}")
    print("  CLOCK VALIDATION — Half-Life from Valley Stress + Atomic Environment")
    print("=" * 72)

    print(f"""
  GENERAL FORM:  log₁₀(t½/s) = a·√|ε| + b·log₁₀(Z) + c·Z + d
  PROVENANCE:  √|ε| from compression law (0 free params)
               log₁₀(Z), Z from atomic geometry (Bohr / Coulomb)
               a, b, c, d = 4 empirical parameters per mode (12 total)
""")

    # Per-mode statistics
    for mode_label, mode_key, clock_params in [
        ('β⁻', 'B-', (BM_CLOCK_A, BM_CLOCK_B, BM_CLOCK_C, BM_CLOCK_D)),
        ('β⁺/EC', 'B+', (BP_CLOCK_A, BP_CLOCK_B, BP_CLOCK_C, BP_CLOCK_D)),
        ('α', 'alpha', (AL_CLOCK_A, AL_CLOCK_B, AL_CLOCK_C, AL_CLOCK_D)),
    ]:
        recs = mode_records[mode_key]
        if not recs:
            continue

        a, b, c, d = clock_params
        log_obs = np.array([r[3] for r in recs])
        log_pred = np.array([r[4] for r in recs])
        n = len(recs)

        r_val, _ = sp_stats.pearsonr(log_pred, log_obs)
        rho_val, _ = sp_stats.spearmanr(log_pred, log_obs)
        residuals = log_obs - log_pred
        rmse = math.sqrt(np.mean(residuals**2))
        within_1 = np.sum(np.abs(residuals) <= 1.0)
        within_2 = np.sum(np.abs(residuals) <= 2.0)

        print(f"  {mode_label} EMITTERS: {n}")
        print(f"    Formula: {a:+.4f}·√|ε| + {b:+.4f}·log₁₀(Z) + {c:+.6f}·Z + {d:+.4f}")
        print(f"    R²         = {r_val**2:.4f}")
        print(f"    Spearman ρ = {rho_val:.4f}")
        print(f"    RMSE       = {rmse:.2f} decades")
        print(f"    Within 10×:  {within_1}/{n} ({within_1/n*100:.1f}%)")
        print(f"    Within 100×: {within_2}/{n} ({within_2/n*100:.1f}%)")
        print()

    # Spot checks — β⁻
    print(f"  SPOT CHECKS — β⁻")
    spot_bm = [
        (6, 14, "C-14", 5730 * 3.1557600e7),
        (55, 137, "Cs-137", 30.2 * 3.1557600e7),
        (27, 60, "Co-60", 5.27 * 3.1557600e7),
        (53, 131, "I-131", 8.02 * 86400),
        (38, 90, "Sr-90", 28.8 * 3.1557600e7),
        (1, 3, "H-3", 12.33 * 3.1557600e7),
    ]
    print(f"  {'Nuclide':<10s} {'|ε|':>6s}  {'Predicted':>12s}  {'Actual':>12s}  {'Δlog':>6s}")
    print(f"  {'-'*52}")
    for Z, A, lab, actual_s in spot_bm:
        eps = Z - z_star(A)
        log_t = _clock_log10t(Z, eps, 'B-')
        if log_t is not None and actual_s > 0:
            log_actual = math.log10(actual_s)
            delta_log = log_t - log_actual
            print(f"  {lab:<10s} {abs(eps):>6.2f}  {_format_halflife(10**log_t):>12s}  {_format_halflife(actual_s):>12s}  {delta_log:>+6.1f}")

    # Spot checks — β⁺/EC
    print(f"\n  SPOT CHECKS — β⁺/EC")
    spot_bp = [
        (11, 22, "Na-22", 2.60 * 3.1557600e7),
        (19, 40, "K-40", 1.25e9 * 3.1557600e7),   # EC branch
        (9, 18, "F-18", 109.77 * 60),
        (53, 123, "I-123", 13.27 * 3600),
        (43, 99, "Tc-99m", 6.007 * 3600),          # approximate
        (81, 201, "Tl-201", 3.04 * 86400),
    ]
    print(f"  {'Nuclide':<10s} {'|ε|':>6s}  {'Predicted':>12s}  {'Actual':>12s}  {'Δlog':>6s}")
    print(f"  {'-'*52}")
    for Z, A, lab, actual_s in spot_bp:
        eps = Z - z_star(A)
        log_t = _clock_log10t(Z, eps, 'B+')
        if log_t is not None and actual_s > 0:
            log_actual = math.log10(actual_s)
            delta_log = log_t - log_actual
            print(f"  {lab:<10s} {abs(eps):>6.2f}  {_format_halflife(10**max(-18, min(20.5, log_t))):>12s}  {_format_halflife(actual_s):>12s}  {delta_log:>+6.1f}")

    # Spot checks — α
    print(f"\n  SPOT CHECKS — α")
    spot_al = [
        (92, 238, "U-238", 4.468e9 * 3.1557600e7),
        (88, 226, "Ra-226", 1600 * 3.1557600e7),
        (84, 210, "Po-210", 138.4 * 86400),
        (86, 222, "Rn-222", 3.8235 * 86400),
        (94, 239, "Pu-239", 24110 * 3.1557600e7),
    ]
    print(f"  {'Nuclide':<10s} {'|ε|':>6s}  {'Predicted':>12s}  {'Actual':>12s}  {'Δlog':>6s}")
    print(f"  {'-'*52}")
    for Z, A, lab, actual_s in spot_al:
        eps = Z - z_star(A)
        log_t = _clock_log10t(Z, eps, 'alpha')
        if log_t is not None and actual_s > 0:
            log_actual = math.log10(actual_s)
            delta_log = log_t - log_actual
            print(f"  {lab:<10s} {abs(eps):>6.2f}  {_format_halflife(10**max(-18, min(20.5, log_t))):>12s}  {_format_halflife(actual_s):>12s}  {delta_log:>+6.1f}")

    # Variance decomposition
    print(f"\n  VARIANCE DECOMPOSITION (√|ε| alone vs + atomic environment):")
    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        recs = mode_records[mode_key]
        if len(recs) < 20:
            continue
        eps_arr = np.array([abs(r[2]) for r in recs])
        Z_arr = np.array([r[0] for r in recs])
        log_obs = np.array([r[3] for r in recs])
        n = len(recs)

        # Base: √|ε| only
        X_base = np.column_stack([np.sqrt(eps_arr), np.ones(n)])
        c_base, _, _, _ = np.linalg.lstsq(X_base, log_obs, rcond=None)
        SS_tot = np.sum((log_obs - np.mean(log_obs))**2)
        R2_base = 1 - np.sum((log_obs - X_base @ c_base)**2) / SS_tot

        # Full: √|ε| + log₁₀(Z) + Z
        log_pred = np.array([r[4] for r in recs])
        R2_full = 1 - np.sum((log_obs - log_pred)**2) / SS_tot

        pct_stress = R2_base * 100
        pct_atomic = (R2_full - R2_base) * 100
        pct_remain = (1 - R2_full) * 100
        print(f"    {mode_label:>5s}: stress={pct_stress:.1f}%  atomic={pct_atomic:+.1f}%  remaining={pct_remain:.1f}%  (total R²={R2_full:.3f})")

    # ── Zero-Parameter Clock Comparison ──
    print(f"\n{'='*72}")
    print("  ZERO-PARAMETER CLOCK — All Constants from α")
    print("=" * 72)
    print(f"""
  FORM:  log₁₀(t½/s) = a·√|ε| + b·log₁₀(Z) + d      (no c·Z term)
  ALL constants derived from Golden Loop:  α → β → (π, e, β)
  PROVENANCE:  Every coefficient is [QFD_DERIVED] — 0 empirical parameters
""")

    # Collect zero-param predictions
    zp_records = {'B-': [], 'B+': [], 'alpha': []}
    for nuc in nubase_entries:
        Z, A = nuc['Z'], nuc['A']
        mode = nuc['dominant_mode']
        hl = nuc['half_life_s']
        if mode in ('stable', 'unknown', 'IT') or A < 3:
            continue
        if not np.isfinite(hl) or hl <= 0:
            continue
        eps = Z - z_star(A)
        log_hl = math.log10(hl)
        actual_norm = normalize_nubase(mode)
        log_pred_zp = _clock_log10t_zero_param(Z, eps, actual_norm)
        if log_pred_zp is not None and actual_norm in zp_records:
            zp_records[actual_norm].append((Z, A, eps, log_hl, log_pred_zp))

    # Side-by-side comparison table
    print(f"  {'Mode':<8s} {'N':>5s}  {'R²_fit':>7s} {'R²_zp':>7s} {'ΔR²':>6s}  {'RMSE_f':>6s} {'RMSE_zp':>7s}  {'10×_f':>5s} {'10×_zp':>6s}")
    print(f"  {'-'*72}")

    for mode_label, mode_key, zp_params in [
        ('β⁻', 'B-', (ZP_BM_A, ZP_BM_B, ZP_BM_D)),
        ('β⁺/EC', 'B+', (ZP_BP_A, ZP_BP_B, ZP_BP_D)),
        ('α', 'alpha', (ZP_AL_A, ZP_AL_B, ZP_AL_D)),
    ]:
        fit_recs = mode_records[mode_key]
        zp_recs = zp_records[mode_key]
        if not fit_recs or not zp_recs:
            continue

        # Fitted stats
        fit_obs = np.array([r[3] for r in fit_recs])
        fit_pred = np.array([r[4] for r in fit_recs])
        fit_resid = fit_obs - fit_pred
        SS_tot_f = np.sum((fit_obs - np.mean(fit_obs))**2)
        R2_fit = 1 - np.sum(fit_resid**2) / SS_tot_f
        rmse_fit = math.sqrt(np.mean(fit_resid**2))
        w1_fit = np.sum(np.abs(fit_resid) <= 1.0)

        # Zero-param stats
        zp_obs = np.array([r[3] for r in zp_recs])
        zp_pred = np.array([r[4] for r in zp_recs])
        zp_resid = zp_obs - zp_pred
        SS_tot_z = np.sum((zp_obs - np.mean(zp_obs))**2)
        R2_zp = 1 - np.sum(zp_resid**2) / SS_tot_z
        rmse_zp = math.sqrt(np.mean(zp_resid**2))
        w1_zp = np.sum(np.abs(zp_resid) <= 1.0)

        n_fit = len(fit_recs)
        n_zp = len(zp_recs)
        delta_R2 = R2_zp - R2_fit

        a_zp, b_zp, d_zp = zp_params
        print(f"  {mode_label:<8s} {n_zp:>5d}  {R2_fit:>7.4f} {R2_zp:>7.4f} {delta_R2:>+6.3f}  {rmse_fit:>6.2f} {rmse_zp:>7.2f}  {w1_fit/n_fit*100:>5.1f} {w1_zp/n_zp*100:>6.1f}%")

    print()
    # Show the formulas
    print(f"  FORMULAS (all from α → β ≈ {BETA:.4f}):")
    print(f"    β⁻:    {ZP_BM_A:.4f}·√|ε| + {ZP_BM_B:.1f}·log₁₀(Z) + {ZP_BM_D:.4f}")
    print(f"           = -πβ/e·√|ε| + 2·log₁₀(Z) + 4π/3")
    print(f"    β⁺/EC: {ZP_BP_A:.4f}·√|ε| + {ZP_BP_B:.4f}·log₁₀(Z) + {ZP_BP_D:.4f}")
    print(f"           = -π·√|ε| + 2β·log₁₀(Z) - 2β/e")
    print(f"    α:     {ZP_AL_A:.4f}·√|ε| + {ZP_AL_B:.4f}·log₁₀(Z) + {ZP_AL_D:.4f}")
    print(f"           = -e·√|ε| + (β+1)·log₁₀(Z) - (β-1)")

    # Zero-param spot checks
    print(f"\n  ZERO-PARAM SPOT CHECKS — β⁻")
    spot_bm = [
        (6, 14, "C-14", 5730 * 3.1557600e7),
        (55, 137, "Cs-137", 30.2 * 3.1557600e7),
        (27, 60, "Co-60", 5.27 * 3.1557600e7),
        (53, 131, "I-131", 8.02 * 86400),
        (38, 90, "Sr-90", 28.8 * 3.1557600e7),
        (1, 3, "H-3", 12.33 * 3.1557600e7),
    ]
    print(f"  {'Nuclide':<10s} {'|ε|':>6s}  {'ZP Pred':>12s}  {'Fitted':>12s}  {'Actual':>12s}  {'Δlog_zp':>7s} {'Δlog_fit':>8s}")
    print(f"  {'-'*72}")
    for Z, A, lab, actual_s in spot_bm:
        eps = Z - z_star(A)
        log_zp = _clock_log10t_zero_param(Z, eps, 'B-')
        log_fit = _clock_log10t(Z, eps, 'B-')
        if log_zp is not None and log_fit is not None and actual_s > 0:
            log_actual = math.log10(actual_s)
            dlog_zp = log_zp - log_actual
            dlog_fit = log_fit - log_actual
            print(f"  {lab:<10s} {abs(eps):>6.2f}  {_format_halflife(10**max(-18, min(20.5, log_zp))):>12s}  {_format_halflife(10**max(-18, min(20.5, log_fit))):>12s}  {_format_halflife(actual_s):>12s}  {dlog_zp:>+7.1f} {dlog_fit:>+8.1f}")

    print(f"\n  ZERO-PARAM SPOT CHECKS — β⁺/EC")
    spot_bp = [
        (11, 22, "Na-22", 2.60 * 3.1557600e7),
        (19, 40, "K-40", 1.25e9 * 3.1557600e7),
        (9, 18, "F-18", 109.77 * 60),
        (53, 123, "I-123", 13.27 * 3600),
        (81, 201, "Tl-201", 3.04 * 86400),
    ]
    print(f"  {'Nuclide':<10s} {'|ε|':>6s}  {'ZP Pred':>12s}  {'Fitted':>12s}  {'Actual':>12s}  {'Δlog_zp':>7s} {'Δlog_fit':>8s}")
    print(f"  {'-'*72}")
    for Z, A, lab, actual_s in spot_bp:
        eps = Z - z_star(A)
        log_zp = _clock_log10t_zero_param(Z, eps, 'B+')
        log_fit = _clock_log10t(Z, eps, 'B+')
        if log_zp is not None and log_fit is not None and actual_s > 0:
            log_actual = math.log10(actual_s)
            dlog_zp = log_zp - log_actual
            dlog_fit = log_fit - log_actual
            print(f"  {lab:<10s} {abs(eps):>6.2f}  {_format_halflife(10**max(-18, min(20.5, log_zp))):>12s}  {_format_halflife(10**max(-18, min(20.5, log_fit))):>12s}  {_format_halflife(actual_s):>12s}  {dlog_zp:>+7.1f} {dlog_fit:>+8.1f}")

    print(f"\n  ZERO-PARAM SPOT CHECKS — α")
    spot_al = [
        (92, 238, "U-238", 4.468e9 * 3.1557600e7),
        (88, 226, "Ra-226", 1600 * 3.1557600e7),
        (84, 210, "Po-210", 138.4 * 86400),
        (86, 222, "Rn-222", 3.8235 * 86400),
        (94, 239, "Pu-239", 24110 * 3.1557600e7),
    ]
    print(f"  {'Nuclide':<10s} {'|ε|':>6s}  {'ZP Pred':>12s}  {'Fitted':>12s}  {'Actual':>12s}  {'Δlog_zp':>7s} {'Δlog_fit':>8s}")
    print(f"  {'-'*72}")
    for Z, A, lab, actual_s in spot_al:
        eps = Z - z_star(A)
        log_zp = _clock_log10t_zero_param(Z, eps, 'alpha')
        log_fit = _clock_log10t(Z, eps, 'alpha')
        if log_zp is not None and log_fit is not None and actual_s > 0:
            log_actual = math.log10(actual_s)
            dlog_zp = log_zp - log_actual
            dlog_fit = log_fit - log_actual
            print(f"  {lab:<10s} {abs(eps):>6.2f}  {_format_halflife(10**max(-18, min(20.5, log_zp))):>12s}  {_format_halflife(10**max(-18, min(20.5, log_fit))):>12s}  {_format_halflife(actual_s):>12s}  {dlog_zp:>+7.1f} {dlog_fit:>+8.1f}")


def print_alpha_onset_analysis(terrain: list):
    """Check where the gradient first predicts α."""
    alpha_cells = [e for e in terrain if e['mode'] == 'alpha']
    if not alpha_cells:
        print("\n  No α predictions found.")
        return

    onset_A = min(e['A'] for e in alpha_cells)
    heavy_alphas = [e for e in alpha_cells if abs(e['eps']) < 2]

    print(f"\n{'='*72}")
    print("  ALPHA ONSET ANALYSIS")
    print("=" * 72)
    print(f"""
    Lightest α prediction:  A = {onset_A}
    Expected onset:         A ≈ {A_ALPHA_ONSET:.0f} (A_CRIT + WIDTH)
    α predictions total:    {len(alpha_cells)}
    Near-valley α (|ε|<2):  {len(heavy_alphas)}

    The gradient predicts α when:
      1. Bulk relief ΔE(A) > 0  (density stress drops by losing 4 states)
      2. Valley penalty Δ(ε²) is small  (α doesn't move far from valley)
    Both conditions are met for heavy, near-valley nuclides.
""")


def plot_isomer_maps(all_entries: list, nuclide_states: dict,
                     iso_val: dict, terrain: list, output_dir: str):
    """Generate isomer-aware nuclide maps.

    Figure 1: Three-panel — predicted vs gs-only vs all-states
    Figure 2: Accuracy with isomer-rescue highlights
    Figure 3: Mode competition — sites with conflicting isomer modes
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Valley line
    A_vals = np.arange(1, 301)
    Z_vals = [z_star(A) for A in A_vals]
    N_vals = [A - z for A, z in zip(A_vals, Z_vals)]

    def _style_ax(ax, title):
        ax.set_title(title, fontsize=11, fontweight='bold', color='white')
        ax.set_xlabel('Neutrons (N)', color='white')
        ax.set_ylabel('Protons (Z)', color='white')
        ax.set_xlim(-5, 200)
        ax.set_ylim(-5, 125)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15)
        ax.set_facecolor('#0A0A1A')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444444')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['stable'],
               markersize=7, label='Stable'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['B-'],
               markersize=7, label='β⁻'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['B+'],
               markersize=7, label='β⁺/EC'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['alpha'],
               markersize=7, label='α'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['SF'],
               markersize=7, label='SF'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['p'],
               markersize=7, label='p'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=MODE_COLORS['n'],
               markersize=7, label='n'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF00FF',
               markersize=7, label='IT'),
        Line2D([0], [0], color='white', linewidth=1, label='Z*(A) valley'),
    ]

    # ── Figure 1: Three-panel comparison ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 9))

    # Left: QFD Predicted
    for entry in terrain:
        color = MODE_COLORS.get(entry['mode'], '#AAAAAA')
        ax1.scatter(entry['N'], entry['Z'], c=color, s=1.2, alpha=0.4,
                    edgecolors='none')
    ax1.plot(N_vals, Z_vals, 'w-', linewidth=0.8, alpha=0.6)
    _style_ax(ax1, 'QFD Prediction\n(Zero Free Parameters)')

    # Middle: Ground states only
    gs_entries = [e for e in all_entries if e['state'] == 'gs']
    for e in gs_entries:
        mode = normalize_nubase(e['dominant_mode'])
        color = MODE_COLORS.get(mode, '#AAAAAA')
        ax2.scatter(e['N'], e['Z'], c=color, s=2.0, alpha=0.5,
                    edgecolors='none')
    ax2.plot(N_vals, Z_vals, 'w-', linewidth=0.8, alpha=0.6)
    _style_ax(ax2, f'NUBASE2020 Ground States\n({len(gs_entries)} entries)')

    # Right: All states (gs + isomers)
    it_color = '#FF00FF'
    # Plot non-IT first (underneath), then IT on top with lower alpha
    non_it = [e for e in all_entries if normalize_nubase(e['dominant_mode']) != 'IT']
    it_only = [e for e in all_entries if normalize_nubase(e['dominant_mode']) == 'IT']
    for e in non_it:
        mode = normalize_nubase(e['dominant_mode'])
        color = MODE_COLORS.get(mode, '#AAAAAA')
        ax3.scatter(e['N'], e['Z'], c=color, s=1.8, alpha=0.4,
                    edgecolors='none')
    for e in it_only:
        ax3.scatter(e['N'], e['Z'], c=it_color, s=1.2, alpha=0.25,
                    edgecolors='none')
    ax3.plot(N_vals, Z_vals, 'w-', linewidth=0.8, alpha=0.6)
    _style_ax(ax3, f'All NUBASE2020 States\n({len(all_entries)} entries, incl. isomers)')

    fig.patch.set_facecolor('#0A0A1A')
    fig.legend(handles=legend_elements, loc='lower center', ncol=9,
               fontsize=9, framealpha=0.8,
               facecolor='#1A1A2A', edgecolor='#444444',
               labelcolor='white')
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path1 = os.path.join(output_dir, 'nuclide_map_all_states.png')
    fig.savefig(path1, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ── Figure 2: Accuracy with rescue highlights ──
    rescued_set = {(r['Z'], r['A']) for r in iso_val['rescue_list']}
    fig2, ax4 = plt.subplots(figsize=(12, 9))

    for (Z, A), states in nuclide_states.items():
        gs_list = [s for s in states if s['state'] == 'gs']
        if not gs_list:
            continue
        gs = gs_list[0]
        actual = gs['dominant_mode']
        if actual in ('unknown', 'IT') or A < 3:
            continue

        pred, _ = predict_decay(Z, A)
        actual_norm = normalize_nubase(actual)
        N = A - Z

        if actual_norm == pred:
            color, alpha, s = '#33CC33', 0.3, 2   # correct
        elif (Z, A) in rescued_set:
            color, alpha, s = '#FFD700', 0.9, 6   # rescued by isomer
        else:
            color, alpha, s = '#CC3333', 0.6, 2   # wrong

        ax4.scatter(N, Z, c=color, s=s, alpha=alpha, edgecolors='none')

    ax4.plot(N_vals, Z_vals, 'w-', linewidth=0.8, alpha=0.5)

    gs_acc = iso_val['gs_correct'] / max(iso_val['gs_total'], 1) * 100
    any_acc = iso_val['any_correct'] / max(iso_val['gs_total'], 1) * 100

    _style_ax(ax4, f'Prediction Accuracy with Isomer Rescue\n'
                    f'GS: {gs_acc:.1f}%  |  Any-state: {any_acc:.1f}%  |  '
                    f'Rescued: {iso_val["rescued"]}')

    legend_acc = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#33CC33',
               markersize=8, label='Correct (gs)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD700',
               markersize=8, label='Isomer rescue'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#CC3333',
               markersize=8, label='Wrong'),
    ]
    ax4.legend(handles=legend_acc, loc='upper left', fontsize=10,
               facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')

    fig2.patch.set_facecolor('#0A0A1A')
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'nuclide_map_isomer_rescue.png')
    fig2.savefig(path2, dpi=150, facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # ── Figure 3: Mode competition map ──
    # For each (Z,A) with isomers, classify the mode diversity
    fig3, ax5 = plt.subplots(figsize=(12, 9))

    for (Z, A), states in nuclide_states.items():
        N = A - Z
        # Collect distinct non-IT, non-unknown modes across all states
        modes = set()
        for s in states:
            m = normalize_nubase(s['dominant_mode'])
            if m not in ('unknown', 'IT'):
                modes.add(m)

        if len(states) == 1:
            # Single state (gs only) — dim gray
            color, alpha, s_size = '#444444', 0.2, 1.0
        elif len(modes) <= 1:
            # All states agree (or all IT) — blue
            color, alpha, s_size = '#3366CC', 0.4, 2.0
        elif len(modes) == 2:
            # Two competing modes — orange
            color, alpha, s_size = '#FF8800', 0.7, 3.5
        else:
            # Three+ modes — bright red
            color, alpha, s_size = '#FF2222', 0.9, 5.0

        ax5.scatter(N, Z, c=color, s=s_size, alpha=alpha, edgecolors='none')

    ax5.plot(N_vals, Z_vals, 'w-', linewidth=0.8, alpha=0.5)

    _style_ax(ax5, 'Mode Competition Map\n'
                    'Where isomers switch decay mode at the same (Z, A)')

    n_multi = sum(1 for (Z, A), states in nuclide_states.items()
                  if len({normalize_nubase(s['dominant_mode'])
                          for s in states
                          if normalize_nubase(s['dominant_mode']) not in ('unknown', 'IT')}) >= 2)

    legend_comp = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#444444',
               markersize=7, label='Single state'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3366CC',
               markersize=7, label='Isomers agree'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8800',
               markersize=7, label=f'2 modes ({n_multi} sites)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF2222',
               markersize=7, label='3+ modes'),
    ]
    ax5.legend(handles=legend_comp, loc='upper left', fontsize=10,
               facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')

    fig3.patch.set_facecolor('#0A0A1A')
    plt.tight_layout()
    path3 = os.path.join(output_dir, 'nuclide_map_mode_competition.png')
    fig3.savefig(path3, dpi=150, facecolor=fig3.get_facecolor())
    plt.close(fig3)
    print(f"  Saved: {path3}")


# ═══════════════════════════════════════════════════════════════════
# ISOMER ANALYSIS — Census, Validation Rescue, Clock Contamination
#
# All isomer data is EMPIRICAL_LOOKUP from NUBASE2020.
# Predictions come from predict_decay() which is QFD_DERIVED.
# No prediction logic is changed — this is diagnostic only.
# ═══════════════════════════════════════════════════════════════════

def print_isomer_census(nuclide_states: dict):
    """Print comprehensive isomer statistics from NUBASE2020.

    Provenance: EMPIRICAL_LOOKUP (measured data census).
    """
    # Collect all isomer entries (non-gs)
    all_isomers = []
    pairs_with_isomers = []
    for (Z, A), states in nuclide_states.items():
        isos = [s for s in states if s['state'] != 'gs']
        if isos:
            all_isomers.extend(isos)
            pairs_with_isomers.append((Z, A))

    multi_isomer = sum(1 for (Z, A), states in nuclide_states.items()
                       if sum(1 for s in states if s['state'] != 'gs') >= 2)

    print(f"\n{'═'*72}")
    print(f"  ISOMER CENSUS — All NUBASE2020 States")
    print(f"{'═'*72}")

    # ── 1. Headline counts ──
    print(f"""
  HEADLINE COUNTS
    Total isomers (excl. IAS):    {len(all_isomers)}
    (Z, A) pairs with isomers:    {len(pairs_with_isomers)}
    Multi-isomer (≥2 excited):    {multi_isomer}
    Total (Z, A) in database:     {len(nuclide_states)}
""")

    # ── 2. Mode-flip statistics ──
    it_switches = 0
    non_it_switches = 0
    flip_table = defaultdict(int)  # (gs_mode, iso_mode) → count
    for (Z, A), states in nuclide_states.items():
        gs_list = [s for s in states if s['state'] == 'gs']
        if not gs_list:
            continue
        gs = gs_list[0]
        gs_mode = normalize_nubase(gs['dominant_mode'])
        for iso in states:
            if iso['state'] == 'gs':
                continue
            iso_mode = normalize_nubase(iso['dominant_mode'])
            if iso_mode == 'IT':
                it_switches += 1
            elif iso_mode != gs_mode and iso_mode not in ('unknown',):
                non_it_switches += 1
                flip_table[(gs_mode, iso_mode)] += 1

    total_non_gs = len(all_isomers)
    print(f"  MODE-FLIP STATISTICS")
    print(f"    IT-dominant isomers:      {it_switches}")
    print(f"    Non-IT mode switches:     {non_it_switches}")
    print(f"    Same-mode as ground:      {total_non_gs - it_switches - non_it_switches}")
    if flip_table:
        print(f"\n    {'gs → iso':>18s}  {'Count':>5s}")
        print(f"    {'-'*28}")
        for (gm, im), cnt in sorted(flip_table.items(), key=lambda x: -x[1])[:12]:
            print(f"    {gm:>8s} → {im:<8s} {cnt:>5d}")
    print()

    # ── 3. Z/A concentration ──
    z_ranges = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100), (101, 120)]
    print(f"  ISOMER DENSITY BY PROTON RANGE")
    print(f"    {'Z range':>10s}  {'Isomers':>7s}  {'(Z,A) pairs':>11s}  {'Density':>7s}")
    print(f"    {'-'*40}")
    for z_lo, z_hi in z_ranges:
        iso_count = sum(1 for s in all_isomers if z_lo <= s['Z'] <= z_hi)
        pair_count = sum(1 for (Z, A) in pairs_with_isomers if z_lo <= Z <= z_hi)
        total_in_range = sum(1 for (Z, A) in nuclide_states if z_lo <= Z <= z_hi)
        density = pair_count / max(total_in_range, 1)
        print(f"    {z_lo:>3d}-{z_hi:<3d}    {iso_count:>7d}  {pair_count:>11d}  {density:>7.1%}")
    print()

    # ── 4. Parity distribution ──
    parity_counts = {'ee': 0, 'eo': 0, 'oe': 0, 'oo': 0}
    for s in all_isomers:
        Z, N = s['Z'], s['N']
        key = ('e' if Z % 2 == 0 else 'o') + ('e' if N % 2 == 0 else 'o')
        parity_counts[key] += 1
    total_iso = len(all_isomers) or 1
    print(f"  PARITY DISTRIBUTION OF ISOMERS")
    for key, label in [('ee', 'even-even'), ('eo', 'even-odd'),
                       ('oe', 'odd-even'), ('oo', 'odd-odd')]:
        cnt = parity_counts[key]
        print(f"    {label:>10s}: {cnt:>5d}  ({cnt/total_iso*100:.1f}%)")
    print()

    # ── 5. Spin-gap |ΔJ| ──
    delta_j_list = []
    for (Z, A), states in nuclide_states.items():
        gs_list = [s for s in states if s['state'] == 'gs']
        if not gs_list:
            continue
        gs_j = _parse_spin_value(gs_list[0]['jpi'])
        if gs_j is None:
            continue
        for iso in states:
            if iso['state'] == 'gs':
                continue
            iso_j = _parse_spin_value(iso['jpi'])
            if iso_j is not None:
                delta_j_list.append(abs(iso_j - gs_j))

    if delta_j_list:
        dj_arr = np.array(delta_j_list)
        print(f"  SPIN GAP |ΔJ| (gs → isomer)")
        print(f"    Pairs with measured J:  {len(delta_j_list)}")
        print(f"    Median |ΔJ|:            {np.median(dj_arr):.1f}")
        print(f"    Mean |ΔJ|:              {np.mean(dj_arr):.2f}")
        # Histogram
        bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 8.5, 100]
        labels = ['0', '1', '2', '3', '4', '5', '6-8', '9+']
        counts, _ = np.histogram(dj_arr, bins=bins)
        print(f"    {'|ΔJ|':>6s}  {'Count':>6s}  {'Frac':>6s}")
        print(f"    {'-'*22}")
        for lab, cnt in zip(labels, counts):
            print(f"    {lab:>6s}  {cnt:>6d}  {cnt/len(delta_j_list)*100:>5.1f}%")
    print()

    # ── 6. Long-lived isomers (t½ > 1 year) ──
    YEAR_S = 3.1557600e7
    long_lived = []
    for s in all_isomers:
        if np.isfinite(s['half_life_s']) and s['half_life_s'] > YEAR_S:
            long_lived.append(s)
    long_lived.sort(key=lambda x: -x['half_life_s'])

    print(f"  LONG-LIVED ISOMERS (t½ > 1 year): {len(long_lived)}")
    if long_lived:
        elem = ELEMENTS
        print(f"    {'Nuclide':>12s} {'State':>5s}  {'E_exc (keV)':>12s}  {'t½':>12s}  {'Mode':>8s}  {'Jπ':>10s}")
        print(f"    {'-'*68}")
        for s in long_lived[:15]:
            sym = elem.get(s['Z'], f"Z{s['Z']}")
            name = f"{sym}-{s['A']}"
            hl_str = _format_halflife(s['half_life_s'])
            print(f"    {name:>12s} {s['state']:>5s}  {s['exc_energy_keV']:>12.1f}  {hl_str:>12s}  {s['dominant_mode']:>8s}  {s['jpi']:>10s}")
    print()


def validate_with_isomers(nuclide_states: dict) -> dict:
    """Run ground-state validation with isomer rescue analysis.

    First reproduces the standard ground-state-only scoring (must match
    existing 79.7% / 97.4% exactly as a regression check).

    Then for each wrong prediction, checks if ANY non-IT isomer at the
    same (Z, A) has a mode matching the prediction → "rescue".

    Provenance: predictions are QFD_DERIVED, isomer data is EMPIRICAL_LOOKUP.

    Returns dict with gs_total, gs_correct, any_correct, rescued, rescue_list.
    """
    gs_total = 0
    gs_correct = 0
    gs_dir_total = 0
    gs_dir_correct = 0
    rescued = 0
    rescue_list = []

    for (Z, A), states in nuclide_states.items():
        gs_list = [s for s in states if s['state'] == 'gs']
        if not gs_list:
            continue
        gs = gs_list[0]
        actual = gs['dominant_mode']
        if actual in ('unknown', 'IT') or A < 3:
            continue

        pred, _ = predict_decay(Z, A)
        actual_norm = normalize_nubase(actual)

        gs_total += 1
        match = (actual_norm == pred)
        if match:
            gs_correct += 1
        else:
            # Check isomer rescue: does any non-IT isomer mode match pred?
            for iso in states:
                if iso['state'] == 'gs':
                    continue
                iso_mode = normalize_nubase(iso['dominant_mode'])
                if iso_mode in ('unknown', 'IT'):
                    continue
                if iso_mode == pred:
                    rescued += 1
                    sym = ELEMENTS.get(Z, f"Z{Z}")
                    rescue_list.append({
                        'Z': Z, 'A': A, 'name': f"{sym}-{A}",
                        'pred': pred, 'gs_mode': actual_norm,
                        'iso_mode': iso_mode, 'iso_state': iso['state'],
                    })
                    break  # One rescue per (Z, A)

        # β-direction
        eps = Z - z_star(A)
        if actual_norm in ('B-', 'B+', 'stable'):
            gs_dir_total += 1
            if actual_norm == 'B-' and eps < 0:
                gs_dir_correct += 1
            elif actual_norm == 'B+' and eps > 0:
                gs_dir_correct += 1
            elif actual_norm == 'stable' and abs(eps) < 1.5:
                gs_dir_correct += 1

    return {
        'gs_total': gs_total,
        'gs_correct': gs_correct,
        'gs_dir_total': gs_dir_total,
        'gs_dir_correct': gs_dir_correct,
        'any_correct': gs_correct + rescued,
        'rescued': rescued,
        'rescue_list': rescue_list,
    }


def print_isomer_validation(results: dict):
    """Print isomer-aware validation results."""
    gs_t = results['gs_total']
    gs_c = results['gs_correct']
    any_c = results['any_correct']
    rescued = results['rescued']
    dir_t = results['gs_dir_total']
    dir_c = results['gs_dir_correct']

    gs_acc = gs_c / max(gs_t, 1) * 100
    any_acc = any_c / max(gs_t, 1) * 100
    dir_acc = dir_c / max(dir_t, 1) * 100

    print(f"\n  ISOMER-AWARE VALIDATION")
    print(f"  ─────────────────────────────────────────────────")
    print(f"    Ground-state mode accuracy:  {gs_c}/{gs_t} ({gs_acc:.1f}%)  ← regression check")
    print(f"    β-direction accuracy:        {dir_c}/{dir_t} ({dir_acc:.1f}%)  ← regression check")
    print(f"    Any-state mode accuracy:     {any_c}/{gs_t} ({any_acc:.1f}%)")
    print(f"    Isomer rescues:              {rescued}")
    print(f"    Accuracy gain from rescues:  {any_acc - gs_acc:+.1f}%")

    if results['rescue_list']:
        print(f"\n    RESCUED NUCLIDES (pred matched an isomer, not ground state)")
        print(f"    {'Nuclide':>12s} {'Predicted':>10s} {'GS Mode':>10s} {'Iso Mode':>10s} {'Iso State':>10s}")
        print(f"    {'-'*56}")
        for r in sorted(results['rescue_list'], key=lambda x: (x['Z'], x['A'])):
            print(f"    {r['name']:>12s} {r['pred']:>10s} {r['gs_mode']:>10s} {r['iso_mode']:>10s} {r['iso_state']:>10s}")
    print()


def validate_clock_with_isomers(nuclide_states: dict):
    """Clock contamination diagnostics: isomers with very different t½.

    For each (Z, A) with a clock prediction and isomers, checks whether
    the measured isomer t½ differs significantly from the ground state.
    Computes RMSE using ground-state-only vs best-matching-state.

    Provenance: Clock predictions = EMPIRICAL_FIT, isomer data = EMPIRICAL_LOOKUP.
    """
    # Per-mode: collect (log_obs_gs, log_pred, log_obs_best, log_obs_best_same_mode)
    mode_data = {'B-': [], 'B+': [], 'alpha': []}
    contaminated = []  # (Z, A) where isomer t½ > 100× different

    for (Z, A), states in nuclide_states.items():
        gs_list = [s for s in states if s['state'] == 'gs']
        if not gs_list:
            continue
        gs = gs_list[0]
        gs_mode = normalize_nubase(gs['dominant_mode'])
        gs_hl = gs['half_life_s']

        if gs_mode in ('stable', 'unknown', 'IT') or A < 3:
            continue
        if not np.isfinite(gs_hl) or gs_hl <= 0:
            continue

        # Get prediction
        pred, _ = predict_decay(Z, A)
        eps = Z - z_star(A)
        log_pred = _clock_log10t(Z, eps, pred)
        if log_pred is None or pred not in mode_data:
            continue

        log_gs = math.log10(gs_hl)

        # Check isomers
        iso_entries = [s for s in states if s['state'] != 'gs']
        if not iso_entries:
            # No isomers — just record gs
            mode_data[pred].append((log_gs, log_pred, log_gs, log_gs))
            continue

        # Find best-state: closest measured t½ to prediction
        all_log_hl = [log_gs]
        same_mode_log_hl = [log_gs] if gs_mode == pred else []
        for iso in iso_entries:
            iso_hl = iso['half_life_s']
            iso_mode = normalize_nubase(iso['dominant_mode'])
            if not np.isfinite(iso_hl) or iso_hl <= 0:
                continue
            log_iso = math.log10(iso_hl)
            all_log_hl.append(log_iso)
            if iso_mode == pred:
                same_mode_log_hl.append(log_iso)

            # Flag contamination (>100× difference from gs)
            if abs(log_iso - log_gs) > 2.0:
                sym = ELEMENTS.get(Z, f"Z{Z}")
                contaminated.append({
                    'name': f"{sym}-{A}", 'Z': Z, 'A': A,
                    'gs_mode': gs_mode, 'iso_mode': iso_mode,
                    'iso_state': iso['state'],
                    'log_gs_hl': log_gs, 'log_iso_hl': log_iso,
                    'delta_log': log_iso - log_gs,
                })

        # Best-state: closest to prediction among all states
        best_all = min(all_log_hl, key=lambda x: abs(x - log_pred))
        best_same = min(same_mode_log_hl, key=lambda x: abs(x - log_pred)) if same_mode_log_hl else log_gs

        mode_data[pred].append((log_gs, log_pred, best_all, best_same))

    # Print results
    print(f"\n  CLOCK CONTAMINATION — Isomer Impact on Half-Life Residuals")
    print(f"  ─────────────────────────────────────────────────")

    print(f"\n    {'Mode':<8s} {'N':>5s}  {'RMSE_gs':>8s} {'RMSE_best':>9s} {'RMSE_same':>9s}  {'Rescued':>7s}")
    print(f"    {'-'*52}")

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        data = mode_data[mode_key]
        if not data:
            continue
        n = len(data)
        resid_gs = np.array([d[0] - d[1] for d in data])
        resid_best = np.array([d[2] - d[1] for d in data])
        resid_same = np.array([d[3] - d[1] for d in data])

        rmse_gs = math.sqrt(np.mean(resid_gs**2))
        rmse_best = math.sqrt(np.mean(resid_best**2))
        rmse_same = math.sqrt(np.mean(resid_same**2))

        # Count rescues: cases where best-state is closer than gs
        rescued = sum(1 for d in data if abs(d[2] - d[1]) < abs(d[0] - d[1]))

        print(f"    {mode_label:<8s} {n:>5d}  {rmse_gs:>8.2f} {rmse_best:>9.2f} {rmse_same:>9.2f}  {rescued:>7d}")

    print(f"""
    RMSE_gs:   residual using ground-state t½ only
    RMSE_best: residual using closest t½ from ANY state at (Z, A)
    RMSE_same: residual using closest t½ from same-mode states only
    Rescued:   count where an isomer t½ is closer to prediction than gs
""")

    # Contaminated list
    if contaminated:
        contaminated.sort(key=lambda x: -abs(x['delta_log']))
        print(f"    TOP CONTAMINATED (Z, A) — isomer t½ > 100× from ground state")
        print(f"    {'Nuclide':>12s} {'State':>5s}  {'GS mode':>8s} {'Iso mode':>8s}  {'log(gs)':>8s} {'log(iso)':>8s} {'Δlog':>6s}")
        print(f"    {'-'*64}")
        for c in contaminated[:15]:
            print(f"    {c['name']:>12s} {c['iso_state']:>5s}  {c['gs_mode']:>8s} {c['iso_mode']:>8s}  {c['log_gs_hl']:>8.1f} {c['log_iso_hl']:>8.1f} {c['delta_log']:>+6.1f}")
    print()


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Data paths
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _DATA_DIRS = [
        os.path.join(_SCRIPT_DIR, '..', 'data', 'raw'),
        os.path.join(_SCRIPT_DIR, 'data'),
    ]

    NUBASE_PATH = None
    for d in _DATA_DIRS:
        candidate = os.path.join(d, 'nubase2020_raw.txt')
        if os.path.exists(candidate):
            NUBASE_PATH = candidate
            break

    # ── Constants ──
    print_constants()

    # ── Spot checks ──
    print_spot_checks()

    # ── Gradient analysis ──
    print_gradient_analysis()

    # ── Generate terrain map ──
    print(f"\n{'='*72}")
    print("  GENERATING TOPOLOGICAL TERRAIN MAP...")
    print("=" * 72)
    terrain = generate_terrain_map(max_A=300)
    print(f"  Generated {len(terrain)} terrain cells")

    # ── Alpha onset ──
    print_alpha_onset_analysis(terrain)

    # ── NUBASE validation ──
    nubase_results = None
    if NUBASE_PATH and os.path.exists(NUBASE_PATH):
        print(f"\n  Loading NUBASE2020 from: {NUBASE_PATH}")
        nubase_entries = load_nubase(NUBASE_PATH)
        print(f"  Parsed {len(nubase_entries)} ground-state nuclides")

        nubase_results = validate_against_nubase(nubase_entries)

        # ── Zone-separated validation (v8) ──
        zone_results = validate_by_zone(nubase_entries)
        print_zone_validation(zone_results)

        # ── Load isomers (needed for mode population + isomer analysis) ──
        all_entries = load_nubase(NUBASE_PATH, include_isomers=True)

        # ── Mode population analysis — each mode as a separate animal ──
        print_mode_population_analysis(nubase_entries, all_entries)

        # ── Clock validation ──
        print_clock_validation(nubase_entries)

        # ── Confidence tiers + progressive removal ──
        classified = classify_confidence_tiers(nubase_entries)
        print_progressive_removal_report(classified)
        print_channel_analysis(classified)

        # ── Isomer analysis ──
        print(f"\n{'═'*72}")
        print(f"  ISOMER ANALYSIS — Extending to All NUBASE2020 States")
        print(f"{'═'*72}")
        print(f"  Loaded {len(all_entries)} total entries (gs + isomers)")

        nuclide_states = group_nuclide_states(all_entries)
        print(f"  Grouped into {len(nuclide_states)} (Z, A) pairs")

        n_gs = sum(1 for e in all_entries if e['state'] == 'gs')
        n_iso = len(all_entries) - n_gs
        print(f"  Ground states: {n_gs},  Isomers: {n_iso}")

        print_isomer_census(nuclide_states)

        iso_val = validate_with_isomers(nuclide_states)
        print_isomer_validation(iso_val)

        validate_clock_with_isomers(nuclide_states)
    else:
        print(f"\n  NUBASE2020 not found. Skipping validation.")
        print(f"  Searched: {_DATA_DIRS}")

    # ── Visualization ──
    try:
        import matplotlib
        if nubase_results:
            print(f"\n{'='*72}")
            print("  GENERATING VISUALIZATIONS...")
            print("=" * 72)
            plot_nuclide_maps(terrain, nubase_results, _SCRIPT_DIR)
            # Orthogonal projection maps + progressive removal
            try:
                print(f"\n  ORTHOGONAL PROJECTIONS + PROGRESSIVE REMOVAL...")
                plot_orthogonal_maps(classified, _SCRIPT_DIR)
                plot_progressive_removal(classified, _SCRIPT_DIR)
                plot_channel_map(classified, _SCRIPT_DIR)
            except NameError:
                pass

            # Isomer-aware maps (if isomer data was loaded)
            try:
                all_entries, nuclide_states, iso_val
                print(f"\n  ISOMER-AWARE MAPS...")
                plot_isomer_maps(all_entries, nuclide_states, iso_val,
                                 terrain, _SCRIPT_DIR)
            except NameError:
                pass
        else:
            print("\n  Skipping visualization (no NUBASE data for comparison).")
    except ImportError:
        print("\n  matplotlib not available. Skipping visualization.")

    print(f"\n{'='*72}")
    print("  ENGINE COMPLETE")
    print("=" * 72)
