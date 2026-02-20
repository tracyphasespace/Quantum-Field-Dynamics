#!/usr/bin/env python3
"""
A-B Connection-Level Barrier Modification for the Rift Abundance Model
======================================================================

QUESTION: Does the ψ-field overlap between two BHs at L1 lower the
effective escape barrier, and by how much?

PHYSICS:
  From U.2, each BH has a ψ-field tail: δψ/ψ₀ = R_s/(ξ_QFD × r)
  where ξ_QFD = k_geom² × 5/6 = 4.4032² × 5/6 = 16.154

  At L1 (r = d/2 from each BH), the tails superpose:
    δψ_gap/ψ₀ = 4R_s/(ξ_QFD × d)

  The gravitational potential at L1 from both BHs:
    |Φ_L1|/c² = 2R_s/d   (each contributes R_s/(d/2), but includes L1 saddle)

  KEY RATIO (constant, independent of d):
    (δψ/ψ₀) / (|Φ_L1|/c²) = (4R_s/(ξ_QFD × d)) / (2R_s/d)
                             = 2/ξ_QFD

  In QFD, the ψ-overlap RAISES the vacuum field at L1, softening soliton
  binding and reducing the effective escape barrier.

  Fractional barrier reduction = 2/ξ_QFD ≈ 12.4%
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# 1. QFD CONSTANTS
# ═══════════════════════════════════════════════════════════════

ALPHA = 1.0 / 137.035999
BETA = 3.043233053
K_GEOM = 4.4032
XI_QFD = K_GEOM**2 * 5.0 / 6.0  # = 16.154
ETA_TOPO = 0.02985

# Masses (AMU)
M_E = 0.00054858
M_H = 1.00784
M_HE = 4.00260

# Current calibrated value
RIFT_K = 5.48

# Barrier reductions per cycle
BARRIERS = {'Shallow': 0.950, 'Deep': 0.985, 'Cataclysmic': 0.998}
FREQS = {'Shallow': 3, 'Deep': 1, 'Cataclysmic': 1}

# Interior pools (Tracy v3)
POOLS = {
    'Shallow':     {'H': 89.9, 'He': 10.1},
    'Deep':        {'H': 47.7, 'He': 52.3},
    'Cataclysmic': {'H': 45.8, 'He': 54.2},
}


# ═══════════════════════════════════════════════════════════════
# 2. ψ-OVERLAP AT L1 (from U.2)
# ═══════════════════════════════════════════════════════════════

def psi_overlap_at_L1(d_over_Rs):
    """δψ/ψ₀ at L1 point for two equal-mass BHs at separation d.

    Both BH tails contribute: δψ_total = 2 × (1/ξ) × (R_s/(d/2))
    = 4R_s / (ξ × d)
    """
    return 4.0 / (XI_QFD * d_over_Rs)


def grav_potential_at_L1(d_over_Rs):
    """|Φ_L1|/c² for two equal-mass BHs at separation d.

    Each contributes GM/(d/2) = c²R_s/d.  Total = 2R_s/d.
    (Ignoring centrifugal — this is the depth of the saddle.)
    """
    return 2.0 / d_over_Rs


# ═══════════════════════════════════════════════════════════════
# 3. BARRIER MODIFICATION
# ═══════════════════════════════════════════════════════════════

# The key result: since both scale as 1/d, the ratio is constant
CONNECTION_BARRIER_FRAC = 2.0 / XI_QFD

print("=" * 74)
print("  A-B CONNECTION-LEVEL BARRIER MODIFICATION")
print("  Rift Abundance Model: Escape Velocity Correction")
print("=" * 74)

print(f"\n  ξ_QFD = k_geom² × 5/6 = {K_GEOM}² × 5/6 = {XI_QFD:.3f}")
print(f"  Connection barrier fraction = 2/ξ_QFD = {CONNECTION_BARRIER_FRAC:.4f} = {CONNECTION_BARRIER_FRAC*100:.1f}%")

print(f"\n{'─'*74}")
print(f"  A. ψ-OVERLAP vs GRAVITY AT L1 (showing d-independence)")
print(f"{'─'*74}")
print(f"  {'d/R_s':>8s} │ {'δψ/ψ₀':>10s} {'|Φ|/c²':>10s} {'Ratio':>10s} {'= 2/ξ?':>8s}")
print(f"  {'─'*8} │ {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

for d in [3.45, 5.0, 8.3, 10.0, 20.0, 50.0]:
    dpsi = psi_overlap_at_L1(d)
    phi = grav_potential_at_L1(d)
    ratio = dpsi / phi
    check = "✓" if abs(ratio - CONNECTION_BARRIER_FRAC) < 1e-10 else "✗"
    print(f"  {d:8.2f} │ {dpsi:10.6f} {phi:10.6f} {ratio:10.6f} {check:>8s}")

# ═══════════════════════════════════════════════════════════════
# 4. EFFECT ON ESCAPE PROBABILITY
# ═══════════════════════════════════════════════════════════════

print(f"\n{'─'*74}")
print(f"  B. EFFECT ON ESCAPE PROBABILITY")
print(f"{'─'*74}")

print(f"\n  The ψ-overlap reduces the effective gravitational barrier by {CONNECTION_BARRIER_FRAC*100:.1f}%.")
print(f"  This is equivalent to replacing k → k_eff = k × (1 - 2/ξ_QFD)")
print(f"")
print(f"  k_calibrated (current) = {RIFT_K}")
print(f"  k_eff (with A-B)       = {RIFT_K * (1 - CONNECTION_BARRIER_FRAC):.3f}")
print(f"")
print(f"  INTERPRETATION:")
print(f"  If k=5.48 was calibrated to match the observed 75/25 ratio,")
print(f"  it already INCLUDES the connection-level effect implicitly.")
print(f"  The 'bare' classical barrier (without A-B) would be:")
print(f"  k_bare = k_total / (1 - 2/ξ_QFD) = {RIFT_K / (1 - CONNECTION_BARRIER_FRAC):.3f}")


def escape_prob(mass, barrier, k):
    return np.exp(-mass * k * (1 - barrier))


def selectivity_ratio(m1, m2, barrier, k):
    return np.exp((m2 - m1) * k * (1 - barrier))


def run_model(k_val, label):
    """Run the Rift Abundance model with given k value."""
    total_H = total_He = 0
    results = {}
    for cycle in ['Shallow', 'Deep', 'Cataclysmic']:
        b = BARRIERS[cycle]
        p_H = escape_prob(M_H, b, k_val)
        p_He = escape_prob(M_HE, b, k_val)
        S = selectivity_ratio(M_H, M_HE, b, k_val)

        pool_H = POOLS[cycle]['H']
        pool_He = POOLS[cycle]['He']
        out_H = pool_H * p_H
        out_He = pool_He * p_He
        tot = out_H + out_He
        pct_H = out_H / tot * 100
        pct_He = out_He / tot * 100

        f = FREQS[cycle]
        total_H += out_H * f
        total_He += out_He * f

        results[cycle] = {
            'H%': pct_H, 'He%': pct_He, 'S': S,
            'P_H': p_H, 'P_He': p_He
        }

    cosmic_H = total_H / (total_H + total_He) * 100
    cosmic_He = total_He / (total_H + total_He) * 100
    results['GLOBAL'] = {'H%': cosmic_H, 'He%': cosmic_He}
    return results


# ═══════════════════════════════════════════════════════════════
# 5. COMPARISON: THREE SCENARIOS
# ═══════════════════════════════════════════════════════════════

k_bare = RIFT_K / (1 - CONNECTION_BARRIER_FRAC)  # Classical only
k_total = RIFT_K  # Current calibrated (implicitly includes A-B?)
k_ab = RIFT_K * (1 - CONNECTION_BARRIER_FRAC)  # If k=5.48 is bare, add A-B on top

print(f"\n{'─'*74}")
print(f"  C. THREE SCENARIOS — H/He RATIO")
print(f"{'─'*74}")
print(f"\n  Scenario 1: k={k_bare:.3f} (bare classical — NO connection effect)")
print(f"  Scenario 2: k={k_total:.3f} (current calibration — status quo)")
print(f"  Scenario 3: k={k_ab:.3f} (k=5.48 is bare + A-B correction on top)")

r1 = run_model(k_bare, "Bare Classical")
r2 = run_model(k_total, "Current k=5.48")
r3 = run_model(k_ab, "k=5.48 + A-B")

print(f"\n  {'Cycle':>12s} │ {'S1 H% (bare)':>13s} {'S2 H% (k=5.48)':>15s} {'S3 H% (+A-B)':>14s}")
print(f"  {'─'*12} │ {'─'*13} {'─'*15} {'─'*14}")
for cycle in ['Shallow', 'Deep', 'Cataclysmic']:
    print(f"  {cycle:>12s} │ {r1[cycle]['H%']:12.1f}% {r2[cycle]['H%']:14.1f}% {r3[cycle]['H%']:13.1f}%")
print(f"  {'─'*12} │ {'─'*13} {'─'*15} {'─'*14}")
print(f"  {'GLOBAL':>12s} │ {r1['GLOBAL']['H%']:12.2f}% {r2['GLOBAL']['H%']:14.2f}% {r3['GLOBAL']['H%']:13.2f}%")

print(f"\n  Selectivities S(H/He):")
print(f"  {'Cycle':>12s} │ {'S1 (bare)':>10s} {'S2 (k=5.48)':>12s} {'S3 (+A-B)':>12s}")
print(f"  {'─'*12} │ {'─'*10} {'─'*12} {'─'*12}")
for cycle in ['Shallow', 'Deep', 'Cataclysmic']:
    print(f"  {cycle:>12s} │ {r1[cycle]['S']:10.3f} {r2[cycle]['S']:12.3f} {r3[cycle]['S']:12.3f}")


# ═══════════════════════════════════════════════════════════════
# 6. ESCAPE VELOCITY REDUCTION
# ═══════════════════════════════════════════════════════════════

print(f"\n{'─'*74}")
print(f"  D. ESCAPE VELOCITY MODIFICATION")
print(f"{'─'*74}")
print(f"\n  Classical escape velocity at L1: v_esc = c × √(2R_s/d)")
print(f"  Connection-modified:             v_esc' = v_esc × √(1 - 2/ξ_QFD)")
print(f"                                         = v_esc × {np.sqrt(1 - CONNECTION_BARRIER_FRAC):.4f}")
print(f"  Reduction factor: {(1 - np.sqrt(1 - CONNECTION_BARRIER_FRAC))*100:.1f}%")
print(f"")
print(f"  Physical mechanism:")
print(f"  The ψ-overlap at L1 RAISES the vacuum field, softening the")
print(f"  soliton's internal binding. A particle at L1 is already")
print(f"  partially destabilized by the connection-level field — it")
print(f"  needs {(1 - np.sqrt(1 - CONNECTION_BARRIER_FRAC))*100:.1f}% less velocity to escape than the force picture predicts.")


# ═══════════════════════════════════════════════════════════════
# 7. CROSS-CHECK: d_topo vs d_tidal
# ═══════════════════════════════════════════════════════════════

print(f"\n{'─'*74}")
print(f"  E. CONSISTENCY WITH U.2 RIFT OPENING DISTANCE")
print(f"{'─'*74}")

d_topo = 4.0 / (XI_QFD * ETA_TOPO)
d_tidal = 3.45  # Classical Roche limit in R_s units

print(f"\n  d_topo  = 4/(ξ_QFD × η_topo) = 4/({XI_QFD:.3f} × {ETA_TOPO}) = {d_topo:.2f} R_s")
print(f"  d_tidal = {d_tidal} R_s (classical Roche limit)")
print(f"  Ratio:    d_topo / d_tidal = {d_topo/d_tidal:.2f}")
print(f"")
print(f"  At d = d_topo ({d_topo:.1f} R_s):")
print(f"    δψ/ψ₀ = {psi_overlap_at_L1(d_topo):.6f} = η_topo ✓ (rift channel opens)")
print(f"    v_esc reduction = {(1 - np.sqrt(1 - CONNECTION_BARRIER_FRAC))*100:.1f}% (same at all d)")
print(f"")
print(f"  At d = d_tidal ({d_tidal} R_s):")
print(f"    δψ/ψ₀ = {psi_overlap_at_L1(d_tidal):.6f} (>> η_topo → deep disruption)")
print(f"    v_esc reduction = {(1 - np.sqrt(1 - CONNECTION_BARRIER_FRAC))*100:.1f}% (same at all d)")


# ═══════════════════════════════════════════════════════════════
# 8. THE REAL QUESTION: CAN WE DERIVE k FROM FIRST PRINCIPLES?
# ═══════════════════════════════════════════════════════════════

print(f"\n{'─'*74}")
print(f"  F. TOWARD A FIRST-PRINCIPLES k")
print(f"{'─'*74}")
print(f"""
  The calibrated k=5.48 encodes: E_barrier / (k_B T_rift)

  If we could compute the barrier and rift temperature from QFD:
    E_barrier = m_p c² × (R_s/r_surface - R_s/(d/2))
    T_rift    = energy released by binary inspiral / (degrees of freedom)

  The A-B correction modifies E_barrier:
    E_barrier_AB = E_barrier × (1 - 2/ξ_QFD)
                 = E_barrier × {1 - CONNECTION_BARRIER_FRAC:.4f}

  Currently: k = 5.48 is ONE calibration parameter.
  Target:    k = f(β, k_geom, α) with ZERO free parameters.

  The connection-level correction 2/ξ_QFD = {CONNECTION_BARRIER_FRAC:.4f} is
  already a zero-parameter QFD prediction. It tells us that the
  'force picture' overestimates the barrier by {CONNECTION_BARRIER_FRAC*100:.1f}%.
""")

# ═══════════════════════════════════════════════════════════════
# 9. SEPARATION DISTANCE EXTENSION
# ═══════════════════════════════════════════════════════════════

# Both gravity and ψ-tail go as 1/d, so the barrier scales as 1/d.
# For the same escape probability at larger d:
#   f(d_new) × (1 - corr) = f(d_old)  →  d_new = d_old / (1 - corr)
D_FACTOR = 1.0 / (1.0 - CONNECTION_BARRIER_FRAC)
V_FACTOR = D_FACTOR**3

d_tidal_ab = d_tidal * D_FACTOR
d_topo_ab = d_topo * D_FACTOR

print(f"\n{'─'*74}")
print(f"  G. SEPARATION DISTANCE EXTENSION")
print(f"{'─'*74}")
print(f"\n  Since both gravity and ψ-tail scale as 1/d, the 12.4% barrier")
print(f"  reduction translates to a {(D_FACTOR-1)*100:.1f}% increase in maximum separation")
print(f"  at which escape is possible.")
print(f"")
print(f"  Separation factor: 1/(1 - 2/ξ_QFD) = {D_FACTOR:.4f} (+{(D_FACTOR-1)*100:.1f}%)")
print(f"  Volume factor:     {D_FACTOR:.4f}³ = {V_FACTOR:.3f} (+{(V_FACTOR-1)*100:.0f}%)")

print(f"\n  {'':>28s} │ {'Classical':>10s} {'+ A-B':>10s} {'Change':>10s}")
print(f"  {'─'*28} │ {'─'*10} {'─'*10} {'─'*10}")
print(f"  {'Tidal disruption (d_tidal)':>28s} │ {d_tidal:8.2f} Rs {d_tidal_ab:8.2f} Rs  +{(D_FACTOR-1)*100:.1f}%")
print(f"  {'Topological channel (d_topo)':>28s} │ {d_topo:8.2f} Rs {d_topo_ab:8.2f} Rs  +{(D_FACTOR-1)*100:.1f}%")

print(f"\n  Processing VOLUME increase:")
print(f"    Tidal sphere:  {4/3*np.pi*d_tidal**3:7.0f} Rs³ → {4/3*np.pi*d_tidal_ab**3:7.0f} Rs³  (+{(V_FACTOR-1)*100:.0f}%)")
print(f"    Topo sphere:   {4/3*np.pi*d_topo**3:7.0f} Rs³ → {4/3*np.pi*d_topo_ab**3:7.0f} Rs³  (+{(V_FACTOR-1)*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════
# 10. THREE-ZONE STRUCTURE
# ═══════════════════════════════════════════════════════════════

print(f"\n{'─'*74}")
print(f"  H. THREE NESTED RIFT PROCESSING ZONES")
print(f"{'─'*74}")

total_vol_ratio = (d_topo_ab / d_tidal)**3

print(f"""
  Zone 1 — CLASSICAL TIDAL (d ≤ {d_tidal:.2f} R_s)
    Mechanism:  Force-level disruption (tidal stress > soliton binding)
    Physics:    F_tidal ∝ 1/d³ (derivative of potential)
    Volume:     {4/3*np.pi*d_tidal**3:.0f} R_s³

  Zone 2 — TOPOLOGICAL CHANNEL ({d_tidal:.2f} < d ≤ {d_topo:.2f} R_s)
    Mechanism:  ψ-overlap opens connection-level channel
    Threshold:  δψ_gap/ψ₀ = η_topo = {ETA_TOPO} (separatrix condition)
    Physics:    Potential-level (A-B); F = 0 but connection ≠ 0
    Volume:     {4/3*np.pi*d_topo**3:.0f} R_s³  ({(d_topo/d_tidal)**3:.1f}× Zone 1)

  Zone 3 — A-B EXTENDED ({d_topo:.2f} < d ≤ {d_topo_ab:.2f} R_s)
    Mechanism:  Escape velocity lowered by {(1-np.sqrt(1-CONNECTION_BARRIER_FRAC))*100:.1f}%
    Physics:    ψ-softening of soliton binding at L1
    Volume:     {4/3*np.pi*d_topo_ab**3:.0f} R_s³  ({(d_topo_ab/d_tidal)**3:.1f}× Zone 1)

  TOTAL: Zone 3 / Zone 1 = {total_vol_ratio:.1f}×  more processing volume
         ({(d_topo/d_tidal)**3:.1f}× from topo channel) × ({V_FACTOR:.3f} from A-B extension)
""")


# ═══════════════════════════════════════════════════════════════
# 11. RIFT THROUGHPUT IMPLICATIONS
# ═══════════════════════════════════════════════════════════════

print(f"{'─'*74}")
print(f"  I. RIFT THROUGHPUT IMPLICATIONS")
print(f"{'─'*74}")
print(f"""
  The rift processes ~{total_vol_ratio:.0f}× more matter per binary interaction
  than the classical force picture predicts.

  This does NOT change the H/He RATIO (attractor is protected — see
  Section C above), but it changes the RATE of cosmic recycling.

  For steady-state cosmology, the rate must be fast enough to replenish
  hydrogen against stellar burndown. The {total_vol_ratio:.0f}× volume enhancement
  means each BH binary interaction reprocesses {total_vol_ratio:.0f}× more matter,
  relaxing the required interaction frequency by the same factor.

  The three-zone structure is a falsifiable prediction:
    - Zone 1 ejecta: collimated, high-velocity (classical jet core)
    - Zone 2 ejecta: broad, moderate-velocity (topological precursor)
    - Zone 3 ejecta: diffuse, low-velocity (connection-level outflow)
  Resolved imaging should reveal this layered structure.
""")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

print("=" * 74)
print("  SUMMARY")
print("=" * 74)
print(f"""
  1. The ψ-overlap at L1 reduces the escape barrier by 2/ξ_QFD = {CONNECTION_BARRIER_FRAC*100:.1f}%
     This is CONSTANT (independent of BH separation d).

  2. Escape velocity is reduced by {(1 - np.sqrt(1 - CONNECTION_BARRIER_FRAC))*100:.1f}%
     (v_esc → v_esc × √(1 - 2/ξ_QFD) = v_esc × {np.sqrt(1 - CONNECTION_BARRIER_FRAC):.4f})

  3. Maximum separation extended by {(D_FACTOR-1)*100:.1f}%:
     d_tidal: {d_tidal:.2f} → {d_tidal_ab:.2f} R_s
     d_topo:  {d_topo:.2f} → {d_topo_ab:.2f} R_s
     Processing volume: +{(V_FACTOR-1)*100:.0f}% (×{V_FACTOR:.3f})

  4. THREE nested zones: tidal ({d_tidal:.2f}) → topo ({d_topo:.2f}) → A-B ({d_topo_ab:.2f} R_s)
     Total volume ratio vs classical: {total_vol_ratio:.0f}×

  5. H/He ratio UNCHANGED: 74.5–74.8% across all scenarios
     (topological protection from discrete interior stratification)

  6. Throughput gain: {total_vol_ratio:.0f}× more matter processed per interaction
     (changes recycling RATE, not composition RATIO)

  7. All from ξ_QFD = k_geom² × 5/6 — ZERO free parameters.
""")
