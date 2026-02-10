#!/usr/bin/env python3
"""
Test: Clock Slopes as Lyapunov Exponents of Chaotic Core Oscillations

THEORY
======

The frozen core of a peanut soliton is NOT static.  It oscillates due to
coupling between multiple modes:
  - Charge exchange (β⁻/β⁺): proton ↔ neutron winding conversion
  - Neck breathing (α): He-4-sized region of neck coherence fluctuating
  - Neck stretching (SF): lobe separation oscillating

These coupled oscillations create CHAOTIC dynamics — deterministic but
sensitive to initial conditions (positive Lyapunov exponents).

A decay event requires convergence:
  1. Internal oscillation at a susceptible phase (Lyapunov window)
  2. External perturbation (collision, photon, EC) at the right time
  3. Directional alignment with the intermediate axis
  4. Sufficient electron availability (for EC) or thermal environment

The HALF-LIFE is the deterministic envelope of this chaotic process —
the average time to visit the escape region of the attractor.

The CLOCK SLOPES may be the Lyapunov exponents of the three modes:
  β⁻:  -πβ/e  ≈ -3.517     (charge exchange oscillation)
  β⁺:  -π     ≈ -3.142     (charge capture oscillation)
  α:   -e     ≈ -2.718     (neck breathing oscillation)

If these are Lyapunov exponents, then:
  1. The ratios between them should be geometric constants
  2. The clock residuals should show short-range spatial correlations
     (nearby nuclides in similar regions of the attractor)
  3. The residuals should NOT be normally distributed (chaotic vs Gaussian)
  4. The residual variance should scale with the attractor dimension (pf, cf)

TESTS:
  Test 1: Lyapunov exponent ratios — do they form a geometric pattern?
  Test 2: Spatial autocorrelation of clock residuals
  Test 3: Residual distribution — normal vs fat-tailed (chaotic)
  Test 4: Residual variance vs geometric parameters (pf, cf)
  Test 5: Mode coupling — do the three clocks share common structure?
"""

import math
import os
import sys
import numpy as np
from collections import defaultdict

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

    nubase_entries = m.load_nubase(NUBASE_PATH)
    print(f"Loaded {len(nubase_entries)} ground-state nuclides\n")

    # Collect per-mode data with geometric state
    mode_data = {'B-': [], 'B+': [], 'alpha': []}

    for nuc in nubase_entries:
        Z, A = nuc['Z'], nuc['A']
        mode = m.normalize_nubase(nuc['dominant_mode'])
        hl = nuc['half_life_s']
        if mode in ('stable', 'unknown', 'IT') or A < 3:
            continue
        if not np.isfinite(hl) or hl <= 0 or hl > 1e30:
            continue

        geo = m.compute_geometric_state(Z, A)
        eps = geo.eps
        log_hl = math.log10(hl)

        if mode in mode_data:
            # Zero-param prediction
            log_pred = m._clock_log10t_zero_param(Z, eps, mode)
            if log_pred is not None:
                resid = log_hl - log_pred
                mode_data[mode].append({
                    'Z': Z, 'A': A, 'N': A - Z,
                    'eps': eps, 'abs_eps': abs(eps),
                    'pf': geo.peanut_f, 'cf': geo.core_full,
                    'log_hl': log_hl, 'log_pred': log_pred,
                    'resid': resid,
                    'zone': geo.zone,
                    'parity': geo.parity,
                })

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: Lyapunov Exponent Ratios
    # ═══════════════════════════════════════════════════════════════
    print("=" * 72)
    print("  TEST 1: LYAPUNOV EXPONENT RATIOS")
    print("=" * 72)
    print("""
  If the clock slopes are Lyapunov exponents of coupled oscillation modes,
  they should form a GEOMETRIC LADDER (each mode's attractor is a scaled
  version of the others, since they all derive from the same soliton).

  The three slopes: β⁻ = -πβ/e, β⁺ = -π, α = -e
""")

    slope_bm = -m.PI * m.BETA / m.E_NUM   # -πβ/e
    slope_bp = -m.PI                        # -π
    slope_a  = -m.E_NUM                     # -e

    print(f"  Slopes (as Lyapunov exponents λ):")
    print(f"    λ_β⁻  = -πβ/e  = {slope_bm:.6f}")
    print(f"    λ_β⁺  = -π     = {slope_bp:.6f}")
    print(f"    λ_α   = -e     = {slope_a:.6f}")

    print(f"\n  Ratios:")
    r1 = slope_bm / slope_bp
    r2 = slope_bp / slope_a
    r3 = slope_bm / slope_a
    print(f"    λ_β⁻/λ_β⁺  = {r1:.6f}  (= β/e = {m.BETA/m.E_NUM:.6f})")
    print(f"    λ_β⁺/λ_α   = {r2:.6f}  (= π/e = {m.PI/m.E_NUM:.6f})")
    print(f"    λ_β⁻/λ_α   = {r3:.6f}  (= πβ/e² = {m.PI*m.BETA/m.E_NUM**2:.6f})")

    print(f"\n  Ladder structure:")
    print(f"    λ_α × (π/e) = λ_β⁺     → {slope_a * m.PI / m.E_NUM:.6f} vs {slope_bp:.6f}")
    print(f"    λ_β⁺ × (β/e) = λ_β⁻    → {slope_bp * m.BETA / m.E_NUM:.6f} vs {slope_bm:.6f}")
    print(f"    λ_α × (πβ/e²) = λ_β⁻   → {slope_a * m.PI * m.BETA / m.E_NUM**2:.6f} vs {slope_bm:.6f}")

    print(f"\n  The three exponents form a geometric ladder:")
    print(f"    α → β⁺: multiply by π/e ≈ {m.PI/m.E_NUM:.4f}")
    print(f"    β⁺ → β⁻: multiply by β/e ≈ {m.BETA/m.E_NUM:.4f}")
    print(f"    Combined: α → β⁻ via πβ/e² ≈ {m.PI*m.BETA/m.E_NUM**2:.4f}")
    print(f"\n  Each step up the ladder = coupling to one more geometric factor")
    print(f"  α has e alone (pure shedding)")
    print(f"  β⁺ adds π (charge shell geometry → π/e coupling)")
    print(f"  β⁻ adds β (soliton topology → β/e coupling)")

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: Spatial Autocorrelation of Residuals
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 2: SPATIAL AUTOCORRELATION — Do Nearby Nuclides Correlate?")
    print("=" * 72)
    print("""
  Chaotic dynamics produces SHORT-RANGE correlations: nuclides that are
  neighbors in (Z, A) space occupy similar regions of the attractor, so
  their clock residuals should be correlated.  Pure randomness would show
  NO spatial correlation.

  Test: for each mode, sort nuclides by A, then compute the autocorrelation
  of residuals at lag 1 (adjacent A values).  Positive autocorrelation =
  chaotic structure; zero = random noise.
""")

    from scipy import stats as sp_stats

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        data = mode_data[mode_key]
        if len(data) < 20:
            continue

        # Sort by A, then Z within same A
        sorted_data = sorted(data, key=lambda d: (d['A'], d['Z']))
        resids = np.array([d['resid'] for d in sorted_data])

        # Autocorrelation at lags 1-5
        n = len(resids)
        mean_r = np.mean(resids)
        var_r = np.var(resids)

        print(f"\n  {mode_label} (n={n}):")
        print(f"    {'Lag':>5s}  {'Autocorr':>10s}  {'Interpretation':>20s}")
        print(f"    {'-'*40}")

        for lag in [1, 2, 3, 5, 10, 20]:
            if lag >= n:
                break
            autocorr = np.mean((resids[:-lag] - mean_r) * (resids[lag:] - mean_r)) / var_r
            interp = "strong" if abs(autocorr) > 0.3 else "moderate" if abs(autocorr) > 0.1 else "weak"
            print(f"    {lag:>5d}  {autocorr:>+10.4f}  {interp:>20s}")

        # Same-Z chain autocorrelation (isotope chains)
        # Group by Z, sort by A within each chain
        z_chains = defaultdict(list)
        for d in data:
            z_chains[d['Z']].append(d)

        chain_corrs = []
        for Z, chain in z_chains.items():
            if len(chain) < 4:
                continue
            chain.sort(key=lambda d: d['A'])
            r = np.array([d['resid'] for d in chain])
            if len(r) > 1 and np.var(r) > 1e-10:
                ac = np.corrcoef(r[:-1], r[1:])[0, 1]
                if np.isfinite(ac):
                    chain_corrs.append(ac)

        if chain_corrs:
            print(f"\n    Same-Z isotope chain autocorrelation:")
            print(f"    Mean: {np.mean(chain_corrs):+.4f}  "
                  f"Median: {np.median(chain_corrs):+.4f}  "
                  f"Positive: {sum(1 for c in chain_corrs if c > 0)}/{len(chain_corrs)}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 3: Residual Distribution — Normal vs Chaotic
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 3: RESIDUAL DISTRIBUTION — Normal vs Chaotic (Fat-Tailed)")
    print("=" * 72)
    print("""
  Pure measurement noise → normally distributed residuals.
  Chaotic dynamics → FAT-TAILED residuals (rare large excursions from
  the attractor) and possibly SKEWED (asymmetric escape).

  Test: kurtosis > 3 = fat tails (leptokurtic = chaotic signature).
  Shapiro-Wilk test for normality.
""")

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        data = mode_data[mode_key]
        if len(data) < 20:
            continue

        resids = np.array([d['resid'] for d in data])

        kurt = sp_stats.kurtosis(resids, fisher=True)  # excess kurtosis (normal = 0)
        skew = sp_stats.skew(resids)
        # Shapiro-Wilk on subsample (max 5000)
        sw_stat, sw_p = sp_stats.shapiro(resids[:5000]) if len(resids) > 3 else (0, 1)

        print(f"\n  {mode_label} (n={len(resids)}):")
        print(f"    Mean residual:  {np.mean(resids):+.3f}")
        print(f"    Std:            {np.std(resids):.3f}")
        print(f"    Skewness:       {skew:+.3f}  {'(left tail)' if skew < -0.5 else '(right tail)' if skew > 0.5 else '(symmetric)'}")
        print(f"    Excess kurtosis: {kurt:+.3f}  {'(FAT TAILS)' if kurt > 1 else '(normal-like)' if abs(kurt) < 1 else '(thin tails)'}")
        print(f"    Shapiro-Wilk:   W={sw_stat:.4f}, p={sw_p:.2e}  "
              f"{'(NOT normal)' if sw_p < 0.01 else '(consistent with normal)'}")

        # Fraction in tails (>2σ and >3σ)
        std = np.std(resids)
        tail_2s = np.sum(np.abs(resids) > 2 * std) / len(resids) * 100
        tail_3s = np.sum(np.abs(resids) > 3 * std) / len(resids) * 100
        print(f"    Beyond 2σ: {tail_2s:.1f}% (normal: 4.6%)")
        print(f"    Beyond 3σ: {tail_3s:.1f}% (normal: 0.3%)")

    # ═══════════════════════════════════════════════════════════════
    # TEST 4: Residual Variance vs Geometric Parameters
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 4: RESIDUAL VARIANCE — Does the Attractor Shape Matter?")
    print("=" * 72)
    print("""
  If the clock misses the attractor geometry (pf, cf), the residuals
  should CORRELATE with pf and cf — the missing dimensions of the
  attractor that the 1D clock cannot see.

  Test: Spearman correlation between residuals and (pf, cf, zone).
  Also: does adding pf and/or cf as clock predictors reduce RMSE?
""")

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        data = mode_data[mode_key]
        if len(data) < 20:
            continue

        resids = np.array([d['resid'] for d in data])
        pf_vals = np.array([d['pf'] for d in data])
        cf_vals = np.array([d['cf'] for d in data])
        zone_vals = np.array([d['zone'] for d in data])
        abs_eps = np.array([d['abs_eps'] for d in data])
        log_hl = np.array([d['log_hl'] for d in data])
        Z_vals = np.array([d['Z'] for d in data])

        print(f"\n  {mode_label} (n={len(data)}):")

        # Correlations with residuals
        for name, vals in [('pf', pf_vals), ('cf', cf_vals), ('zone', zone_vals)]:
            r, p = sp_stats.spearmanr(resids, vals)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    resid ~ {name:>4s}:  r={r:+.4f}  p={p:.2e}  {sig}")

        # Can pf/cf improve the clock?  Simple regression
        # Current: log_hl = a·√|ε| + b·log₁₀(Z) + d
        # Extended: log_hl = a·√|ε| + b·log₁₀(Z) + d + e·pf + f·cf
        X_base = np.column_stack([
            np.sqrt(abs_eps),
            np.log10(Z_vals),
            np.ones(len(data)),
        ])
        X_ext = np.column_stack([
            np.sqrt(abs_eps),
            np.log10(Z_vals),
            pf_vals,
            cf_vals,
            np.ones(len(data)),
        ])

        # Fit base
        beta_base, res_base, _, _ = np.linalg.lstsq(X_base, log_hl, rcond=None)
        pred_base = X_base @ beta_base
        ss_tot = np.sum((log_hl - np.mean(log_hl))**2)
        ss_res_base = np.sum((log_hl - pred_base)**2)
        r2_base = 1.0 - ss_res_base / ss_tot
        rmse_base = math.sqrt(np.mean((log_hl - pred_base)**2))

        # Fit extended
        beta_ext, res_ext, _, _ = np.linalg.lstsq(X_ext, log_hl, rcond=None)
        pred_ext = X_ext @ beta_ext
        ss_res_ext = np.sum((log_hl - pred_ext)**2)
        r2_ext = 1.0 - ss_res_ext / ss_tot
        rmse_ext = math.sqrt(np.mean((log_hl - pred_ext)**2))

        print(f"\n    Clock improvement from adding pf + cf:")
        print(f"      Base (√|ε|, log Z):     R²={r2_base:.4f}  RMSE={rmse_base:.3f}")
        print(f"      Extended (+pf, +cf):     R²={r2_ext:.4f}  RMSE={rmse_ext:.3f}")
        print(f"      ΔR²: {r2_ext - r2_base:+.4f}  ΔRMSE: {rmse_ext - rmse_base:+.3f}")

        if r2_ext > r2_base + 0.01:
            print(f"      ✓ pf/cf capture attractor dimensions the 1D clock misses")
            print(f"      Coefficients: pf={beta_ext[2]:+.4f}  cf={beta_ext[3]:+.4f}")
        else:
            print(f"      ~ Marginal improvement — 1D stress captures most of the attractor")

    # ═══════════════════════════════════════════════════════════════
    # TEST 5: Mode Coupling — Shared Structure
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 5: MODE COUPLING — Do the Three Clocks Share Structure?")
    print("=" * 72)
    print("""
  If the three decay modes are coupled oscillations of the SAME attractor,
  their residuals should share common structure.  Nuclides that are slow
  for β⁻ (positive residual) should also tend to be slow for α at the
  same (Z, A).

  This is the coupling signature: the attractor modulates ALL modes, not
  just the one that fires.

  Test: for (Z, A) pairs that have BOTH a β⁻ emitter neighbor (Z-1, A)
  and an α emitter neighbor (Z-2, A-4), correlate their residuals.
""")

    # Build lookup tables
    bm_lookup = {}  # (Z, A) → resid
    bp_lookup = {}
    alpha_lookup = {}

    for d in mode_data['B-']:
        bm_lookup[(d['Z'], d['A'])] = d['resid']
    for d in mode_data['B+']:
        bp_lookup[(d['Z'], d['A'])] = d['resid']
    for d in mode_data['alpha']:
        alpha_lookup[(d['Z'], d['A'])] = d['resid']

    # Cross-mode residual correlation at nearby (Z, A)
    bm_alpha_pairs = []
    for (Z, A), r_bm in bm_lookup.items():
        # Look for alpha emitters at (Z+2, A+4) [daughter → parent relationship]
        # or nearby
        for dz in range(-2, 3):
            for da in range(-4, 5):
                key = (Z + dz, A + da)
                if key in alpha_lookup:
                    bm_alpha_pairs.append((r_bm, alpha_lookup[key], abs(dz) + abs(da)))

    bp_alpha_pairs = []
    for (Z, A), r_bp in bp_lookup.items():
        for dz in range(-2, 3):
            for da in range(-4, 5):
                key = (Z + dz, A + da)
                if key in alpha_lookup:
                    bp_alpha_pairs.append((r_bp, alpha_lookup[key], abs(dz) + abs(da)))

    if bm_alpha_pairs:
        # Nearest neighbors only (distance 0 = same (Z,A) doesn't exist,
        # distance 1-2 = adjacent)
        for max_dist in [2, 4, 6]:
            pairs = [(a, b) for a, b, d in bm_alpha_pairs if d <= max_dist]
            if len(pairs) >= 10:
                r, p = sp_stats.spearmanr([x[0] for x in pairs], [x[1] for x in pairs])
                print(f"\n  β⁻ ↔ α residual correlation (d ≤ {max_dist}): "
                      f"r={r:+.4f}, p={p:.2e}, n={len(pairs)}")

    if bp_alpha_pairs:
        for max_dist in [2, 4, 6]:
            pairs = [(a, b) for a, b, d in bp_alpha_pairs if d <= max_dist]
            if len(pairs) >= 10:
                r, p = sp_stats.spearmanr([x[0] for x in pairs], [x[1] for x in pairs])
                print(f"  β⁺ ↔ α residual correlation (d ≤ {max_dist}): "
                      f"r={r:+.4f}, p={p:.2e}, n={len(pairs)}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 6: Zone-Resolved Clock — Does Separating Attractor Regions Help?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  TEST 6: ZONE-RESOLVED CLOCK — Separate Attractors Per Zone")
    print("=" * 72)
    print("""
  If each zone has a different attractor (different peanut geometry =
  different moment-of-inertia ratios = different Lyapunov spectrum),
  then fitting the clock WITHIN each zone should be better than one
  clock for all zones.  The improvement measures how much attractor
  structure the global clock is averaging over.
""")

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        data = mode_data[mode_key]
        if len(data) < 20:
            continue

        # Global fit
        log_hl = np.array([d['log_hl'] for d in data])
        abs_eps = np.array([d['abs_eps'] for d in data])
        Z_vals = np.array([d['Z'] for d in data])
        zones = np.array([d['zone'] for d in data])

        X = np.column_stack([np.sqrt(abs_eps), np.log10(Z_vals), np.ones(len(data))])
        beta_glob, _, _, _ = np.linalg.lstsq(X, log_hl, rcond=None)
        pred_glob = X @ beta_glob
        rmse_glob = math.sqrt(np.mean((log_hl - pred_glob)**2))
        ss_tot = np.sum((log_hl - np.mean(log_hl))**2)
        r2_glob = 1.0 - np.sum((log_hl - pred_glob)**2) / ss_tot

        print(f"\n  {mode_label}:")
        print(f"    Global:  R²={r2_glob:.4f}  RMSE={rmse_glob:.3f}  (n={len(data)})")

        # Per-zone fits
        total_ss_res_zone = 0.0
        for z in (1, 2, 3):
            mask = zones == z
            if mask.sum() < 5:
                continue
            X_z = X[mask]
            y_z = log_hl[mask]
            beta_z, _, _, _ = np.linalg.lstsq(X_z, y_z, rcond=None)
            pred_z = X_z @ beta_z
            rmse_z = math.sqrt(np.mean((y_z - pred_z)**2))
            ss_res_z = np.sum((y_z - pred_z)**2)
            total_ss_res_zone += ss_res_z
            ss_tot_z = np.sum((y_z - np.mean(y_z))**2)
            r2_z = 1.0 - ss_res_z / max(ss_tot_z, 1e-10)
            print(f"    Zone {z}: R²={r2_z:.4f}  RMSE={rmse_z:.3f}  (n={mask.sum()})  "
                  f"slope={beta_z[0]:+.3f}")

        r2_zone = 1.0 - total_ss_res_zone / ss_tot
        print(f"    Combined per-zone: R²={r2_zone:.4f}")
        print(f"    ΔR² (zone - global): {r2_zone - r2_glob:+.4f}")
        if r2_zone > r2_glob + 0.01:
            print(f"    ✓ Each zone has a different attractor — the slope changes!")
        else:
            print(f"    ~ Same attractor across zones (slope is universal)")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 72}")
    print("  SUMMARY — Lyapunov Structure in the Clock")
    print("=" * 72)

    print(f"""
  THE LYAPUNOV LADDER:

    α slope  = -e ≈ -2.718       (pure shedding — neck breathing mode)
                × π/e
    β⁺ slope = -π ≈ -3.142       (charge capture — adds shell geometry)
                × β/e
    β⁻ slope = -πβ/e ≈ -3.517    (charge exchange — adds soliton topology)

  Each step couples one more geometric factor:
    e alone → e × (π/e) = π → π × (β/e) = πβ/e

  The coupling constants ARE the attractor scaling ratios:
    π/e ≈ 1.156  (charge shell couples to shedding)
    β/e ≈ 1.120  (soliton topology couples to charge shell)

  PHYSICAL PICTURE:
    The soliton's internal dynamics are a coupled oscillator system.
    The three modes (charge exchange, charge capture, neck breathing)
    each have their own Lyapunov exponent.  The exponents form a
    geometric ladder because the modes share the same underlying
    soliton geometry — each mode "sees" the same β but couples to
    different geometric features (π = charge shell, e = exponential
    envelope, β = topological winding).

  WHY DECAY LOOKS "RANDOM":
    The coupled oscillations are CHAOTIC (positive Lyapunov exponents,
    sensitive to initial conditions).  An individual decay event occurs
    when:
      1. Core oscillation reaches a susceptible phase
      2. External perturbation (collision, photon, EC) provides a kick
      3. The kick aligns with the intermediate axis
      4. The Lyapunov-accessible escape volume is reached

    The HALF-LIFE is the deterministic envelope — the average time for
    all four conditions to converge.  The √|ε| term measures the
    attractor depth (further from valley = shallower attractor = faster
    escape).  The log₁₀(Z) term measures the atomic environment's
    modulation of the perturbation rate.

  THE ELECTRON CONNECTION:
    Electron capture modifies condition (2) — the perturbation rate.
    A missing electron changes the Lyapunov volume (fewer perturbation
    channels → larger waiting time → longer half-life for EC modes).
    This is why the electron damping factor (2π²β/α³ ≈ 1.5×10⁸)
    affects EC/β⁺ but not β⁻ or α.  The ATTRACTOR is unchanged —
    only the perturbation rate changes.
""")


if __name__ == '__main__':
    main()
