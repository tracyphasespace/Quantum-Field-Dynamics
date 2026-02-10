#!/usr/bin/env python3
"""
Diameter Ceiling Test — Does the neutral core saturate at maximum size?
=======================================================================

The frozen core conjecture predicts:
  1. At each density level, the core has a maximum diameter
  2. When the core hits the diameter ceiling, it must either:
     - Increase to a higher density shell (costs energy)
     - Deform into a peanut (two centers, A > ~150)
     - Shed mass (alpha/fission)
  3. This should show up as:
     - N_max(Z) showing step/plateau structure
     - The neutron drip line flattening at specific A values
     - N/Z ratio at the drip line showing discrete shelves
     - Spacing between shelves matching the geometric shell ratio (~1.288)

Two complementary tests:
  TEST A (Empirical): Map the observed drip line — N_max(Z), band edges,
     and look for plateaus, steps, or saturation in the neutron count.
  TEST B (Geometric): Check if step spacings match the density shell ratio
     of 1.288 (from the alpha ln(A) periodicity test).

Provenance: EMPIRICAL_LOOKUP (drip line from NUBASE2020) + QFD_DERIVED (valley)
"""

from __future__ import annotations
import math
import os
import sys

import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_nuclide_topology import (
    ALPHA, BETA, PI, E_NUM,
    z_star, normalize_nubase, load_nubase,
    ELEMENTS,
)
from isomer_clock_analysis import find_nubase, build_dataframe


def main():
    print("=" * 80)
    print("  DIAMETER CEILING TEST")
    print("  Does the neutral core saturate at a maximum size?")
    print("=" * 80)

    # Load ground states only (we want the full isotope chart, not isomers)
    nubase_path = find_nubase()
    entries = load_nubase(nubase_path, include_isomers=False)
    print(f"\n  Loaded {len(entries)} ground-state nuclides")

    # Organize by Z
    by_Z = defaultdict(list)
    for e in entries:
        by_Z[e['Z']].append(e)

    # Also organize by (Z, A) for lookup
    by_ZA = {}
    for e in entries:
        by_ZA[(e['Z'], e['A'])] = e

    # ═══════════════════════════════════════════════════════════════
    # TEST A: Map the observed drip line — N_max(Z) and N_min(Z)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST A: Neutron Drip Line Structure")
    print("  N_max(Z) = most neutron-rich observed isotope for each element")
    print("=" * 80)

    Z_values = sorted(by_Z.keys())
    N_max = {}  # Z -> max N observed
    N_min = {}  # Z -> min N observed
    A_max = {}  # Z -> max A observed
    N_stable_max = {}  # Z -> max N among stable isotopes
    N_count = {}  # Z -> number of known isotopes

    for Z in Z_values:
        isotopes = by_Z[Z]
        Ns = [e['A'] - e['Z'] for e in isotopes]
        N_max[Z] = max(Ns)
        N_min[Z] = min(Ns)
        A_max[Z] = max(e['A'] for e in isotopes)
        N_count[Z] = len(isotopes)

        stable = [e['A'] - e['Z'] for e in isotopes if e['is_stable']]
        N_stable_max[Z] = max(stable) if stable else None

    # Print N_max progression
    print(f"\n  {'Z':>4s} {'El':>3s} {'N_max':>6s} {'A_max':>6s} {'N/Z':>6s} {'ΔN':>4s} "
          f"{'N_stab':>7s} {'#iso':>5s} {'z*(A)':>6s} {'ε':>6s}")
    print(f"  {'-'*62}")

    prev_N = 0
    Zs_arr = []
    Nmax_arr = []
    NoverZ_arr = []
    delta_N_arr = []

    for Z in Z_values:
        if Z < 1:
            continue
        el = ELEMENTS.get(Z, f"Z{Z}")
        n_max = N_max[Z]
        a_max = A_max[Z]
        nz = n_max / Z if Z > 0 else 0
        delta = n_max - prev_N
        zs = z_star(a_max)
        eps = Z - zs
        ns = N_stable_max.get(Z)
        ns_str = f"{ns:>7d}" if ns is not None else f"{'—':>7s}"

        Zs_arr.append(Z)
        Nmax_arr.append(n_max)
        NoverZ_arr.append(nz)
        delta_N_arr.append(delta)

        # Only print every 5th element for readability, plus notable ones
        if Z % 5 == 0 or Z <= 10 or delta <= 0 or Z in [20, 28, 50, 82]:
            marker = ""
            if delta <= 0:
                marker = " ← PLATEAU"
            elif delta >= 4:
                marker = " ← JUMP"
            print(f"  {Z:>4d} {el:>3s} {n_max:>6d} {a_max:>6d} {nz:>6.2f} {delta:>+4d} "
                  f"{ns_str} {N_count[Z]:>5d} {zs:>6.1f} {eps:>+6.1f}{marker}")
        prev_N = n_max

    Zs_arr = np.array(Zs_arr)
    Nmax_arr = np.array(Nmax_arr)
    NoverZ_arr = np.array(NoverZ_arr)
    delta_N_arr = np.array(delta_N_arr)

    # ═══════════════════════════════════════════════════════════════
    # TEST A2: N/Z ratio at the drip line — does it saturate?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST A2: N/Z Ratio at the Neutron Drip Line")
    print("  If diameter ceiling exists, N/Z should plateau or step")
    print("=" * 80)

    # Fit piecewise: is N/Z flattening above some Z?
    # Test: N/Z = a + b*Z for Z < Z_break vs N/Z = c for Z > Z_break
    for Z_break in [20, 28, 40, 50, 60, 82]:
        mask_above = Zs_arr >= Z_break
        mask_below = Zs_arr < Z_break
        if np.sum(mask_above) < 5 or np.sum(mask_below) < 5:
            continue

        # Slope above break
        if np.sum(mask_above) >= 5:
            slope_above, intercept_above = np.polyfit(Zs_arr[mask_above], NoverZ_arr[mask_above], 1)
            # Slope below
            slope_below, intercept_below = np.polyfit(Zs_arr[mask_below], NoverZ_arr[mask_below], 1)

            print(f"  Break at Z={Z_break}: slope_below={slope_below:+.4f}/Z, "
                  f"slope_above={slope_above:+.4f}/Z  "
                  f"(ratio={slope_above/slope_below:.2f})" if slope_below != 0 else "")

    # Overall trend
    valid = Zs_arr >= 8  # skip very light
    slope, intercept = np.polyfit(Zs_arr[valid], NoverZ_arr[valid], 1)
    print(f"\n  Overall N_max/Z trend (Z>=8): slope = {slope:+.5f} per Z")
    print(f"  At Z=20: N/Z = {slope*20+intercept:.2f}")
    print(f"  At Z=50: N/Z = {slope*50+intercept:.2f}")
    print(f"  At Z=82: N/Z = {slope*82+intercept:.2f}")
    print(f"  At Z=110: N/Z = {slope*110+intercept:.2f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST A3: ΔN plateaus — where does adding Z NOT add N?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST A3: ΔN_max Plateaus")
    print("  Plateaus (ΔN ≤ 0) = diameter ceiling events")
    print("  Jumps (ΔN ≥ 3) = new density shell opening")
    print("=" * 80)

    plateaus = []
    jumps = []
    for i, Z in enumerate(Zs_arr):
        if i == 0:
            continue
        dN = delta_N_arr[i]
        if dN <= 0:
            plateaus.append((Z, N_max.get(int(Z), 0), dN))
        elif dN >= 3:
            jumps.append((Z, N_max.get(int(Z), 0), dN))

    print(f"\n  Plateaus (ΔN_max ≤ 0): {len(plateaus)} events")
    if plateaus:
        print(f"  {'Z':>4s} {'El':>3s} {'N_max':>6s} {'ΔN':>4s} {'A':>6s}")
        print(f"  {'-'*28}")
        for Z, nm, dn in plateaus:
            el = ELEMENTS.get(int(Z), '?')
            print(f"  {int(Z):>4d} {el:>3s} {nm:>6d} {dn:>+4d} {int(Z)+nm:>6d}")

    print(f"\n  Jumps (ΔN_max ≥ 3): {len(jumps)} events")
    if jumps:
        print(f"  {'Z':>4s} {'El':>3s} {'N_max':>6s} {'ΔN':>4s} {'A':>6s}")
        print(f"  {'-'*28}")
        for Z, nm, dn in jumps:
            el = ELEMENTS.get(int(Z), '?')
            print(f"  {int(Z):>4d} {el:>3s} {nm:>6d} {dn:>+4d} {int(Z)+nm:>6d}")

    # ═══════════════════════════════════════════════════════════════
    # TEST A4: Band width vs A — does it narrow in steps?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST A4: Survival Band Width — Step Structure")
    print("  Band width = N_max - N_min for each Z")
    print("  Steps in width = density shell transitions")
    print("=" * 80)

    band_width = []
    band_Z = []
    for Z in Z_values:
        if Z < 3:
            continue
        w = N_max[Z] - N_min[Z]
        band_width.append(w)
        band_Z.append(Z)

    band_width = np.array(band_width)
    band_Z = np.array(band_Z)

    # Running difference of band width
    dwidth = np.diff(band_width)

    # Find step events: large changes in band width
    print(f"\n  Large band width changes (|ΔW| ≥ 3):")
    print(f"  {'Z':>4s} {'El':>3s} {'Width':>6s} {'ΔWidth':>7s} {'A_max':>6s}")
    print(f"  {'-'*32}")
    for i in range(len(dwidth)):
        if abs(dwidth[i]) >= 3:
            Z = band_Z[i+1]
            el = ELEMENTS.get(int(Z), '?')
            w = band_width[i+1]
            a = A_max.get(int(Z), 0)
            print(f"  {int(Z):>4d} {el:>3s} {w:>6d} {int(dwidth[i]):>+7d} {a:>6d}")

    # ═══════════════════════════════════════════════════════════════
    # TEST B: Geometric spacing of plateaus and jumps
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST B: Geometric Shell Spacing")
    print("  If density shells are geometric (ratio 1.288), then plateau A values")
    print("  should be at geometric series: A₀, 1.288·A₀, 1.288²·A₀, ...")
    print("=" * 80)

    # Collect A values at plateaus and jumps
    if plateaus:
        plateau_A = np.array([int(Z) + nm for Z, nm, dn in plateaus], dtype=float)
        print(f"\n  Plateau A values: {plateau_A.astype(int)}")

        if len(plateau_A) >= 3:
            # Test: are ratios between consecutive plateau A values constant?
            ratios = plateau_A[1:] / plateau_A[:-1]
            print(f"  Consecutive ratios: {np.array2string(ratios, precision=3)}")
            print(f"  Mean ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
            print(f"  Expected (density shell): 1.288")
            print(f"  CV of ratios: {np.std(ratios)/np.mean(ratios)*100:.1f}%")

            # ln(A) spacing
            lnA_plateau = np.log(plateau_A)
            lnA_spacing = np.diff(lnA_plateau)
            print(f"\n  ln(A) spacings: {np.array2string(lnA_spacing, precision=4)}")
            print(f"  Mean ln(A) spacing: {np.mean(lnA_spacing):.4f}")
            print(f"  Expected (from alpha shells): 0.2534")

    if jumps:
        jump_A = np.array([int(Z) + nm for Z, nm, dn in jumps], dtype=float)
        print(f"\n  Jump A values: {jump_A.astype(int)}")

        if len(jump_A) >= 3:
            ratios = jump_A[1:] / jump_A[:-1]
            print(f"  Consecutive ratios: {np.array2string(ratios, precision=3)}")
            print(f"  Mean ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST B2: N_max derivative — looking for discrete density levels
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST B2: dN_max/dZ — Discrete Density Levels")
    print("  Smooth: dN/dZ grows monotonically")
    print("  Density shells: dN/dZ has plateaus (constant within each shell)")
    print("=" * 80)

    # Smooth N_max with running mean (window 5)
    smoothed_N = np.convolve(Nmax_arr.astype(float), np.ones(5)/5, mode='same')
    dN_dZ = np.diff(smoothed_N) / np.diff(Zs_arr.astype(float))

    # Bin dN/dZ by Z range
    Z_mid = (Zs_arr[:-1] + Zs_arr[1:]) / 2.0

    bins = [(3, 10), (10, 20), (20, 28), (28, 40), (40, 50), (50, 60),
            (60, 70), (70, 82), (82, 92), (92, 105), (105, 120)]

    print(f"\n  {'Z range':>12s} {'mean dN/dZ':>11s} {'std':>6s} {'n':>4s} {'N/Z at end':>11s}")
    print(f"  {'-'*50}")

    dNdZ_values = []
    for lo, hi in bins:
        mask = (Z_mid >= lo) & (Z_mid < hi)
        if np.sum(mask) < 2:
            continue
        mean_dndz = np.mean(dN_dZ[mask])
        std_dndz = np.std(dN_dZ[mask])

        # N/Z at the end of this range
        z_end_mask = Zs_arr == min(hi, Zs_arr.max())
        if np.sum(z_end_mask) > 0:
            nz_end = NoverZ_arr[z_end_mask][0]
        else:
            nz_end = 0

        dNdZ_values.append((lo, hi, mean_dndz))
        print(f"  {f'{lo}-{hi}':>12s} {mean_dndz:>+11.3f} {std_dndz:>6.3f} {np.sum(mask):>4d} {nz_end:>11.2f}")

    # Check if dN/dZ is stepping or smooth
    if len(dNdZ_values) >= 3:
        means = [v[2] for v in dNdZ_values]
        diffs = np.diff(means)
        print(f"\n  Changes in dN/dZ between bins:")
        for i, ((lo1, hi1, m1), (lo2, hi2, m2)) in enumerate(zip(dNdZ_values[:-1], dNdZ_values[1:])):
            print(f"    {lo1}-{hi1} → {lo2}-{hi2}: Δ(dN/dZ) = {m2-m1:+.3f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST C: Core size proxy — N_excess = N - Z at drip line
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST C: Core Size Proxy — N_excess = N_max - Z")
    print("  N_excess ≈ number of neutral core winding states")
    print("  Diameter ceiling → N_excess should show saturation / steps")
    print("=" * 80)

    N_excess = Nmax_arr - Zs_arr

    # Does N_excess grow linearly, or does it flatten?
    # Test: fit N_excess = a·Z + b for different ranges
    ranges = [(3, 30), (30, 60), (60, 90), (90, 120)]
    print(f"\n  {'Z range':>12s} {'slope':>8s} {'intercept':>10s} {'R²':>6s}")
    print(f"  {'-'*40}")

    for lo, hi in ranges:
        mask = (Zs_arr >= lo) & (Zs_arr < hi)
        if np.sum(mask) < 5:
            continue
        z_r = Zs_arr[mask].astype(float)
        ne_r = N_excess[mask].astype(float)
        slope, intercept = np.polyfit(z_r, ne_r, 1)
        pred = slope * z_r + intercept
        ss_res = np.sum((ne_r - pred)**2)
        ss_tot = np.sum((ne_r - np.mean(ne_r))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  {f'{lo}-{hi}':>12s} {slope:>+8.3f} {intercept:>+10.2f} {r2:>6.3f}")

    # Key question: does the slope decrease? (flattening = diameter ceiling)
    slopes = []
    for lo, hi in ranges:
        mask = (Zs_arr >= lo) & (Zs_arr < hi)
        if np.sum(mask) < 5:
            continue
        slope, _ = np.polyfit(Zs_arr[mask].astype(float), N_excess[mask].astype(float), 1)
        slopes.append((lo, hi, slope))

    if len(slopes) >= 2:
        print(f"\n  Slope trend:")
        for (lo, hi, s) in slopes:
            print(f"    Z={lo}-{hi}: dN_excess/dZ = {s:+.3f}")

        # Is the slope decreasing? (diameter ceiling would cause this)
        slope_values = [s[2] for s in slopes]
        if all(slope_values[i] >= slope_values[i+1] for i in range(len(slope_values)-1)):
            print(f"  → Slope MONOTONICALLY DECREASING → consistent with diameter ceiling")
        else:
            print(f"  → Slope non-monotonic")

    # ═══════════════════════════════════════════════════════════════
    # TEST D: A_max vs Z — what is the heaviest isotope per element?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST D: A_max vs Z — Envelope of the nuclear chart")
    print("  The outer edge of the chart is the effective drip line")
    print("  If diameter ceiling exists, A_max should show step structure")
    print("=" * 80)

    Amax_arr = np.array([A_max.get(int(Z), 0) for Z in Zs_arr])

    # Fit A_max = a·Z + b (linear expectation) and look at residuals
    valid = Zs_arr >= 8
    slope_A, intercept_A = np.polyfit(Zs_arr[valid].astype(float), Amax_arr[valid].astype(float), 1)
    pred_A = slope_A * Zs_arr[valid] + intercept_A
    resid_A = Amax_arr[valid] - pred_A

    print(f"\n  Linear fit A_max = {slope_A:.3f}·Z + {intercept_A:.1f}")
    print(f"  Residual std = {np.std(resid_A):.1f} AMU")

    # Find step features in A_max residuals
    # Smooth and look for jumps
    smoothed_resid = np.convolve(resid_A, np.ones(5)/5, mode='same')
    d_resid = np.diff(smoothed_resid)

    print(f"\n  Step features in A_max (|Δresidual| > 3 AMU):")
    print(f"  {'Z':>4s} {'El':>3s} {'A_max':>6s} {'Residual':>9s} {'ΔResid':>7s}")
    print(f"  {'-'*35}")
    Z_valid = Zs_arr[valid]
    for i in range(len(d_resid)):
        if abs(d_resid[i]) > 3:
            Z_here = Z_valid[i+1]
            el = ELEMENTS.get(int(Z_here), '?')
            print(f"  {int(Z_here):>4d} {el:>3s} {Amax_arr[valid][i+1]:>6d} "
                  f"{smoothed_resid[i+1]:>+9.1f} {d_resid[i]:>+7.1f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST E: ln(A_max) spacing — geometric steps?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST E: Geometric Steps in A_max")
    print("  If density shells set the drip line, steps in A_max should")
    print("  be at geometric ratios ≈ 1.288")
    print("=" * 80)

    # Find major steps in A_max (ΔA ≥ 5 above linear trend)
    step_Z = []
    step_A = []
    for i in range(1, len(Amax_arr)):
        if Amax_arr[i] - Amax_arr[i-1] >= 5:
            step_Z.append(Zs_arr[i])
            step_A.append(Amax_arr[i])

    if len(step_A) >= 3:
        step_A = np.array(step_A, dtype=float)
        ratios = step_A[1:] / step_A[:-1]
        ln_spacings = np.diff(np.log(step_A))

        print(f"\n  Major steps (ΔA_max ≥ 5):")
        print(f"  {'Z':>4s} {'A_max':>6s} {'Ratio':>7s} {'ln(A) spacing':>14s}")
        print(f"  {'-'*35}")
        for i, (Z, A) in enumerate(zip(step_Z, step_A)):
            r_str = f"{ratios[i-1]:.3f}" if i > 0 else "—"
            ln_str = f"{ln_spacings[i-1]:.4f}" if i > 0 else "—"
            print(f"  {int(Z):>4d} {int(A):>6d} {r_str:>7s} {ln_str:>14s}")

        print(f"\n  Mean ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
        print(f"  Expected (density shell): 1.288")
        print(f"  Mean ln(A) spacing: {np.mean(ln_spacings):.4f}")
        print(f"  Expected: 0.2534")

    # ═══════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        output_dir = os.path.dirname(os.path.abspath(__file__))

        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.patch.set_facecolor('#0A0A1A')
        fig.suptitle('Diameter Ceiling Test — Neutral Core Saturation',
                      fontsize=14, fontweight='bold', color='white', y=0.98)

        def _style(ax, title):
            ax.set_facecolor('#0A0A1A')
            ax.set_title(title, fontsize=10, fontweight='bold', color='white')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#444444')
            ax.grid(True, alpha=0.15)

        # 1: N_max vs Z
        ax = axes[0, 0]
        ax.scatter(Zs_arr, Nmax_arr, s=8, c='cyan', alpha=0.7, edgecolors='none')
        # Valley line
        A_line = np.arange(1, 350)
        Z_line = np.array([z_star(A) for A in A_line])
        N_line = A_line - Z_line
        ax.plot(Z_line, N_line, 'w--', linewidth=0.8, alpha=0.5, label='Valley N(Z)')
        # Mark plateaus
        for Z, nm, dn in plateaus:
            ax.scatter([Z], [nm], c='red', s=40, zorder=5, marker='v', edgecolors='white', linewidths=0.5)
        ax.set_xlabel('Proton number Z')
        ax.set_ylabel('N_max (most neutron-rich)')
        ax.legend(fontsize=8, facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')
        _style(ax, 'N_max(Z) — Neutron Drip Line\nRed triangles = plateaus (ΔN ≤ 0)')

        # 2: N/Z ratio at drip line
        ax = axes[0, 1]
        valid_nz = Zs_arr >= 3
        ax.scatter(Zs_arr[valid_nz], NoverZ_arr[valid_nz], s=8, c='lime', alpha=0.7, edgecolors='none')
        # Trend line
        z_fit = np.linspace(3, 120, 100)
        ax.plot(z_fit, slope * z_fit + intercept, 'w--', linewidth=1, alpha=0.5, label=f'Linear fit')
        # Mark the three transitions
        for a_trans, label, color in [(123, 'A≈123\n(surf→dens)', '#FF6600'),
                                       (150, 'A≈150\n(peanut)', '#FF0066'),
                                       (318, 'A≈318\n(drip)', '#FF00FF')]:
            # Find Z at this A on the valley
            z_at = z_star(a_trans)
            ax.axvline(z_at, color=color, linewidth=1, linestyle=':', alpha=0.7)
            ax.text(z_at, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 2.5, label,
                    color=color, fontsize=7, ha='center', va='bottom')
        ax.set_xlabel('Proton number Z')
        ax.set_ylabel('N_max / Z')
        ax.legend(fontsize=8, facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')
        _style(ax, 'N/Z Ratio at Drip Line\nShould plateau if diameter ceiling exists')

        # 3: ΔN_max vs Z
        ax = axes[0, 2]
        ax.bar(Zs_arr[1:], delta_N_arr[1:], width=0.8, color='cyan', alpha=0.5, edgecolor='none')
        ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
        ax.axhline(1, color='yellow', linewidth=0.5, alpha=0.3, linestyle='--')
        # Highlight plateaus
        for Z, nm, dn in plateaus:
            ax.bar([Z], [dn], width=0.8, color='red', alpha=0.8, edgecolor='none')
        ax.set_xlabel('Proton number Z')
        ax.set_ylabel('ΔN_max (change per Z)')
        ax.set_ylim(-5, 10)
        _style(ax, 'ΔN_max per element\nRed = plateau events (ΔN ≤ 0)')

        # 4: Band width vs Z
        ax = axes[1, 0]
        ax.scatter(band_Z, band_width, s=8, c='gold', alpha=0.7, edgecolors='none')
        # sqrt(Z) envelope
        z_env = np.linspace(3, 120, 100)
        ax.plot(z_env, 2.5 * np.sqrt(z_env), 'w--', linewidth=1, alpha=0.4, label='~2.5√Z')
        ax.set_xlabel('Proton number Z')
        ax.set_ylabel('Band width (N_max - N_min)')
        ax.legend(fontsize=8, facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')
        _style(ax, 'Survival Band Width\nSteps = density shell boundaries')

        # 5: N_excess = N_max - Z
        ax = axes[1, 1]
        ax.scatter(Zs_arr, N_excess, s=8, c='#FF6600', alpha=0.7, edgecolors='none')
        # Piecewise fits
        colors_fit = ['#3366CC', '#33CC33', '#CC3333', '#DDAA00']
        for j, (lo, hi) in enumerate(ranges):
            mask = (Zs_arr >= lo) & (Zs_arr < hi)
            if np.sum(mask) < 5:
                continue
            z_r = Zs_arr[mask].astype(float)
            ne_r = N_excess[mask].astype(float)
            s, inter = np.polyfit(z_r, ne_r, 1)
            ax.plot(z_r, s * z_r + inter, color=colors_fit[j % len(colors_fit)],
                    linewidth=2, alpha=0.7, label=f'Z={lo}-{hi}: slope={s:.2f}')
        ax.set_xlabel('Proton number Z')
        ax.set_ylabel('N_excess = N_max - Z')
        ax.legend(fontsize=7, facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')
        _style(ax, 'Core Size Proxy (N_max - Z)\nSlope should decrease if diameter ceiling')

        # 6: A_max residuals
        ax = axes[1, 2]
        ax.scatter(Z_valid, resid_A, s=8, c='magenta', alpha=0.6, edgecolors='none')
        ax.plot(Z_valid, smoothed_resid, 'w-', linewidth=1.5, alpha=0.7, label='Smoothed')
        ax.axhline(0, color='yellow', linewidth=0.5, alpha=0.3, linestyle='--')
        ax.set_xlabel('Proton number Z')
        ax.set_ylabel('A_max residual (vs linear)')
        ax.legend(fontsize=8, facecolor='#1A1A2A', edgecolor='#444444', labelcolor='white')
        _style(ax, 'A_max Detrended — Step Structure\nJumps = new shell opening')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(output_dir, 'diameter_ceiling_test.png')
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  Saved: {path}")

        # ── Additional: N_max vs ln(A_max) to look for geometric structure ──
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
        fig2.patch.set_facecolor('#0A0A1A')

        valid_a = Amax_arr > 0
        ln_Amax = np.log(Amax_arr[valid_a].astype(float))

        ax2.scatter(ln_Amax, Nmax_arr[valid_a], s=12, c='cyan', alpha=0.7, edgecolors='none')

        # Mark geometric shell boundaries (from alpha test: period = 0.2534 in ln(A))
        ln_A0 = np.log(20.0)  # starting point
        shell = ln_A0
        while shell < 6.0:
            A_shell = math.exp(shell)
            ax2.axvline(shell, color='gold', linewidth=0.8, linestyle=':', alpha=0.4)
            ax2.text(shell, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 170,
                     f'A≈{A_shell:.0f}', color='gold', fontsize=7, rotation=90,
                     ha='right', va='top')
            shell += 0.2534

        ax2.set_xlabel('ln(A_max)', color='white')
        ax2.set_ylabel('N_max', color='white')
        _style(ax2, 'N_max vs ln(A_max) with Geometric Shell Boundaries\n'
                     'Gold lines at spacing 0.2534 (from alpha density shell test)')

        path2 = os.path.join(output_dir, 'diameter_ceiling_geometric.png')
        fig2.savefig(path2, dpi=150, facecolor=fig2.get_facecolor())
        plt.close(fig2)
        print(f"  Saved: {path2}")

    except ImportError:
        print("  matplotlib not available — skipping plots")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print("=" * 80)
    print(f"""
  Diameter ceiling predictions:
    1. N/Z at drip line should plateau → TEST ABOVE
    2. ΔN_max should show plateau events (ΔN ≤ 0) → {len(plateaus)} found
    3. N_excess slope should decrease with Z → TEST ABOVE
    4. Steps should be at geometric spacing (ratio ~1.288)
    5. Band width should narrow in steps, not smoothly

  Three transitions predicted:
    A ≈ 123 (Z ≈ {z_star(123):.0f}): surface → density crossover
    A ≈ 150 (Z ≈ {z_star(150):.0f}): diameter ceiling → peanut onset
    A ≈ 318 (Z ≈ {z_star(318):.0f}): total drip line (band → 0)
    """)


if __name__ == '__main__':
    main()
