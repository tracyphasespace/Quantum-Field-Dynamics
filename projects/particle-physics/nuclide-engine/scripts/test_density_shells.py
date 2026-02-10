#!/usr/bin/env python3
"""
Density Shell Test — Is the ~36 AMU periodicity fixed or geometric?
====================================================================

Two competing hypotheses for the residual periodicity:

  EMPIRICAL (fixed period):  The ~36 AMU period is constant across A.
    → Suggests an additive, Fourier-type superposition on the valley.
    → Period in ln(A) space would INCREASE with A.

  GEOMETRIC (scaling period): The period scales with A (constant in ln(A)).
    → Suggests discrete density shells with geometric (multiplicative) spacing.
    → Denser cores can accommodate proportionally fewer additional soliton states.
    → The frozen core conjecture: core density ceiling → melting → charge production.

Method: Sliding-window FFT in both A-space and ln(A)-space.
If the dominant period is constant in A-space → empirical/Fourier.
If the dominant period is constant in ln(A)-space → geometric/density shells.

Provenance: EMPIRICAL_FIT (clock residuals) + QFD_DERIVED (ε from valley)
"""

from __future__ import annotations
import math
import os
import sys

import numpy as np
from numpy.linalg import lstsq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_nuclide_topology import (
    ALPHA, BETA, PI, E_NUM,
    z_star, normalize_nubase, load_nubase,
    ELEMENTS, _format_halflife,
)
from isomer_clock_analysis import (
    find_nubase, build_dataframe, fit_clock, CLOCK_VARIANTS,
)


def sliding_fft_A_space(A_arr, resids, window_half=40, step=5):
    """
    Sliding-window FFT in A-space.
    Returns: window_centers, dominant_periods, dominant_amplitudes
    """
    A_min, A_max = int(A_arr.min()), int(A_arr.max())
    centers = list(range(A_min + window_half, A_max - window_half + 1, step))
    dom_periods = []
    dom_amps = []

    for center in centers:
        lo, hi = center - window_half, center + window_half
        mask = (A_arr >= lo) & (A_arr <= hi)
        if np.sum(mask) < 30:
            dom_periods.append(np.nan)
            dom_amps.append(np.nan)
            continue

        # Bin residuals by integer A within window
        binned = np.full(hi - lo + 1, 0.0)
        counts = np.full(hi - lo + 1, 0)
        for a_val, r_val in zip(A_arr[mask], resids[mask]):
            idx = int(a_val) - lo
            if 0 <= idx < len(binned):
                binned[idx] += r_val
                counts[idx] += 1
        has_data = counts > 0
        if np.sum(has_data) < 20:
            dom_periods.append(np.nan)
            dom_amps.append(np.nan)
            continue

        # Average in each bin, interpolate empty
        for j in range(len(binned)):
            if counts[j] > 0:
                binned[j] /= counts[j]
            else:
                # Nearest-neighbor interpolation
                valid_idx = np.where(has_data)[0]
                nearest = valid_idx[np.argmin(np.abs(valid_idx - j))]
                binned[j] = binned[nearest]

        binned -= np.mean(binned)

        # FFT
        fft_mag = np.abs(np.fft.rfft(binned))
        freqs = np.fft.rfftfreq(len(binned))
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)

        # Find dominant period (exclude DC, periods > half-window, and < 4 AMU)
        valid_p = (periods > 4) & (periods < window_half)
        if np.sum(valid_p) == 0:
            dom_periods.append(np.nan)
            dom_amps.append(np.nan)
            continue

        best = np.argmax(fft_mag[valid_p])
        valid_indices = np.where(valid_p)[0]
        dom_periods.append(periods[valid_indices[best]])
        dom_amps.append(fft_mag[valid_indices[best]])

    return np.array(centers), np.array(dom_periods), np.array(dom_amps)


def sliding_fft_lnA_space(A_arr, resids, window_half_lnA=0.4, step_lnA=0.05):
    """
    Sliding-window FFT in ln(A)-space.
    Resamples residuals onto uniform ln(A) grid, then FFT.
    Returns: window_centers_lnA, dominant_periods_lnA, dominant_amplitudes
    """
    lnA = np.log(A_arr.astype(float))
    lnA_min, lnA_max = lnA.min(), lnA.max()

    centers = np.arange(lnA_min + window_half_lnA, lnA_max - window_half_lnA, step_lnA)
    dom_periods = []
    dom_amps = []

    # Uniform ln(A) grid spacing
    dlnA = 0.01  # ~1% spacing

    for center in centers:
        lo, hi = center - window_half_lnA, center + window_half_lnA
        mask = (lnA >= lo) & (lnA <= hi)
        if np.sum(mask) < 20:
            dom_periods.append(np.nan)
            dom_amps.append(np.nan)
            continue

        # Resample onto uniform ln(A) grid
        grid = np.arange(lo, hi, dlnA)
        gridded = np.full(len(grid), 0.0)
        grid_counts = np.full(len(grid), 0)

        for la, rv in zip(lnA[mask], resids[mask]):
            idx = int((la - lo) / dlnA)
            if 0 <= idx < len(grid):
                gridded[idx] += rv
                grid_counts[idx] += 1

        has_data = grid_counts > 0
        if np.sum(has_data) < 15:
            dom_periods.append(np.nan)
            dom_amps.append(np.nan)
            continue

        for j in range(len(gridded)):
            if grid_counts[j] > 0:
                gridded[j] /= grid_counts[j]
            else:
                valid_idx = np.where(has_data)[0]
                if len(valid_idx) == 0:
                    continue
                nearest = valid_idx[np.argmin(np.abs(valid_idx - j))]
                gridded[j] = gridded[nearest]

        gridded -= np.mean(gridded)

        # FFT in ln(A) space
        fft_mag = np.abs(np.fft.rfft(gridded))
        freqs = np.fft.rfftfreq(len(gridded), d=dlnA)
        periods_lnA = np.where(freqs > 0, 1.0 / freqs, np.inf)

        # Find dominant period in ln(A) units
        valid_p = (periods_lnA > 0.05) & (periods_lnA < window_half_lnA)
        if np.sum(valid_p) == 0:
            dom_periods.append(np.nan)
            dom_amps.append(np.nan)
            continue

        best = np.argmax(fft_mag[valid_p])
        valid_indices = np.where(valid_p)[0]
        dom_periods.append(periods_lnA[valid_indices[best]])
        dom_amps.append(fft_mag[valid_indices[best]])

    return centers, np.array(dom_periods), np.array(dom_amps)


def main():
    print("=" * 80)
    print("  DENSITY SHELL TEST")
    print("  Is the ~36 AMU periodicity FIXED (Fourier) or SCALING (geometric)?")
    print("=" * 80)

    # Load data
    nubase_path = find_nubase()
    entries = load_nubase(nubase_path, include_isomers=True)
    data = build_dataframe(entries)

    # Fit V5 clocks (best available)
    print("\n  Fitting V5 clocks...")
    results = {}
    for mode in ['B-', 'B+', 'alpha']:
        r = fit_clock(data, mode, 'V5_corrections')
        if r is not None:
            results[mode] = r
            print(f"    {mode}: R²={r['r2']:.4f}, RMSE={r['rmse']:.3f}, n={r['n']}")

    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: EMPIRICAL — Sliding FFT in A-space
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 1: EMPIRICAL — Sliding FFT in A-space")
    print("  If period is CONSTANT across windows → Fourier superposition")
    print("  If period CHANGES with A → NOT simple Fourier")
    print("=" * 80)

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        r = results.get(mode_key)
        if not r:
            continue

        gs_idx = r['gs_indices']
        A_arr = data['A'][gs_idx].astype(float)
        resids = r['residuals']

        centers, periods, amps = sliding_fft_A_space(A_arr, resids, window_half=40, step=5)
        valid = np.isfinite(periods)

        if np.sum(valid) < 3:
            print(f"\n  {mode_label}: insufficient data for sliding FFT")
            continue

        # Statistics
        mean_p = np.nanmean(periods[valid])
        std_p = np.nanstd(periods[valid])
        cv = std_p / mean_p * 100 if mean_p > 0 else 999

        # Trend: fit period vs center A
        if np.sum(valid) >= 5:
            c_valid = centers[valid]
            p_valid = periods[valid]
            slope, intercept = np.polyfit(c_valid, p_valid, 1)
            # Convert slope to % change per 100 AMU
            pct_per_100 = slope * 100 / mean_p * 100

            print(f"\n  {mode_label} — A-space periods:")
            print(f"    Mean period  = {mean_p:.1f} ± {std_p:.1f} AMU")
            print(f"    CV           = {cv:.1f}%")
            print(f"    Trend        = {slope:+.3f} AMU/AMU ({pct_per_100:+.1f}% per 100 AMU)")
            if abs(cv) < 20:
                print(f"    → CONSISTENT with FIXED period (Fourier)")
            else:
                print(f"    → INCONSISTENT with fixed period")

            # Print window-by-window
            print(f"\n    Window centers and dominant periods:")
            print(f"    {'A_center':>10s} {'Period (AMU)':>14s} {'Amplitude':>10s}")
            print(f"    {'-'*38}")
            for c, p, a in zip(centers[valid], periods[valid], amps[valid]):
                marker = " ←" if abs(p - mean_p) > 2 * std_p else ""
                print(f"    {c:>10.0f} {p:>14.1f} {a:>10.2f}{marker}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: GEOMETRIC — Sliding FFT in ln(A)-space
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 2: GEOMETRIC — Sliding FFT in ln(A)-space")
    print("  If period is CONSTANT in ln(A) → geometric/multiplicative scaling")
    print("  This would mean each density shell holds a FIXED RATIO of mass, not fixed AMU")
    print("=" * 80)

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        r = results.get(mode_key)
        if not r:
            continue

        gs_idx = r['gs_indices']
        A_arr = data['A'][gs_idx].astype(float)
        resids = r['residuals']

        centers, periods, amps = sliding_fft_lnA_space(A_arr, resids,
                                                        window_half_lnA=0.4, step_lnA=0.05)
        valid = np.isfinite(periods)

        if np.sum(valid) < 3:
            print(f"\n  {mode_label}: insufficient data for ln(A) FFT")
            continue

        mean_p = np.nanmean(periods[valid])
        std_p = np.nanstd(periods[valid])
        cv = std_p / mean_p * 100 if mean_p > 0 else 999

        if np.sum(valid) >= 5:
            c_valid = centers[valid]
            p_valid = periods[valid]
            slope, intercept = np.polyfit(c_valid, p_valid, 1)

            # Convert ln(A) period to equivalent A-space period at different A values
            A_50 = math.exp(math.log(50))  # = 50
            A_100 = 100.0
            A_200 = 200.0

            print(f"\n  {mode_label} — ln(A)-space periods:")
            print(f"    Mean period  = {mean_p:.4f} (in ln(A) units)")
            print(f"    CV           = {cv:.1f}%")
            print(f"    Trend        = {slope:+.4f} per ln(A)")

            # Convert to A-space equivalent at representative masses
            print(f"\n    Equivalent A-space periods (if geometric):")
            print(f"      At A=50:   {mean_p * 50:.1f} AMU")
            print(f"      At A=100:  {mean_p * 100:.1f} AMU")
            print(f"      At A=200:  {mean_p * 200:.1f} AMU")

            if abs(cv) < 20:
                print(f"    → CONSISTENT with CONSTANT ln(A) period (geometric scaling)")
            else:
                print(f"    → INCONSISTENT with constant ln(A) period")

            # What ratio does this correspond to?
            ratio = math.exp(mean_p)
            print(f"\n    Shell ratio: e^{mean_p:.4f} = {ratio:.4f}")
            print(f"    Each density shell holds {(ratio-1)*100:.1f}% more mass than the previous")

            # Test against QFD constants
            candidates = [
                (2 * PI * ALPHA, '2πα'),
                (ALPHA * BETA, 'αβ'),
                (1.0 / BETA, '1/β'),
                (PI / (BETA * E_NUM), 'π/(βe)'),
                (2 * ALPHA, '2α'),
                (1.0 / (2 * PI), '1/(2π)'),
                (ALPHA * PI, 'απ'),
                (math.log(2) / BETA, 'ln2/β'),
                (1.0 / E_NUM, '1/e'),
                (BETA / (2 * PI * E_NUM), 'β/(2πe)'),
            ]

            print(f"\n    Algebraic candidates for ln(A) period = {mean_p:.4f}:")
            for val, name in sorted(candidates, key=lambda x: abs(x[0] - mean_p)):
                err = abs(val - mean_p) / mean_p * 100
                if err < 30:
                    print(f"      {name:>12s} = {val:.4f}  ({err:>5.1f}%)")

            # Window-by-window
            print(f"\n    Window centers and dominant periods:")
            print(f"    {'ln(A)':>10s} {'A≈':>6s} {'Period(lnA)':>14s} {'≈AMU':>8s} {'Amplitude':>10s}")
            print(f"    {'-'*52}")
            for c, p, a in zip(centers[valid], periods[valid], amps[valid]):
                A_equiv = math.exp(c)
                amu_equiv = p * A_equiv
                print(f"    {c:>10.3f} {A_equiv:>6.0f} {p:>14.4f} {amu_equiv:>8.1f} {a:>10.3f}")

    # ═══════════════════════════════════════════════════════════════
    # TEST 3: DIRECT COMPARISON — period vs A correlation
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 3: DIRECT — Does period grow with A?")
    print("  Fixed period → slope ≈ 0 in A-space")
    print("  Geometric → period ∝ A (slope > 0 in A-space)")
    print("=" * 80)

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        r = results.get(mode_key)
        if not r:
            continue

        gs_idx = r['gs_indices']
        A_arr = data['A'][gs_idx].astype(float)
        resids = r['residuals']

        centers, periods, amps = sliding_fft_A_space(A_arr, resids, window_half=30, step=3)
        valid = np.isfinite(periods) & (amps > np.nanpercentile(amps[np.isfinite(amps)], 25))

        if np.sum(valid) < 5:
            continue

        c_v = centers[valid]
        p_v = periods[valid]

        # Pearson correlation of period with A
        if len(c_v) >= 5:
            corr = np.corrcoef(c_v, p_v)[0, 1]
            slope, intercept = np.polyfit(c_v, p_v, 1)

            # For geometric: period = k*A, so test ratio period/A
            ratios = p_v / c_v
            mean_ratio = np.mean(ratios)
            cv_ratio = np.std(ratios) / mean_ratio * 100

            print(f"\n  {mode_label}:")
            print(f"    Correlation(period, A)     = {corr:+.3f}")
            print(f"    Slope (dP/dA)              = {slope:+.4f}")
            print(f"    If geometric, P/A ratio    = {mean_ratio:.4f} ± CV {cv_ratio:.1f}%")
            print(f"    Period at A=50 (fit)       = {slope*50 + intercept:.1f} AMU")
            print(f"    Period at A=100 (fit)      = {slope*100 + intercept:.1f} AMU")
            print(f"    Period at A=200 (fit)      = {slope*200 + intercept:.1f} AMU")

            if abs(corr) < 0.3:
                print(f"    → VERDICT: Period DOES NOT correlate with A → FIXED (Fourier)")
            elif corr > 0.5:
                print(f"    → VERDICT: Period GROWS with A → GEOMETRIC (density shells)")
            elif corr < -0.3:
                print(f"    → VERDICT: Period SHRINKS with A → neither model (compression?)")
            else:
                print(f"    → VERDICT: AMBIGUOUS (|r| in 0.3-0.5)")

    # ═══════════════════════════════════════════════════════════════
    # TEST 4: ZERO-CROSSING ANALYSIS — direct period measurement
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  TEST 4: ZERO-CROSSING — Direct period measurement")
    print("  Measure spacing between residual sign changes → local half-period")
    print("  Advantage: no FFT artifacts, directly measures local oscillation")
    print("=" * 80)

    for mode_label, mode_key in [('β⁻', 'B-'), ('β⁺/EC', 'B+'), ('α', 'alpha')]:
        r = results.get(mode_key)
        if not r:
            continue

        gs_idx = r['gs_indices']
        A_arr = data['A'][gs_idx].astype(float)
        resids = r['residuals']

        # Sort by A
        order = np.argsort(A_arr)
        A_sorted = A_arr[order]
        r_sorted = resids[order]

        # Smooth with running mean (window=5)
        smoothed = np.convolve(r_sorted, np.ones(5) / 5, mode='same')

        # Find zero crossings of smoothed residual
        signs = np.sign(smoothed)
        crossings = []
        for j in range(1, len(signs)):
            if signs[j] != signs[j - 1] and signs[j] != 0 and signs[j - 1] != 0:
                # Linear interpolation for crossing point
                A_cross = A_sorted[j - 1] + (A_sorted[j] - A_sorted[j - 1]) * abs(smoothed[j - 1]) / (abs(smoothed[j - 1]) + abs(smoothed[j]))
                crossings.append(A_cross)

        if len(crossings) < 4:
            print(f"\n  {mode_label}: only {len(crossings)} crossings — insufficient")
            continue

        crossings = np.array(crossings)
        spacings = np.diff(crossings)  # half-periods
        full_periods = spacings * 2  # estimated full periods

        # Filter out very short spacings (noise)
        real = full_periods > 4
        if np.sum(real) < 3:
            print(f"\n  {mode_label}: too few real periods")
            continue

        fp = full_periods[real]
        cp = (crossings[:-1][real] + crossings[1:][real]) / 2  # center A of each period

        print(f"\n  {mode_label} — Zero-crossing periods (n={len(fp)}):")
        print(f"    Mean half-period spacing = {np.mean(spacings):.1f} AMU")
        print(f"    Mean full period = {np.mean(fp):.1f} ± {np.std(fp):.1f} AMU")
        print(f"    Median full period = {np.median(fp):.1f} AMU")

        # Correlation of period with A
        if len(fp) >= 5:
            corr_p_A = np.corrcoef(cp, fp)[0, 1]
            print(f"    Correlation(period, A) = {corr_p_A:+.3f}")

            if corr_p_A > 0.3:
                print(f"    → Period GROWS with A (geometric)")
                # Fit: period = k * A
                k_fit = np.mean(fp / cp)
                print(f"    → Geometric ratio P/A = {k_fit:.4f}")
                print(f"    → Shell ratio = e^(P/A) ≈ {math.exp(k_fit):.4f}")
            elif corr_p_A < -0.3:
                print(f"    → Period SHRINKS with A (compression)")
            else:
                print(f"    → Period is FLAT — fixed Fourier")

        # Bin by A range
        A_bins = [(20, 80), (80, 140), (140, 200), (200, 260)]
        print(f"\n    Period by mass range:")
        print(f"    {'A range':>15s} {'Mean period':>14s} {'n':>5s}")
        print(f"    {'-'*38}")
        for lo, hi in A_bins:
            bin_mask = (cp >= lo) & (cp < hi)
            if np.sum(bin_mask) >= 2:
                print(f"    {f'{lo}-{hi}':>15s} {np.mean(fp[bin_mask]):>14.1f} {np.sum(bin_mask):>5d}")

    # ═══════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.patch.set_facecolor('#0A0A1A')
        fig.suptitle('Density Shell Test: Fixed vs Geometric Periodicity',
                      fontsize=14, fontweight='bold', color='white', y=0.98)

        def _style(ax, title):
            ax.set_facecolor('#0A0A1A')
            ax.set_title(title, fontsize=10, fontweight='bold', color='white')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#444444')
            ax.grid(True, alpha=0.1)

        row = 0
        for mode_label, mode_key, color in [('β⁻', 'B-', '#3366CC'),
                                              ('β⁺/EC', 'B+', '#CC3333'),
                                              ('α', 'alpha', '#DDAA00')]:
            r = results.get(mode_key)
            if not r:
                row += 1
                continue

            gs_idx = r['gs_indices']
            A_arr = data['A'][gs_idx].astype(float)
            resids = r['residuals']

            # Left: A-space sliding periods
            ax = axes[row, 0]
            centers_a, periods_a, amps_a = sliding_fft_A_space(
                A_arr, resids, window_half=40, step=3)
            valid = np.isfinite(periods_a)
            if np.sum(valid) > 2:
                ax.scatter(centers_a[valid], periods_a[valid],
                           c=amps_a[valid], cmap='plasma', s=20, alpha=0.7,
                           edgecolors='white', linewidths=0.3)
                # Mean line
                mean_p = np.nanmean(periods_a[valid])
                ax.axhline(mean_p, color=color, linewidth=1.5, linestyle='--',
                           label=f'Mean = {mean_p:.1f} AMU')
                # Trend line
                if np.sum(valid) >= 5:
                    slope, intercept = np.polyfit(centers_a[valid], periods_a[valid], 1)
                    ax.plot(centers_a[valid],
                            slope * centers_a[valid] + intercept,
                            'w-', linewidth=1.0, alpha=0.6,
                            label=f'Trend: {slope:+.3f} AMU/AMU')
                ax.set_xlabel('Window center (A)')
                ax.set_ylabel('Dominant FFT period (AMU)')
                ax.set_ylim(0, 80)
                ax.legend(fontsize=8, facecolor='#1A1A2A', edgecolor='#444444',
                          labelcolor='white')
            _style(ax, f'{mode_label} — Period vs A (constant = Fourier)')

            # Right: ln(A)-space sliding periods
            ax = axes[row, 1]
            centers_ln, periods_ln, amps_ln = sliding_fft_lnA_space(
                A_arr, resids, window_half_lnA=0.4, step_lnA=0.03)
            valid = np.isfinite(periods_ln)
            if np.sum(valid) > 2:
                A_equiv = np.exp(centers_ln[valid])
                ax.scatter(A_equiv, periods_ln[valid],
                           c=amps_ln[valid], cmap='plasma', s=20, alpha=0.7,
                           edgecolors='white', linewidths=0.3)
                mean_p_ln = np.nanmean(periods_ln[valid])
                ax.axhline(mean_p_ln, color=color, linewidth=1.5, linestyle='--',
                           label=f'Mean = {mean_p_ln:.4f} ln(A)')
                ax.set_xlabel('Window center (A)')
                ax.set_ylabel('Dominant FFT period (ln(A) units)')
                ax.legend(fontsize=8, facecolor='#1A1A2A', edgecolor='#444444',
                          labelcolor='white')
            _style(ax, f'{mode_label} — Period in ln(A) space (constant = geometric)')

            row += 1

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(output_dir, 'density_shell_test.png')
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n  Saved: {path}")

        # ── Additional figure: Zero-crossing period vs A ──
        fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
        fig2.patch.set_facecolor('#0A0A1A')

        for i, (mode_label, mode_key, color) in enumerate(
                [('β⁻', 'B-', '#3366CC'), ('β⁺/EC', 'B+', '#CC3333'), ('α', 'alpha', '#DDAA00')]):
            ax = axes2[i]
            r = results.get(mode_key)
            if not r:
                _style(ax, f'{mode_label} — no data')
                continue

            gs_idx = r['gs_indices']
            A_arr = data['A'][gs_idx].astype(float)
            resids = r['residuals']

            # Sort, smooth, find crossings
            order = np.argsort(A_arr)
            A_sorted = A_arr[order]
            r_sorted = resids[order]
            smoothed = np.convolve(r_sorted, np.ones(5) / 5, mode='same')

            signs = np.sign(smoothed)
            crossings = []
            for j in range(1, len(signs)):
                if signs[j] != signs[j - 1] and signs[j] != 0 and signs[j - 1] != 0:
                    A_cross = A_sorted[j - 1] + (A_sorted[j] - A_sorted[j - 1]) * abs(smoothed[j - 1]) / (abs(smoothed[j - 1]) + abs(smoothed[j]))
                    crossings.append(A_cross)

            if len(crossings) >= 4:
                crossings = np.array(crossings)
                spacings = np.diff(crossings) * 2  # full periods
                centers_zc = (crossings[:-1] + crossings[1:]) / 2
                real = spacings > 4
                if np.sum(real) > 2:
                    ax.scatter(centers_zc[real], spacings[real],
                               c=color, s=30, alpha=0.7, edgecolors='white', linewidths=0.3)
                    mean_s = np.mean(spacings[real])
                    ax.axhline(mean_s, color=color, linestyle='--', linewidth=1.5,
                               label=f'Mean = {mean_s:.1f} AMU')
                    # Trend
                    if np.sum(real) >= 5:
                        slope, intercept = np.polyfit(centers_zc[real], spacings[real], 1)
                        ax.plot(centers_zc[real], slope * centers_zc[real] + intercept,
                                'w-', linewidth=1.0, alpha=0.6,
                                label=f'slope={slope:+.3f}')
                    # Geometric reference: period = 0.36 * A
                    A_ref = np.linspace(20, 260, 50)
                    ax.plot(A_ref, 0.36 * A_ref, 'g--', linewidth=0.8, alpha=0.4,
                            label='P=0.36·A (geometric)')
                    ax.set_ylim(0, 100)
                    ax.legend(fontsize=8, facecolor='#1A1A2A', edgecolor='#444444',
                              labelcolor='white')

            ax.set_xlabel('A (center of period)')
            ax.set_ylabel('Full period (AMU)')
            _style(ax, f'{mode_label} — Zero-crossing periods vs A')

        plt.tight_layout()
        path2 = os.path.join(output_dir, 'density_shell_zerocrossing.png')
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
    print("""
  The frozen core conjecture predicts GEOMETRIC scaling:
    - Core density has a ceiling (observational)
    - At ceiling, core melts → produces charge → β⁺/EC
    - Each density shell accommodates a RATIO of mass (not fixed AMU)
    - Period in A-space should GROW with A
    - Period in ln(A)-space should be CONSTANT

  The alternative (Fourier superposition) predicts FIXED period:
    - Valley backbone has harmonic content from rational compression law
    - The periodicity is an additive oscillation on a smooth function
    - Period in A-space should be CONSTANT (~36 AMU everywhere)
    - Period in ln(A)-space would decrease with A
    """)


if __name__ == '__main__':
    main()
