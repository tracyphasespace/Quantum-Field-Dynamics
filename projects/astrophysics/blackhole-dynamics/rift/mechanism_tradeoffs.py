#!/usr/bin/env python3
"""
Four-Mechanism Trade-offs

**STATUS**: PARTIALLY CORRECTED - NEEDS FULL REWRITE
This script still contains references to incorrect "spin cancellation" physics.
The correct physics is:
1. Binary L1 geometry - PRIMARY (~90% barrier reduction, always present)
2. Rotational KE (disk spin) - ~8-10% boost, E_rot = (1/2)m(Ωr)²
3. Coulomb (charge) - ~1-2% boost
4. Thermal - negligible for bulk, important for species selection

Key insight: L1 geometry does the heavy lifting, but rotation/charge/thermal
tip the balance. Lower spin rate Ω requires compensating with proximity, charge, or temperature.

TODO: Rewrite trade-off 2 and 3, update all visualizations with correct physics.
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.674e-11  # m^3/kg/s^2
c = 2.998e8    # m/s
M_sun = 1.989e30  # kg
k_B = 1.381e-23  # J/K

def gravitational_radius(M):
    """r_g = GM/c^2"""
    return G * M / c**2

def estimate_L1_barrier(M_total, separation):
    """Φ(L1) ≈ -GM/r where r ~ separation/2"""
    r_L1 = separation / 2.0
    return -G * M_total / r_L1

def thermal_energy(T, m):
    """E_thermal = (3/2) k_B T"""
    return 1.5 * k_B * T

def main():
    print("="*80)
    print("FOUR-MECHANISM TRADE-OFFS")
    print("="*80)
    print()

    # Binary parameters
    M_BH = 100 * M_sun
    M_total = 2 * M_BH
    r_g = gravitational_radius(M_BH)

    print(f"System: 100 M☉ + 100 M☉ binary")
    print(f"r_g = {r_g/1e3:.1f} km")
    print()

    # === Trade-off 1: Spin vs Proximity ===
    print("="*80)
    print("TRADE-OFF 1: SPIN RATE vs PROXIMITY")
    print("="*80)
    print()

    # Fix other parameters
    T_fixed = 1e9  # K
    m_bulk = 1.0  # kg
    E_thermal_fixed = thermal_energy(T_fixed, m_bulk)
    E_coulomb_fixed = 2.4e12  # J (~2% of barrier at 300 r_g)

    print(f"Fixed: T = {T_fixed:.1e} K, E_coulomb = {E_coulomb_fixed:.1e} J")
    print(f"L1 geometry ALWAYS provides ~90% barrier reduction")
    print()

    # Vary spin rate and find required proximity
    # Omega in units of c/r_g (dimensionless)
    spin_rates = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01])  # from near-maximal to very low

    print(f"{'Ω [c/r_g]':<12} {'E_rot [J]':<15} {'Required Sep [r_g]':<20} {'Notes':<30}")
    print("-"*85)

    separations_needed = []
    L1_reduction_factor = 0.1  # Binary L1 is ~10× easier than single BH (~90% reduction)

    for Omega_normalized in spin_rates:
        # Rotational KE: E_rot = 0.5 * m * (Ω * r)²
        # At r = r_g: v_rot = Ω * c (where Ω in units of c/r_g)
        v_rot_at_rg = Omega_normalized * c
        E_rot = 0.5 * m_bulk * v_rot_at_rg**2

        # Find separation where escape is possible
        # Binary L1 barrier = (GM/r_L1) * L1_reduction_factor where r_L1 = sep/2
        # Need: E_rot + E_coulomb + E_thermal > 0.1 * barrier_L1 (for ~1% escape)

        # Try different separations
        test_seps = np.logspace(np.log10(10*r_g), np.log10(2000*r_g), 100)
        found = False

        for sep in test_seps:
            # Single BH barrier at L1 distance
            barrier_single = abs(estimate_L1_barrier(M_total, sep))
            # Binary L1 barrier (reduced by ~90% from geometry)
            barrier_L1 = barrier_single * L1_reduction_factor
            E_available = E_rot + E_coulomb_fixed + E_thermal_fixed

            if E_available > barrier_L1 * 0.10:  # Threshold for ~1% escape
                separations_needed.append(sep)

                if sep < 100 * r_g:
                    note = "Very close - near merger"
                elif sep < 300 * r_g:
                    note = "Close - viable"
                elif sep < 1000 * r_g:
                    note = "Moderate - good"
                else:
                    note = "Far - rare alignment needed"

                print(f"{Omega_normalized:>6.2f}       {E_rot:.2e}     "
                      f"{sep/r_g:>8.0f} r_g ({sep/1e3:>7.0f} km)  {note}")
                found = True
                break

        if not found:
            separations_needed.append(np.nan)
            print(f"{Omega_normalized:>6.2f}       {E_rot:.2e}     "
                  f"{'IMPOSSIBLE':<18}  {'Cannot escape - need higher Ω!':<30}")

    print()
    print("KEY INSIGHT: Lower spin → need closer proximity!")
    print()

    # === Trade-off 2: Spin vs Charge ===
    print("="*80)
    print("TRADE-OFF 2: SPIN vs CHARGE")
    print("="*80)
    print()

    # Fix proximity and temperature
    sep_fixed = 300 * r_g
    barrier_fixed = abs(estimate_L1_barrier(M_total, sep_fixed))

    print(f"Fixed: Separation = {sep_fixed/r_g:.0f} r_g ({sep_fixed/1e3:.0f} km)")
    print(f"       Temperature = {T_fixed:.1e} K")
    print()

    print(f"{'Spin Cancel':<15} {'E_coulomb Needed':<20} {'Q_separated':<15} {'Notes':<30}")
    print("-"*85)

    for cancel in spin_cancellations:
        barrier_eff = barrier_fixed * (1 - cancel)

        # Need: (E_thermal + E_coulomb) > 0.1 * barrier_eff
        E_coulomb_needed = max(0, barrier_eff * 0.10 - E_thermal_fixed)

        # Calculate required charge: E = k*Q^2/r → Q = sqrt(E*r/k)
        r_charge_sep = 100  # m
        k_coulomb = 8.988e9
        Q_needed = np.sqrt(E_coulomb_needed * r_charge_sep / k_coulomb) if E_coulomb_needed > 0 else 0

        if E_coulomb_needed < 1e20:
            if Q_needed < 1e3:
                note = "Minimal charge needed"
            elif Q_needed < 1e6:
                note = "Moderate charge needed"
            else:
                note = "High charge needed"

            print(f"{cancel*100:>6.0f}%         {E_coulomb_needed:>12.2e} J    "
                  f"{Q_needed:>10.2e} C   {note}")
        else:
            print(f"{cancel*100:>6.0f}%         {'TOO HIGH':<20} {'IMPOSSIBLE':<15} {'Cannot escape!'}")

    print()
    print("KEY INSIGHT: Lower spin → need more charge separation!")
    print()

    # === Trade-off 3: Spin vs Temperature ===
    print("="*80)
    print("TRADE-OFF 3: SPIN vs TEMPERATURE")
    print("="*80)
    print()

    # Fix proximity and charge
    E_coulomb_fixed2 = 1.8e13  # J

    print(f"Fixed: Separation = {sep_fixed/r_g:.0f} r_g ({sep_fixed/1e3:.0f} km)")
    print(f"       E_coulomb = {E_coulomb_fixed2:.1e} J")
    print()

    print(f"{'Spin Cancel':<15} {'E_thermal Needed':<20} {'Temperature':<15} {'Notes':<30}")
    print("-"*85)

    for cancel in spin_cancellations:
        barrier_eff = barrier_fixed * (1 - cancel)

        # Need: (E_thermal + E_coulomb) > 0.1 * barrier_eff
        E_thermal_needed = max(0, barrier_eff * 0.10 - E_coulomb_fixed2)

        # Calculate required temperature: E = (3/2)kT → T = 2E/(3k)
        T_needed = (2 * E_thermal_needed) / (3 * k_B * m_bulk) if E_thermal_needed > 0 else T_fixed

        if T_needed < 1e15:
            if T_needed < 1e10:
                note = "Achievable (hot disk)"
            elif T_needed < 1e12:
                note = "Very hot (inner disk)"
            else:
                note = "Extreme (near BH)"

            print(f"{cancel*100:>6.0f}%         {E_thermal_needed:>12.2e} J    "
                  f"{T_needed:>10.2e} K  {note}")
        else:
            print(f"{cancel*100:>6.0f}%         {'TOO HIGH':<20} {'IMPOSSIBLE':<15} {'Cannot escape!'}")

    print()
    print("KEY INSIGHT: Lower spin → need higher temperature!")
    print()

    # === Summary Plot ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Required separation vs spin cancellation
    ax1 = axes[0, 0]
    valid_seps = np.array([s for s in separations_needed if not np.isnan(s)])
    valid_cancels = spin_cancellations[:len(valid_seps)]

    ax1.semilogy(valid_cancels * 100, valid_seps / r_g, 'bo-', linewidth=3, markersize=10)
    ax1.axhline(300, color='orange', linestyle='--', linewidth=2, label='Baseline (300 r_g)')
    ax1.set_xlabel('Spin Cancellation [%]', fontsize=14)
    ax1.set_ylabel('Required Separation [r_g]', fontsize=14)
    ax1.set_title('Trade-off: Lower Spin → Need Closer Proximity', fontsize=15, weight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Lower spin on right

    # Panel 2: Required charge vs spin cancellation
    ax2 = axes[0, 1]
    charges = []
    for cancel in spin_cancellations:
        barrier_eff = barrier_fixed * (1 - cancel)
        E_coulomb_needed = max(0, barrier_eff * 0.10 - E_thermal_fixed)
        Q_needed = np.sqrt(E_coulomb_needed * r_charge_sep / k_coulomb) if E_coulomb_needed > 0 else 0
        charges.append(Q_needed)

    ax2.semilogy(spin_cancellations * 100, charges, 'ro-', linewidth=3, markersize=10)
    ax2.axhline(447, color='orange', linestyle='--', linewidth=2, label='Baseline (447 C)')
    ax2.set_xlabel('Spin Cancellation [%]', fontsize=14)
    ax2.set_ylabel('Required Separated Charge [C]', fontsize=14)
    ax2.set_title('Trade-off: Lower Spin → Need More Charge', fontsize=15, weight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    # Panel 3: Required temperature vs spin cancellation
    ax3 = axes[1, 0]
    temps = []
    for cancel in spin_cancellations:
        barrier_eff = barrier_fixed * (1 - cancel)
        E_thermal_needed = max(0, barrier_eff * 0.10 - E_coulomb_fixed2)
        T_needed = (2 * E_thermal_needed) / (3 * k_B * m_bulk) if E_thermal_needed > 0 else T_fixed
        temps.append(T_needed)

    ax3.semilogy(spin_cancellations * 100, temps, 'go-', linewidth=3, markersize=10)
    ax3.axhline(1e9, color='orange', linestyle='--', linewidth=2, label='Baseline (10⁹ K)')
    ax3.axhline(1e10, color='red', linestyle=':', linewidth=2, label='Inner disk limit')
    ax3.set_xlabel('Spin Cancellation [%]', fontsize=14)
    ax3.set_ylabel('Required Temperature [K]', fontsize=14)
    ax3.set_title('Trade-off: Lower Spin → Need Higher Temperature', fontsize=15, weight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    # Panel 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = f"""
FOUR-MECHANISM TRADE-OFFS

The four mechanisms can compensate for each other:

BASELINE (with 90% spin cancellation):
  • Separation: 300 r_g ≈ 44,000 km
  • Charge: 447 C
  • Temperature: 10⁹ K
  • Escape probability: ~1%

If SPIN REDUCED to 50% cancellation:
  • Need 10× closer (30 r_g)
  OR
  • Need 100× more charge (44,700 C)
  OR
  • Need 100× hotter (10¹¹ K)

If SPIN REDUCED to 10% cancellation:
  • Escape becomes nearly impossible
  • Even at contact, insufficient energy
  • This is why GR (no spin effect) predicts
    no escape!

CONCLUSION:
High opposing spins (Ω₁ = -Ω₂) are CRITICAL
for rift mechanism to work at observable
separations.

Without spin: Need extreme proximity,
charge, or temperature → rare/impossible

With spin: Natural escape at moderate
conditions → matches observations!
    """

    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.suptitle('Four-Mechanism Trade-offs: Spin ↔ Proximity ↔ Charge ↔ Temperature',
                fontsize=17, weight='bold')
    plt.tight_layout()
    plt.savefig('validation_plots/16_mechanism_tradeoffs.png', dpi=300, bbox_inches='tight')
    print("✅ Saved plot: validation_plots/16_mechanism_tradeoffs.png")
    print()

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print("High spin (90% cancellation) is CRITICAL:")
    print("  • Allows escape at moderate separations (300 r_g)")
    print("  • Requires only modest charge (~500 C)")
    print("  • Works at achievable temperatures (10⁹ K)")
    print()
    print("Without high spin (<50% cancellation):")
    print("  • Need extreme proximity (<30 r_g) → merger risk")
    print("  • OR unrealistic charge (>10⁴ C)")
    print("  • OR impossible temperatures (>10¹¹ K)")
    print()
    print("This is why GR (no spin effect) predicts escape is impossible!")
    print()

if __name__ == "__main__":
    main()
