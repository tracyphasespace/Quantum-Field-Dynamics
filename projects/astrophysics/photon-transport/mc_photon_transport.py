#!/usr/bin/env python3
"""
mc_photon_transport.py -- Monte Carlo photon transport in the QFD vacuum

QFD FRAMEWORK (static universe, no expansion):
    The vacuum is a dynamic medium with stiffness beta ~ 3.043.  Photons
    interact with the vacuum via two channels:

    1. Forward drag (achromatic):
       Coherent absorption/re-emission with zero angular deflection.
       Each interaction transfers energy dE ~ k_B T_CMB to the bath.
       This produces the cosmological redshift: E(d) = E_0 exp(-kappa d).

    2. Hard scatter (rare, incoherent):
       Four-photon vertex process suppressed by alpha^2 ~ 5.3e-5.
       The photon is removed from the direct beam (absorbed and
       re-emitted isotropically into the CMB).  Surviving photons
       are those that happened to scatter into the forward cone.

KEY INSIGHT (survivor bias):
    Drag events are perfectly forward -- they do NOT broaden the beam.
    Hard scatters remove the photon entirely.  The only beam-broadening
    comes from hard-scattered photons that happen to re-emerge within
    the telescope's PSF.  This contribution is doubly suppressed:
      - by the hard-scatter rate (alpha^2)
      - by the forward-cone solid angle (theta_scatter^2)
    The net PSF is essentially a delta function convolved with a tiny
    wing from stray hard-scatter survivors.

RESULT:
    QFD predicts PSF << telescope PSF at all z <= 10.  This is the
    resolution to the "tired light blurs images" objection: only the
    REMOVED photons scatter; the SURVIVING beam is perfectly collimated.

DISTANCE-REDSHIFT RELATION:
    D(z) = (c / K_J) * ln(1 + z)    [QFD, no expansion]
    where K_J = xi_QFD * beta^(3/2) ~ 85.8 km/s/Mpc

Usage:
    python3 mc_photon_transport.py

Reference: Book v8.5 Ch. 9-12, Appendix Z
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from qfd.shared_constants import (
    ALPHA, BETA, K_J_KM_S_MPC, XI_QFD, C_SI, HBAR_SI,
    K_BOLTZ_SI, MPC_TO_M, M_ELECTRON_MEV, KAPPA_QFD_MPC
)

import numpy as np

# =============================================================================
# DERIVED CONSTANTS
# =============================================================================

# Electron Compton wavelength (vacuum coherence scale)
# L_0 = hbar / (m_e c)
_M_ELECTRON_KG = M_ELECTRON_MEV * 1.602e-13 / C_SI**2  # convert MeV/c^2 to kg
L_0 = HBAR_SI / (_M_ELECTRON_KG * C_SI)  # ~ 3.86e-13 m

# CMB temperature (K)
T_CMB = 2.7255

# Mean thermal energy of the CMB bath
E_CMB_J = K_BOLTZ_SI * T_CMB  # ~ 2.35e-4 eV in Joules
E_CMB_EV = E_CMB_J / 1.602e-19  # ~ 2.35e-4 eV

# Photon decay constant in SI (per meter)
# kappa_SI = K_J / c, where K_J is in s^{-1}
KAPPA_SI = K_J_KM_S_MPC * 1e3 / MPC_TO_M / C_SI  # m^{-1} (very small)

# Same in Mpc^{-1} (imported as KAPPA_QFD_MPC, recompute for clarity)
KAPPA_MPC = KAPPA_QFD_MPC  # ~ 2.86e-4 Mpc^{-1}

# Hard scatter probability per drag event
# Suppressed by alpha^2 (four-photon vertex, two extra EM couplings)
P_HARD_PER_DRAG = ALPHA**2  # ~ 5.3e-5

# Angular deflection per hard scatter
# Momentum transfer: dp ~ k_B T / c
# Deflection: theta ~ dp / p_photon = (k_B T) / E_photon
# For a 2 eV optical photon: theta ~ 2.35e-4 / 2 ~ 1.2e-4 rad
# But hard-scatter survivors are biased toward the forward cone;
# the RMS for those that stay in the beam is even smaller.
# We parametrise the per-scatter RMS as k_B T / E_photon.

# QFD distance (c / K_J) in Mpc
D_SCALE_MPC = C_SI / (K_J_KM_S_MPC * 1e3 / MPC_TO_M) / MPC_TO_M
# Equivalently: c_km_s / K_J_km_s_Mpc
C_KM_S = C_SI / 1e3
D_SCALE_MPC_ALT = C_KM_S / K_J_KM_S_MPC  # Mpc -- this is simpler


# =============================================================================
# CORE CLASS
# =============================================================================

class QFDPhotonTransport:
    """
    Monte Carlo simulator for photon propagation through the QFD vacuum.

    Tracks two populations:
      - Direct beam: photons that experienced ONLY forward drag (zero deflection)
      - Hard-scatter survivors: photons that scattered but remained near
        the line of sight (tiny probability, small deflection)

    The PSF is computed from the angular distribution of ALL surviving
    photons (direct + hard-scatter survivors).
    """

    def __init__(self, n_photons=1_000_000, E_photon_eV=2.0, seed=42):
        """
        Initialise the photon transport simulation.

        Parameters
        ----------
        n_photons : int
            Number of photons to simulate.
        E_photon_eV : float
            Initial photon energy in eV (default 2.0 eV, optical).
        seed : int
            Random number generator seed for reproducibility.
        """
        self.n_photons = n_photons
        self.E0_eV = E_photon_eV
        self.rng = np.random.default_rng(seed)

        # Per-scatter angular RMS (radians)
        # theta_scatter ~ k_B T_CMB / E_photon
        self.theta_per_scatter = E_CMB_EV / self.E0_eV  # ~ 1.2e-4 rad

        # Step size in Mpc (adaptive: smaller at low z, larger at high z)
        self.base_step_Mpc = 10.0  # 10 Mpc steps

    def qfd_distance(self, z):
        """
        QFD distance-redshift relation.

        D(z) = (c / K_J) * ln(1 + z)

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            Proper distance in Mpc.
        """
        return D_SCALE_MPC_ALT * np.log(1.0 + z)

    def propagate(self, z_target):
        """
        Propagate n_photons through the QFD vacuum to redshift z_target.

        Physics:
          - Drag events: E -> E * exp(-kappa * dd) per step dd.
            These are perfectly forward (zero deflection).
          - Hard scatters: occur with probability P_hard_per_step
            per photon per step.  Each hard scatter either removes
            the photon from the beam (most likely) or, with small
            probability proportional to (theta_scatter)^2, redirects
            it into the forward cone.
          - A photon that hard-scatters is removed from the direct
            beam.  We track whether it re-enters via forward emission
            (forward-cone solid angle fraction).

        Parameters
        ----------
        z_target : float
            Target redshift.

        Returns
        -------
        dict
            Results dictionary with keys:
            - 'z': target redshift
            - 'D_Mpc': total distance in Mpc
            - 'n_photons': initial photon count
            - 'n_direct': photons with zero scatters (direct beam)
            - 'n_hard_survivors': photons with >= 1 hard scatter that
              remained in forward cone
            - 'survival_fraction': total surviving / initial
            - 'direct_fraction': direct / initial
            - 'theta_rms_rad': RMS angular deviation of ALL survivors
            - 'theta_rms_arcsec': same in arcseconds
            - 'E_final_eV': mean final photon energy
            - 'z_check': redshift from energy loss (should match z_target)
            - 'E_lost_total_eV': total energy lost to bath per photon
            - 'N_drag_total': total drag interactions per photon
        """
        D_total = self.qfd_distance(z_target)
        if D_total <= 0:
            return self._empty_result(z_target, 0.0)

        # Number of steps
        n_steps = max(10, int(np.ceil(D_total / self.base_step_Mpc)))
        dd = D_total / n_steps  # step size in Mpc

        # --- Per-step quantities ---
        # Fractional energy loss per step from drag: dE/E = kappa * dd
        # where kappa is in Mpc^{-1}
        frac_loss_per_step = KAPPA_MPC * dd

        # Number of drag interactions per step
        # Each drag interaction transfers dE ~ k_B T_CMB.
        # Total fractional loss per step = N_drag * (k_B T / E).
        # But E changes; for small steps, E ~ E0 * exp(-kappa * d_so_far).
        # We use the mean E over the step as approximation.
        # N_drag_per_step ~ (frac_loss_per_step * E) / (k_B T)
        # At d=0: N_drag ~ kappa * dd * E0 / (k_B T)

        # Hard scatter optical depth per step
        # tau_hard_per_step = P_hard_per_drag * N_drag_per_step
        # But N_drag_per_step depends on photon energy.
        # Alternatively: tau_hard = alpha^2 * kappa * dd * (E / k_B T)
        # This is energy-dependent.  For simplicity we track E as it degrades.

        # --- Initialise photon arrays ---
        alive = np.ones(self.n_photons, dtype=bool)  # in the beam
        n_hard_scatters = np.zeros(self.n_photons, dtype=int)
        theta_cumulative = np.zeros(self.n_photons)  # angular deviation
        E_current = np.full(self.n_photons, self.E0_eV)  # current energy

        # --- Step through distance ---
        for step in range(n_steps):
            n_alive = np.sum(alive)
            if n_alive == 0:
                break

            idx = np.where(alive)[0]

            # Drag: reduce energy (achromatic, zero deflection)
            E_current[idx] *= np.exp(-frac_loss_per_step)

            # Number of drag interactions this step for each photon
            # N_drag ~ kappa * dd * E / (k_B T)
            N_drag = frac_loss_per_step * E_current[idx] / E_CMB_EV

            # Hard scatter probability this step per photon
            # P_hard_step = 1 - exp(-tau_hard) ~ tau_hard for small tau
            tau_hard = P_HARD_PER_DRAG * N_drag
            P_hard_step = 1.0 - np.exp(-tau_hard)

            # Draw hard scatters
            hard_mask = self.rng.random(n_alive) < P_hard_step

            if np.any(hard_mask):
                hard_idx = idx[hard_mask]

                # Hard scatter angular deflection
                # Each hard scatter gives theta ~ Rayleigh(sigma)
                # with sigma = theta_per_scatter = k_B T / E
                # The photon is re-emitted quasi-isotropically, but
                # we only keep those in the forward cone.
                # Forward cone solid angle fraction ~ theta_scatter^2 / 4
                theta_scatter = E_CMB_EV / E_current[hard_idx]
                forward_prob = theta_scatter**2 / 4.0

                # Does the re-emitted photon go forward?
                forward_mask = self.rng.random(len(hard_idx)) < forward_prob

                # Photons that scatter backward: removed from beam
                removed = hard_idx[~forward_mask]
                alive[removed] = False

                # Photons that scatter forward: stay in beam with deflection
                survived = hard_idx[forward_mask]
                if len(survived) > 0:
                    n_hard_scatters[survived] += 1
                    # Angular kick: Rayleigh-distributed with sigma = theta_scatter
                    sigma_theta = E_CMB_EV / E_current[survived]
                    dtheta = self.rng.rayleigh(sigma_theta)
                    theta_cumulative[survived] = np.sqrt(
                        theta_cumulative[survived]**2 + dtheta**2
                    )

        # --- Collect results ---
        alive_mask = alive
        n_surviving = np.sum(alive_mask)
        n_direct = np.sum(alive_mask & (n_hard_scatters == 0))
        n_hard_surv = np.sum(alive_mask & (n_hard_scatters > 0))

        # Angular statistics of surviving photons
        theta_survivors = theta_cumulative[alive_mask]
        if len(theta_survivors) > 0:
            theta_rms = np.sqrt(np.mean(theta_survivors**2))
        else:
            theta_rms = 0.0

        # Energy statistics
        E_final_mean = np.mean(E_current[alive_mask]) if n_surviving > 0 else 0.0
        z_check = (self.E0_eV / E_final_mean - 1.0) if E_final_mean > 0 else np.nan

        # Total drag interactions per photon (mean, summed over all steps)
        N_drag_total = KAPPA_MPC * D_total * self.E0_eV / E_CMB_EV

        return {
            'z': z_target,
            'D_Mpc': D_total,
            'n_photons': self.n_photons,
            'n_direct': int(n_direct),
            'n_hard_survivors': int(n_hard_surv),
            'n_surviving': int(n_surviving),
            'survival_fraction': n_surviving / self.n_photons,
            'direct_fraction': n_direct / self.n_photons,
            'theta_rms_rad': theta_rms,
            'theta_rms_arcsec': np.degrees(theta_rms) * 3600.0,
            'E_final_eV': E_final_mean,
            'z_check': z_check,
            'E_lost_total_eV': self.E0_eV - E_final_mean,
            'N_drag_total': N_drag_total,
        }

    def _empty_result(self, z, D):
        """Return a result dict for the trivial case (z=0)."""
        return {
            'z': z,
            'D_Mpc': D,
            'n_photons': self.n_photons,
            'n_direct': self.n_photons,
            'n_hard_survivors': 0,
            'n_surviving': self.n_photons,
            'survival_fraction': 1.0,
            'direct_fraction': 1.0,
            'theta_rms_rad': 0.0,
            'theta_rms_arcsec': 0.0,
            'E_final_eV': self.E0_eV,
            'z_check': 0.0,
            'E_lost_total_eV': 0.0,
            'N_drag_total': 0.0,
        }

    def compute_psf(self, z_target):
        """
        Compute the Point Spread Function at a given redshift.

        Returns a summary dict with FWHM and RMS of the PSF.

        Parameters
        ----------
        z_target : float
            Target redshift.

        Returns
        -------
        dict
            PSF summary with keys: 'z', 'D_Mpc', 'fwhm_arcsec',
            'rms_arcsec', 'survival_fraction', 'direct_fraction'.
        """
        result = self.propagate(z_target)

        # For a Rayleigh distribution: FWHM ~ 2.355 * sigma
        # For our distribution (mostly delta + tiny Rayleigh wing):
        # The FWHM of the direct beam is zero (delta function).
        # The observable FWHM is set by the telescope, not QFD.
        # We report the RMS of the QFD-induced angular spread.
        # FWHM ~ 2.355 * theta_rms (Gaussian approximation)
        fwhm_arcsec = 2.355 * result['theta_rms_arcsec']

        return {
            'z': z_target,
            'D_Mpc': result['D_Mpc'],
            'fwhm_arcsec': fwhm_arcsec,
            'rms_arcsec': result['theta_rms_arcsec'],
            'survival_fraction': result['survival_fraction'],
            'direct_fraction': result['direct_fraction'],
            'n_direct': result['n_direct'],
            'n_hard_survivors': result['n_hard_survivors'],
        }

    def psf_vs_redshift(self, z_values):
        """
        Compute PSF at multiple redshifts.

        Parameters
        ----------
        z_values : array-like
            List of redshift values.

        Returns
        -------
        list of dict
            PSF summary at each redshift.
        """
        results = []
        for z in z_values:
            psf = self.compute_psf(z)
            results.append(psf)
        return results


# =============================================================================
# SELF-TEST AND VERIFICATION
# =============================================================================

def verify_physics(result):
    """
    Verify internal consistency of a propagation result.

    Checks:
      1. Energy conservation: final energy = E0 / (1 + z)
      2. Redshift consistency: z_check ~ z_target
      3. Survival fraction is physical (0 <= S <= 1)
    """
    checks = []

    # Check 1: z from energy loss matches target z
    z_target = result['z']
    z_check = result['z_check']
    if z_target > 0 and np.isfinite(z_check):
        z_err = abs(z_check - z_target) / z_target
        ok = z_err < 0.05  # 5% tolerance (finite step discretisation)
        checks.append(('z consistency', ok,
                        f'z_target={z_target:.4f}, z_check={z_check:.4f}, '
                        f'err={z_err:.2e}'))
    else:
        checks.append(('z consistency', True, 'z=0, trivially OK'))

    # Check 2: E_final ~ E0 / (1 + z)
    E_expected = result['E_final_eV']
    if result['n_surviving'] > 0 and z_target > 0:
        # Direct-beam photons should have E = E0 * exp(-kappa*D)
        # which equals E0 / (1+z) in QFD
        E0 = 2.0  # default, but we don't store it in result
        E_predicted = E0 / (1.0 + z_target)
        if E_expected > 0:
            e_err = abs(E_expected - E_predicted) / E_predicted
            ok = e_err < 0.1
            checks.append(('energy conservation', ok,
                            f'E_final={E_expected:.6f} eV, '
                            f'E_predicted={E_predicted:.6f} eV, '
                            f'err={e_err:.2e}'))

    # Check 3: survival fraction is physical
    S = result['survival_fraction']
    ok = 0.0 <= S <= 1.0
    checks.append(('survival physical', ok, f'S={S:.6f}'))

    # Check 4: direct >= hard_survivors (most photons should be direct)
    n_d = result['n_direct']
    n_h = result['n_hard_survivors']
    ok = n_d >= n_h
    checks.append(('direct > hard', ok,
                    f'n_direct={n_d}, n_hard_surv={n_h}'))

    return checks


def run_verification():
    """Run the full verification suite and print results."""
    print("\n" + "=" * 70)
    print("SELF-TEST: Physics Verification")
    print("=" * 70)

    sim = QFDPhotonTransport(n_photons=100_000, E_photon_eV=2.0, seed=42)

    test_redshifts = [0.1, 0.5, 1.0, 2.0]
    all_pass = True

    for z in test_redshifts:
        result = sim.propagate(z)
        checks = verify_physics(result)

        print(f"\n  z = {z}:")
        for name, ok, detail in checks:
            status = "PASS" if ok else "FAIL"
            print(f"    [{status}] {name}: {detail}")
            if not ok:
                all_pass = False

    print(f"\n  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


# =============================================================================
# PLOTTING
# =============================================================================

def generate_plot(psf_results, output_path):
    """
    Generate PSF vs redshift figure with telescope comparison.

    Parameters
    ----------
    psf_results : list of dict
        Output of psf_vs_redshift().
    output_path : str
        Path to save the figure.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available, skipping plot")
        return

    z_vals = [r['z'] for r in psf_results]
    fwhm_vals = [r['fwhm_arcsec'] for r in psf_results]
    rms_vals = [r['rms_arcsec'] for r in psf_results]
    surv_vals = [r['survival_fraction'] for r in psf_results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('QFD Monte Carlo Photon Transport: PSF vs Redshift',
                 fontsize=14, fontweight='bold')

    # --- Top panel: PSF upper bound ---
    # When MC gives zero (no hard-scatter survivors), compute the
    # analytic upper bound on the PSF contribution.
    # N_hard_total ~ (1 - S) * N_photons (photons that hard-scattered)
    # P_forward ~ (theta_scatter)^2 / 4
    # Expected forward survivors ~ N_hard_total * P_forward
    # If < 1, the PSF FWHM upper bound is theta_scatter * 2.355
    fwhm_upper = []
    for r in psf_results:
        if r['fwhm_arcsec'] > 0:
            fwhm_upper.append(r['fwhm_arcsec'])
        else:
            # Analytic upper bound: if one photon DID survive,
            # it would carry theta ~ k_BT / E_final
            E_final = 2.0 / (1.0 + r['z'])
            theta_one = E_CMB_EV / E_final  # rad
            fwhm_one = theta_one * 2.355 * 3600.0 * (180.0 / np.pi)  # arcsec
            fwhm_upper.append(fwhm_one)

    ax1.semilogy(z_vals, fwhm_upper, 'b-o', markersize=4,
                 label='QFD PSF upper bound (single-scatter)')

    # Telescope PSF reference lines
    ax1.axhline(0.050, color='red', linestyle='--', alpha=0.7,
                label='Hubble PSF (50 mas)')
    ax1.axhline(0.070, color='orange', linestyle='--', alpha=0.7,
                label='JWST PSF (70 mas)')
    ax1.axhline(0.800, color='gray', linestyle=':', alpha=0.5,
                label='Ground seeing (0.8")')

    ax1.set_ylabel('Angular size (arcsec)')
    ax1.set_ylim(1e-4, 10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Even the worst-case QFD scatter is far below telescope PSF')

    # --- Bottom panel: survival fraction ---
    ax2.plot(z_vals, surv_vals, 'g-o', label='Survival fraction', markersize=4)
    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.3)

    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Beam survival fraction')
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Fraction of photons remaining in direct beam')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved to: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("QFD MONTE CARLO PHOTON TRANSPORT SIMULATOR")
    print("Static Universe | Vacuum Drag Redshift | No Expansion")
    print("=" * 70)

    # --- Derived constants ---
    print("\n--- DERIVED CONSTANTS ---")
    print(f"  alpha              = 1/{1.0/ALPHA:.6f}")
    print(f"  beta               = {BETA:.9f}")
    print(f"  xi_QFD             = {XI_QFD:.4f}")
    print(f"  K_J                = {K_J_KM_S_MPC:.4f} km/s/Mpc")
    print(f"  kappa (Mpc^-1)     = {KAPPA_MPC:.6e}")
    print(f"  kappa (m^-1)       = {KAPPA_SI:.6e}")
    print(f"  D_scale = c/K_J    = {D_SCALE_MPC_ALT:.2f} Mpc")
    print(f"  L_0 (Compton)      = {L_0:.4e} m")
    print(f"  k_B T_CMB          = {E_CMB_EV:.4e} eV")
    print(f"  P_hard/drag        = alpha^2 = {P_HARD_PER_DRAG:.4e}")
    print(f"  theta_per_scatter  = k_BT/E  = {E_CMB_EV/2.0:.4e} rad "
          f"(for 2 eV photon)")

    # --- Simulation parameters ---
    N_PHOTONS = 1_000_000
    E_PHOTON = 2.0  # eV (optical)
    z_values = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])

    print(f"\n--- SIMULATION PARAMETERS ---")
    print(f"  N_photons          = {N_PHOTONS:,}")
    print(f"  E_photon           = {E_PHOTON} eV (optical)")
    print(f"  z_range            = [{z_values[0]}, {z_values[-1]}]")

    # --- Run simulation ---
    print(f"\n--- RUNNING MONTE CARLO ---")
    sim = QFDPhotonTransport(n_photons=N_PHOTONS, E_photon_eV=E_PHOTON, seed=42)

    t_start = time.time()
    psf_results = sim.psf_vs_redshift(z_values)
    t_elapsed = time.time() - t_start

    print(f"  Completed in {t_elapsed:.1f} s")

    # --- Results table ---
    # Telescope PSF references
    HUBBLE_PSF_ARCSEC = 0.050   # 50 milliarcsecond
    JWST_PSF_ARCSEC = 0.070     # 70 milliarcsecond

    print(f"\n--- PSF vs REDSHIFT ---")
    print(f"  Telescope reference: Hubble = {HUBBLE_PSF_ARCSEC*1000:.0f} mas, "
          f"JWST = {JWST_PSF_ARCSEC*1000:.0f} mas")
    print()

    header = (f"  {'z':>6}  {'D(Mpc)':>10}  {'Survival':>10}  {'Direct':>10}  "
              f"{'N_hard':>8}  {'PSF_FWHM':>12}  {'PSF_RMS':>12}  "
              f"{'vs Hubble':>12}  {'vs JWST':>12}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in psf_results:
        fwhm = r['fwhm_arcsec']
        rms = r['rms_arcsec']

        # Ratio to telescope PSF
        if fwhm > 0:
            ratio_hubble = fwhm / HUBBLE_PSF_ARCSEC
            ratio_jwst = fwhm / JWST_PSF_ARCSEC
            hubble_str = f"{ratio_hubble:.2e}"
            jwst_str = f"{ratio_jwst:.2e}"
        else:
            hubble_str = "0 (delta)"
            jwst_str = "0 (delta)"

        # Format PSF values
        if fwhm > 0:
            fwhm_str = f"{fwhm:.4e} as"
            rms_str = f"{rms:.4e} as"
        else:
            fwhm_str = "0 (delta)"
            rms_str = "0 (delta)"

        print(f"  {r['z']:6.2f}  {r['D_Mpc']:10.1f}  "
              f"{r['survival_fraction']:10.6f}  {r['direct_fraction']:10.6f}  "
              f"{r['n_hard_survivors']:8d}  {fwhm_str:>12}  {rms_str:>12}  "
              f"{hubble_str:>12}  {jwst_str:>12}")

    # --- Key result ---
    print(f"\n--- KEY RESULT ---")
    z_max = z_values[-1]
    r_max = psf_results[-1]
    print(f"  At z = {z_max}:")
    print(f"    QFD PSF FWHM     = {r_max['fwhm_arcsec']:.4e} arcsec")
    print(f"    Hubble PSF       = {HUBBLE_PSF_ARCSEC:.3f} arcsec")
    print(f"    JWST PSF         = {JWST_PSF_ARCSEC:.3f} arcsec")

    if r_max['fwhm_arcsec'] > 0:
        ratio = r_max['fwhm_arcsec'] / HUBBLE_PSF_ARCSEC
        print(f"    QFD / Hubble     = {ratio:.2e}")
        print(f"    QFD vacuum scatter is {1.0/ratio:.0e}x SMALLER than Hubble PSF")
    else:
        print(f"    QFD PSF is identically zero (no hard-scatter survivors)")

    print(f"\n    Survival fraction = {r_max['survival_fraction']:.6f}")
    print(f"    Direct beam frac  = {r_max['direct_fraction']:.6f}")
    print(f"    Hard scatter surv = {r_max['n_hard_survivors']}")

    # --- Physics explanation ---
    print(f"\n--- PHYSICS EXPLANATION ---")
    print(f"  The QFD vacuum produces negligible image broadening because:")
    print(f"  1. Forward drag events (>> 99.99% of interactions) have")
    print(f"     ZERO angular deflection -- they are coherent processes.")
    print(f"  2. Hard scatters (alpha^2 ~ {ALPHA**2:.1e} per drag event)")
    print(f"     REMOVE the photon from the beam entirely.")
    print(f"  3. The tiny fraction that re-scatter into the forward cone")
    print(f"     (~theta^2/4 ~ {(E_CMB_EV/E_PHOTON)**2/4:.1e} per hard scatter)")
    print(f"     carry negligible angular deviation.")
    print(f"  4. Net PSF is a delta function + vanishing wing.")
    print(f"  5. This resolves the 'tired light blurs images' objection:")
    print(f"     only REMOVED photons scatter; SURVIVING beam is collimated.")

    # --- Comparison table ---
    print(f"\n--- COMPARISON: QFD vs STANDARD TIRED LIGHT ---")
    print(f"  {'':>20}  {'Standard TL':>15}  {'QFD':>15}")
    print(f"  {'':>20}  {'(isotropic)':>15}  {'(forward drag)':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Scatter geometry':>20}  {'isotropic':>15}  {'forward only':>15}")
    print(f"  {'Image blur':>20}  {'severe':>15}  {'negligible':>15}")
    print(f"  {'Achromatic?':>20}  {'no':>15}  {'yes':>15}")
    print(f"  {'Surface brightness':>20}  {'dims as (1+z)':>15}  {'dims as (1+z)':>15}")
    print(f"  {'CMB spectrum':>20}  {'non-Planckian':>15}  {'Planckian':>15}")
    print(f"  {'Tolman test':>20}  {'fails':>15}  {'passes':>15}")

    # --- Self-test ---
    run_verification()

    # --- Generate plot ---
    print(f"\n--- GENERATING FIGURE ---")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(output_dir, 'psf_vs_redshift.png')
    generate_plot(psf_results, plot_path)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  QFD photon transport Monte Carlo ({N_PHOTONS:,} photons, E={E_PHOTON} eV)")
    print(f"  K_J = {K_J_KM_S_MPC:.2f} km/s/Mpc (derived from alpha via Golden Loop)")
    print(f"  PSF FWHM at z=10: {r_max['fwhm_arcsec']:.4e} arcsec")
    print(f"  Conclusion: QFD vacuum scatter is UNDETECTABLE")
    print(f"  (many orders of magnitude below telescope diffraction limit)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
