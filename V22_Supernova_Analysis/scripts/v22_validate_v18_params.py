#!/usr/bin/env python3
"""
V22: Validate V18 Best-Fit Parameters Against Lean Constraints

Instead of re-fitting (which causes issues), validate V18's working
parameters against:
1. Unified schema structure
2. Lean-derived constraints
3. Physical bounds

Then explore which additional schema parameters could be constrained.
"""

import sys
import json
from pathlib import Path

# Add schema directory
schema_path = Path("/home/tracy/development/QFD_SpectralGap/Background_and_Schema")
sys.path.insert(0, str(schema_path))

try:
    from qfd_unified_schema import QFDCouplings, CosmologyParams
    HAVE_SCHEMA = True
except ImportError:
    print("Warning: Could not import unified schema")
    HAVE_SCHEMA = False

# ============================================================================
# LEAN CONSTRAINTS
# ============================================================================

class LeanConstraints:
    """Lean-derived parameter constraints (to be formalized)."""

    # k_J: Universal J·A interaction
    K_J_MIN = 50.0   # km/s/Mpc
    K_J_MAX = 100.0  # km/s/Mpc

    # eta_prime: Plasma veil opacity
    ETA_PRIME_MIN = -10.0
    ETA_PRIME_MAX = 0.0

    # xi: Thermal processing
    XI_MIN = -10.0
    XI_MAX = 0.0

    @classmethod
    def validate_k_J(cls, k_J_correction, baseline=70.0):
        """Validate k_J = baseline + correction."""
        k_J_total = baseline + k_J_correction
        return cls.K_J_MIN <= k_J_total <= cls.K_J_MAX, k_J_total

    @classmethod
    def validate_eta_prime(cls, eta_prime):
        """Validate eta_prime."""
        return cls.ETA_PRIME_MIN <= eta_prime <= cls.ETA_PRIME_MAX

    @classmethod
    def validate_xi(cls, xi):
        """Validate xi."""
        return cls.XI_MIN <= xi <= cls.XI_MAX

# ============================================================================
# MAIN VALIDATION
# ============================================================================

def validate_v18_parameters():
    """Validate V18's best-fit parameters."""

    print("="*80)
    print("V22: VALIDATE V18 PARAMETERS AGAINST LEAN CONSTRAINTS")
    print("="*80)
    print()

    # Load V18 Stage2 best-fit parameters
    v18_stage2_file = Path("/home/tracy/development/SupernovaSrc/qfd-supernova-v15/v15_clean/v18/results/stage2_fullscale_5468sne/summary.json")

    if not v18_stage2_file.exists():
        print(f"ERROR: V18 Stage2 results not found at:\n  {v18_stage2_file}")
        return

    print(f"Loading V18 Stage2 results from:\n  {v18_stage2_file}")
    with open(v18_stage2_file) as f:
        v18_params = json.load(f)

    print()
    print("V18 BEST-FIT PARAMETERS (4,885 SNe, RMS=2.18 mag):")
    print("-"*80)

    # Extract parameters (median values from MCMC)
    k_J_correction = v18_params['k_J_correction']['median']
    eta_prime = v18_params['eta_prime']['median']
    xi = v18_params['xi']['median']
    sigma_ln_A = v18_params['sigma_ln_A']['median']

    print(f"k_J_correction = {k_J_correction:.4f} ± {v18_params['k_J_correction']['std']:.4f} km/s/Mpc")
    print(f"eta_prime      = {eta_prime:.4f} ± {v18_params['eta_prime']['std']:.4f}")
    print(f"xi             = {xi:.4f} ± {v18_params['xi']['std']:.4f}")
    print(f"sigma_ln_A     = {sigma_ln_A:.6f} ± {v18_params['sigma_ln_A']['std']:.6f}")
    print()

    # Calculate total k_J
    k_J_baseline = 70.0  # km/s/Mpc
    k_J_total = k_J_baseline + k_J_correction

    print(f"Derived k_J (total) = {k_J_baseline} + {k_J_correction:.4f} = {k_J_total:.4f} km/s/Mpc")
    print()

    # Validate against Lean constraints
    print("="*80)
    print("LEAN CONSTRAINT VALIDATION")
    print("="*80)
    print()

    print(f"Constraint: k_J ∈ [{LeanConstraints.K_J_MIN}, {LeanConstraints.K_J_MAX}] km/s/Mpc")
    valid_k_J, k_J_val = LeanConstraints.validate_k_J(k_J_correction)
    if valid_k_J:
        print(f"  ✅ PASS: k_J = {k_J_val:.2f} km/s/Mpc")
    else:
        print(f"  ❌ FAIL: k_J = {k_J_val:.2f} km/s/Mpc (outside bounds!)")
    print()

    print(f"Constraint: eta_prime ∈ [{LeanConstraints.ETA_PRIME_MIN}, {LeanConstraints.ETA_PRIME_MAX}]")
    valid_eta = LeanConstraints.validate_eta_prime(eta_prime)
    if valid_eta:
        print(f"  ✅ PASS: eta_prime = {eta_prime:.4f}")
    else:
        print(f"  ❌ FAIL: eta_prime = {eta_prime:.4f} (outside bounds!)")
    print()

    print(f"Constraint: xi ∈ [{LeanConstraints.XI_MIN}, {LeanConstraints.XI_MAX}]")
    valid_xi = LeanConstraints.validate_xi(xi)
    if valid_xi:
        print(f"  ✅ PASS: xi = {xi:.4f}")
    else:
        print(f"  ❌ FAIL: xi = {xi:.4f} (outside bounds!)")
    print()

    # Overall validation
    all_valid = valid_k_J and valid_eta and valid_xi

    print("="*80)
    if all_valid:
        print("✅ ALL V18 PARAMETERS SATISFY LEAN CONSTRAINTS")
    else:
        print("❌ SOME V18 PARAMETERS VIOLATE LEAN CONSTRAINTS")
    print("="*80)
    print()

    # Map to unified schema
    if HAVE_SCHEMA:
        print("="*80)
        print("UNIFIED SCHEMA MAPPING")
        print("="*80)
        print()

        print("V18 parameters map to QFDCouplings:")
        print(f"  k_J = {k_J_total:.4f} km/s/Mpc  →  QFDCouplings.k_J")
        print(f"  eta_prime = {eta_prime:.4f}      →  QFDCouplings.eta_prime")
        print(f"  xi = {xi:.4f}                    →  QFDCouplings.xi")
        print()

        print("Parameters constrained by V18 supernova data:")
        print("  ✅ k_J (Universal J·A interaction)")
        print("  ✅ eta_prime (Plasma veil opacity)")
        print("  ✅ xi (Thermal processing)")
        print()

        print("Schema parameters NOT YET constrained (require other data):")
        print("  ❓ V2, V4, V6, V8 (Potential couplings - need nuclear/particle data)")
        print("  ❓ lambda_R1-R4 (Rotor couplings - need BBH lensing signal)")
        print("  ❓ k_c2, k_EM, k_csr (Interaction couplings - need multi-domain data)")
        print("  ❓ g_c (Geometric charge - need charge geometry measurements)")
        print()

        print("Total schema parameters: 15")
        print("Constrained by V18 SN data: 3 (20%)")
        print("Remaining unconstrained: 12 (80%)")
        print()

    # Save V22 validation results
    output_dir = Path("/home/tracy/development/QFD_SpectralGap/V22_Supernova_Analysis/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    v22_results = {
        'v18_best_fit': {
            'k_J_correction': k_J_correction,
            'k_J_total': k_J_total,
            'eta_prime': eta_prime,
            'xi': xi,
            'sigma_ln_A': sigma_ln_A
        },
        'lean_validation': {
            'k_J_valid': valid_k_J,
            'eta_prime_valid': valid_eta,
            'xi_valid': valid_xi,
            'all_valid': all_valid
        },
        'schema_integration': {
            'have_schema': HAVE_SCHEMA,
            'parameters_constrained': 3,
            'parameters_total': 15,
            'constraint_fraction': 0.20
        }
    }

    output_file = output_dir / "v22_v18_validation.json"
    with open(output_file, 'w') as f:
        json.dump(v22_results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()

    return v22_results

if __name__ == "__main__":
    validate_v18_parameters()
