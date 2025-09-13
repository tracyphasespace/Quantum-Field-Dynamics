#!/usr/bin/env python3
"""
test_genesis_constants.py (v3.2)

Automated validation of the Genesis Constants discovery.

GENESIS CONSTANTS:
- alpha = 4.0 (electrostatic coupling)
- gamma_e_target = 6.0 (electron quartic coupling)
- Expected virial residual: ~0.047 (excellent stability)

This test validates that the "Gentle Equilibrium" regime produces
physically stable atomic structures with the discovered parameters.

CURRENT STATE:
- ✓ 15-minute timeout for thorough convergence
- ✓ Subprocess output logging for diagnostics
- ✓ Proper exit codes for CI integration
- ✓ Path-independent execution
"""

import subprocess
import sys
import json
import os
from qfd_result_schema import genesis_constants, validate_result_schema

def run_genesis_test():
    """Run a single test with the Genesis Constants."""
    gc = genesis_constants()
    print("=" * 60)
    print("TESTING GENESIS CONSTANTS")
    print(f"alpha={gc['alpha']}, gamma_e_target={gc['gamma_e_target']}")
    print(f"Expected: virial < 0.1 (reference: {gc['reference_virial']})")
    print(f"Status: {gc['status']}")
    print("=" * 60)
    
    # Find Deuterium.py relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    deuterium_path = os.path.join(script_dir, "Deuterium.py")
    
    # Run Deuterium.py with Genesis Constants
    cmd = [
        sys.executable, deuterium_path,
        "--mode", "single",
        "--mass", "1.0",      # Hydrogen
        "--charge", "1",
        "--electrons", "1", 
        "--alpha", str(gc["alpha"]),     # Genesis Constant
        "--ge", str(gc["gamma_e_target"]),        # Genesis Constant
        "--iters", "600",     # Reasonable test duration
        "--tol", "1e-7",
        "--outdir", "genesis_test",
        "--outfile", "genesis_result.json"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)  # 15 minutes
        
        if result.returncode == 0:
            print("✓ Run completed successfully")
            # Print subprocess output for quick triage
            if result.stdout.strip():
                print("\nSubprocess output (last 10 lines):")
                lines = result.stdout.strip().split('\n')
                for line in lines[-10:]:
                    print(f"  {line}")
            
            # Load and analyze results
            json_path = os.path.join("genesis_test", "genesis_result.json")
            print(f"Reading results from: {json_path}")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Validate standardized schema
                if not validate_result_schema(data):
                    print(f"[WARN] Result does not conform to standard schema")
                
                # Use standardized fields (no fallbacks needed)
                virial = data.get('virial', float('inf'))
                converged = data.get('converged', False)
                physical_success = data.get('physical_success', False)
                virial_ok = data.get('virial_ok', False)
                penalties_ok = data.get('penalties_ok', False)
                
                print(f"\nRESULTS:")
                print(f"  Virial residual: {virial:.6f}")
                print(f"  Converged flag:  {converged}")
                print(f"  Physical success: {physical_success}")
                
                # Reference comparison
                ref_virial = gc["reference_virial"]
                performance_ratio = virial / ref_virial if ref_virial > 0 else float('inf')
                
                print(f"\nEVALUATION:")
                print(f"  Virial OK:        {'✓' if virial_ok else '✗'}")
                print(f"  Penalties OK:     {'✓' if penalties_ok else '✗'}")
                print(f"  Physical Success: {'✓' if physical_success else '✗'}")
                print(f"  vs Reference:     {performance_ratio:.2f}x (target: ~1.0)")
                
                if physical_success:
                    if performance_ratio < 1.5:  # Within 50% of reference
                        print(f"\n🎉 EXCELLENT! Virial {virial:.4f} matches Genesis Constants reference!")
                        return True
                    else:
                        print(f"\n✓ GOOD! Genesis Constants working (virial {virial:.4f}).")
                        return True
                else:
                    print(f"\n⚠️  Genesis Constants may need adjustment or longer run time.")
                    return False
                    
            else:
                print("⚠️  Results file not found")
                return False
                
        else:
            print("✗ Run failed")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  Run timed out (15 minutes)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = run_genesis_test()
    sys.exit(0 if success else 1)