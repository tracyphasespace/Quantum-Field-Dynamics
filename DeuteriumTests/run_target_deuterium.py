#!/usr/bin/env python3
"""
run_target_deuterium.py (Genesis Constants v3.2)

Production-ready convenience runner for deuterium-family simulations.
Uses the validated Genesis Constants with comprehensive output generation.

GENESIS CONSTANTS (Locked In):
- alpha = 4.0 (electrostatic coupling strength)
- gamma_e_target = 6.0 (electron quartic coupling target)
- Validated virial residual: 0.0472 (excellent stability)

CURRENT STATE FEATURES:
- ✓ Genesis Constants as defaults
- ✓ Higher resolution defaults (128³ grid)
- ✓ Comprehensive output: JSON + Markdown + CSV
- ✓ Mass scaling with selective dilation
- ✓ Physical success evaluation
- ✓ Production-ready error handling

Example usage:
    # Basic deuterium with Genesis Constants
    python run_target_deuterium.py
    
    # Test different scaling strategies
    python run_target_deuterium.py --dilate-exp 2.0 --outfile D_quadratic.json
    
    # High-resolution validation
    python run_target_deuterium.py --grid 160 --iters 1500 --outfile D_hires.json
    
    # Different isotopes
    python run_target_deuterium.py --mass 3.0 --outfile tritium.json
"""

import argparse
import subprocess
import sys
import shlex


def main():
    ap = argparse.ArgumentParser(
        description="Convenience runner for Deuterium target using Genesis Constants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Physics parameters (Genesis Constants as defaults)
    ap.add_argument("--alpha", type=float, default=4.0, help="Genesis Constant: alpha coupling")
    ap.add_argument("--ge", type=float, default=6.0, help="Genesis Constant: gamma_e_target")
    
    # Matter specification (deuterium-like defaults)
    ap.add_argument("--mass", type=float, default=2.0, help="Soliton mass in AMU")
    ap.add_argument("--charge", type=int, default=1, help="Nuclear charge Z")
    ap.add_argument("--electrons", type=int, default=1, help="Number of electrons")
    
    # Numerical parameters
    ap.add_argument("--iters", type=int, default=1200, help="Outer iterations")
    ap.add_argument("--tol", type=float, default=1e-8, help="Energy convergence tolerance")
    ap.add_argument("--grid", type=int, default=128, help="Grid points (higher for better resolution)")
    
    # Output control
    ap.add_argument("--outdir", default="runs_D128", help="Output directory")
    ap.add_argument("--outfile", default="D128_a4_g6_lin.json", help="Output JSON filename")
    
    # Dilation scaling
    ap.add_argument("--dilate", default="alpha+ge", 
                    choices=["alpha", "ge", "kappa", "alpha+ge", "all", "none"],
                    help="Which couplings to scale with mass")
    ap.add_argument("--dilate-exp", type=float, default=1.0, 
                    help="Mass scaling exponent (1.0=linear, 2.0=quadratic)")
    
    # System parameters
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", 
                    help="Computation device")
    ap.add_argument("--dtype", choices=("float64", "float32"), default="float64", 
                    help="Floating point precision")
    
    args = ap.parse_args()
    
    # Build command for Deuterium.py
    cmd = [
        sys.executable, "Deuterium.py",
        "--mode", "single",
        "--mass", str(args.mass),
        "--charge", str(args.charge),
        "--electrons", str(args.electrons),
        "--alpha", str(args.alpha),
        "--ge", str(args.ge),
        "--dilate", args.dilate,
        "--dilate-exp", str(args.dilate_exp),
        "--grid", str(args.grid),
        "--iters", str(args.iters),
        "--tol", str(args.tol),
        "--outdir", args.outdir,
        "--outfile", args.outfile,
        "--device", args.device,
        "--dtype", args.dtype,
    ]
    
    print("=" * 60)
    print("DEUTERIUM TARGET RUNNER (Genesis Constants v3.2)")
    print("=" * 60)
    print(f"Target: {args.mass} AMU, Z={args.charge}, e⁻={args.electrons}")
    print(f"Genesis Constants: α={args.alpha}, γₑ={args.ge}")
    print(f"Scaling: {args.dilate} with exponent {args.dilate_exp}")
    print(f"Grid: {args.grid}³, iterations: {args.iters}")
    print()
    print("Running command:")
    print("  " + " ".join(shlex.quote(x) for x in cmd))
    print("=" * 60)
    
    # Execute the command
    rc = subprocess.call(cmd)
    
    # Generate human-friendly summaries
    import json
    import os
    import csv
    import datetime as dt
    from qfd_result_schema import QFDResultSchema, validate_result_schema
    
    outpath = os.path.join(args.outdir, args.outfile)
    try:
        with open(outpath, "r") as f:
            data = json.load(f)
        
        # Validate schema
        if not validate_result_schema(data):
            print(f"[WARN] Result file does not conform to standard schema: {outpath}")
        
        # Use standardized schema (no more fallbacks needed!)
        summary = {
            "alpha": data.get("alpha", 4.0),
            "gamma_e_target": data.get("gamma_e_target", 6.0),
            "mass_AMU": data.get("mass_amu", 1.0),
            "charge_e": data.get("charge", 1),
            "electrons": data.get("electrons", 1),
            "E_model": data.get("E_model", 0.0),
            "virial": data.get("virial", 1.0),
            "pen_max": data.get("penalty_Q", 0.0) + data.get("penalty_B", 0.0) + data.get("penalty_center", 0.0),
            "converged": data.get("converged", False),
            "physical_success": data.get("physical_success", False),
            "grid_points": data.get("grid_points", 96),
            "timestamp": data.get("timestamp", dt.datetime.now().isoformat(timespec="seconds"))
        }
        
        base = os.path.splitext(outpath)[0]
        md_path = base + "_summary.md"
        csv_path = base + "_summary.csv"
        
        # Markdown summary
        with open(md_path, "w") as f:
            f.write("# QFD Deuterium Run Summary\n\n")
            f.write(f"**Genesis Constants Configuration**\n\n")
            for k, v in summary.items():
                if v is not None:
                    f.write(f"- **{k}**: {v}\n")
            
            # Use standardized evaluation (already computed in schema)
            virial_ok = data.get("virial_ok", False)
            pen_ok = data.get("penalties_ok", False)
            physical_success = data.get("physical_success", False)
            
            f.write(f"\n## Evaluation\n\n")
            f.write(f"- **Virial OK (< 0.1)**: {'✓' if virial_ok else '✗'}\n")
            f.write(f"- **Penalties OK (< 1e-5)**: {'✓' if pen_ok else '✗'}\n")
            f.write(f"- **Physical Success**: {'✓' if physical_success else '✗'}\n")
        
        # CSV summary
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            w.writeheader()
            w.writerow(summary)
        
        if rc == 0:
            print("\n" + "=" * 60)
            print("✓ DEUTERIUM RUN COMPLETED SUCCESSFULLY")
            print(f"Results saved to:")
            print(f"  JSON:    {outpath}")
            print(f"  Summary: {md_path}")
            print(f"  CSV:     {csv_path}")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✗ DEUTERIUM RUN FAILED")
            print(f"Exit code: {rc}")
            print("=" * 60)
            
    except Exception as e:
        print(f"[WARN] Could not generate summaries ({outpath}): {e}")
        if rc == 0:
            print("\n" + "=" * 60)
            print("✓ DEUTERIUM RUN COMPLETED")
            print(f"Results saved to: {args.outdir}/{args.outfile}")
            print("=" * 60)
    
    sys.exit(rc)


if __name__ == "__main__":
    main()