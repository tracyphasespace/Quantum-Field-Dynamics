#!/usr/bin/env python3
import sys
import subprocess
import concurrent.futures
from pathlib import Path
from datetime import datetime

def print_header(title):
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")

def run_validation(script_name, description, args=None, cwd=None):
    base_dir = Path(__file__).parent.resolve()
    script_path = (base_dir / script_name).resolve()
    if not script_path.exists():
        return False, f"⚠️ WARNING: {script_path} not found"

    cmd = [sys.executable, str(script_path)]
    if args: cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(cwd or script_path.parent))
        status = result.returncode == 0
        output = result.stdout + (result.stderr if result.stderr else "")
        return status, output
    except Exception as e:
        return False, str(e)

def main():
    start_time = datetime.now()
    print_header("QFD PARALLEL VALIDATION SUITE (6 Workers)")
    
    tasks = [
        ('beta_from_alpha', '../../simulation/scripts/derive_beta_from_alpha.py', 'Derive Beta from Alpha'),
        ('hbar_topology', '../../simulation/scripts/derive_hbar_from_topology_parallel.py', 'Derive Planck (Parallel)', ['--relax', '--relax_steps', '500']),
        ('verify_golden', '../../simulation/scripts/verify_golden_loop.py', 'Verify Golden Loop'),
        ('alpha_derived', '../../simulation/scripts/QFD_ALPHA_DERIVED_CONSTANTS.py', 'Calculate Constants'),
        ('beta_tension', '../../simulation/scripts/explore_beta_tension.py', 'Explore Beta Tension'),
        ('integer_ladder', '../nuclear/scripts/integer_ladder_test.py', 'Integer Ladder', ['--scores', '../data/harmonic_scores.parquet', '--out', '../results/']),
        ('fission', '../nuclear/scripts/validate_fission_pythagorean.py', 'Fission Resonance'),
        ('decay_rules', '../nuclear/scripts/analyze_all_decay_transitions.py', 'Decay Selection Rules'),
        ('proton', 'validate_proton_engine.py', 'Proton Drip Engine'),
        ('g2_anomaly', 'validate_g2_corrected.py', 'g-2 Anomaly'),
        ('lepton_stability', 'lepton_stability.py', 'Lepton Stability'),
        ('cmb', 'derive_cmb_temperature.py', 'CMB Temperature')
    ]

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        future_to_task = {executor.submit(run_validation, t[1], t[2], t[3] if len(t)>3 else None): t[0] for t in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            name = future_to_task[future]
            status, output = future.result()
            results[name] = status
            print(f"Completed: {name} ({'PASSED' if status else 'FAILED'})")
            if not status: print(f"Error in {name}:\n{output}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print_header("VALIDATION SUMMARY")
    print(f"Runtime: {elapsed:.1f} seconds using 6 workers\n")
    
    passed = sum(1 for v in results.values() if v)
    for t in tasks:
        print(f"  {t[2]:<40} {'PASSED' if results.get(t[0]) else 'FAILED'}")
    
    print(f"\nTotal: {passed}/{len(tasks)} tests passed")
    return 0 if passed == len(tasks) else 1

if __name__ == '__main__':
    sys.exit(main())
