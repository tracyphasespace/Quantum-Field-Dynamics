#!/usr/bin/env python3
"""Monitor Stage 1 pipeline progress."""

import subprocess
import sys
from pathlib import Path
import time

LOG_FILE = Path("results/v15_production/stage1.log")
TOTAL_SNE = 5468

def get_process_info():
    """Get Stage 1 process info."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if 'stage1_optimize.py' in line and 'grep' not in line:
                parts = line.split()
                return {
                    'pid': parts[1],
                    'cpu': parts[2],
                    'mem_mb': int(parts[5]) // 1024,
                    'running': True
                }
        return {'running': False}
    except Exception:
        return {'running': False}

def parse_log():
    """Parse log file for progress stats."""
    if not LOG_FILE.exists():
        return None

    with open(LOG_FILE) as f:
        lines = f.readlines()

    current = sum(1 for line in lines if 'Optimizing SNID' in line)
    ok_count = sum(1 for line in lines if '✓ OK' in line)
    fail_count = sum(1 for line in lines if '✗ FAIL' in line)

    recent = [line.strip() for line in lines if ('✓ OK' in line or '✗ FAIL' in line)][-10:]

    return {
        'current': current,
        'ok': ok_count,
        'fail': fail_count,
        'recent': recent
    }

def main():
    print("=" * 60)
    print("  QFD V15 Stage 1 Pipeline Monitor")
    print("=" * 60)
    print()

    # Process status
    proc = get_process_info()
    if proc['running']:
        print(f"✓ Stage 1 is RUNNING")
        print(f"  PID: {proc['pid']}")
        print(f"  CPU: {proc['cpu']}%")
        print(f"  Memory: {proc['mem_mb']}MB")
    else:
        print("✗ Stage 1 is NOT running")
        print()
        print("To start:")
        print("  nohup python src/stage1_optimize.py \\")
        print("      --lightcurves data/lightcurves_unified_v2_min3.csv \\")
        print("      --out results/v15_production/stage1 \\")
        print("      --global \"70.0,0.01,30.0\" \\")
        print("      --tol 1e-4 --max-iters 500 --verbose \\")
        print("      > results/v15_production/stage1.log 2>&1 &")
        sys.exit(1)

    print()
    print("-" * 60)
    print()

    # Progress
    stats = parse_log()
    if stats is None:
        print("⚠ Log file not found")
        sys.exit(1)

    pct = int(stats['current'] * 100 / TOTAL_SNE)
    print(f"Progress: {stats['current']} / {TOTAL_SNE} SNe ({pct}%)")

    total_processed = stats['ok'] + stats['fail']
    if total_processed > 0:
        success_rate = int(stats['ok'] * 100 / total_processed)
        print(f"Success rate: {stats['ok']} OK / {stats['fail']} FAIL = {success_rate}%")
    else:
        print("Success rate: (starting...)")

    # Estimate time remaining
    if stats['current'] > 10:
        # Rough estimate based on current progress
        # (This is approximate - actual time varies)
        remaining_sne = TOTAL_SNE - stats['current']
        print(f"Remaining: ~{remaining_sne} SNe")

    print()
    print("-" * 60)
    print()
    print("Recent activity (last 10):")
    print()
    for line in stats['recent']:
        if '✓ OK' in line:
            print(f"  {line}")
        else:
            print(f"  {line}")

    print()
    print("-" * 60)
    print()
    print("Commands:")
    print(f"  watch -n 5 python monitor_pipeline.py  # Auto-refresh every 5s")
    print(f"  tail -f {LOG_FILE}                     # Follow log in real-time")
    if proc['running']:
        print(f"  kill {proc['pid']}                         # Stop pipeline")
    print()

if __name__ == "__main__":
    main()
