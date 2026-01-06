#!/usr/bin/env python3
"""
Real-time monitoring for batch optimization.

Usage:
    python monitor_batch.py results/batch_optimization
    python monitor_batch.py results/batch_optimization --watch
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime


def print_status(checkpoint_file: Path):
    """Print current optimization status."""

    if not checkpoint_file.exists():
        print(f"❌ Checkpoint not found: {checkpoint_file}")
        return

    with open(checkpoint_file, 'r') as f:
        data = json.load(f)

    # Header
    print("\n" + "="*70)
    print("BATCH OPTIMIZATION STATUS")
    print("="*70)

    # Timing
    timestamp = datetime.fromisoformat(data['timestamp'])
    elapsed_hours = data['elapsed_hours']

    print(f"\nLast update: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed: {elapsed_hours:.2f} hours")

    # Progress
    n_completed = data['n_completed']
    results = data['results']

    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') != 'success']

    print(f"\nProgress:")
    print(f"  Total jobs: {n_completed}")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed: {len(failed)}")

    if not successful:
        print("\n⚠️  No successful results yet")
        return

    # Best results so far
    successful_sorted = sorted(successful, key=lambda r: r.get('score', 1000))

    print(f"\n🏆 Best Result So Far:")
    best = successful_sorted[0]

    metrics = best.get('metrics', {})
    params = best.get('parameters', {})

    print(f"  Job ID: {best.get('job_id', 'N/A')}")
    print(f"  Score: {best.get('score', 0):.2f}")
    print(f"  Mean Error: {metrics.get('mean_error_pct', 0):.2f}%")
    print(f"  Std Error: {metrics.get('std_error_pct', 0):.2f}%")
    print(f"  Max Error: {metrics.get('max_abs_error_pct', 0):.2f}%")
    print(f"  Mean Virial: {metrics.get('mean_virial', 0):.3f}")

    print(f"\n  Key parameters:")
    for key in ['c_v2_base', 'c_v4_base', 'c_v2_iso', 'c_v4_size']:
        if key in params:
            print(f"    {key:15s}: {params[key]:.6f}")

    # Recent activity
    if len(results) >= 3:
        print(f"\n📊 Recent Jobs (last 3):")
        for r in results[-3:]:
            status_icon = "✓" if r.get('status') == 'success' else "✗"
            job_id = r.get('job_id', 'N/A')
            elapsed_min = r.get('elapsed_seconds', 0) / 60
            score = r.get('score', 0)

            print(f"  {status_icon} Job {job_id:2d}: {elapsed_min:5.1f} min, score={score:.2f}")

    # Error rate
    if len(results) > 0:
        success_rate = 100 * len(successful) / len(results)
        print(f"\nSuccess rate: {success_rate:.1f}%")

    print("\n" + "="*70 + "\n")


def watch_mode(checkpoint_file: Path, interval: int = 30):
    """Watch checkpoint file and update display."""

    print("🔍 Monitoring batch optimization...")
    print(f"   Checkpoint: {checkpoint_file}")
    print(f"   Update interval: {interval} seconds")
    print(f"   Press Ctrl+C to exit")
    print()

    try:
        while True:
            print_status(checkpoint_file)
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n⏸️  Monitoring stopped")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Monitor batch optimization progress"
    )

    parser.add_argument(
        'output_dir',
        nargs='?',
        default='results/batch_optimization',
        help='Output directory to monitor'
    )

    parser.add_argument(
        '--watch',
        action='store_true',
        help='Watch mode: continuously update display'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Update interval in seconds (default: 30)'
    )

    args = parser.parse_args()

    # Find checkpoint file
    output_dir = Path(args.output_dir)
    checkpoint_file = output_dir / 'checkpoint.json'

    if args.watch:
        watch_mode(checkpoint_file, args.interval)
    else:
        print_status(checkpoint_file)


if __name__ == '__main__':
    main()
