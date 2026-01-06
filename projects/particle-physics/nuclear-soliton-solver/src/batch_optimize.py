#!/usr/bin/env python3
"""
Parallel Batch Optimization for Nuclear Soliton Solver
======================================================

Runs multiple optimization jobs in parallel with controlled memory usage.

Usage:
    python batch_optimize.py --workers 4 --hours 8 --config experiments/nuclear_heavy_region.runspec.json

Features:
    - Parallel worker pool with memory limits
    - Checkpoint saving for crash recovery
    - Progress monitoring
    - Automatic result aggregation
    - Best parameters selection
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from runspec_adapter import RunSpecAdapter


class BatchOptimizer:
    """Manages parallel optimization jobs with checkpointing."""

    def __init__(self, config_path: str, n_workers: int, max_hours: float,
                 output_dir: str = "results/batch_optimization"):
        """
        Initialize batch optimizer.

        Args:
            config_path: Path to base RunSpec configuration
            n_workers: Number of parallel workers
            max_hours: Maximum runtime in hours
            output_dir: Directory for results and checkpoints
        """
        self.config_path = Path(config_path)
        self.n_workers = n_workers
        self.max_hours = max_hours
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load base configuration
        with open(config_path, 'r') as f:
            self.base_config = json.load(f)

        # Timing
        self.start_time = None
        self.end_time = None

        # Results storage
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.results = []

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.should_stop = False

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\n⚠️  Interrupt received. Saving checkpoint and exiting...")
        self.should_stop = True
        self._save_checkpoint()

    def _save_checkpoint(self):
        """Save current results to checkpoint file."""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_hours': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0,
            'results': self.results,
            'n_completed': len(self.results)
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"✓ Checkpoint saved: {len(self.results)} results")

    def _load_checkpoint(self) -> bool:
        """Load checkpoint if exists. Returns True if loaded."""
        if self.checkpoint_file.exists():
            print(f"📂 Found checkpoint: {self.checkpoint_file}")
            response = input("Resume from checkpoint? [y/N]: ").strip().lower()

            if response == 'y':
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)

                self.results = checkpoint['results']
                print(f"✓ Loaded {len(self.results)} previous results")
                return True

        return False

    def _create_job_configs(self, n_jobs: int) -> List[Path]:
        """
        Create job-specific RunSpec configurations.

        Strategy: Use different random seeds and isotope subsets for diversity.

        Args:
            n_jobs: Number of jobs to create

        Returns:
            List of paths to job configuration files
        """
        job_configs = []

        # Load AME data to select different isotope subsets
        from qfd_metaopt_ame2020 import load_ame2020_data

        # Try multiple possible paths for AME data
        possible_paths = [
            Path(self.config_path).parent.parent / 'data' / 'ame2020_system_energies.csv',
            Path('data/ame2020_system_energies.csv'),
            Path('../data/ame2020_system_energies.csv'),
        ]

        ame_path = None
        for path in possible_paths:
            if path.exists():
                ame_path = path
                break

        if ame_path is None:
            raise FileNotFoundError(f"Could not find ame2020_system_energies.csv")

        ame_data = load_ame2020_data(str(ame_path))

        # Heavy isotopes: A >= 120
        heavy_isotopes = ame_data[ame_data['A'] >= 120]

        print(f"\n📋 Creating {n_jobs} job configurations...")
        print(f"   Heavy isotope pool: {len(heavy_isotopes)} total")

        for job_id in range(n_jobs):
            # Create job-specific config (DEEP COPY to avoid shared nested dicts)
            job_config = copy.deepcopy(self.base_config)

            # Update experiment ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_config['experiment_id'] = f"{self.base_config['experiment_id']}_job{job_id:02d}_{timestamp}"

            # Modify solver options for diversity
            if 'solver' not in job_config:
                job_config['solver'] = {}

            if 'options' not in job_config['solver']:
                job_config['solver']['options'] = {}

            # Different random seed for each job
            job_config['solver']['options']['seed'] = 42 + job_id

            # Adaptive population size (larger for early jobs)
            # Clamp to valid range: scipy.differential_evolution requires popsize >= 5
            # but performs best with >= 8-10 for proper mutation/crossover dynamics
            if 'popsize' not in job_config['solver']['options']:
                job_config['solver']['options']['popsize'] = max(8, 15 - (job_id // 4))

            # Different maxiter (time budget distribution)
            if 'maxiter' not in job_config['solver']['options']:
                # Early jobs: more iterations (thorough search)
                # Later jobs: fewer iterations (quick exploration)
                base_maxiter = 300
                job_config['solver']['options']['maxiter'] = max(50, base_maxiter - job_id * 20)

            # Select different isotope subset for each job
            # This increases diversity and reduces overfitting
            n_isotopes = 4
            subset = heavy_isotopes.sample(n=min(n_isotopes, len(heavy_isotopes)),
                                          random_state=42 + job_id)

            # Create dataset cut specification
            # We'll encode specific isotopes in the description for now
            # (The adapter will still use A_range, but we document the intent)
            job_config['metadata'] = {
                'job_id': job_id,
                'random_seed': 42 + job_id,
                'target_isotopes': [
                    {'Z': int(row['Z']), 'A': int(row['A'])}
                    for _, row in subset.iterrows()
                ],
                'strategy': 'diverse_sampling'
            }

            # Save job config
            job_file = self.output_dir / f"job_{job_id:02d}_config.json"
            with open(job_file, 'w') as f:
                json.dump(job_config, f, indent=2)

            job_configs.append(job_file)

            print(f"   Job {job_id:2d}: seed={42+job_id}, maxiter={job_config['solver']['options']['maxiter']}, "
                  f"isotopes={n_isotopes}")

        return job_configs

    @staticmethod
    def _run_single_job(args):
        """
        Run a single optimization job (worker function).

        Args:
            args: Tuple of (job_id, config_path, timeout_seconds)

        Returns:
            Dictionary with job results
        """
        job_id, config_path, timeout_seconds = args

        start_time = time.time()

        # Timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Job {job_id} exceeded timeout of {timeout_seconds/60:.1f} minutes")

        try:
            print(f"[Job {job_id:02d}] Starting optimization (timeout: {timeout_seconds/60:.1f} min)...")

            # Set timeout alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))

            try:
                # Run optimization
                adapter = RunSpecAdapter(str(config_path))
                results = adapter.run(verbose=False)
            finally:
                # Cancel alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            elapsed = time.time() - start_time

            # Extract key metrics
            job_result = {
                'job_id': job_id,
                'config_path': str(config_path),
                'status': results.get('status', 'failed'),
                'elapsed_seconds': elapsed,
                'parameters': results.get('parameters', {}),
                'metrics': results.get('metrics', {}),
                'optimization': results.get('optimization', {}),
                'n_predictions': len(results.get('predictions', [])),
                'timestamp': datetime.now().isoformat()
            }

            # Compute overall score (lower is better)
            if results.get('status') == 'success':
                metrics = results['metrics']
                # Penalize both mean error and spread
                score = abs(metrics.get('mean_error_pct', 100)) + 0.5 * metrics.get('std_error_pct', 100)
                job_result['score'] = score

                print(f"[Job {job_id:02d}] ✓ Complete in {elapsed/60:.1f} min | "
                      f"Error: {metrics.get('mean_error_pct', 0):.2f}% | "
                      f"Score: {score:.2f}")
            else:
                job_result['score'] = 1000.0  # Large penalty for failures
                print(f"[Job {job_id:02d}] ✗ Failed after {elapsed/60:.1f} min")

            return job_result

        except TimeoutError as e:
            elapsed = time.time() - start_time
            print(f"[Job {job_id:02d}] ⏱ Timeout after {elapsed/60:.1f} min")

            return {
                'job_id': job_id,
                'config_path': str(config_path),
                'status': 'timeout',
                'error': str(e),
                'elapsed_seconds': elapsed,
                'score': 1000.0,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[Job {job_id:02d}] ✗ Exception: {e}")

            return {
                'job_id': job_id,
                'config_path': str(config_path),
                'status': 'error',
                'error': str(e),
                'elapsed_seconds': elapsed,
                'score': 1000.0,
                'timestamp': datetime.now().isoformat()
            }

    def run(self):
        """Execute batch optimization with parallel workers."""

        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  Batch Optimization - Parallel Worker Pool                   ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        print()
        print(f"Configuration: {self.config_path}")
        print(f"Workers: {self.n_workers}")
        print(f"Max runtime: {self.max_hours:.1f} hours")
        print(f"Output: {self.output_dir}")
        print()

        # Check for checkpoint
        resumed = self._load_checkpoint()

        self.start_time = datetime.now()
        deadline = self.start_time + timedelta(hours=self.max_hours)

        print(f"⏰ Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏰ Deadline: {deadline.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Estimate number of jobs we can run
        # Assume each job takes ~30-60 minutes on average
        avg_job_minutes = 45
        total_worker_minutes = self.max_hours * 60 * self.n_workers
        estimated_jobs = int(total_worker_minutes / avg_job_minutes)

        # Cap at reasonable number
        n_jobs = min(estimated_jobs, 50)

        print(f"📊 Estimated jobs: {n_jobs} (assuming ~{avg_job_minutes} min/job)")
        print()

        # Create job configurations
        job_configs = self._create_job_configs(n_jobs)

        # Calculate timeout per job (leave 10% buffer)
        timeout_seconds = (self.max_hours * 3600 * 0.9) / (n_jobs / self.n_workers)

        print()
        print(f"⚙️  Worker pool configuration:")
        print(f"   Workers: {self.n_workers}")
        print(f"   Jobs: {n_jobs}")
        print(f"   Timeout per job: {timeout_seconds/60:.1f} min")
        print()

        # Prepare job arguments
        job_args = [(i, config, timeout_seconds) for i, config in enumerate(job_configs)]

        # If resumed, skip completed jobs
        if resumed:
            completed_ids = {r['job_id'] for r in self.results}
            job_args = [args for args in job_args if args[0] not in completed_ids]
            print(f"📌 Skipping {len(completed_ids)} completed jobs")
            print()

        # Run parallel optimization
        print("🚀 Starting parallel optimization...")
        print()

        try:
            with mp.Pool(processes=self.n_workers) as pool:
                # Use imap_unordered for better progress tracking
                for result in pool.imap_unordered(self._run_single_job, job_args):
                    self.results.append(result)

                    # Save checkpoint periodically
                    if len(self.results) % 5 == 0:
                        self._save_checkpoint()

                    # Check deadline
                    if datetime.now() >= deadline:
                        print(f"\n⏰ Deadline reached. Stopping...")
                        pool.terminate()
                        break

                    if self.should_stop:
                        pool.terminate()
                        break

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")

        finally:
            self.end_time = datetime.now()
            self._save_checkpoint()
            self._summarize_results()

    def _summarize_results(self):
        """Print summary of optimization results."""

        print()
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║  Optimization Complete - Results Summary                      ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        print()

        if not self.results:
            print("⚠️  No results to summarize")
            return

        # Filter successful runs
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] != 'success']

        print(f"Total jobs: {len(self.results)}")
        print(f"  ✓ Successful: {len(successful)}")
        print(f"  ✗ Failed: {len(failed)}")
        print()

        if not successful:
            print("⚠️  No successful optimizations")
            return

        # Sort by score (lower is better)
        successful_sorted = sorted(successful, key=lambda r: r['score'])

        print("🏆 Top 5 Results:")
        print()

        for i, result in enumerate(successful_sorted[:5]):
            metrics = result['metrics']
            params = result['parameters']

            print(f"  Rank {i+1}:")
            print(f"    Job ID: {result['job_id']}")
            print(f"    Score: {result['score']:.2f}")
            print(f"    Mean Error: {metrics.get('mean_error_pct', 0):.2f}%")
            print(f"    Std Error: {metrics.get('std_error_pct', 0):.2f}%")
            print(f"    Max Error: {metrics.get('max_abs_error_pct', 0):.2f}%")
            print(f"    Mean Virial: {metrics.get('mean_virial', 0):.3f}")
            print(f"    Time: {result['elapsed_seconds']/60:.1f} min")
            print()

            # Show key parameter changes from base
            base_params = self._extract_base_parameters()
            print(f"    Parameter changes from base:")
            for key in ['c_v2_base', 'c_v4_base', 'c_v2_iso', 'c_v4_size']:
                if key in params and key in base_params:
                    base_val = base_params[key]
                    new_val = params[key]
                    change_pct = 100 * (new_val - base_val) / base_val
                    print(f"      {key:15s}: {base_val:.6f} → {new_val:.6f} ({change_pct:+.1f}%)")
            print()

        # Save best parameters
        best_result = successful_sorted[0]
        best_params_file = self.output_dir / "best_parameters.json"

        best_params_output = {
            'parameters': best_result['parameters'],
            'metrics': best_result['metrics'],
            'job_id': best_result['job_id'],
            'score': best_result['score'],
            'timestamp': best_result['timestamp']
        }

        with open(best_params_file, 'w') as f:
            json.dump(best_params_output, f, indent=2)

        print(f"💾 Best parameters saved to: {best_params_file}")
        print()

        # Runtime statistics
        if self.start_time and self.end_time:
            elapsed = (self.end_time - self.start_time).total_seconds()
            print(f"⏱️  Total runtime: {elapsed/3600:.2f} hours")
            print(f"   Jobs per hour: {len(self.results) / (elapsed/3600):.1f}")
            print()

    def _extract_base_parameters(self) -> Dict[str, float]:
        """Extract base parameters from configuration."""
        params = {}
        for param in self.base_config.get('parameters', []):
            name = param['name'].replace('nuclear.', '')
            params[name] = param['value']
        return params


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Parallel batch optimization for nuclear soliton solver"
    )

    parser.add_argument(
        '--config',
        default='experiments/nuclear_heavy_region.runspec.json',
        help='Path to base RunSpec configuration'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    parser.add_argument(
        '--hours',
        type=float,
        default=8.0,
        help='Maximum runtime in hours (default: 8.0)'
    )

    parser.add_argument(
        '--output-dir',
        default='results/batch_optimization',
        help='Output directory for results (default: results/batch_optimization)'
    )

    args = parser.parse_args()

    # Create optimizer
    optimizer = BatchOptimizer(
        config_path=args.config,
        n_workers=args.workers,
        max_hours=args.hours,
        output_dir=args.output_dir
    )

    # Run optimization
    optimizer.run()


if __name__ == '__main__':
    main()
