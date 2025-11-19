#!/usr/bin/env python3
"""
Parallel Runner for Stage 1 V20 with WIDENED STRETCH BOUNDS

V20 CRITICAL FIX: Stretch bounds widened from [0.3, 3.0] to [0.3, 10.0]
This resolves the 60% railing issue identified in Vall forensics.
"""
import argparse
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys

# Add stages and core directories to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from stage1_v20_fullscale import run_single_sn_optimization
from v17_data import LightcurveLoader

def check_memory_safety(batch_size, num_cores):
    """
    Pre-flight memory check to warn about potential OOM issues.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)

        print(f"\n{'='*80}")
        print("MEMORY SAFETY CHECK")
        print(f"{'='*80}")
        print(f"Total RAM: {total_gb:.2f} GB")
        print(f"Available RAM: {available_gb:.2f} GB ({mem.percent:.1f}% used)")
        print(f"Batch size: {batch_size} SNe")
        print(f"Worker cores: {num_cores}")

        # Estimate memory usage per batch
        # Rough estimate: ~50KB per observation, ~100 obs per SN = 5MB per SN
        # Plus JAX overhead per worker: ~200MB per core
        estimated_batch_mb = batch_size * 5 + num_cores * 200
        estimated_batch_gb = estimated_batch_mb / 1024

        print(f"\nEstimated peak memory per batch: ~{estimated_batch_gb:.2f} GB")

        if estimated_batch_gb > available_gb * 0.8:
            print(f"\n⚠️  WARNING: Estimated memory usage ({estimated_batch_gb:.2f} GB)")
            print(f"   exceeds 80% of available RAM ({available_gb:.2f} GB)!")
            print(f"\n   RECOMMENDATIONS:")
            print(f"   1. Reduce batch size: --batch-size {batch_size // 2}")
            print(f"   2. Reduce cores: --ncores {max(1, num_cores // 2)}")
            print(f"   3. Close other applications to free memory")
            print(f"\n   Continue anyway? (Ctrl+C to cancel)")
            import time
            time.sleep(5)
        elif estimated_batch_gb > available_gb * 0.6:
            print(f"\n⚠️  CAUTION: Memory usage will be ~{100*estimated_batch_gb/available_gb:.0f}% of available RAM")
            print(f"   Should be safe, but monitor for slowdowns")
        else:
            print(f"\n✓ Memory usage looks safe ({100*estimated_batch_gb/available_gb:.0f}% of available)")

        print(f"{'='*80}\n")

    except ImportError:
        print("\n⚠️  psutil not available, skipping memory check")
        print("   Install with: pip install psutil\n")

def main():
    parser = argparse.ArgumentParser(description="Stage 1 V20 Fullscale Runner")
    parser.add_argument('--lightcurves', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--ncores', type=int, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Number of SNe to process per batch (default: 500)')
    parser.add_argument('--save-incremental', action='store_true')
    args = parser.parse_args()

    print("="*80)
    print("V20 STAGE 1: FULLSCALE RUN WITH WIDENED STRETCH BOUNDS")
    print("="*80)
    print("\nCRITICAL FIX: Stretch bounds [0.3, 10.0] (was [0.3, 3.0])\n")

    # Auto-detect cores or use specified
    if args.ncores:
        num_cores = args.ncores
    else:
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            # MEMORY FIX: More conservative - 1 core per 2GB RAM (was 1.5GB)
            # Also cap at 4 cores to prevent memory pressure from JAX workers
            memory_limited_cores = max(1, int(available_gb / 2.0))
            num_cores = min(cpu_count(), memory_limited_cores, 4)
            print(f"Available memory: {available_gb:.1f} GB -> using {num_cores} cores (capped at 4)")
        except:
            num_cores = 2  # Conservative default (was 4)

    print(f"Using {num_cores} CPU cores\n")
    print(f"Batch size: {args.batch_size} SNe per batch\n")

    # Pre-flight memory safety check
    check_memory_safety(args.batch_size, num_cores)

    # MEMORY OPTIMIZATION: Get list of SNIDs without loading full data
    print(f"Reading SNID list from: {args.lightcurves}")
    loader = LightcurveLoader(Path(args.lightcurves))
    all_snids = loader.get_snid_list()

    if args.limit:
        all_snids = all_snids[:args.limit]
        print(f"  Limited to {len(all_snids)} SNe")

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check existing results to skip already-fitted SNe
    existing_snids = set()
    if args.save_incremental:
        existing_snids = {f.stem for f in output_dir.glob("*.json")}
        if existing_snids:
            print(f"  Found {len(existing_snids)} existing results")

    # Filter out already-fitted SNe
    snids_to_fit = [snid for snid in all_snids if snid not in existing_snids]
    print(f"  Remaining to fit: {len(snids_to_fit)} SNe")

    # Calculate number of batches
    num_batches = (len(snids_to_fit) + args.batch_size - 1) // args.batch_size
    print(f"\nProcessing in {num_batches} batches of {args.batch_size} SNe each\n")

    # Global parameters for all fits
    global_params = {'eta_prime': 0.0}
    config = {'max_iter': 1000, 'ftol': 1e-6, 'nu': 5.0}

    # Process in batches
    total_fitted = 0
    for batch_idx in range(num_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(snids_to_fit))
        batch_snids = snids_to_fit[batch_start:batch_end]

        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx + 1}/{num_batches}: Processing SNe {batch_start}-{batch_end-1}")
        print(f"{'='*80}")

        # Load only this batch of data
        print(f"Loading batch data...")
        try:
            import psutil
            mem_before = psutil.virtual_memory().available / (1024**3)
            print(f"  Memory available before load: {mem_before:.2f} GB")
        except:
            pass

        batch_lightcurves = loader.load_batch(
            snid_list=snids_to_fit,
            batch_size=args.batch_size,
            batch_index=batch_idx
        )

        try:
            import psutil
            mem_after = psutil.virtual_memory().available / (1024**3)
            print(f"  Memory available after load: {mem_after:.2f} GB (used {mem_before - mem_after:.2f} GB)")
        except:
            pass

        # Convert to Photometry objects
        batch_photometry = {
            snid: lc.to_photometry()
            for snid, lc in batch_lightcurves.items()
        }

        # Create tasks for this batch
        tasks = [
            {
                'snid': snid,
                'photometry': phot,
                'global_params': global_params,
                'config': config
            }
            for snid, phot in batch_photometry.items()
        ]

        print(f"Fitting {len(tasks)} SNe in parallel...")

        # Fit this batch
        with Pool(processes=num_cores) as pool:
            for result in tqdm(
                pool.imap_unordered(run_single_sn_optimization, tasks),
                total=len(tasks),
                desc=f"Batch {batch_idx + 1}/{num_batches}",
                unit="SN"
            ):
                output_file = output_dir / f"{result['snid']}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                total_fitted += 1

        # Clean up batch data to free memory
        del batch_lightcurves
        del batch_photometry
        del tasks

        # Force garbage collection
        import gc
        gc.collect()

        print(f"Batch {batch_idx + 1} complete. Total fitted so far: {total_fitted}/{len(snids_to_fit)}")

    # Final summary
    print("\n" + "="*80)
    print("ALL BATCHES COMPLETE - Computing final summary...")
    print("="*80)

    results = [json.load(open(f)) for f in output_dir.glob("*.json")]
    successful = [r for r in results if r.get('success', False)]

    print(f"\nTotal: {len(results)}, Successful: {len(successful)} ({100*len(successful)/len(results):.1f}%)")

    if successful:
        import numpy as np
        stretches = [r['best_fit_params']['stretch'] for r in successful]
        print(f"\nStretch Statistics:")
        print(f"  Mean: {np.mean(stretches):.3f}")
        print(f"  Median: {np.median(stretches):.3f}")
        print(f"  Range: [{np.min(stretches):.3f}, {np.max(stretches):.3f}]")
        print(f"  Railing: <0.35: {np.sum(np.array(stretches)<0.35)}, >9.5: {np.sum(np.array(stretches)>9.5)}")

    print(f"\nResults saved to: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
