#!/usr/bin/env python3
"""
Comprehensive replication test to match paper's 1.107 Z result
"""
import subprocess
import json
import pandas as pd
from pathlib import Path

DATA_PATH = "/home/tracy/development/QFD_SpectralGap/projects/particle-physics/nuclide-prediction/NuMass.csv"
RESULTS_DIR = Path("replication_tests")
RESULTS_DIR.mkdir(exist_ok=True)

results = []

print("="*80)
print("COMPREHENSIVE REPLICATION TEST")
print("="*80)
print(f"Target: RMSE_soft = 1.107 Z (paper claim)")
print(f"Data: {DATA_PATH}")
print("="*80)

# Test 1: Base case (what we just ran)
print("\n[1/6] Base case (K=3, no extras, default seed)")
cmd = f"python mixture_core_compression.py --csv {DATA_PATH} --out {RESULTS_DIR}/test1_base --K 3"
subprocess.run(cmd, shell=True, capture_output=True)
metrics = pd.read_csv(RESULTS_DIR / "test1_base" / "metrics_summary.csv")
rmse_soft = metrics.loc[metrics['Model']=='A-only', 'RMSE_soft'].values[0]
rmse_hard = metrics.loc[metrics['Model']=='A-only', 'RMSE_hard'].values[0]
print(f"   RMSE_soft = {rmse_soft:.4f} Z, RMSE_hard = {rmse_hard:.4f} Z")
results.append({'test': 'Base (K=3)', 'rmse_soft': rmse_soft, 'rmse_hard': rmse_hard, 'gap': abs(rmse_soft - 1.107)})

# Test 2: With pairing term
print("\n[2/6] With pairing correction")
cmd = f"python mixture_core_compression.py --csv {DATA_PATH} --out {RESULTS_DIR}/test2_pair --K 3 --with-pair"
subprocess.run(cmd, shell=True, capture_output=True)
metrics = pd.read_csv(RESULTS_DIR / "test2_pair" / "metrics_summary.csv")
rmse_soft = metrics.loc[metrics['Model']=='Augmented', 'RMSE_soft'].values[0]
rmse_hard = metrics.loc[metrics['Model']=='Augmented', 'RMSE_hard'].values[0]
print(f"   RMSE_soft = {rmse_soft:.4f} Z, RMSE_hard = {rmse_hard:.4f} Z")
results.append({'test': 'Pairing (K=3)', 'rmse_soft': rmse_soft, 'rmse_hard': rmse_hard, 'gap': abs(rmse_soft - 1.107)})

# Test 3: K=2 (simpler model)
print("\n[3/6] K=2 (two components)")
cmd = f"python mixture_core_compression.py --csv {DATA_PATH} --out {RESULTS_DIR}/test3_k2 --K 2"
subprocess.run(cmd, shell=True, capture_output=True)
metrics = pd.read_csv(RESULTS_DIR / "test3_k2" / "metrics_summary.csv")
rmse_soft = metrics.loc[metrics['Model']=='A-only', 'RMSE_soft'].values[0]
rmse_hard = metrics.loc[metrics['Model']=='A-only', 'RMSE_hard'].values[0]
print(f"   RMSE_soft = {rmse_soft:.4f} Z, RMSE_hard = {rmse_hard:.4f} Z")
results.append({'test': 'K=2', 'rmse_soft': rmse_soft, 'rmse_hard': rmse_hard, 'gap': abs(rmse_soft - 1.107)})

# Test 4: K=4 (more components)
print("\n[4/6] K=4 (four components)")
cmd = f"python mixture_core_compression.py --csv {DATA_PATH} --out {RESULTS_DIR}/test4_k4 --K 4"
subprocess.run(cmd, shell=True, capture_output=True)
metrics = pd.read_csv(RESULTS_DIR / "test4_k4" / "metrics_summary.csv")
rmse_soft = metrics.loc[metrics['Model']=='A-only', 'RMSE_soft'].values[0]
rmse_hard = metrics.loc[metrics['Model']=='A-only', 'RMSE_hard'].values[0]
print(f"   RMSE_soft = {rmse_soft:.4f} Z, RMSE_hard = {rmse_hard:.4f} Z")
results.append({'test': 'K=4', 'rmse_soft': rmse_soft, 'rmse_hard': rmse_hard, 'gap': abs(rmse_soft - 1.107)})

# Test 5: K=5
print("\n[5/6] K=5 (five components)")
cmd = f"python mixture_core_compression.py --csv {DATA_PATH} --out {RESULTS_DIR}/test5_k5 --K 5"
subprocess.run(cmd, shell=True, capture_output=True)
metrics = pd.read_csv(RESULTS_DIR / "test5_k5" / "metrics_summary.csv")
rmse_soft = metrics.loc[metrics['Model']=='A-only', 'RMSE_soft'].values[0]
rmse_hard = metrics.loc[metrics['Model']=='A-only', 'RMSE_hard'].values[0]
print(f"   RMSE_soft = {rmse_soft:.4f} Z, RMSE_hard = {rmse_hard:.4f} Z")
results.append({'test': 'K=5', 'rmse_soft': rmse_soft, 'rmse_hard': rmse_hard, 'gap': abs(rmse_soft - 1.107)})

# Test 6: Expert Model verification
print("\n[6/6] Expert Model (best 2400)")
cmd = f"python experiment_best2400_clean90.py --csv {DATA_PATH} --out {RESULTS_DIR}/test6_expert --K 3"
subprocess.run(cmd, shell=True, capture_output=True)
metrics = pd.read_csv(RESULTS_DIR / "test6_expert" / "metrics_summary.csv")
rmse_train = metrics.loc[metrics['split']=='train2400', 'RMSE'].values[0]
rmse_holdout = metrics.loc[metrics['split']=='holdout_all', 'RMSE'].values[0]
rmse_clean90 = metrics.loc[metrics['split']=='holdout_clean90', 'RMSE'].values[0]
print(f"   Train (2400) = {rmse_train:.4f} Z")
print(f"   Holdout (all) = {rmse_holdout:.4f} Z")
print(f"   Holdout (clean90) = {rmse_clean90:.4f} Z")
results.append({'test': 'Expert Train', 'rmse_soft': rmse_train, 'rmse_hard': rmse_train, 'gap': abs(rmse_train - 0.5225)})
results.append({'test': 'Expert Holdout', 'rmse_soft': rmse_holdout, 'rmse_hard': rmse_holdout, 'gap': abs(rmse_holdout - 1.8069)})

# Summary
print("\n" + "="*80)
print("REPLICATION SUMMARY")
print("="*80)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('gap')

print("\nRanked by closeness to paper (Global Model target: 1.107 Z):")
for i, row in df_results.head(8).iterrows():
    status = "✅ EXACT" if row['gap'] < 0.001 else ("✓ Close" if row['gap'] < 0.05 else "")
    print(f"  {row['test']:20s}: RMSE_soft={row['rmse_soft']:.4f} Z  (gap={row['gap']:.4f} Z) {status}")

# Save summary
df_results.to_csv(RESULTS_DIR / "summary.csv", index=False)
print(f"\n✅ Full results saved to {RESULTS_DIR}/summary.csv")

# Best match
best = df_results.iloc[0]
if 'Global' in best['test'] or best['test'].startswith('K=') or 'Base' in best['test'] or 'Pairing' in best['test']:
    print(f"\n🎯 Best Global Model match: {best['test']} with RMSE_soft = {best['rmse_soft']:.4f} Z")
    print(f"   Gap to paper (1.107 Z): {best['gap']:.4f} Z ({100*best['gap']/1.107:.1f}%)")
