#!/usr/bin/env python3
"""
generate_viewer.py — Build interactive nuclide species heatmap viewer.

Fits per-species ridge regression models on the tracked population,
computes per-nuclide predictions, and generates a self-contained HTML
viewer with:
  - Nuclear chart (N vs Z) colored by species or residual
  - Predicted vs observed half-life scatter plot
  - Species toggles with live R²/RMSE statistics

Usage: python generate_viewer.py
Output: results/nuclide_viewer.html
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# ── Path setup ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.environ.get('TLG_DATA_DIR', os.path.join(_ROOT_DIR, 'data'))
RESULTS_DIR = os.environ.get('TLG_RESULTS_DIR', os.path.join(_ROOT_DIR, 'results'))
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(_ROOT_DIR, '..', '..', '..'))
from qfd.shared_constants import BETA as _BETA_IMPORT

# ── Constants ──
BETA = _BETA_IMPORT
N_MAX = 2 * np.pi * BETA**3
A_CRIT = 2 * np.e**2 * BETA**2
WIDTH = 2 * np.pi * BETA**2

# ── Load data ──
print("Loading data...")
cs = pd.read_csv(os.path.join(DATA_DIR, 'clean_species_sorted.csv'))
cs['N'] = cs['A'] - cs['Z']
print(f"  {len(cs)} nuclides loaded")

# ── Feature functions ──
def f_sqrt_eps(df): return np.sqrt(np.abs(df['epsilon'].values))
def f_abs_eps(df): return np.abs(df['epsilon'].values)
def f_logZ(df): return np.log10(df['Z'].values.astype(float))
def f_Z(df): return df['Z'].values.astype(float)
def f_N_NMAX(df): return df['N'].values / N_MAX
def f_N_Z(df): return df['N'].values / df['Z'].values.astype(float)
def f_lnA(df): return np.log(df['A'].values.astype(float))
def f_ee(df): return (df['parity'] == 'ee').astype(float).values
def f_oo(df): return (df['parity'] == 'oo').astype(float).values
def f_logQ(df): return np.log10(np.maximum(df['Q_keV'].values.astype(float), 1.0))
def f_inv_sqrtQ(df):
    Q = df['Q_keV'].values.astype(float) / 1000
    return 1.0 / np.sqrt(np.maximum(Q, 0.01))
def f_log_pen(df):
    Q = df['Q_keV'].values.astype(float)
    V = df['V_coulomb_keV'].values.astype(float)
    return np.log10(np.maximum(Q, 1.0) / np.maximum(V, 1.0))
def f_deficit(df):
    return (df['V_coulomb_keV'].values - df['Q_keV'].values) / 1000
def f_log_trans_E(df):
    E = df['transition_energy_keV'].values.astype(float)
    return np.log10(np.maximum(E, 1.0))
def f_lambda(df):
    med = df['correct_lambda'].median()
    if pd.isna(med): med = 3.0
    return df['correct_lambda'].fillna(med).values
def f_lambda_sq(df):
    med = df['correct_lambda'].median()
    if pd.isna(med): med = 3.0
    return df['correct_lambda'].fillna(med).values ** 2

# ── Species configurations (same as tracked_channel_fits.py) ──
species_configs = {
    'beta-': {
        'features': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                     ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 1.0
    },
    'beta+': {
        'features': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ), ('Z', f_Z),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z), ('lnA', f_lnA),
                     ('logQ', f_logQ), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 1.0
    },
    'alpha': {
        'features': [('abs_eps', f_abs_eps), ('logZ', f_logZ),
                     ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                     ('1/sqrtQ', f_inv_sqrtQ), ('log_pen', f_log_pen),
                     ('deficit', f_deficit), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 5.0
    },
    'IT': {
        'features': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
                     ('lambda_sq', f_lambda_sq), ('sqrt_eps', f_sqrt_eps),
                     ('logZ', f_logZ), ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo)],
        'ridge': 2.0
    },
    'IT_platypus': {
        'features': [('log_transE', f_log_trans_E), ('lambda', f_lambda),
                     ('lambda_sq', f_lambda_sq), ('sqrt_eps', f_sqrt_eps),
                     ('logZ', f_logZ), ('N_NMAX', f_N_NMAX), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo)],
        'ridge': 2.0
    },
    'SF': {
        'features': [('N_NMAX', f_N_NMAX), ('abs_eps', f_abs_eps),
                     ('logZ', f_logZ), ('N_Z', f_N_Z),
                     ('ee', f_ee), ('oo', f_oo)],
        'ridge': 5.0
    },
    'proton': {
        'features': [('sqrt_eps', f_sqrt_eps), ('logZ', f_logZ),
                     ('N_NMAX', f_N_NMAX), ('ee', f_ee), ('oo', f_oo)],
        'ridge': 5.0
    },
}

# ── Tracking bins ──
has_hl = cs['log_hl'].notna()
is_stable = cs['clean_species'] == 'stable'
is_rare = cs['clean_species'] == 'rare'
radioactive = has_hl & ~is_stable & ~is_rare
ephemeral = radioactive & (cs['log_hl'] < -6)
far_exotic = radioactive & (np.abs(cs['epsilon']) > 8)
higher_iso_short = radioactive & (cs['az_order'] >= 2) & (cs['log_hl'] < -3)
neutron_sp = radioactive & (cs['clean_species'] == 'neutron')
suspect = (far_exotic | higher_iso_short | neutron_sp) & ~ephemeral
tracked = radioactive & ~ephemeral & ~suspect

# ── Fit models and compute predictions ──
print("Fitting models...")
cs['log_hl_pred'] = np.nan
cs['residual'] = np.nan
species_stats = {}

ALL_SPECIES = ['beta-', 'beta+', 'alpha', 'IT', 'IT_platypus', 'SF',
               'proton', 'beta-_iso', 'beta+_iso', 'alpha_iso',
               'SF_iso', 'proton_iso', 'neutron', 'rare']

for sp in ALL_SPECIES:
    sp_mask = (cs['clean_species'] == sp) & tracked
    sp_data = cs[sp_mask].copy()
    if len(sp_data) < 8:
        continue

    base_sp = sp.replace('_iso', '')
    config = species_configs.get(base_sp, species_configs['beta-'])
    features = config['features']
    ridge_alpha = config['ridge']

    y = sp_data['log_hl'].values
    X_parts = [func(sp_data) for _, func in features]
    X_parts.append(np.ones(len(sp_data)))
    X = np.column_stack(X_parts)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    n = valid.sum()
    if n < len(features) + 2:
        continue

    X_v, y_v = X[valid], y[valid]
    I_mat = np.eye(X_v.shape[1])
    I_mat[-1, -1] = 0
    try:
        coef = np.linalg.solve(X_v.T @ X_v + ridge_alpha * I_mat, X_v.T @ y_v)
    except np.linalg.LinAlgError:
        coef, _, _, _ = np.linalg.lstsq(X_v, y_v, rcond=None)

    # Predict for ALL nuclides of this species
    all_sp_mask = cs['clean_species'] == sp
    all_sp = cs[all_sp_mask]
    X_all_parts = [func(all_sp) for _, func in features]
    X_all_parts.append(np.ones(len(all_sp)))
    X_all = np.column_stack(X_all_parts)
    all_valid = np.isfinite(X_all).all(axis=1)
    pred_all = X_all @ coef
    pred_all[~all_valid] = np.nan
    cs.loc[all_sp_mask, 'log_hl_pred'] = pred_all

    obs = cs.loc[all_sp_mask, 'log_hl'].values
    cs.loc[all_sp_mask, 'residual'] = obs - pred_all

    # Stats on tracked subset
    pred_v = X_v @ coef
    resid_v = y_v - pred_v
    ss_res = np.sum(resid_v**2)
    ss_tot = np.sum((y_v - y_v.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse_val = np.sqrt(np.mean(resid_v**2))
    solved = int(np.sum(np.abs(resid_v) < 1.0))

    # Coverage: tracked / total for this species
    total_sp = int((cs['clean_species'] == sp).sum())
    tracked_sp = int(sp_mask.sum())

    species_stats[sp] = {
        'n': int(n), 'r2': round(float(r2), 4),
        'rmse': round(float(rmse_val), 3),
        'solved': solved, 'pct': round(100 * solved / n, 1),
        'total': total_sp, 'tracked': tracked_sp,
        'coverage': round(100 * tracked_sp / total_sp, 1) if total_sp > 0 else 0,
    }
    print(f"  {sp:15s}  n={n:5d}  R2={r2:.3f}  RMSE={rmse_val:.2f}  <1dec={solved}/{n} ({100*solved/n:.1f}%)  coverage={tracked_sp}/{total_sp} ({100*tracked_sp/total_sp:.1f}%)")

# Add stable to stats (100% coverage, no half-life prediction)
n_stable = int(is_stable.sum())
species_stats['stable'] = {
    'n': n_stable, 'r2': None, 'rmse': None,
    'solved': n_stable, 'pct': 100.0,
    'total': n_stable, 'tracked': n_stable, 'coverage': 100.0,
}
print(f"  {'stable':15s}  n={n_stable:5d}  coverage=100.0%  (all found)")

# Add coverage-only stats for species that weren't fitted
for sp in ALL_SPECIES:
    if sp in species_stats:
        continue
    total_sp = int((cs['clean_species'] == sp).sum())
    sp_mask_all = (cs['clean_species'] == sp) & tracked
    tracked_sp = int(sp_mask_all.sum())
    species_stats[sp] = {
        'n': 0, 'r2': None, 'rmse': None,
        'solved': 0, 'pct': None,
        'total': total_sp, 'tracked': tracked_sp,
        'coverage': round(100 * tracked_sp / total_sp, 1) if total_sp > 0 else 0,
    }
    print(f"  {sp:15s}  total={total_sp:5d}  tracked={tracked_sp}  (no model, coverage only)")

# ── Build JSON ──
print("Building JSON data...")
nuclides = []
for _, row in cs.iterrows():
    d = {
        'A': int(row['A']), 'Z': int(row['Z']), 'N': int(row['N']),
        'el': str(row['element']), 'sp': str(row['clean_species']),
        'par': str(row['parity']), 'bin': str(row['tracking_bin']),
    }
    if pd.notna(row.get('epsilon')): d['eps'] = round(float(row['epsilon']), 2)
    if pd.notna(row.get('log_hl')): d['hl'] = round(float(row['log_hl']), 2)
    if pd.notna(row.get('log_hl_pred')): d['pred'] = round(float(row['log_hl_pred']), 2)
    if pd.notna(row.get('residual')): d['res'] = round(float(row['residual']), 2)
    nuclides.append(d)

data_json = json.dumps(nuclides, separators=(',', ':'))
stats_json = json.dumps(species_stats, separators=(',', ':'))
print(f"  {len(nuclides)} nuclides, {len(data_json)//1024} KB JSON")

# ══════════════════════════════════════════════════════════════════════
# HTML TEMPLATE — assembled in parts to avoid f-string brace conflicts
# ══════════════════════════════════════════════════════════════════════

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Three-Layer LaGrangian — Species Heatmap Viewer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root {
  --bg: #050608; --panel: #0d1117; --border: #30363d;
  --text: #c9d1d9; --muted: #8b949e; --mono: 'SF Mono','Courier New',monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  display: grid; grid-template-columns: 300px 1fr;
  height: 100vh; overflow: hidden;
}
.sidebar {
  background: var(--panel); border-right: 1px solid var(--border);
  padding: 14px; display: flex; flex-direction: column; gap: 10px;
  overflow-y: auto;
}
.sidebar h1 {
  font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px;
  color: white; border-bottom: 1px solid #333; padding-bottom: 8px;
}
.ctrl {
  background: rgba(255,255,255,0.03); padding: 10px;
  border-radius: 6px; border: 1px solid var(--border);
}
.ctrl-label {
  font-size: 0.7em; color: var(--muted); text-transform: uppercase;
  margin-bottom: 6px; display: flex; justify-content: space-between;
}
.sp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 3px; }
.sp-item {
  display: flex; align-items: center; gap: 5px; font-size: 0.78em;
  padding: 3px 5px; border-radius: 4px; cursor: pointer;
  transition: background 0.15s;
}
.sp-item:hover { background: rgba(255,255,255,0.05); }
.sp-item input { margin: 0; cursor: pointer; }
.sp-swatch {
  width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
}
.sp-count { color: var(--muted); font-size: 0.85em; margin-left: auto; }
select {
  width: 100%; background: #0b0f14; color: var(--text);
  border: 1px solid #2a2f36; border-radius: 5px; padding: 5px;
  font-size: 0.82em;
}
.stat-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
}
.stat-box .stat-label {
  font-size: 0.65em; color: var(--muted); text-transform: uppercase;
}
.stat-box .stat-val {
  font-size: 1.15em; font-weight: bold; font-family: var(--mono);
}
.stat-table {
  width: 100%; font-size: 0.72em; border-collapse: collapse; margin-top: 6px;
}
.stat-table td { padding: 2px 4px; border-bottom: 1px solid #1a1f26; }
.stat-table .sp-name { color: #aaa; }
.stat-table .num { text-align: right; font-family: var(--mono); }

.main {
  display: grid; grid-template-rows: 58% 42%; gap: 10px; padding: 12px;
  height: 100vh; overflow: hidden;
}
.panel {
  background: var(--panel); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px; position: relative;
  display: flex; flex-direction: column; min-height: 0;
}
.panel-title {
  font-size: 0.72em; color: var(--muted); text-transform: uppercase;
  font-weight: bold; margin-bottom: 4px;
  display: flex; justify-content: space-between;
}
.panel-title .pill {
  display: inline-block; padding: 1px 7px; border-radius: 999px;
  font-size: 0.9em; border: 1px solid rgba(46,160,67,0.25);
  background: rgba(46,160,67,0.1); color: #2ea043;
}
.canvas-wrap { flex: 1; position: relative; min-height: 0; }
.canvas-wrap canvas { display: block; width: 100%; height: 100%; }
#tooltip {
  display: none; position: fixed; z-index: 100;
  background: #1c2128; border: 1px solid #444; border-radius: 6px;
  padding: 8px 10px; font-size: 0.78em; line-height: 1.5;
  pointer-events: none; max-width: 280px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
.legend-row {
  display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px;
}
.legend-chip {
  display: flex; align-items: center; gap: 3px;
  font-size: 0.65em; color: var(--muted); padding: 1px 5px;
  border: 1px solid #222; border-radius: 3px;
}
.legend-dot { width: 7px; height: 7px; border-radius: 2px; }
.btn-row { display: flex; gap: 4px; margin-top: 4px; }
.btn-sm {
  font-size: 0.7em; padding: 2px 8px; border-radius: 4px;
  border: 1px solid var(--border); background: rgba(255,255,255,0.04);
  color: var(--muted); cursor: pointer;
}
.btn-sm:hover { background: rgba(255,255,255,0.08); color: white; }
</style>
</head>
<body>

<div class="sidebar">
  <h1>Three-Layer LaGrangian</h1>

  <div class="ctrl">
    <div class="ctrl-label"><span>Decay Species</span></div>
    <div class="sp-grid" id="speciesGrid"></div>
    <div class="btn-row">
      <button class="btn-sm" onclick="toggleAll(true)">All</button>
      <button class="btn-sm" onclick="toggleAll(false)">None</button>
      <button class="btn-sm" onclick="toggleGroup('ground')">Ground</button>
      <button class="btn-sm" onclick="toggleGroup('iso')">Isomers</button>
    </div>
  </div>

  <div class="ctrl">
    <div class="ctrl-label"><span>Color Mode</span></div>
    <select id="colorMode">
      <option value="species" selected>By Species</option>
      <option value="residual">By Residual (|pred - obs|)</option>
      <option value="halflife">By Half-Life</option>
      <option value="parity">By Parity (ee/eo/oe/oo)</option>
      <option value="stress">By Stress |epsilon|</option>
    </select>
    <div class="legend-row" id="legendRow"></div>
  </div>

  <div class="ctrl" style="border-left: 3px solid #2ea043;">
    <div class="ctrl-label"><span>Statistics (visible)</span></div>
    <div class="stat-grid">
      <div class="stat-box">
        <div class="stat-label">Nuclides</div>
        <div class="stat-val" id="statN">--</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Found / Tracked</div>
        <div class="stat-val" style="color:#f1c40f" id="statFound">--</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Coverage</div>
        <div class="stat-val" style="color:#f1c40f" id="statCoverage">--</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">R² (weighted)</div>
        <div class="stat-val" style="color:#2ea043" id="statR2">--</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">RMSE (decades)</div>
        <div class="stat-val" id="statRMSE">--</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">&lt;1 decade (%)</div>
        <div class="stat-val" id="statSolved">--</div>
      </div>
    </div>
    <table class="stat-table" id="speciesTable"></table>
  </div>

  <div class="ctrl">
    <div class="ctrl-label"><span>About</span></div>
    <div style="font-size:0.72em; color:#666; line-height:1.5;">
      McSheery (2026). Three-Layer LaGrangian for Nuclear Half-Lives.
      Data: NUBASE2020 + AME2020. Ridge-regularized OLS, ~10 params/species.
    </div>
  </div>
</div>

<div class="main">
  <div class="panel">
    <div class="panel-title">
      <span>Chart of Nuclides (N vs Z)</span>
      <span class="pill" id="chartCount">--</span>
    </div>
    <div class="canvas-wrap">
      <canvas id="nuclearChart"></canvas>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">
      <span>Predicted vs Observed log&#8321;&#8320;(T&#189;)</span>
      <span class="pill" id="scatterCount">--</span>
    </div>
    <div class="canvas-wrap">
      <canvas id="scatterChart"></canvas>
    </div>
  </div>
</div>

<div id="tooltip"></div>
"""

HTML_JS = r"""
// ══════════════════════════════════════════════
// Species configuration
// ══════════════════════════════════════════════
const SP = {
  'stable':      {c:'#6b7280', label:'Stable',   grp:'ground'},
  'beta-':       {c:'#3b82f6', label:'\u03b2\u207b',      grp:'ground'},
  'beta+':       {c:'#ef4444', label:'\u03b2\u207a',      grp:'ground'},
  'alpha':       {c:'#22c55e', label:'\u03b1',        grp:'ground'},
  'IT':          {c:'#eab308', label:'IT',        grp:'ground'},
  'IT_platypus': {c:'#f97316', label:'IT plat',   grp:'ground'},
  'SF':          {c:'#a855f7', label:'SF',        grp:'ground'},
  'proton':      {c:'#06b6d4', label:'Proton',    grp:'ground'},
  'neutron':     {c:'#d946ef', label:'Neutron',   grp:'ground'},
  'beta-_iso':   {c:'#60a5fa', label:'\u03b2\u207b iso', grp:'iso'},
  'beta+_iso':   {c:'#f87171', label:'\u03b2\u207a iso', grp:'iso'},
  'alpha_iso':   {c:'#4ade80', label:'\u03b1 iso',   grp:'iso'},
  'SF_iso':      {c:'#c084fc', label:'SF iso',    grp:'iso'},
  'proton_iso':  {c:'#67e8f9', label:'p iso',     grp:'iso'},
  'rare':        {c:'#374151', label:'Rare',      grp:'other'},
};

const SP_ORDER = ['stable','beta-','beta+','alpha','IT','IT_platypus','SF',
                  'proton','neutron','beta-_iso','beta+_iso','alpha_iso',
                  'SF_iso','proton_iso','rare'];

// ══════════════════════════════════════════════
// Global state
// ══════════════════════════════════════════════
let activeSpecies = new Set(SP_ORDER);
let colorMode = 'species';

// Pre-compute indices per species
const spIdx = {};
DATA.forEach((d,i) => {
  if (!spIdx[d.sp]) spIdx[d.sp] = [];
  spIdx[d.sp].push(i);
});

// Species counts
const spCounts = {};
SP_ORDER.forEach(sp => { spCounts[sp] = (spIdx[sp]||[]).length; });

// Grid for hover: key = N*1000+Z -> array of indices
const grid = {};
DATA.forEach((d,i) => {
  const k = d.N*1000+d.Z;
  if (!grid[k]) grid[k] = [];
  grid[k].push(i);
});

// Data ranges
let maxN = 0, maxZ = 0;
DATA.forEach(d => { if(d.N>maxN) maxN=d.N; if(d.Z>maxZ) maxZ=d.Z; });
maxN += 2; maxZ += 2;

// ══════════════════════════════════════════════
// Build sidebar species grid
// ══════════════════════════════════════════════
function buildSpeciesGrid() {
  const el = document.getElementById('speciesGrid');
  let html = '';
  SP_ORDER.forEach(sp => {
    const cfg = SP[sp];
    const n = spCounts[sp] || 0;
    html += `<label class="sp-item">
      <input type="checkbox" data-sp="${sp}" checked onchange="onToggle()">
      <span class="sp-swatch" style="background:${cfg.c}"></span>
      ${cfg.label}
      <span class="sp-count">${n}</span>
    </label>`;
  });
  el.innerHTML = html;
}

function onToggle() {
  activeSpecies.clear();
  document.querySelectorAll('#speciesGrid input').forEach(cb => {
    if (cb.checked) activeSpecies.add(cb.dataset.sp);
  });
  updateAll();
}

function toggleAll(on) {
  document.querySelectorAll('#speciesGrid input').forEach(cb => { cb.checked = on; });
  onToggle();
}

function toggleGroup(grp) {
  document.querySelectorAll('#speciesGrid input').forEach(cb => {
    const sp = cb.dataset.sp;
    if (grp === 'ground') cb.checked = SP[sp].grp === 'ground';
    else if (grp === 'iso') cb.checked = SP[sp].grp === 'iso';
  });
  onToggle();
}

// ══════════════════════════════════════════════
// Color functions
// ══════════════════════════════════════════════
function getColor(d) {
  if (colorMode === 'species') return SP[d.sp]?.c || '#333';
  if (colorMode === 'residual') {
    if (d.res == null) return '#222';
    const a = Math.abs(d.res);
    if (a < 0.5) return '#22c55e';
    if (a < 1.0) return '#84cc16';
    if (a < 1.5) return '#eab308';
    if (a < 2.0) return '#f97316';
    return '#ef4444';
  }
  if (colorMode === 'halflife') {
    if (d.hl == null) return '#222';
    // Blue (short) -> White (medium) -> Red (long)
    const t = Math.max(0, Math.min(1, (d.hl + 6) / 30));  // -6 to 24
    const r = Math.round(40 + 215 * t);
    const b = Math.round(255 - 215 * t);
    const g = Math.round(t < 0.5 ? 40 + 430*t : 255 - 430*(t-0.5));
    return `rgb(${r},${g},${b})`;
  }
  if (colorMode === 'parity') {
    const pc = {'ee':'#3b82f6','eo':'#22c55e','oe':'#f97316','oo':'#ef4444'};
    return pc[d.par] || '#333';
  }
  if (colorMode === 'stress') {
    if (d.eps == null) return '#222';
    const a = Math.min(Math.abs(d.eps) / 8, 1);
    const r = Math.round(34 + 221*a);
    const g = Math.round(197 - 129*a);
    const b = Math.round(94 - 50*a);
    return `rgb(${r},${g},${b})`;
  }
  return '#444';
}

function updateLegend() {
  const el = document.getElementById('legendRow');
  let html = '';
  if (colorMode === 'species') {
    SP_ORDER.forEach(sp => {
      if (activeSpecies.has(sp))
        html += `<span class="legend-chip"><span class="legend-dot" style="background:${SP[sp].c}"></span>${SP[sp].label}</span>`;
    });
  } else if (colorMode === 'residual') {
    const items = [['#22c55e','<0.5'],['#84cc16','0.5-1'],['#eab308','1-1.5'],['#f97316','1.5-2'],['#ef4444','>2'],['#222','N/A']];
    items.forEach(([c,l]) => { html += `<span class="legend-chip"><span class="legend-dot" style="background:${c}"></span>${l}</span>`; });
  } else if (colorMode === 'halflife') {
    html += '<span class="legend-chip" style="color:#2860f0">Short</span>';
    html += '<span class="legend-chip" style="color:#ccc">Med</span>';
    html += '<span class="legend-chip" style="color:#ef4444">Long</span>';
  } else if (colorMode === 'parity') {
    const items = [['#3b82f6','ee'],['#22c55e','eo'],['#f97316','oe'],['#ef4444','oo']];
    items.forEach(([c,l]) => { html += `<span class="legend-chip"><span class="legend-dot" style="background:${c}"></span>${l}</span>`; });
  } else if (colorMode === 'stress') {
    html += '<span class="legend-chip" style="color:#22c55e">Low |&epsilon;|</span>';
    html += '<span class="legend-chip" style="color:#ef4444">High |&epsilon;|</span>';
  }
  el.innerHTML = html;
}

// ══════════════════════════════════════════════
// Nuclear chart (custom Canvas)
// ══════════════════════════════════════════════
const nucCanvas = document.getElementById('nuclearChart');
const nucCtx = nucCanvas.getContext('2d');
const tooltip = document.getElementById('tooltip');

const MAGIC = [2, 8, 20, 28, 50, 82, 126];
const MARGIN = {top:12, right:12, bottom:36, left:44};

function setupHiDPI(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
  return rect;
}

function cellSize() {
  const rect = nucCanvas.getBoundingClientRect();
  const w = rect.width - MARGIN.left - MARGIN.right;
  const h = rect.height - MARGIN.top - MARGIN.bottom;
  return Math.max(1, Math.min(w / (maxN+1), h / (maxZ+1)));
}

function nToX(n) { return MARGIN.left + n * cellSize(); }
function zToY(z) {
  const rect = nucCanvas.getBoundingClientRect();
  return rect.height - MARGIN.bottom - (z+1) * cellSize();
}

function renderNuclearChart() {
  const rect = setupHiDPI(nucCanvas);
  const ctx = nucCtx;
  const cs = cellSize();
  const w = rect.width, h = rect.height;

  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0,0,w,h);

  // Magic number lines
  ctx.lineWidth = 0.5;
  MAGIC.forEach(m => {
    if (m <= maxN) {
      const x = MARGIN.left + m * cs;
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.beginPath(); ctx.moveTo(x, MARGIN.top); ctx.lineTo(x, h-MARGIN.bottom); ctx.stroke();
    }
    if (m <= maxZ) {
      const y = h - MARGIN.bottom - (m+1)*cs;
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.beginPath(); ctx.moveTo(MARGIN.left, y+cs/2); ctx.lineTo(w-MARGIN.right, y+cs/2); ctx.stroke();
    }
  });

  // N=Z line
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4,4]);
  ctx.beginPath();
  const diagMax = Math.min(maxN, maxZ);
  ctx.moveTo(MARGIN.left, h-MARGIN.bottom-cs);
  ctx.lineTo(MARGIN.left + diagMax*cs, h-MARGIN.bottom-(diagMax+1)*cs);
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw nuclides
  let visCount = 0;
  const gap = Math.max(0.3, cs > 3 ? 0.5 : 0);
  DATA.forEach(d => {
    if (!activeSpecies.has(d.sp)) return;
    visCount++;
    const x = MARGIN.left + d.N * cs;
    const y = h - MARGIN.bottom - (d.Z+1) * cs;
    ctx.fillStyle = getColor(d);
    ctx.fillRect(x, y, Math.max(cs-gap, 0.8), Math.max(cs-gap, 0.8));
  });

  // Axes
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'center';
  for (let n = 0; n <= maxN; n += 20) {
    ctx.fillText(n, MARGIN.left + n*cs + cs/2, h - MARGIN.bottom + 14);
  }
  ctx.textAlign = 'right';
  for (let z = 0; z <= maxZ; z += 10) {
    const y = h - MARGIN.bottom - (z+1)*cs + cs/2 + 3;
    ctx.fillText(z, MARGIN.left - 4, y);
  }

  // Axis titles
  ctx.fillStyle = '#666';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Neutron Number (N)', w/2, h - 3);
  ctx.save();
  ctx.translate(10, h/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('Proton Number (Z)', 0, 0);
  ctx.restore();

  document.getElementById('chartCount').textContent = visCount + ' visible';
}

// Hover
nucCanvas.addEventListener('mousemove', (e) => {
  const rect = nucCanvas.getBoundingClientRect();
  const cs_val = cellSize();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  const n = Math.floor((mx - MARGIN.left) / cs_val);
  const z = Math.floor((rect.height - MARGIN.bottom - my) / cs_val);
  const k = n*1000+z;
  const indices = grid[k];
  if (indices) {
    const vis = indices.filter(i => activeSpecies.has(DATA[i].sp));
    if (vis.length > 0) {
      showTooltip(e, vis);
      return;
    }
  }
  tooltip.style.display = 'none';
});
nucCanvas.addEventListener('mouseleave', () => { tooltip.style.display='none'; });

function showTooltip(e, indices) {
  let html = '';
  indices.forEach((idx, i) => {
    if (i > 0) html += '<hr style="border:none;border-top:1px solid #333;margin:4px 0">';
    const d = DATA[idx];
    const sup = d.A < 10 ? '\u2070\u2071\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079'[d.A] : d.A;
    html += `<b>${d.el}-${d.A}</b> <span style="color:${SP[d.sp]?.c||'#888'}">${d.sp}</span><br>`;
    html += `Z=${d.Z}, N=${d.N}, ${d.par}`;
    if (d.hl != null) html += `<br>log\u2081\u2080(T\u00bd) = ${d.hl.toFixed(2)}`;
    if (d.pred != null) html += `<br>Predicted = ${d.pred.toFixed(2)}`;
    if (d.res != null) {
      const sign = d.res >= 0 ? '+' : '';
      html += `<br>Residual = <b>${sign}${d.res.toFixed(2)}</b> dec`;
    }
    if (d.eps != null) html += `<br>\u03b5 = ${d.eps.toFixed(2)}`;
  });
  tooltip.innerHTML = html;
  tooltip.style.display = 'block';
  // Position tooltip near cursor, clamped to viewport
  let tx = e.clientX + 14;
  let ty = e.clientY - 10;
  if (tx + 260 > window.innerWidth) tx = e.clientX - 270;
  if (ty + 200 > window.innerHeight) ty = window.innerHeight - 210;
  tooltip.style.left = tx + 'px';
  tooltip.style.top = ty + 'px';
}

// ══════════════════════════════════════════════
// Scatter plot (Chart.js)
// ══════════════════════════════════════════════
let scatterChart = null;

function initScatter() {
  const ctx = document.getElementById('scatterChart').getContext('2d');
  scatterChart = new Chart(ctx, {
    type: 'scatter',
    data: { datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 200 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(context) {
              const d = context.raw.d;
              return `${d.el}-${d.A} (${d.sp}): obs=${d.hl.toFixed(1)}, pred=${d.pred.toFixed(1)}, res=${d.res.toFixed(1)}`;
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Observed log\u2081\u2080(T\u00bd)', color: '#666' },
          grid: { color: '#1a1f26' }, ticks: { color: '#8b949e' },
          min: -8, max: 26
        },
        y: {
          title: { display: true, text: 'Predicted log\u2081\u2080(T\u00bd)', color: '#666' },
          grid: { color: '#1a1f26' }, ticks: { color: '#8b949e' },
          min: -8, max: 26
        }
      }
    }
  });
}

function updateScatter() {
  // Build one dataset per active species (for coloring)
  const datasets = [];
  let totalVis = 0;

  // Diagonal reference line
  datasets.push({
    type: 'line', data: [{x:-10,y:-10},{x:30,y:30}],
    borderColor: 'rgba(255,255,255,0.15)', borderWidth: 1,
    pointRadius: 0, borderDash: [5,5], order: 0
  });

  // +/- 1 decade band
  datasets.push({
    type: 'line', data: [{x:-10,y:-9},{x:30,y:31}],
    borderColor: 'rgba(46,160,67,0.1)', borderWidth: 1,
    pointRadius: 0, borderDash: [3,3], order: 0
  });
  datasets.push({
    type: 'line', data: [{x:-10,y:-11},{x:30,y:29}],
    borderColor: 'rgba(46,160,67,0.1)', borderWidth: 1,
    pointRadius: 0, borderDash: [3,3], order: 0
  });

  SP_ORDER.forEach(sp => {
    if (!activeSpecies.has(sp)) return;
    const pts = [];
    (spIdx[sp]||[]).forEach(i => {
      const d = DATA[i];
      if (d.hl != null && d.pred != null) {
        pts.push({x: d.hl, y: d.pred, d: d});
      }
    });
    if (pts.length === 0) return;
    totalVis += pts.length;
    datasets.push({
      label: SP[sp].label,
      data: pts,
      backgroundColor: SP[sp].c + '99',
      borderColor: SP[sp].c,
      pointRadius: 2, pointHoverRadius: 5,
      order: 1
    });
  });

  scatterChart.data.datasets = datasets;
  scatterChart.update();
  document.getElementById('scatterCount').textContent = totalVis + ' predicted';
}

// ══════════════════════════════════════════════
// Statistics
// ══════════════════════════════════════════════
function updateStats() {
  let totalN = 0, totalSolved = 0;
  let totalFound = 0, totalTotal = 0;

  // Per-species stats for visible species
  let tableHTML = '';
  SP_ORDER.forEach(sp => {
    if (!activeSpecies.has(sp)) return;
    const st = STATS[sp];
    if (!st) return;

    // Coverage
    totalFound += st.tracked;
    totalTotal += st.total;
    totalSolved += st.solved;

    // Table row: species, found/total, coverage%, R², RMSE, solved%
    const r2str = st.r2 != null ? st.r2.toFixed(3) : '--';
    const rmsestr = st.rmse != null ? st.rmse.toFixed(2) : '--';
    const solstr = (st.pct != null && st.n > 0) ? st.pct.toFixed(0)+'%' : (sp === 'stable' ? '\u2713' : '--');
    tableHTML += `<tr>
      <td class="sp-name"><span style="color:${SP[sp].c}">\u25A0</span> ${SP[sp].label}</td>
      <td class="num">${st.tracked}/${st.total}</td>
      <td class="num" style="color:#f1c40f">${st.coverage.toFixed(0)}%</td>
      <td class="num">${r2str}</td>
      <td class="num">${solstr}</td>
    </tr>`;
  });

  // Compute global weighted R² from visible species
  let visObs = [], visPred = [];
  SP_ORDER.forEach(sp => {
    if (!activeSpecies.has(sp)) return;
    (spIdx[sp]||[]).forEach(i => {
      const d = DATA[i];
      if (d.hl != null && d.pred != null) {
        visObs.push(d.hl);
        visPred.push(d.pred);
      }
    });
  });

  totalN = 0;
  SP_ORDER.forEach(sp => { if (activeSpecies.has(sp)) totalN += spCounts[sp]; });

  let r2 = 0, rmse = 0;
  if (visObs.length > 1) {
    const mean = visObs.reduce((a,b)=>a+b,0) / visObs.length;
    let ssR = 0, ssT = 0;
    for (let i = 0; i < visObs.length; i++) {
      const res = visObs[i] - visPred[i];
      ssR += res*res;
      ssT += (visObs[i]-mean)*(visObs[i]-mean);
    }
    r2 = ssT > 0 ? 1 - ssR/ssT : 0;
    rmse = Math.sqrt(ssR / visObs.length);
  }

  const coveragePct = totalTotal > 0 ? (100*totalFound/totalTotal).toFixed(1)+'%' : '--';
  const solvedPct = totalFound > 0 ? (100*totalSolved/totalFound).toFixed(1)+'%' : '--';

  document.getElementById('statN').textContent = totalN;
  document.getElementById('statFound').textContent = totalFound + ' / ' + totalTotal;
  document.getElementById('statCoverage').textContent = coveragePct;
  document.getElementById('statR2').textContent = visObs.length > 1 ? r2.toFixed(3) : '--';
  document.getElementById('statRMSE').textContent = visObs.length > 1 ? rmse.toFixed(2) : '--';
  document.getElementById('statSolved').textContent = solvedPct;

  // Species table header + rows
  let headerHTML = '<tr style="font-size:0.9em;color:#555"><td></td><td class="num">found</td><td class="num">cov%</td><td class="num">R\u00b2</td><td class="num">sol%</td></tr>';
  document.getElementById('speciesTable').innerHTML = headerHTML + tableHTML;
}

// ══════════════════════════════════════════════
// Master update
// ══════════════════════════════════════════════
function updateAll() {
  renderNuclearChart();
  updateScatter();
  updateStats();
  updateLegend();
}

// Color mode change
document.getElementById('colorMode').addEventListener('change', (e) => {
  colorMode = e.target.value;
  renderNuclearChart();
  updateLegend();
});

// Window resize
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(updateAll, 150);
});

// ══════════════════════════════════════════════
// Init
// ══════════════════════════════════════════════
buildSpeciesGrid();
initScatter();
updateAll();
"""

# ══════════════════════════════════════════════════════════════════════
# Assemble and write HTML
# ══════════════════════════════════════════════════════════════════════
print("Generating HTML...")
html = HTML_HEAD
html += '\n<script>\n'
html += f'const DATA = {data_json};\n'
html += f'const STATS = {stats_json};\n'
html += HTML_JS
html += '\n</script>\n</body>\n</html>\n'

output_path = os.path.join(RESULTS_DIR, 'nuclide_viewer.html')
with open(output_path, 'w') as f:
    f.write(html)

size_kb = os.path.getsize(output_path) // 1024
print(f"\nViewer written to {output_path} ({size_kb} KB)")
print("Open in a browser to explore.")
