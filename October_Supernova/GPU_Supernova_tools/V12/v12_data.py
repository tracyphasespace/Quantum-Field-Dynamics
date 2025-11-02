import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
import jax.numpy as jnp
import v12_systematics as sysw

# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_lightcurves(filepath, n_sne, start_sne,
                     z_min, z_max,
                     k_J, delta_k_J, z_trans,
                     gamma_thermo, xi_fixed,
                     with_survey_weights,
                     time_window_days,
                     downsample_factor,
                     indices_path,
                     no_presort,
                     save_effective_set,
                     exclude_snids: set[str] | None = None,
                     is_prefiltered: bool = False):
    """
    Loads and processes the lightcurve data from the CSV.
    Prepares the data structures for the fitter.
    It returns a list of dictionaries, where each dict contains numpy arrays
    for a single supernova.
    """
    print(f"Loading: {filepath}")
    try:
        suffix = Path(filepath).suffix.lower()
        if suffix in {'.parquet', '.pq'}:
            df = pd.read_parquet(filepath)
        elif suffix in {'.feather', '.ft'}:
            df = pd.read_feather(filepath)
        else:
            df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"ERROR: Lightcurve file not found at {filepath}")
        return None, None
    
    print(f"  Total observations: {len(df)}")
    print(f"  Total unique SNe (raw): {df['snid'].nunique()}")

    # Optional: exclude specific SNIDs before any other filtering
    if exclude_snids:
        # try to find the SNID column in several common casings
        sn_cols = [c for c in df.columns if c.lower() in ('snid', 'sn_id', 'snname', 'sn_name')]
        if sn_cols:
            snc = sn_cols[0]
            before = len(df)
            df = df[~df[snc].astype(str).isin({str(x) for x in exclude_snids})].copy()
            removed = before - len(df)
            print(f"  [filter] Excluded {removed} rows via {len(exclude_snids)} SNIDs (column='{snc}').")
        else:
            print("  [warn] --exclude-snids provided, but no SNID-like column found in dataframe;"
                  " skipping exclusion.")

    # Ensure consistent dtypes
    df['snid'] = df['snid'].astype(str)
    if 'source' in df.columns:
        df['source'] = df['source'].fillna("UNKNOWN").astype(str)
    else:
        df['source'] = "UNKNOWN"

    indices_list = None
    if indices_path:
        try:
            with open(indices_path, 'r') as fh:
                indices_raw = json.load(fh)
            indices_list = [str(x) for x in indices_raw]
            print(f"  Using precomputed indices ({len(indices_list)} entries).")
        except Exception as exc:
            print(f"ERROR: Failed to read indices file {indices_path}: {exc}")
            return None, None

    # --- helper: choose redshift column robustly ---
    def _pick_z_col(frame, preferred=None):
        if preferred and preferred in frame.columns:
            return preferred
        candidates = ["zcmb", "z_cmb", "zHD", "z_helio", "z_hel", "z", "redshift"]
        for c in candidates:
            if c in frame.columns:
                return c
        return None

    # --- 1. Apply Filters (ALWAYS if provided, even when prefiltered) ---
    zcol = _pick_z_col(df)
    if (z_min is not None or z_max is not None):
        if zcol is None:
            raise KeyError(
                "No redshift column found. Tried: zcmb, z_cmb, zHD, z_helio, z_hel, z, redshift. "
                f"Available columns: {list(df.columns)[:20]}…"
            )
        if z_min is not None:
            df = df[df[zcol] >= float(z_min)]
        if z_max is not None:
            df = df[df[zcol] <= float(z_max)]
    # (Keep indices_list behavior below; it further refines selection)

    # Log the filter actually applied (and sanity numbers)
    count = len(df['snid'].unique()) if 'snid' in df.columns else len(df)
    z_med = float(df[zcol].median())
    z_min_actual = float(df[zcol].min())
    z_max_actual = float(df[zcol].max())

    print(f"[V12] Using z_col='{zcol}'")
    print(f"[V12] After filters: {count} SNe | median_z={z_med:.4f} "
          f"(min={z_min_actual:.4f}, max={z_max_actual:.4f})")
    if z_min is not None: 
        assert z_min_actual >= z_min - 1e-9, "z_min filter not applied!"
    if z_max is not None: 
        assert z_max_actual <= z_max + 1e-9, "z_max filter not applied!"

    # Filter for SNe with a minimum number of observations
    obs_counts = df['snid'].value_counts()
    valid_sne_ids = obs_counts[obs_counts >= 5].index # At least 5 observations
    df = df[df['snid'].isin(valid_sne_ids)]

    # Optional explicit SN index selection
    if indices_list is not None:
        available = set(df['snid'].unique())
        missing = [sn for sn in indices_list if sn not in available]
        if missing:
            print(f"  WARNING: {len(missing)} SN IDs from indices file not found after filtering.")
        ordered_ids = [sn for sn in indices_list if sn in available]
    else:
        if no_presort:
            ordered_ids = df['snid'].drop_duplicates().tolist()
        else:
            df = df.sort_values([zcol, 'snid'])
            ordered_ids = df['snid'].drop_duplicates().tolist()

    if start_sne < 0:
        start_sne = 0

    total_available = len(ordered_ids)
    if total_available == 0:
        print("Error: No SNe available after filtering.")
        return None, None

    max_end = total_available
    if n_sne is not None and n_sne > 0:
        max_end = min(total_available, start_sne + n_sne)

    sne_ids_to_process = ordered_ids[start_sne:max_end]

    if not sne_ids_to_process:
        print("Error: No SNe selected for processing after applying start/n limits.")
        return None, None

    if save_effective_set:
        try:
            os.makedirs(os.path.dirname(save_effective_set), exist_ok=True)
        except Exception:
            pass
        try:
            with open(save_effective_set, 'w') as fh:
                json.dump(sne_ids_to_process, fh, indent=2)
        except Exception as exc:
            print(f"  WARNING: Failed to write effective set file {save_effective_set}: {exc}")

    if indices_list is None:
        print(f"\nProcessing {len(sne_ids_to_process)} SNe (from index {start_sne}).")
    else:
        print(f"\nProcessing {len(sne_ids_to_process)} SNe (from indices file).")

    # Unit guard: if errors look way too big relative to flux, interpret them as mag errors and convert.
    # Heuristic: median(err/|flux|) >> 1 means err is not in Jy.
    MAG2FLUX = 0.921034
    flux = df['flux_nu_jy'].to_numpy()
    err  = df['flux_nu_jy_err'].to_numpy()

    # Filter out non-positive mag_err values before processing
    if 'mag_err' in df.columns:
        initial_rows = len(df)
        df = df[df['mag_err'] > 0].copy()
        if len(df) < initial_rows:
            print(f"  [filter] Removed {initial_rows - len(df)} rows with non-positive mag_err.")
        
        # Recalculate flux_nu_jy and flux_nu_jy_err from mag, mag_err, and zp
        if 'mag' in df.columns and 'zp' in df.columns:
            # Flux = 10^((zp - mag) / 2.5)
            df['flux_nu_jy'] = 10**((df['zp'] - df['mag']) / 2.5)
            # Flux_err = MAG2FLUX * mag_err * Flux
            df['flux_nu_jy_err'] = MAG2FLUX * df['mag_err'] * df['flux_nu_jy']
            print("  [recalc] Recalculated flux_nu_jy and flux_nu_jy_err from mag, mag_err, and zp.")
            
            # Update flux and err arrays for the unit guard check
            flux = df['flux_nu_jy'].to_numpy()
            err  = df['flux_nu_jy_err'].to_numpy()

    # Proceed with unit guard check only if flux and err are valid
    if len(flux) > 0 and len(err) > 0:
        # Avoid division by zero or NaN issues
        ratio_mask = np.abs(flux) > 0
        if np.any(ratio_mask):
            ratio = np.nanmedian(np.where(ratio_mask, err / np.abs(flux), np.nan))
            print(f"[V12] flux median={np.nanmedian(flux):.3e} Jy, err median={np.nanmedian(err):.3e} Jy, med(err/|flux|)≈{ratio:.2e}")

            if np.isfinite(ratio) and ratio > 1e3:
                # Treat incoming error as magnitude uncertainty; convert to Jy
                # This block might be redundant if mag_err recalculation already happened,
                # but serves as a fallback/double-check.
                print("  [unit-guard] Detected potentially mis-scaled errors (ratio > 1e3). Interpreting as magnitude errors.")
                # Assuming 'flux_nu_jy_err' was actually 'mag_err'
                err_mag = err / (MAG2FLUX * np.abs(flux)) # Attempt to reverse engineer mag_err
                err = MAG2FLUX * err_mag * np.abs(flux)
                df['flux_nu_jy_err'] = err
        else:
            print("[V12] Warning: All flux values are zero, cannot compute err/|flux| ratio for unit guard.")
    else:
        print("[V12] Warning: No valid flux or error data for unit guard check.")

    # --- 2. Apply --with-survey-weights ---
    if with_survey_weights:
        print("Applying survey-specific systematic floors...")
        # This is the key modification: call apply_systematic_floor from v8_systematics
        df['flux_nu_jy_err_weighted'] = df.apply(sysw.apply_systematic_floor, axis=1).astype(np.float32)
        print("  ...done.")
    else:
        print("Skipping survey weights. Using raw flux_nu_jy_err.")
        df['flux_nu_jy_err_weighted'] = df['flux_nu_jy_err'].astype(np.float32)

    df_sample = df[df['snid'].isin(sne_ids_to_process)].copy()

    # --- 4. Prepare data structures for each SN ---
    sn_data_list = []
    initial_nuisance_list = []

    for snid in sne_ids_to_process:
        sn_df_full = df_sample[df_sample['snid'] == snid].sort_values('mjd')

        if sn_df_full.empty:
            continue

        # Initial guesses from full lightcurve
        peak_idx = sn_df_full['flux_nu_jy'].idxmax()
        t_peak_guess = np.float32(sn_df_full.loc[peak_idx, 'mjd'])
        A_0_guess = np.float32(sn_df_full.loc[peak_idx, 'flux_nu_jy'])
        tau_0_guess = np.float32(20.0) # Standard SN rise/fall time in days (rest frame) 

        sn_df = sn_df_full

        if time_window_days and time_window_days > 0:
            t_lo = float(t_peak_guess - time_window_days)
            t_hi = float(t_peak_guess + time_window_days)
            windowed = sn_df_full[(sn_df_full['mjd'] >= t_lo) & (sn_df_full['mjd'] <= t_hi)].copy()
            if len(windowed) >= 5:
                sn_df = windowed

        if downsample_factor and downsample_factor > 1:
            downsampled = sn_df.iloc[::int(downsample_factor)].copy()
            if len(downsampled) >= 5:
                sn_df = downsampled

        if len(sn_df) < 5:
            sn_df = sn_df_full

        z_sn = sn_df[zcol].iloc[0]

        initial_nuisance = np.array([t_peak_guess, A_0_guess, tau_0_guess], dtype=np.float32)

        # Pack data into a dictionary with numpy arrays
        sn_data = {
            'mjd': sn_df['mjd'].values.astype(np.float32),
            'flux_nu_jy': sn_df['flux_nu_jy'].values.astype(np.float32),
            'flux_nu_jy_err': sn_df['flux_nu_jy_err'].values.astype(np.float32),
            'flux_nu_jy_err_weighted': sn_df['flux_nu_jy_err_weighted'].values.astype(np.float32),
            'wavelength_eff_nm': sn_df['wavelength_eff_nm'].values.astype(np.float32),
            'z': np.float32(z_sn),
            'snid': str(snid),
            'survey': sn_df['source'].values[0] if 'source' in sn_df.columns else "UNKNOWN", # Store for later diagnostics
            # Pass fixed physics knobs (these are now part of the sn_data dict for convenience in chi2_likelihood)
            'k_J': np.float32(k_J),
            'delta_k_J': np.float32(delta_k_J),
            'z_trans': np.float32(z_trans),
            'gamma_thermo': np.float32(gamma_thermo),
            'xi': np.float32(xi_fixed)
        }
        
        sn_data_list.append(sn_data)
        initial_nuisance_list.append(initial_nuisance)

    if not sn_data_list:
        print("Error: No SNe available for processing.")
        return None, None

    # =========================================================================
    # V12 VECTORIZATION: Pad data to create fixed-shape batches
    # =========================================================================
    print("Padding data for vectorization...")

    # Find max number of observations in the batch
    max_len = max(len(sn['mjd']) for sn in sn_data_list)
    num_sne = len(sn_data_list)

    # Define which keys to pad (per-observation data)
    keys_to_pad = ['mjd', 'flux_nu_jy', 'flux_nu_jy_err',
                   'flux_nu_jy_err_weighted', 'wavelength_eff_nm']
    # Keys with per-SN scalar data
    keys_to_stack = ['z', 'k_J', 'delta_k_J', 'z_trans', 'gamma_thermo', 'xi']

    # Initialize batch_data dictionary
    batch_data = {}

    # Process padded keys (create 2D arrays: [num_sne, max_len])
    for key in keys_to_pad:
        padded_arrays = []
        for sn in sn_data_list:
            arr = sn[key]
            pad_width = max_len - len(arr)
            # Pad with zeros (will be masked out)
            padded_arr = np.pad(arr, (0, pad_width), 'constant', constant_values=0.0)
            padded_arrays.append(padded_arr)
        # Stack and convert to JAX array
        batch_data[key] = jnp.asarray(np.stack(padded_arrays), dtype=jnp.float32)

    # Process stacked scalar keys (create 1D arrays: [num_sne])
    for key in keys_to_stack:
        stacked_vals = [sn[key] for sn in sn_data_list]
        batch_data[key] = jnp.asarray(np.stack(stacked_vals), dtype=jnp.float32)

    # Create the mask (True = valid data, False = padding)
    masks = []
    for sn in sn_data_list:
        mask = np.zeros(max_len, dtype=bool)
        mask[:len(sn['mjd'])] = True
        masks.append(mask)
    batch_data['mask'] = jnp.asarray(np.stack(masks), dtype=jnp.bool_)

    # Store metadata (not used in computation, just for tracking)
    batch_data['snid'] = [sn['snid'] for sn in sn_data_list]
    batch_data['survey'] = [sn['survey'] for sn in sn_data_list]

    # Convert initial nuisances to JAX array
    initial_nuisance_stacked = jnp.asarray(np.stack(initial_nuisance_list), dtype=jnp.float32)

    # Sanity check shapes
    print(f"  Padded batch shape: (num_sne={num_sne}, max_obs={max_len})")
    assert batch_data['mask'].shape == (num_sne, max_len)
    assert batch_data['mjd'].shape == (num_sne, max_len)
    assert batch_data['z'].shape == (num_sne,)
    assert initial_nuisance_stacked.shape == (num_sne, 3)

    print(f"  ✓ Vectorized batch ready: {num_sne} SNe, max {max_len} observations each")

    return batch_data, initial_nuisance_stacked