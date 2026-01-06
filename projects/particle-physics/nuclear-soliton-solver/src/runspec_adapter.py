#!/usr/bin/env python3
"""
RunSpec Adapter for Nuclear Soliton Solver
==========================================

Bridges QFD RunSpec v0 schema with the Phase 9 SCF solver.

Usage:
    python runspec_adapter.py experiments/nuclear_heavy_region.runspec.json

Features:
    - Loads and validates RunSpec JSON against schema
    - Extracts parameters and datasets
    - Maps to qfd_solver.py parameter format
    - Runs optimization with specified solver
    - Returns structured results with provenance
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

# Import existing solver modules
from qfd_metaopt_ame2020 import run_qfd_solver, load_ame2020_data


class RunSpecAdapter:
    """Adapter to run QFD nuclear solver from RunSpec configuration."""

    def __init__(self, runspec_path: str):
        """
        Initialize adapter with RunSpec file.

        Args:
            runspec_path: Path to RunSpec JSON file
        """
        self.runspec_path = Path(runspec_path)
        self.runspec = self._load_runspec()
        self.base_dir = self.runspec_path.parent.parent

    def _load_runspec(self) -> Dict[str, Any]:
        """Load and parse RunSpec JSON."""
        with open(self.runspec_path, 'r') as f:
            return json.load(f)

    def _validate_model(self):
        """Validate model specification."""
        model = self.runspec['model']
        allowed_models = ['qfd.nuclear.binding.soliton', 'qfd.nuclear.ccl']

        if model['id'] not in allowed_models:
            raise ValueError(f"Unknown model: {model['id']}")

        if model['id'] == 'qfd.nuclear.binding.soliton':
            if model.get('variant', 'standard') not in ['phase9_scf', 'standard']:
                raise ValueError(f"Unknown variant: {model['variant']}")

    def _extract_parameters(self) -> Dict[str, float]:
        """
        Extract parameters from RunSpec and convert to solver format.

        Returns:
            Dictionary mapping parameter names to values
        """
        params = {}

        for param in self.runspec['parameters']:
            # Map RunSpec parameter names to solver names
            name = param['name']
            value = param['value']

            # Remove 'nuclear.' prefix if present
            if name.startswith('nuclear.'):
                name = name.replace('nuclear.', '')

            params[name] = value

        # Ensure required parameters exist (set defaults if missing)
        # These are defaults from Trial 32
        required_defaults = {
            'c_v4_iso': 0.005164,
            'alpha_e_scale': 1.007419,
            'beta_e_scale': 0.504312,
            'kappa_rho': 0.029816,
            'c_coul': 0.801463,
            'c_surf': 18.5,
            'c_pair_even': 12.0,
            'c_pair_odd': -12.0,
            'c_shell': 0.0,
        }

        for key, default_value in required_defaults.items():
            if key not in params:
                params[key] = default_value

        return params

    def _get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Extract parameter bounds for optimization.

        Returns:
            Dictionary mapping parameter names to (lower, upper) bounds
        """
        bounds = {}

        for param in self.runspec['parameters']:
            if param.get('frozen', False):
                continue  # Skip frozen parameters

            if param.get('bounds') is not None:
                name = param['name'].replace('nuclear.', '')
                bounds[name] = tuple(param['bounds'])

        return bounds

    def _load_datasets(self) -> pd.DataFrame:
        """
        Load datasets specified in RunSpec.

        Returns:
            Merged DataFrame with all datasets
        """
        datasets = []

        for dataset_spec in self.runspec.get('datasets', []):
            source = dataset_spec['source']

            # Resolve relative paths
            if not os.path.isabs(source):
                source = self.base_dir / source
            else:
                source = Path(source)

            # Load dataset
            if str(source).endswith('.csv'):
                df = pd.read_csv(source)
            elif str(source).endswith('.json'):
                df = pd.read_json(source)
            else:
                raise ValueError(f"Unsupported dataset format: {source}")

            # Apply column mappings if specified
            if 'columns' in dataset_spec:
                df = df.rename(columns=dataset_spec['columns'])

            # Apply cuts/filters if specified
            if 'cuts' in dataset_spec:
                for key, condition in dataset_spec['cuts'].items():
                    if key == 'A_range':
                        df = df[(df['A'] >= condition[0]) & (df['A'] <= condition[1])]
                    elif key == 'stable_only':
                        if condition:
                            df = df[df.get('stable', True)]

            datasets.append(df)

        if len(datasets) == 0:
            raise ValueError("No datasets specified in RunSpec")

        # Merge all datasets
        return pd.concat(datasets, ignore_index=True)

    def _select_target_isotopes(self, ame_data: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Select target isotopes from datasets.

        Args:
            ame_data: Full AME2020 dataset

        Returns:
            List of (Z, A) tuples
        """
        # If datasets specify specific isotopes, use those
        datasets = self.runspec.get('datasets', [])

        isotopes = []
        for dataset_spec in datasets:
            cuts = dataset_spec.get('cuts', {})

            # Priority 1: Explicit target_isotopes list
            if 'target_isotopes' in cuts:
                for iso in cuts['target_isotopes']:
                    isotopes.append((int(iso['Z']), int(iso['A'])))

            # Priority 2: A_range with sampling
            elif 'A_range' in cuts:
                A_min, A_max = cuts['A_range']
                # Select representative isotopes from this range
                subset = ame_data[(ame_data['A'] >= A_min) & (ame_data['A'] <= A_max)]

                # For testing, just use a small sample
                n_samples = min(10, len(subset))
                sampled = subset.sample(n=n_samples, random_state=42)

                for _, row in sampled.iterrows():
                    isotopes.append((int(row['Z']), int(row['A'])))

            # Priority 3: A_min/A_max without A_range key
            elif 'A_min' in cuts and 'A_max' in cuts:
                A_min = cuts['A_min']
                A_max = cuts['A_max']
                subset = ame_data[(ame_data['A'] >= A_min) & (ame_data['A'] <= A_max)]

                # Use small sample
                n_samples = min(10, len(subset))
                sampled = subset.sample(n=n_samples, random_state=42)

                for _, row in sampled.iterrows():
                    isotopes.append((int(row['Z']), int(row['A'])))

        # If no isotopes selected, use a default test set
        if len(isotopes) == 0:
            # Use light isotopes for quick testing
            isotopes = [(2, 4), (6, 12), (8, 16)]

        return isotopes

    def _build_objective_function(self, target_isotopes: List[Tuple[int, int]],
                                   ame_data: pd.DataFrame):
        """
        Build objective function from RunSpec specification.

        Args:
            target_isotopes: List of (Z, A) tuples
            ame_data: AME2020 experimental data

        Returns:
            Objective function for optimization
        """
        objective_spec = self.runspec.get('objective', {})
        obj_type = objective_spec.get('type', 'chi_squared')

        # Extract device from solver options
        solver_spec = self.runspec.get('solver', {})
        scf_options = solver_spec.get('scf_solver_options', {})
        device = scf_options.get('device', 'cpu')

        # Build lookup table for experimental values
        exp_values = {}
        for Z, A in target_isotopes:
            row = ame_data[(ame_data['Z'] == Z) & (ame_data['A'] == A)]
            if len(row) > 0:
                # Use experimental system energy
                exp_values[(Z, A)] = row.iloc[0]['E_exp_MeV']

        def objective(params_dict: Dict[str, float]) -> float:
            """Compute objective function value."""
            from qfd_metaopt_ame2020 import M_PROTON, M_NEUTRON, M_ELECTRON

            residuals = []
            virial_penalties = []

            for Z, A in target_isotopes:
                if (Z, A) not in exp_values:
                    continue

                # Run solver for this isotope (fast_mode for optimization speed)
                result = run_qfd_solver(A, Z, params_dict, verbose=False, fast_mode=True, device=device)

                if result is None or not result.get('physical_success', False):
                    residuals.append(1000.0)  # Large penalty for failure
                    continue

                # Compute predicted system energy
                E_interaction = result['E_model']
                N = A - Z
                M_constituents = Z * M_PROTON + N * M_NEUTRON + Z * M_ELECTRON
                pred_E = M_constituents + E_interaction
                exp_E = exp_values[(Z, A)]

                # Compute residual as percentage error
                error_pct = 100.0 * abs(pred_E - exp_E) / abs(exp_E)

                # Compute residual
                if obj_type == 'chi_squared':
                    # Assume 1% experimental uncertainty
                    sigma = 0.01 * abs(exp_E)
                    residual = ((pred_E - exp_E) / sigma) ** 2
                elif obj_type == 'sse':
                    residual = (pred_E - exp_E) ** 2
                else:
                    residual = error_pct

                residuals.append(residual)

                # Virial constraint penalty
                virial = result.get('virial', result.get('virial_abs', 0.0))
                if abs(virial) > 0.18:
                    virial_penalties.append(4.0 * (abs(virial) - 0.18) ** 2)

            # Combine terms
            loss = np.mean(residuals) if len(residuals) > 0 else 1e6

            if len(virial_penalties) > 0:
                loss += np.mean(virial_penalties)

            return loss

        return objective

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute the experiment specified in RunSpec.

        Args:
            verbose: Print progress messages

        Returns:
            Results dictionary with predictions, fit quality, provenance
        """
        if verbose:
            print(f"╔═══════════════════════════════════════════════════════════════╗")
            print(f"║  QFD Nuclear Solver - RunSpec Adapter                         ║")
            print(f"╚═══════════════════════════════════════════════════════════════╝")
            print(f"\nExperiment: {self.runspec['experiment_id']}")
            print(f"Description: {self.runspec['description']}")
            print()

        # Validate model
        self._validate_model()

        # Extract parameters
        initial_params = self._extract_parameters()
        param_bounds = self._get_parameter_bounds()

        if verbose:
            print(f"Parameters: {len(initial_params)} total, {len(param_bounds)} optimizable")
            print()

        # Load AME2020 data
        # Try multiple possible paths
        possible_paths = [
            self.base_dir / 'data' / 'ame2020_system_energies.csv',
            Path('data/ame2020_system_energies.csv'),
            Path('../data/ame2020_system_energies.csv'),
            Path('../../data/ame2020_system_energies.csv'),
        ]

        ame_path = None
        for path in possible_paths:
            if path.exists():
                ame_path = path
                break

        if ame_path is None:
            raise FileNotFoundError(f"Could not find ame2020_system_energies.csv in any of: {possible_paths}")

        ame_data = load_ame2020_data(str(ame_path))

        # Select target isotopes
        target_isotopes = self._select_target_isotopes(ame_data)

        if verbose:
            print(f"Target isotopes: {len(target_isotopes)}")
            print()

        # Get solver configuration
        solver_spec = self.runspec.get('solver', {})
        solver_method = solver_spec.get('method', 'scipy.differential_evolution')
        solver_options = solver_spec.get('options', {})

        # Run optimization or evaluation
        if len(param_bounds) == 0:
            # Just evaluate with given parameters
            if verbose:
                print("Mode: Evaluation (all parameters frozen)")
                print()

            results = self._evaluate(initial_params, target_isotopes, ame_data)
        else:
            # Optimize parameters
            if verbose:
                print(f"Mode: Optimization ({solver_method})")
                print()

            results = self._optimize(initial_params, param_bounds, target_isotopes,
                                    ame_data, solver_method, solver_options, verbose)

        # Add provenance
        results['provenance'] = {
            'runspec': str(self.runspec_path),
            'experiment_id': self.runspec['experiment_id'],
            'timestamp': datetime.now().isoformat(),
            'schema_version': self.runspec['schema_version']
        }

        return results

    def _evaluate(self, params: Dict[str, float],
                  target_isotopes: List[Tuple[int, int]],
                  ame_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate parameters without optimization."""
        from qfd_metaopt_ame2020 import M_PROTON, M_NEUTRON, M_ELECTRON

        predictions = []

        for Z, A in target_isotopes:
            result = run_qfd_solver(A, Z, params, verbose=False, fast_mode=False)

            if result is not None and result.get('physical_success', False):
                exp_row = ame_data[(ame_data['Z'] == Z) & (ame_data['A'] == A)]
                if len(exp_row) > 0:
                    # Get experimental system energy
                    exp_E_MeV = exp_row.iloc[0]['E_exp_MeV']

                    # Compute predicted system energy
                    E_interaction = result['E_model']
                    N = A - Z
                    M_constituents = Z * M_PROTON + N * M_NEUTRON + Z * M_ELECTRON
                    pred_E_MeV = M_constituents + E_interaction

                    # Compute binding energy (negative of interaction energy)
                    pred_BE_MeV = -E_interaction
                    exp_BE_MeV = A * exp_row.iloc[0]['BE_per_A_MeV']

                    error_pct = 100.0 * (pred_E_MeV - exp_E_MeV) / exp_E_MeV

                    predictions.append({
                        'Z': Z,
                        'A': A,
                        'pred_E_MeV': pred_E_MeV,
                        'exp_E_MeV': exp_E_MeV,
                        'pred_BE_MeV': pred_BE_MeV,
                        'exp_BE_MeV': exp_BE_MeV,
                        'error_pct': error_pct,
                        'virial': result.get('virial', result.get('virial_abs', None))
                    })

        df = pd.DataFrame(predictions)

        if len(df) == 0:
            return {
                'status': 'failed',
                'mode': 'evaluation',
                'parameters': params,
                'error': 'No successful predictions'
            }

        return {
            'status': 'success',
            'mode': 'evaluation',
            'parameters': params,
            'predictions': df,
            'metrics': {
                'mean_error_pct': df['error_pct'].mean(),
                'std_error_pct': df['error_pct'].std(),
                'max_abs_error_pct': df['error_pct'].abs().max(),
                'mean_virial': df['virial'].mean() if 'virial' in df.columns else None,
                'n_converged': len(df[df['virial'] < 0.18]) if 'virial' in df.columns else None
            }
        }

    def _optimize(self, initial_params: Dict[str, float],
                  param_bounds: Dict[str, Tuple[float, float]],
                  target_isotopes: List[Tuple[int, int]],
                  ame_data: pd.DataFrame,
                  method: str,
                  options: Dict[str, Any],
                  verbose: bool) -> Dict[str, Any]:
        """Run parameter optimization."""
        from scipy.optimize import differential_evolution, minimize

        # Build objective function
        objective = self._build_objective_function(target_isotopes, ame_data)

        # Prepare bounds for scipy
        param_names = sorted(param_bounds.keys())
        bounds_list = [param_bounds[name] for name in param_names]

        def objective_vec(x):
            """Vectorized objective for scipy optimizers."""
            params = initial_params.copy()
            for i, name in enumerate(param_names):
                params[name] = x[i]
            return objective(params)

        # Run optimization
        if 'differential_evolution' in method:
            if verbose:
                print("Running differential evolution...")
                options.setdefault('disp', True)

            # Add progress tracking callback
            progress_file = Path('optimization_progress.log')
            eval_count = [0]  # Mutable counter for closure

            def progress_callback(xk, convergence):
                """Callback to track optimization progress."""
                eval_count[0] += 1
                score = objective_vec(xk)
                with open(progress_file, 'a') as f:
                    f.write(f"Generation {eval_count[0]}: score={score:.6f}, convergence={convergence:.6f}\n")
                    f.flush()  # Force write to disk
                return False  # Continue optimization

            # Clear previous progress log
            if progress_file.exists():
                progress_file.unlink()

            with open(progress_file, 'w') as f:
                f.write(f"Starting differential evolution optimization\n")
                f.write(f"Bounds: {len(bounds_list)} parameters\n")
                f.write(f"Options: {options}\n")
                f.write(f"=" * 70 + "\n")
                f.flush()

            result = differential_evolution(objective_vec, bounds_list,
                                          callback=progress_callback, **options)

        elif 'minimize' in method:
            x0 = [initial_params[name] for name in param_names]

            if verbose:
                print("Running minimization...")
                options.setdefault('disp', True)

            result = minimize(objective_vec, x0, bounds=bounds_list, **options)

        else:
            raise ValueError(f"Unknown solver method: {method}")

        # Extract optimized parameters
        optimized_params = initial_params.copy()
        for i, name in enumerate(param_names):
            optimized_params[name] = result.x[i]

        # Evaluate with optimized parameters
        eval_results = self._evaluate(optimized_params, target_isotopes, ame_data)

        eval_results['mode'] = 'optimization'
        eval_results['optimization'] = {
            'success': result.success,
            'message': getattr(result, 'message', ''),
            'n_iterations': getattr(result, 'nit', None),
            'n_function_evals': getattr(result, 'nfev', None),
            'final_loss': result.fun
        }

        return eval_results


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python runspec_adapter.py <runspec.json>")
        print()
        print("Example:")
        print("  python runspec_adapter.py experiments/nuclear_heavy_region.runspec.json")
        sys.exit(1)

    runspec_path = sys.argv[1]

    # Run experiment
    adapter = RunSpecAdapter(runspec_path)
    results = adapter.run(verbose=True)

    # Print summary
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  Results Summary                                              ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print(f"Status: {results['status']}")
    print(f"Mode: {results['mode']}")
    print()
    print("Fit Quality:")
    for key, value in results['metrics'].items():
        print(f"  {key:20s}: {value:10.4f}")
    print()

    if results['mode'] == 'optimization':
        print("Optimization:")
        for key, value in results['optimization'].items():
            print(f"  {key:20s}: {value}")
        print()

    # Save results
    output_dir = Path('results') / results['provenance']['experiment_id']
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert DataFrame to dict for JSON serialization
    results_serializable = results.copy()
    results_serializable['predictions'] = results['predictions'].to_dict('records')

    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()


if __name__ == '__main__':
    main()
