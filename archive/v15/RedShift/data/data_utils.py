#!/usr/bin/env python3
"""
Data Loading Utilities for QFD CMB Module

This module provides utilities for loading and working with sample datasets
generated for the QFD CMB Module.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings


class QFDDataLoader:
    """Utility class for loading QFD CMB sample datasets."""
    
    def __init__(self, data_dir: Union[str, Path] = None):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        data_dir : str or Path, optional
            Path to the data directory. If None, uses default 'data/sample'.
        """
        if data_dir is None:
            # Default to data/sample relative to this file
            self.data_dir = Path(__file__).parent / "sample"
        else:
            self.data_dir = Path(data_dir)
        
        self.metadata = None
        self._load_metadata()
    
    def _load_metadata(self):
        """Load dataset metadata if available."""
        metadata_file = self.data_dir / "dataset_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            warnings.warn(f"Metadata file not found: {metadata_file}")
    
    def list_available_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
        --------
        list
            List of available dataset names
        """
        if not self.data_dir.exists():
            return []
        
        datasets = []
        for file_path in self.data_dir.glob("*.csv"):
            datasets.append(file_path.stem)
        
        # Add JSON datasets
        for file_path in self.data_dir.glob("*.json"):
            if file_path.stem != "dataset_metadata":
                datasets.append(file_path.stem)
        
        return sorted(datasets)
    
    def load_minimal_test_data(self) -> pd.DataFrame:
        """
        Load minimal test dataset for quick validation.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: ell, C_TT, C_EE, C_TE, rho_TE
        """
        file_path = self.data_dir / "minimal_test_spectra.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Minimal test data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        return df
    
    def load_planck_like_data(self) -> pd.DataFrame:
        """
        Load Planck-like dataset for realistic demonstrations.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: ell, C_TT, C_EE, C_TE, error_TT, error_EE, error_TE, rho_TE
        """
        file_path = self.data_dir / "planck_like_spectra.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Planck-like data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        return df
    
    def load_reference_data(self) -> Dict:
        """
        Load reference data for regression testing.
        
        Returns:
        --------
        dict
            Dictionary containing reference data and metadata
        """
        file_path = self.data_dir / "reference_spectra.json"
        if not file_path.exists():
            raise FileNotFoundError(f"Reference data not found: {file_path}")
        
        with open(file_path, 'r') as f:
            reference_data = json.load(f)
        
        return reference_data
    
    def load_parameter_variations(self, parameter: str) -> pd.DataFrame:
        """
        Load parameter variation dataset.
        
        Parameters:
        -----------
        parameter : str
            Parameter name ('rpsi', 'aosc', or 'ns')
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with parameter variations
        """
        valid_params = ['rpsi', 'aosc', 'ns']
        if parameter not in valid_params:
            raise ValueError(f"Parameter must be one of {valid_params}")
        
        file_path = self.data_dir / f"{parameter}_variations.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Parameter variation data not found: {file_path}")
        
        df = pd.read_csv(file_path)
        return df
    
    def load_dataset(self, name: str) -> Union[pd.DataFrame, Dict]:
        """
        Load any dataset by name.
        
        Parameters:
        -----------
        name : str
            Dataset name (without extension)
        
        Returns:
        --------
        pd.DataFrame or dict
            Loaded dataset
        """
        # Try CSV first
        csv_path = self.data_dir / f"{name}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        
        # Try JSON
        json_path = self.data_dir / f"{name}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        
        raise FileNotFoundError(f"Dataset not found: {name}")
    
    def get_dataset_info(self, name: str) -> Dict:
        """
        Get information about a specific dataset.
        
        Parameters:
        -----------
        name : str
            Dataset name
        
        Returns:
        --------
        dict
            Dataset information from metadata
        """
        if self.metadata is None:
            return {"error": "Metadata not available"}
        
        datasets = self.metadata.get('qfd_cmb_sample_data', {}).get('datasets', {})
        
        # Try exact match first
        if f"{name}.csv" in datasets:
            return datasets[f"{name}.csv"]
        elif f"{name}.json" in datasets:
            return datasets[f"{name}.json"]
        
        # Try partial match
        for key, info in datasets.items():
            if name in key:
                return info
        
        return {"error": f"Dataset info not found: {name}"}
    
    def validate_reference_data(self, computed_data: Dict, 
                               tolerance_relative: float = 1e-10,
                               tolerance_absolute: float = 1e-12) -> Dict:
        """
        Validate computed data against reference values.
        
        Parameters:
        -----------
        computed_data : dict
            Dictionary with 'ell', 'C_TT', 'C_EE', 'C_TE' arrays
        tolerance_relative : float
            Relative tolerance for comparison
        tolerance_absolute : float
            Absolute tolerance for comparison
        
        Returns:
        --------
        dict
            Validation results
        """
        reference = self.load_reference_data()
        ref_data = reference['data']
        
        results = {
            'passed': True,
            'details': {},
            'tolerances': {
                'relative': tolerance_relative,
                'absolute': tolerance_absolute
            }
        }
        
        # Check each spectrum type
        for spectrum in ['C_TT', 'C_EE', 'C_TE']:
            if spectrum not in computed_data:
                results['details'][spectrum] = {'error': 'Missing from computed data'}
                results['passed'] = False
                continue
            
            ref_values = np.array(ref_data[spectrum])
            comp_values = np.array(computed_data[spectrum])
            
            if len(ref_values) != len(comp_values):
                results['details'][spectrum] = {
                    'error': f'Length mismatch: ref={len(ref_values)}, computed={len(comp_values)}'
                }
                results['passed'] = False
                continue
            
            # Compute relative and absolute differences
            abs_diff = np.abs(comp_values - ref_values)
            rel_diff = abs_diff / (np.abs(ref_values) + tolerance_absolute)
            
            max_abs_diff = np.max(abs_diff)
            max_rel_diff = np.max(rel_diff)
            
            passed_abs = max_abs_diff <= tolerance_absolute
            passed_rel = max_rel_diff <= tolerance_relative
            passed_spectrum = passed_abs or passed_rel
            
            results['details'][spectrum] = {
                'passed': passed_spectrum,
                'max_absolute_difference': float(max_abs_diff),
                'max_relative_difference': float(max_rel_diff),
                'passed_absolute': passed_abs,
                'passed_relative': passed_rel
            }
            
            if not passed_spectrum:
                results['passed'] = False
        
        return results


def load_sample_data(dataset_name: str = "minimal_test", 
                    data_dir: Union[str, Path] = None) -> pd.DataFrame:
    """
    Convenience function to load sample data.
    
    Parameters:
    -----------
    dataset_name : str
        Name of dataset to load ('minimal_test', 'planck_like', or custom name)
    data_dir : str or Path, optional
        Path to data directory
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    loader = QFDDataLoader(data_dir)
    
    if dataset_name == "minimal_test":
        return loader.load_minimal_test_data()
    elif dataset_name == "planck_like":
        return loader.load_planck_like_data()
    else:
        return loader.load_dataset(dataset_name)


def create_mock_observational_data(base_data: pd.DataFrame, 
                                  noise_level: float = 0.05,
                                  random_seed: int = 42) -> pd.DataFrame:
    """
    Create mock observational data by adding noise to theoretical spectra.
    
    Parameters:
    -----------
    base_data : pd.DataFrame
        Base theoretical data
    noise_level : float
        Relative noise level (default: 5%)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Mock observational data with noise and error bars
    """
    np.random.seed(random_seed)
    
    mock_data = base_data.copy()
    
    # Add noise to each spectrum type
    for spectrum in ['C_TT', 'C_EE', 'C_TE']:
        if spectrum in mock_data.columns:
            true_values = mock_data[spectrum].values
            errors = noise_level * np.abs(true_values)
            noisy_values = true_values + np.random.normal(0, errors)
            
            # Update values and add error columns
            mock_data[spectrum] = noisy_values
            mock_data[f'error_{spectrum[2:]}'] = errors
    
    return mock_data


def extract_multipole_range(data: pd.DataFrame, 
                           ell_min: int = None, 
                           ell_max: int = None) -> pd.DataFrame:
    """
    Extract a specific multipole range from dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    ell_min : int, optional
        Minimum multipole (inclusive)
    ell_max : int, optional
        Maximum multipole (inclusive)
    
    Returns:
    --------
    pd.DataFrame
        Filtered dataset
    """
    if 'ell' not in data.columns:
        raise ValueError("Dataset must contain 'ell' column")
    
    mask = np.ones(len(data), dtype=bool)
    
    if ell_min is not None:
        mask &= (data['ell'] >= ell_min)
    
    if ell_max is not None:
        mask &= (data['ell'] <= ell_max)
    
    return data[mask].copy()


def compute_spectrum_statistics(data: pd.DataFrame) -> Dict:
    """
    Compute basic statistics for CMB spectra.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset with CMB spectra
    
    Returns:
    --------
    dict
        Dictionary with statistics for each spectrum
    """
    stats = {}
    
    for spectrum in ['C_TT', 'C_EE', 'C_TE']:
        if spectrum in data.columns:
            values = data[spectrum].values
            ells = data['ell'].values
            
            # Compute Dl = l(l+1)Cl
            Dl_values = ells * (ells + 1) * values
            
            stats[spectrum] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'peak_ell': int(ells[np.argmax(Dl_values)]),
                'peak_Dl': float(np.max(Dl_values))
            }
    
    return stats


def save_dataset(data: Union[pd.DataFrame, Dict], 
                filename: str, 
                output_dir: Union[str, Path] = "data/sample",
                format: str = "auto") -> Path:
    """
    Save dataset to file.
    
    Parameters:
    -----------
    data : pd.DataFrame or dict
        Data to save
    filename : str
        Output filename
    output_dir : str or Path
        Output directory
    format : str
        Output format ('csv', 'json', or 'auto')
    
    Returns:
    --------
    Path
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine format
    if format == "auto":
        if isinstance(data, pd.DataFrame):
            format = "csv"
        else:
            format = "json"
    
    # Add extension if not present
    if not filename.endswith(('.csv', '.json')):
        filename += f".{format}"
    
    output_path = output_dir / filename
    
    # Save data
    if format == "csv":
        if not isinstance(data, pd.DataFrame):
            raise ValueError("CSV format requires pandas DataFrame")
        data.to_csv(output_path, index=False, float_format='%.6e')
    elif format == "json":
        if isinstance(data, pd.DataFrame):
            data_dict = data.to_dict('records')
        else:
            data_dict = data
        
        with open(output_path, 'w') as f:
            json.dump(data_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return output_path


# Example usage and testing
if __name__ == "__main__":
    print("QFD CMB Data Utilities - Example Usage")
    print("=" * 40)
    
    # Initialize loader
    loader = QFDDataLoader()
    
    # List available datasets
    print("Available datasets:")
    datasets = loader.list_available_datasets()
    for dataset in datasets:
        print(f"  - {dataset}")
    
    # Try to load minimal test data
    try:
        print("\nLoading minimal test data...")
        minimal_data = loader.load_minimal_test_data()
        print(f"Loaded {len(minimal_data)} data points")
        print(f"Columns: {list(minimal_data.columns)}")
        
        # Compute statistics
        stats = compute_spectrum_statistics(minimal_data)
        print("\nSpectrum statistics:")
        for spectrum, stat_dict in stats.items():
            print(f"  {spectrum}: peak at ell={stat_dict['peak_ell']}, Dl={stat_dict['peak_Dl']:.2e}")
        
    except FileNotFoundError as e:
        print(f"Sample data not found: {e}")
        print("Run 'python generate_sample_data.py' to create sample datasets")