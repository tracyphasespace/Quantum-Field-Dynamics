#!/usr/bin/env python3
"""
Comprehensive Error Handling and Logging System
==============================================

Provides robust error handling, logging, and warning systems for the
E144-scaled supernova QVD model to help diagnose numerical issues.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

import logging
import warnings
import numpy as np
from typing import Union, Optional, Dict, Any, Callable
from functools import wraps
import traceback
from pathlib import Path
import json
from datetime import datetime

# Configure logging format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class QVDModelLogger:
    """
    Specialized logger for QVD model operations with context tracking.
    """
    
    def __init__(self, name: str = "QVDModel", level: int = logging.INFO):
        """
        Initialize QVD model logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Context tracking
        self._context = {}
        self._error_counts = {}
        self._warning_counts = {}
    
    def set_context(self, **kwargs):
        """Set context information for logging"""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear context information"""
        self._context.clear()
    
    def _format_message(self, message: str) -> str:
        """Format message with context"""
        if self._context:
            context_str = ", ".join([f"{k}={v}" for k, v in self._context.items()])
            return f"{message} [Context: {context_str}]"
        return message
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.set_context(**kwargs)
        self.logger.debug(self._format_message(message))
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.set_context(**kwargs)
        self.logger.info(self._format_message(message))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context and count"""
        self.set_context(**kwargs)
        formatted_msg = self._format_message(message)
        self.logger.warning(formatted_msg)
        
        # Count warnings
        warning_key = message.split()[0] if message else "unknown"
        self._warning_counts[warning_key] = self._warning_counts.get(warning_key, 0) + 1
    
    def error(self, message: str, **kwargs):
        """Log error message with context and count"""
        self.set_context(**kwargs)
        formatted_msg = self._format_message(message)
        self.logger.error(formatted_msg)
        
        # Count errors
        error_key = message.split()[0] if message else "unknown"
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context"""
        self.set_context(**kwargs)
        self.logger.critical(self._format_message(message))
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts"""
        return self._error_counts.copy()
    
    def get_warning_summary(self) -> Dict[str, int]:
        """Get summary of warning counts"""
        return self._warning_counts.copy()
    
    def reset_counts(self):
        """Reset error and warning counts"""
        self._error_counts.clear()
        self._warning_counts.clear()


class NumericalIssueWarning(UserWarning):
    """Warning for numerical stability issues"""
    pass


class PhysicalBoundsWarning(UserWarning):
    """Warning for physical bounds violations"""
    pass


class ModelConsistencyWarning(UserWarning):
    """Warning for model consistency issues"""
    pass


def handle_numerical_errors(func: Callable) -> Callable:
    """
    Decorator to handle numerical errors gracefully.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            # Check for NaN/Inf in results
            if isinstance(result, (int, float)):
                if not np.isfinite(result):
                    logger = QVDModelLogger()
                    logger.warning(f"Non-finite result from {func.__name__}: {result}")
                    warnings.warn(
                        f"Non-finite result from {func.__name__}",
                        NumericalIssueWarning,
                        stacklevel=2
                    )
            elif isinstance(result, np.ndarray):
                if not np.all(np.isfinite(result)):
                    non_finite_count = np.sum(~np.isfinite(result))
                    logger = QVDModelLogger()
                    logger.warning(f"Non-finite results from {func.__name__}: {non_finite_count}/{len(result)}")
                    warnings.warn(
                        f"Non-finite results from {func.__name__}",
                        NumericalIssueWarning,
                        stacklevel=2
                    )
            elif isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float, np.ndarray)):
                        if not np.all(np.isfinite(value)):
                            logger = QVDModelLogger()
                            logger.warning(f"Non-finite result in {func.__name__}.{key}: {value}")
                            warnings.warn(
                                f"Non-finite result in {func.__name__}.{key}",
                                NumericalIssueWarning,
                                stacklevel=2
                            )
            
            return result
            
        except (ZeroDivisionError, ValueError, OverflowError) as e:
            logger = QVDModelLogger()
            logger.error(f"Numerical error in {func.__name__}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Return safe fallback value based on function name
            if 'cross_section' in func.__name__:
                return 1e-30  # Minimum cross-section
            elif 'transmission' in func.__name__:
                return 0.1    # Reasonable transmission
            elif 'dimming' in func.__name__:
                return 1.0    # Reasonable dimming
            elif 'density' in func.__name__:
                return 1e15   # Reasonable density
            elif 'temperature' in func.__name__:
                return 1e4    # Reasonable temperature
            else:
                # Generic fallback
                return 0.0
            
        except Exception as e:
            logger = QVDModelLogger()
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
    
    return wrapper


def validate_input_parameters(func: Callable) -> Callable:
    """
    Decorator to validate input parameters.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with input validation
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = QVDModelLogger()
        
        # Check for NaN/Inf in inputs
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float)):
                if not np.isfinite(arg):
                    logger.warning(f"Non-finite input to {func.__name__} at position {i}: {arg}")
                    warnings.warn(
                        f"Non-finite input to {func.__name__}",
                        NumericalIssueWarning,
                        stacklevel=2
                    )
            elif isinstance(arg, np.ndarray):
                if not np.all(np.isfinite(arg)):
                    non_finite_count = np.sum(~np.isfinite(arg))
                    logger.warning(f"Non-finite inputs to {func.__name__} at position {i}: {non_finite_count}/{len(arg)}")
                    warnings.warn(
                        f"Non-finite inputs to {func.__name__}",
                        NumericalIssueWarning,
                        stacklevel=2
                    )
        
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                if not np.isfinite(value):
                    logger.warning(f"Non-finite input to {func.__name__} parameter {key}: {value}")
                    warnings.warn(
                        f"Non-finite input parameter {key} to {func.__name__}",
                        NumericalIssueWarning,
                        stacklevel=2
                    )
        
        return func(*args, **kwargs)
    
    return wrapper


def log_bounds_violations(parameter_name: str, original_value: Union[float, np.ndarray], 
                         bounded_value: Union[float, np.ndarray], 
                         min_bound: float, max_bound: float):
    """
    Log bounds violations with detailed information.
    
    Args:
        parameter_name: Name of the parameter
        original_value: Original value before bounds enforcement
        bounded_value: Value after bounds enforcement
        min_bound: Minimum allowed value
        max_bound: Maximum allowed value
    """
    logger = QVDModelLogger()
    
    if np.any(original_value != bounded_value):
        if isinstance(original_value, np.ndarray):
            violation_count = np.sum(original_value != bounded_value)
            below_min = np.sum(original_value < min_bound)
            above_max = np.sum(original_value > max_bound)
            
            logger.warning(
                f"Bounds violation in {parameter_name}: {violation_count} values clamped "
                f"({below_min} below {min_bound:.2e}, {above_max} above {max_bound:.2e})"
            )
        else:
            if original_value < min_bound:
                logger.warning(
                    f"Bounds violation in {parameter_name}: {original_value:.2e} < {min_bound:.2e}, "
                    f"clamped to {bounded_value:.2e}"
                )
            elif original_value > max_bound:
                logger.warning(
                    f"Bounds violation in {parameter_name}: {original_value:.2e} > {max_bound:.2e}, "
                    f"clamped to {bounded_value:.2e}"
                )
        
        warnings.warn(
            f"Physical bounds violation in {parameter_name}",
            PhysicalBoundsWarning,
            stacklevel=3
        )


def log_model_inconsistency(parameter1_name: str, parameter1_value: float,
                           parameter2_name: str, parameter2_value: float,
                           expected_relationship: str, tolerance: float):
    """
    Log model consistency issues.
    
    Args:
        parameter1_name: Name of first parameter
        parameter1_value: Value of first parameter
        parameter2_name: Name of second parameter
        parameter2_value: Value of second parameter
        expected_relationship: Description of expected relationship
        tolerance: Tolerance for the relationship
    """
    logger = QVDModelLogger()
    
    logger.warning(
        f"Model inconsistency detected: {parameter1_name}={parameter1_value:.3e}, "
        f"{parameter2_name}={parameter2_value:.3e}. Expected: {expected_relationship} "
        f"(tolerance: {tolerance:.3e})"
    )
    
    warnings.warn(
        f"Model inconsistency between {parameter1_name} and {parameter2_name}",
        ModelConsistencyWarning,
        stacklevel=3
    )


class ErrorReporter:
    """
    Collects and reports errors and warnings from QVD model calculations.
    """
    
    def __init__(self):
        """Initialize error reporter"""
        self.errors = []
        self.warnings = []
        self.bounds_violations = []
        self.numerical_issues = []
        self.start_time = datetime.now()
    
    def add_error(self, error_type: str, message: str, context: Dict[str, Any] = None):
        """Add an error to the report"""
        self.errors.append({
            'type': error_type,
            'message': message,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def add_warning(self, warning_type: str, message: str, context: Dict[str, Any] = None):
        """Add a warning to the report"""
        self.warnings.append({
            'type': warning_type,
            'message': message,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def add_bounds_violation(self, parameter: str, original: float, bounded: float, 
                           min_bound: float, max_bound: float):
        """Add a bounds violation to the report"""
        self.bounds_violations.append({
            'parameter': parameter,
            'original_value': original,
            'bounded_value': bounded,
            'min_bound': min_bound,
            'max_bound': max_bound,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_numerical_issue(self, function: str, issue_type: str, details: str):
        """Add a numerical issue to the report"""
        self.numerical_issues.append({
            'function': function,
            'issue_type': issue_type,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        return {
            'report_metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings),
                'total_bounds_violations': len(self.bounds_violations),
                'total_numerical_issues': len(self.numerical_issues)
            },
            'errors': self.errors,
            'warnings': self.warnings,
            'bounds_violations': self.bounds_violations,
            'numerical_issues': self.numerical_issues,
            'summary': {
                'error_types': self._count_by_type(self.errors),
                'warning_types': self._count_by_type(self.warnings),
                'most_violated_bounds': self._count_bounds_violations(),
                'most_problematic_functions': self._count_numerical_issues()
            }
        }
    
    def _count_by_type(self, items: list) -> Dict[str, int]:
        """Count items by type"""
        counts = {}
        for item in items:
            item_type = item.get('type', 'unknown')
            counts[item_type] = counts.get(item_type, 0) + 1
        return counts
    
    def _count_bounds_violations(self) -> Dict[str, int]:
        """Count bounds violations by parameter"""
        counts = {}
        for violation in self.bounds_violations:
            param = violation.get('parameter', 'unknown')
            counts[param] = counts.get(param, 0) + 1
        return counts
    
    def _count_numerical_issues(self) -> Dict[str, int]:
        """Count numerical issues by function"""
        counts = {}
        for issue in self.numerical_issues:
            func = issue.get('function', 'unknown')
            counts[func] = counts.get(func, 0) + 1
        return counts
    
    def save_report(self, filepath: Union[str, Path]):
        """Save error report to JSON file"""
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def print_summary(self):
        """Print a summary of errors and warnings"""
        report = self.generate_report()
        metadata = report['report_metadata']
        
        print("="*60)
        print("QVD MODEL ERROR REPORT SUMMARY")
        print("="*60)
        print(f"Duration: {metadata['duration_seconds']:.2f} seconds")
        print(f"Total Errors: {metadata['total_errors']}")
        print(f"Total Warnings: {metadata['total_warnings']}")
        print(f"Total Bounds Violations: {metadata['total_bounds_violations']}")
        print(f"Total Numerical Issues: {metadata['total_numerical_issues']}")
        
        if report['summary']['error_types']:
            print("\nError Types:")
            for error_type, count in report['summary']['error_types'].items():
                print(f"  {error_type}: {count}")
        
        if report['summary']['warning_types']:
            print("\nWarning Types:")
            for warning_type, count in report['summary']['warning_types'].items():
                print(f"  {warning_type}: {count}")
        
        if report['summary']['most_violated_bounds']:
            print("\nMost Violated Bounds:")
            for param, count in report['summary']['most_violated_bounds'].items():
                print(f"  {param}: {count}")
        
        if report['summary']['most_problematic_functions']:
            print("\nMost Problematic Functions:")
            for func, count in report['summary']['most_problematic_functions'].items():
                print(f"  {func}: {count}")
        
        print("="*60)


# Global error reporter instance
global_error_reporter = ErrorReporter()


def setup_qvd_logging(level: int = logging.INFO, 
                     log_file: Optional[str] = None,
                     enable_warnings: bool = True):
    """
    Set up comprehensive logging for QVD model.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        enable_warnings: Whether to enable warning filters
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    # Configure warning filters
    if enable_warnings:
        warnings.filterwarnings('default', category=NumericalIssueWarning)
        warnings.filterwarnings('default', category=PhysicalBoundsWarning)
        warnings.filterwarnings('default', category=ModelConsistencyWarning)
    
    # Create QVD model logger
    qvd_logger = QVDModelLogger("QVDModel", level)
    qvd_logger.info("QVD model logging initialized")
    
    return qvd_logger