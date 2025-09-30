#!/usr/bin/env python3
"""
Numerical Safety Utilities for QVD Calculations
==============================================

Provides robust mathematical functions to prevent NaN/Inf values in 
scientific calculations, particularly for the E144-scaled supernova
QVD scattering model.

Copyright © 2025 PhaseSpace. All rights reserved.
"""

import numpy as np
import warnings
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

def safe_power(base: Union[float, np.ndarray], 
               exponent: Union[float, np.ndarray], 
               min_base: float = 1e-30,
               max_exponent: float = 700.0) -> Union[float, np.ndarray]:
    """
    Safely compute power operations avoiding negative/zero bases and overflow.
    
    Args:
        base: Base value(s) for power operation
        exponent: Exponent value(s)
        min_base: Minimum allowed base value (prevents zero/negative)
        max_exponent: Maximum allowed exponent (prevents overflow)
        
    Returns:
        Safe power result, always finite and positive
    """
    # Ensure base is positive and above minimum
    safe_base = np.maximum(np.abs(base), min_base)
    
    # Cap exponent to prevent overflow
    safe_exponent = np.clip(exponent, -max_exponent, max_exponent)
    
    try:
        result = np.power(safe_base, safe_exponent)
        
        # Check for any remaining non-finite values
        if np.any(~np.isfinite(result)):
            logger.warning(f"Non-finite values detected in safe_power, using fallback")
            result = np.where(np.isfinite(result), result, 1.0)
            
        return result
        
    except (OverflowError, ZeroDivisionError, ValueError) as e:
        logger.warning(f"Exception in safe_power: {e}, returning safe fallback")
        return np.ones_like(safe_base) if hasattr(safe_base, 'shape') else 1.0


def safe_log10(value: Union[float, np.ndarray], 
               min_value: float = 1e-30) -> Union[float, np.ndarray]:
    """
    Safely compute log10 avoiding zero/negative arguments.
    
    Args:
        value: Input value(s) for logarithm
        min_value: Minimum allowed value (prevents log of zero/negative)
        
    Returns:
        Safe log10 result, always finite
    """
    # Ensure value is positive and above minimum
    safe_value = np.maximum(np.abs(value), min_value)
    
    try:
        result = np.log10(safe_value)
        
        # Check for any remaining non-finite values
        if np.any(~np.isfinite(result)):
            logger.warning(f"Non-finite values detected in safe_log10, using fallback")
            result = np.where(np.isfinite(result), result, np.log10(min_value))
            
        return result
        
    except (ValueError, OverflowError) as e:
        logger.warning(f"Exception in safe_log10: {e}, returning safe fallback")
        fallback = np.log10(min_value)
        return np.full_like(safe_value, fallback) if hasattr(safe_value, 'shape') else fallback


def safe_exp(exponent: Union[float, np.ndarray], 
             max_exponent: float = 700.0,
             min_exponent: float = -700.0) -> Union[float, np.ndarray]:
    """
    Safely compute exponential avoiding overflow/underflow.
    
    Args:
        exponent: Exponent value(s)
        max_exponent: Maximum allowed exponent (prevents overflow)
        min_exponent: Minimum allowed exponent (prevents underflow to zero)
        
    Returns:
        Safe exponential result, always finite and positive
    """
    # Cap exponent to prevent overflow/underflow
    safe_exponent = np.clip(exponent, min_exponent, max_exponent)
    
    try:
        result = np.exp(safe_exponent)
        
        # Check for any remaining non-finite values
        if np.any(~np.isfinite(result)):
            logger.warning(f"Non-finite values detected in safe_exp, using fallback")
            result = np.where(np.isfinite(result), result, 1.0)
            
        return result
        
    except (OverflowError, ValueError) as e:
        logger.warning(f"Exception in safe_exp: {e}, returning safe fallback")
        return np.ones_like(safe_exponent) if hasattr(safe_exponent, 'shape') else 1.0


def safe_divide(numerator: Union[float, np.ndarray], 
                denominator: Union[float, np.ndarray],
                min_denominator: float = 1e-30) -> Union[float, np.ndarray]:
    """
    Safely perform division avoiding divide-by-zero.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        min_denominator: Minimum allowed denominator value
        
    Returns:
        Safe division result, always finite
    """
    # Ensure denominator is not zero
    safe_denominator = np.where(np.abs(denominator) < min_denominator, 
                               min_denominator, denominator)
    
    try:
        result = numerator / safe_denominator
        
        # Check for any remaining non-finite values
        if np.any(~np.isfinite(result)):
            logger.warning(f"Non-finite values detected in safe_divide, using fallback")
            result = np.where(np.isfinite(result), result, 0.0)
            
        return result
        
    except (ZeroDivisionError, ValueError) as e:
        logger.warning(f"Exception in safe_divide: {e}, returning zero")
        return np.zeros_like(numerator) if hasattr(numerator, 'shape') else 0.0


def validate_finite(value: Union[float, np.ndarray], 
                   name: str = "value",
                   replace_with: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
    """
    Validate that all values are finite, optionally replacing non-finite values.
    
    Args:
        value: Value(s) to validate
        name: Name for logging purposes
        replace_with: Value to replace non-finite entries with (if None, raises warning)
        
    Returns:
        Validated value with finite entries
    """
    if not np.all(np.isfinite(value)):
        non_finite_count = np.sum(~np.isfinite(value))
        logger.warning(f"Found {non_finite_count} non-finite values in {name}")
        
        if replace_with is not None:
            return np.where(np.isfinite(value), value, replace_with)
        else:
            warnings.warn(f"Non-finite values detected in {name}")
    
    return value


def clamp_to_range(value: Union[float, np.ndarray], 
                   min_val: float, 
                   max_val: float,
                   name: str = "value") -> Union[float, np.ndarray]:
    """
    Clamp values to a specified range.
    
    Args:
        value: Value(s) to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name for logging purposes
        
    Returns:
        Clamped values within [min_val, max_val]
    """
    clamped = np.clip(value, min_val, max_val)
    
    # Log if clamping occurred
    if np.any(value != clamped):
        clamped_count = np.sum(value != clamped)
        logger.debug(f"Clamped {clamped_count} values in {name} to range [{min_val}, {max_val}]")
    
    return clamped


def safe_sqrt(value: Union[float, np.ndarray], 
              min_value: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safely compute square root avoiding negative arguments.
    
    Args:
        value: Input value(s) for square root
        min_value: Minimum allowed value (prevents sqrt of negative)
        
    Returns:
        Safe square root result, always finite and non-negative
    """
    # Ensure value is non-negative
    safe_value = np.maximum(value, min_value)
    
    try:
        result = np.sqrt(safe_value)
        
        # Check for any remaining non-finite values
        if np.any(~np.isfinite(result)):
            logger.warning(f"Non-finite values detected in safe_sqrt, using fallback")
            result = np.where(np.isfinite(result), result, 0.0)
            
        return result
        
    except (ValueError, OverflowError) as e:
        logger.warning(f"Exception in safe_sqrt: {e}, returning safe fallback")
        return np.zeros_like(safe_value) if hasattr(safe_value, 'shape') else 0.0


# Convenience function for common scientific operations
def safe_scientific_operation(operation: str, *args, **kwargs) -> Union[float, np.ndarray]:
    """
    Dispatcher for common safe scientific operations.
    
    Args:
        operation: Name of operation ('power', 'log10', 'exp', 'divide', 'sqrt')
        *args: Arguments for the operation
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Result of safe operation
    """
    operations = {
        'power': safe_power,
        'log10': safe_log10,
        'exp': safe_exp,
        'divide': safe_divide,
        'sqrt': safe_sqrt
    }
    
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}. Available: {list(operations.keys())}")
    
    return operations[operation](*args, **kwargs)