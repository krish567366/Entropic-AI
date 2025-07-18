"""
License Decorators for Entropic AI

Provides decorators to enforce licensing on functions, methods, and classes.
"""

from functools import wraps
from typing import List, Optional, Callable, Any
from .validation import validate_or_fail


def licensed_function(features: Optional[List[str]] = None):
    """
    Decorator to require license validation for functions.
    
    Args:
        features: List of required features for this function
        
    Usage:
        @licensed_function(['advanced_optimization'])
        def advanced_evolve():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate license before function execution
            validate_or_fail(features)
            return func(*args, **kwargs)
        
        # Store license requirements on function
        wrapper._license_features = features or ['core']
        return wrapper
    return decorator


def licensed_method(features: Optional[List[str]] = None):
    """
    Decorator to require license validation for methods.
    
    Args:
        features: List of required features for this method
        
    Usage:
        @licensed_method(['molecule_evolution'])
        def evolve_molecule(self):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Validate license before method execution
            validate_or_fail(features)
            return func(self, *args, **kwargs)
        
        # Store license requirements on method
        wrapper._license_features = features or ['core']
        return wrapper
    return decorator


def licensed_class(features: Optional[List[str]] = None):
    """
    Class decorator to require license validation for class instantiation.
    
    Args:
        features: List of required features for this class
        
    Usage:
        @licensed_class(['advanced_networks'])
        class AdvancedNetwork:
            pass
    """
    def decorator(cls):
        original_init = cls.__init__
        
        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Validate license before class instantiation
            validate_or_fail(features)
            original_init(self, *args, **kwargs)
        
        cls.__init__ = new_init
        cls._license_features = features or ['core']
        return cls
    return decorator


def licensed_property(features: Optional[List[str]] = None):
    """
    Decorator to require license validation for property access.
    
    Args:
        features: List of required features for this property
        
    Usage:
        @licensed_property(['visualization'])
        def advanced_plot(self):
            pass
    """
    def decorator(func: Callable) -> property:
        def getter(self):
            validate_or_fail(features)
            return func(self)
        
        return property(getter)
    return decorator


class LicenseEnforcer:
    """
    Context manager for license-enforced code blocks.
    
    Usage:
        with LicenseEnforcer(['enterprise']):
            # Enterprise-only code here
            pass
    """
    
    def __init__(self, features: Optional[List[str]] = None):
        self.features = features or ['core']
    
    def __enter__(self):
        validate_or_fail(self.features)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Nothing to clean up
        pass


def require_features(*features: str):
    """
    Simplified decorator for single or multiple features.
    
    Usage:
        @require_features('molecule_evolution', 'advanced_optimization')
        def complex_evolution():
            pass
    """
    return licensed_function(list(features))


def enterprise_only(func: Callable) -> Callable:
    """Decorator for enterprise-only features."""
    return licensed_function(['enterprise'])(func)


def professional_or_higher(func: Callable) -> Callable:
    """Decorator for professional+ features."""
    return licensed_function(['professional'])(func)


def academic_allowed(func: Callable) -> Callable:
    """Decorator for academic+ features."""
    return licensed_function(['academic'])(func)
