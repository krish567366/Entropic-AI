"""
License Enforcement Mixins and Base Classes

These classes ensure NO instantiation occurs without proper licensing.
"""

from .validation import validate_or_fail
from .license_manager import LicenseError


class LicenseEnforcedMixin:
    """
    Mixin that enforces licensing on class instantiation.
    
    NO class inheriting from this can be instantiated without a valid license.
    """
    
    _required_features = ['core']
    
    def __new__(cls, *args, **kwargs):
        """Override __new__ to enforce licensing before object creation."""
        # Determine required features for this class
        required_features = getattr(cls, '_required_features', ['core'])
        
        try:
            # Validate license BEFORE creating object
            validate_or_fail(required_features)
        except LicenseError as e:
            raise RuntimeError(
                f"License required to instantiate {cls.__name__}. "
                f"Required features: {required_features}. "
                f"Error: {e}. "
                f"Contact bajpaikrishna715@gmail.com for licensing."
            )
        
        # Only create object if license is valid
        return super().__new__(cls)


class LicenseEnforcedMetaclass(type):
    """
    Metaclass that enforces licensing on ALL class operations.
    
    Prevents ANY class access without proper licensing.
    """
    
    def __call__(cls, *args, **kwargs):
        """Override class call to enforce licensing."""
        # Get required features for this class
        required_features = getattr(cls, '_required_features', ['core'])
        
        try:
            # Validate license before ANY class instantiation
            validate_or_fail(required_features)
        except LicenseError as e:
            raise RuntimeError(
                f"❌ LICENSE REQUIRED: Cannot instantiate {cls.__name__}\n"
                f"Required features: {required_features}\n"
                f"License error: {e}\n"
                f"📧 Contact bajpaikrishna715@gmail.com for licensing"
            )
        
        # Only proceed if license is valid
        return super().__call__(*args, **kwargs)
    
    def __getattribute__(cls, name):
        """Override attribute access to enforce licensing on class methods."""
        attr = super().__getattribute__(name)
        
        # Check if this is a method that requires licensing
        if callable(attr) and hasattr(attr, '_license_features'):
            required_features = attr._license_features
            
            def license_enforced_method(*args, **kwargs):
                try:
                    validate_or_fail(required_features)
                    return attr(*args, **kwargs)
                except LicenseError as e:
                    raise RuntimeError(
                        f"❌ LICENSE REQUIRED: Cannot access {cls.__name__}.{name}\n"
                        f"Required features: {required_features}\n"
                        f"📧 Contact bajpaikrishna715@gmail.com"
                    )
            
            return license_enforced_method
        
        return attr


class CoreLicenseEnforcedBase(LicenseEnforcedMixin):
    """Base class for core E-AI components requiring 'core' license."""
    _required_features = ['core']


class BasicNetworkLicenseEnforcedBase(LicenseEnforcedMixin):
    """Base class for basic network components."""
    _required_features = ['core', 'basic_networks']


class AdvancedNetworkLicenseEnforcedBase(LicenseEnforcedMixin):
    """Base class for advanced network components."""
    _required_features = ['core', 'advanced_networks']


class ApplicationLicenseEnforcedBase(LicenseEnforcedMixin):
    """Base class for application components."""
    _required_features = ['core', 'applications']


class MoleculeEvolutionLicenseEnforcedBase(LicenseEnforcedMixin):
    """Base class for molecule evolution components."""
    _required_features = ['core', 'molecule_evolution']


class CircuitEvolutionLicenseEnforcedBase(LicenseEnforcedMixin):
    """Base class for circuit evolution components."""
    _required_features = ['core', 'circuit_evolution']


class TheoryDiscoveryLicenseEnforcedBase(LicenseEnforcedMixin):
    """Base class for theory discovery components."""
    _required_features = ['core', 'theory_discovery']


class EnterpriseLicenseEnforcedBase(LicenseEnforcedMixin):
    """Base class for enterprise-only components."""
    _required_features = ['core', 'enterprise']


def enforce_license_on_module_access(module_name, required_features):
    """
    Decorator to enforce licensing on entire module access.
    
    Args:
        module_name: Name of the module being protected
        required_features: List of features required to access this module
    """
    def decorator(module_dict):
        # Validate license when module is accessed
        try:
            validate_or_fail(required_features)
        except LicenseError as e:
            # Replace all module contents with license requirement messages
            for key in list(module_dict.keys()):
                if not key.startswith('_'):
                    def license_required_stub(*args, **kwargs):
                        raise RuntimeError(
                            f"❌ LICENSE REQUIRED: Module '{module_name}' requires features: {required_features}\n"
                            f"License error: {e}\n"
                            f"📧 Contact bajpaikrishna715@gmail.com"
                        )
                    module_dict[key] = license_required_stub
        
        return module_dict
    
    return decorator


class LicenseEnforcedProperty:
    """Property descriptor that enforces licensing on attribute access."""
    
    def __init__(self, required_features, getter_func):
        self.required_features = required_features
        self.getter_func = getter_func
    
    def __get__(self, obj, objtype=None):
        """Enforce license on property access."""
        try:
            validate_or_fail(self.required_features)
            return self.getter_func(obj)
        except LicenseError as e:
            raise RuntimeError(
                f"❌ LICENSE REQUIRED: Property access requires features: {self.required_features}\n"
                f"License error: {e}\n"
                f"📧 Contact bajpaikrishna715@gmail.com"
            )
    
    def __set__(self, obj, value):
        """Enforce license on property setting."""
        try:
            validate_or_fail(self.required_features)
            # Property setting logic would go here
        except LicenseError as e:
            raise RuntimeError(
                f"❌ LICENSE REQUIRED: Property modification requires features: {self.required_features}\n"
                f"📧 Contact bajpaikrishna715@gmail.com"
            )


def license_protected_import(module_name, required_features):
    """
    Function to perform license-protected imports.
    
    Returns None if license is insufficient, preventing import.
    """
    try:
        validate_or_fail(required_features)
        return __import__(module_name)
    except LicenseError:
        print(f"⚠️  Module '{module_name}' requires license features: {required_features}")
        print("Contact bajpaikrishna715@gmail.com for licensing information.")
        return None
