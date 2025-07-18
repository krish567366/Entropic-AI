"""
Entropic AI License Management System

This module provides comprehensive license validation for all E-AI features.
No functionality is available without a valid license.

Author: Krishna Bajpai <bajpaikrishna715@gmail.com>
License: Proprietary - Patent Pending
"""

# License enforcement on import - NO BYPASS ALLOWED
try:
    from .license_manager import LicenseManager, LicenseError, FeatureNotLicensedError
    from .validation import validate_license, requires_license, check_license_status
    from .decorators import licensed_function, licensed_class, licensed_method
    
    # Additional imports that require validation
    from .license_manager import (
        LicenseExpiredError,
        LicenseNotFoundError, 
        InvalidLicenseError,
        get_license_manager
    )
    from .validation import validate_or_fail, show_license_info
    from .decorators import (
        licensed_property,
        LicenseEnforcer,
        require_features,
        enterprise_only,
        professional_or_higher,
        academic_allowed
    )
    from .enforcement import (
        LicenseEnforcedMixin,
        LicenseEnforcedMetaclass,
        CoreLicenseEnforcedBase,
        BasicNetworkLicenseEnforcedBase,
        AdvancedNetworkLicenseEnforcedBase,
        ApplicationLicenseEnforcedBase,
        MoleculeEvolutionLicenseEnforcedBase,
        CircuitEvolutionLicenseEnforcedBase,
        TheoryDiscoveryLicenseEnforcedBase,
        EnterpriseLicenseEnforcedBase,
        enforce_license_on_module_access,
        LicenseEnforcedProperty,
        license_protected_import
    )
    
except ImportError as e:
    # Block all licensing module access if imports fail
    raise SystemExit(f"License system initialization failed: {e}")

__all__ = [
    "LicenseManager",
    "LicenseError", 
    "FeatureNotLicensedError",
    "LicenseExpiredError",
    "LicenseNotFoundError",
    "InvalidLicenseError",
    "get_license_manager",
    "validate_license",
    "validate_or_fail",
    "requires_license",
    "check_license_status",
    "show_license_info",
    "licensed_function",
    "licensed_class", 
    "licensed_method",
    "licensed_property",
    "LicenseEnforcer",
    "require_features",
    "enterprise_only",
    "professional_or_higher",
    "academic_allowed",
    # Enforcement classes
    "LicenseEnforcedMixin",
    "LicenseEnforcedMetaclass",
    "CoreLicenseEnforcedBase",
    "BasicNetworkLicenseEnforcedBase",
    "AdvancedNetworkLicenseEnforcedBase",
    "ApplicationLicenseEnforcedBase",
    "MoleculeEvolutionLicenseEnforcedBase",
    "CircuitEvolutionLicenseEnforcedBase",
    "TheoryDiscoveryLicenseEnforcedBase",
    "EnterpriseLicenseEnforcedBase",
    "enforce_license_on_module_access",
    "LicenseEnforcedProperty",
    "license_protected_import"
]
