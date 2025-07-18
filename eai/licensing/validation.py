"""
License Validation Functions for Entropic AI

Provides validation functions and status checking for all E-AI components.
"""

from typing import List, Optional, Dict, Any
from .license_manager import get_license_manager, LicenseError


def validate_license(features: Optional[List[str]] = None) -> bool:
    """
    Validate license for E-AI usage.
    
    Args:
        features: List of required features. If None, validates basic access.
        
    Returns:
        True if license is valid
        
    Raises:
        LicenseError: If license validation fails
    """
    manager = get_license_manager()
    
    if features is None:
        features = ['core']
    
    return manager.validate_features(features)


def requires_license(features: Optional[List[str]] = None) -> bool:
    """
    Check if license is required and valid for given features.
    
    Args:
        features: List of required features
        
    Returns:
        True if license is valid, False otherwise
    """
    try:
        return validate_license(features)
    except LicenseError:
        return False


def check_license_status() -> Dict[str, Any]:
    """
    Get comprehensive license status information.
    
    Returns:
        Dictionary with license status details
    """
    manager = get_license_manager()
    return manager.get_license_info()


def get_available_features() -> List[str]:
    """
    Get list of features available in current license.
    
    Returns:
        List of available feature names
    """
    manager = get_license_manager()
    return manager.get_available_features()


def show_license_info():
    """Display license information to user."""
    manager = get_license_manager()
    info = manager.get_license_info()
    
    print("\n🔐 Entropic AI License Information")
    print("=" * 50)
    
    if info["status"] == "no_license":
        print("❌ No valid license found")
        print("\n📧 To obtain a license:")
        print("   Email: bajpaikrishna715@gmail.com")
        print("   Web: https://entropicai.com/licensing")
        
    elif info["status"] == "expired":
        print(f"⏰ License expired on {info['expiry_date']}")
        print(f"   ({info['days_expired']} days ago)")
        print("\n📧 To renew your license:")
        print("   Email: bajpaikrishna715@gmail.com")
        print("   Subject: License Renewal Request")
        
    elif info["status"] == "valid":
        print(f"✅ Valid {info['license_type'].title()} License")
        print(f"👤 Licensed to: {info['user_name']}")
        if info.get('organization'):
            print(f"🏢 Organization: {info['organization']}")
        print(f"📅 Expires: {info['expiry_date']}")
        print(f"⏳ Days remaining: {info['days_remaining']}")
        print(f"🔧 Features: {', '.join(info['features'])}")
        
        if info['days_remaining'] <= 7:
            print(f"\n⚠️  License expires soon! Renew at bajpaikrishna715@gmail.com")
    
    print("=" * 50)


def validate_or_fail(features: Optional[List[str]] = None):
    """
    Validate license or raise descriptive error.
    
    Args:
        features: List of required features
        
    Raises:
        LicenseError: With user-friendly error message
    """
    try:
        validate_license(features)
    except LicenseError as e:
        print("\n" + "="*60)
        print("🚫 ENTROPIC AI LICENSE REQUIRED")
        print("="*60)
        print(f"Error: {e}")
        print("\n📧 For licensing information:")
        print("   Email: bajpaikrishna715@gmail.com")
        print("   Web: https://entropicai.com/licensing")
        print("   Phone: +1-XXX-XXX-XXXX")
        print("\n💼 Available License Types:")
        print("   • Academic: For research and education")
        print("   • Professional: For commercial development")  
        print("   • Enterprise: Full features + support")
        print("="*60)
        raise
