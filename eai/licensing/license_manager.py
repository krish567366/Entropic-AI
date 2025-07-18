"""
Core License Management for Entropic AI

Handles license validation, feature checking, and access control.
All E-AI functionality requires valid licensing.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from cryptography.fernet import Fernet
import uuid


class LicenseError(Exception):
    """Base exception for license-related errors."""
    pass


class LicenseExpiredError(LicenseError):
    """Raised when license has expired."""
    pass


class FeatureNotLicensedError(LicenseError):
    """Raised when feature is not included in license."""
    pass


class LicenseNotFoundError(LicenseError):
    """Raised when no valid license is found."""
    pass


class InvalidLicenseError(LicenseError):
    """Raised when license format or signature is invalid."""
    pass


class License:
    """Represents an E-AI license with features and expiration."""
    
    def __init__(self, license_data: Dict[str, Any]):
        self.license_id = license_data.get('license_id')
        self.user_name = license_data.get('user_name')
        self.organization = license_data.get('organization')
        self.email = license_data.get('email')
        self.issue_date = datetime.fromisoformat(license_data.get('issue_date'))
        self.expiry_date = datetime.fromisoformat(license_data.get('expiry_date'))
        self.features = license_data.get('features', [])
        self.license_type = license_data.get('license_type', 'basic')
        self.max_instances = license_data.get('max_instances', 1)
        self.signature = license_data.get('signature')
    
    def is_expired(self) -> bool:
        """Check if license has expired."""
        return datetime.now() > self.expiry_date
    
    def has_feature(self, feature: str) -> bool:
        """Check if license includes specific feature."""
        return feature in self.features
    
    def days_until_expiry(self) -> int:
        """Get number of days until license expires."""
        delta = self.expiry_date - datetime.now()
        return max(0, delta.days)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert license to dictionary."""
        return {
            'license_id': self.license_id,
            'user_name': self.user_name,
            'organization': self.organization,
            'email': self.email,
            'issue_date': self.issue_date.isoformat(),
            'expiry_date': self.expiry_date.isoformat(),
            'features': self.features,
            'license_type': self.license_type,
            'max_instances': self.max_instances,
            'signature': self.signature
        }


class LicenseManager:
    """Manages license loading, validation, and feature checking."""
    
    # Package name for license validation
    PACKAGE_NAME = "entropic-ai"
    
    # License tiers and their features
    LICENSE_TIERS = {
        'academic': [
            'core',
            'basic_networks',
            'basic_optimization',
            'visualization',
            'tutorials'
        ],
        'professional': [
            'core',
            'basic_networks',
            'advanced_networks', 
            'basic_optimization',
            'advanced_optimization',
            'visualization',
            'tutorials',
            'molecule_evolution',
            'circuit_evolution',
            'api_access'
        ],
        'enterprise': [
            'core',
            'basic_networks',
            'advanced_networks',
            'basic_optimization', 
            'advanced_optimization',
            'visualization',
            'tutorials',
            'molecule_evolution',
            'circuit_evolution',
            'theory_discovery',
            'custom_applications',
            'api_access',
            'batch_processing',
            'distributed_computing',
            'priority_support'
        ]
    }
    
    def __init__(self):
        self._license = None
        self._license_path = self._find_license_file()
        self._load_license()
    
    def _find_license_file(self) -> Optional[str]:
        """Find license file in various locations."""
        possible_paths = [
            # Current directory
            "./eai_license.json",
            "./license.eai",
            # User home directory
            os.path.expanduser("~/.eai/license.json"),
            os.path.expanduser("~/.eai_license"),
            # System-wide locations
            "/etc/eai/license.json",
            "C:\\ProgramData\\EntropicAI\\license.json",
            # Environment variable
            os.environ.get('EAI_LICENSE_PATH', '')
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
        
        return None
    
    def _load_license(self):
        """Load and validate license from file."""
        if not self._license_path:
            raise LicenseNotFoundError(
                "No valid license found. Please contact bajpaikrishna715@gmail.com for licensing."
            )
        
        try:
            with open(self._license_path, 'r') as f:
                license_data = json.load(f)
            
            # Validate license signature
            if not self._validate_signature(license_data):
                raise InvalidLicenseError("Invalid license signature")
            
            self._license = License(license_data)
            
            # Check if license is expired
            if self._license.is_expired():
                raise LicenseExpiredError(
                    f"License expired on {self._license.expiry_date.strftime('%Y-%m-%d')}. "
                    f"Please contact bajpaikrishna715@gmail.com for renewal."
                )
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise InvalidLicenseError(f"Invalid license format: {e}")
    
    def _validate_signature(self, license_data: Dict[str, Any]) -> bool:
        """Validate license cryptographic signature."""
        # Extract signature
        signature = license_data.get('signature')
        if not signature:
            return False
        
        # Create validation data (excluding signature)
        validation_data = {k: v for k, v in license_data.items() if k != 'signature'}
        validation_string = json.dumps(validation_data, sort_keys=True)
        
        # For production, use proper cryptographic verification
        # This is simplified for demo purposes
        expected_hash = hashlib.sha256(
            (validation_string + "ENTROPIC_AI_SECRET_KEY").encode()
        ).hexdigest()
        
        return signature == expected_hash
    
    def validate_feature(self, feature: str) -> bool:
        """Validate if current license includes specific feature."""
        if not self._license:
            raise LicenseNotFoundError("No valid license loaded")
        
        if self._license.is_expired():
            raise LicenseExpiredError("License has expired")
        
        if not self._license.has_feature(feature):
            raise FeatureNotLicensedError(
                f"Feature '{feature}' not included in {self._license.license_type} license. "
                f"Please upgrade your license at bajpaikrishna715@gmail.com"
            )
        
        return True
    
    def validate_features(self, features: List[str]) -> bool:
        """Validate multiple features at once."""
        for feature in features:
            self.validate_feature(feature)
        return True
    
    def get_license_info(self) -> Dict[str, Any]:
        """Get current license information."""
        if not self._license:
            return {"status": "no_license"}
        
        if self._license.is_expired():
            return {
                "status": "expired",
                "expiry_date": self._license.expiry_date.isoformat(),
                "days_expired": (datetime.now() - self._license.expiry_date).days
            }
        
        return {
            "status": "valid",
            "license_type": self._license.license_type,
            "user_name": self._license.user_name,
            "organization": self._license.organization,
            "features": self._license.features,
            "expiry_date": self._license.expiry_date.isoformat(),
            "days_remaining": self._license.days_until_expiry()
        }
    
    def get_available_features(self) -> List[str]:
        """Get list of features available in current license."""
        if not self._license or self._license.is_expired():
            return []
        return self._license.features.copy()
    
    def check_license_health(self) -> Dict[str, Any]:
        """Comprehensive license health check."""
        try:
            info = self.get_license_info()
            
            if info["status"] == "no_license":
                return {
                    "healthy": False,
                    "status": "no_license",
                    "message": "No license found. Please contact bajpaikrishna715@gmail.com",
                    "action_required": "obtain_license"
                }
            
            if info["status"] == "expired":
                return {
                    "healthy": False,
                    "status": "expired", 
                    "message": f"License expired {info['days_expired']} days ago",
                    "action_required": "renew_license"
                }
            
            days_remaining = info["days_remaining"]
            if days_remaining <= 7:
                return {
                    "healthy": True,
                    "status": "expiring_soon",
                    "message": f"License expires in {days_remaining} days",
                    "action_required": "renew_soon"
                }
            
            return {
                "healthy": True,
                "status": "valid",
                "message": f"License valid for {days_remaining} days",
                "action_required": None
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "status": "error",
                "message": str(e),
                "action_required": "contact_support"
            }


# Global license manager instance
_license_manager = None

def get_license_manager() -> LicenseManager:
    """Get global license manager instance."""
    global _license_manager
    if _license_manager is None:
        _license_manager = LicenseManager()
    return _license_manager
