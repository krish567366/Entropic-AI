"""
License Enforcement Test Suite

Tests to verify that licensing is properly enforced throughout E-AI.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eai.licensing import (
    LicenseManager,
    LicenseError,
    LicenseExpiredError,
    FeatureNotLicensedError,
    LicenseNotFoundError,
    validate_license,
    requires_license
)
from eai.licensing.decorators import licensed_function, licensed_class


class TestLicenseEnforcement:
    """Test suite for license enforcement."""
    
    def test_no_license_blocks_import(self):
        """Test that missing license blocks package import."""
        with patch('eai.licensing.license_manager.LicenseManager._find_license_file', return_value=None):
            with pytest.raises(SystemExit):
                # This should fail on import due to license check
                import eai
    
    def test_expired_license_blocks_access(self):
        """Test that expired license blocks access."""
        # Mock expired license
        expired_license_data = {
            'license_id': 'TEST-EXPIRED',
            'user_name': 'Test User',
            'organization': 'Test Org',
            'email': 'test@example.com',
            'issue_date': '2024-01-01T00:00:00',
            'expiry_date': '2024-12-31T23:59:59',  # Expired
            'features': ['core'],
            'license_type': 'basic',
            'max_instances': 1,
            'signature': 'test_signature'
        }
        
        with patch('eai.licensing.license_manager.LicenseManager._load_license') as mock_load:
            mock_load.side_effect = LicenseExpiredError("License expired")
            
            with pytest.raises(LicenseExpiredError):
                validate_license(['core'])
    
    def test_feature_not_licensed_blocks_access(self):
        """Test that unlicensed features are blocked."""
        # Mock license with limited features
        limited_license_data = {
            'license_id': 'TEST-LIMITED',
            'user_name': 'Test User',
            'organization': 'Test Org', 
            'email': 'test@example.com',
            'issue_date': '2025-01-01T00:00:00',
            'expiry_date': '2025-12-31T23:59:59',
            'features': ['core'],  # Only basic features
            'license_type': 'basic',
            'max_instances': 1,
            'signature': 'test_signature'
        }
        
        with patch('eai.licensing.license_manager.LicenseManager.validate_feature') as mock_validate:
            mock_validate.side_effect = FeatureNotLicensedError("Feature not licensed")
            
            with pytest.raises(FeatureNotLicensedError):
                validate_license(['enterprise'])
    
    def test_function_decorator_enforcement(self):
        """Test that function decorators enforce licensing."""
        
        @licensed_function(['premium_feature'])
        def premium_function():
            return "Premium functionality"
        
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            mock_validate.side_effect = FeatureNotLicensedError("Premium feature not licensed")
            
            with pytest.raises(FeatureNotLicensedError):
                premium_function()
    
    def test_class_decorator_enforcement(self):
        """Test that class decorators enforce licensing."""
        
        @licensed_class(['advanced_networks'])
        class AdvancedNetwork:
            def __init__(self):
                self.value = "advanced"
        
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            mock_validate.side_effect = FeatureNotLicensedError("Advanced networks not licensed")
            
            with pytest.raises(FeatureNotLicensedError):
                AdvancedNetwork()
    
    def test_valid_license_allows_access(self):
        """Test that valid license allows access."""
        # Mock valid license
        valid_license_data = {
            'license_id': 'TEST-VALID',
            'user_name': 'Test User',
            'organization': 'Test Org',
            'email': 'test@example.com',
            'issue_date': '2025-01-01T00:00:00',
            'expiry_date': '2025-12-31T23:59:59',
            'features': ['core', 'advanced_networks'],
            'license_type': 'professional',
            'max_instances': 1,
            'signature': 'test_signature'
        }
        
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            mock_validate.return_value = True
            
            @licensed_function(['advanced_networks'])
            def advanced_function():
                return "Advanced functionality"
            
            # Should not raise an exception
            result = advanced_function()
            assert result == "Advanced functionality"
    
    def test_license_manager_initialization(self):
        """Test license manager initialization."""
        with patch('eai.licensing.license_manager.LicenseManager._find_license_file', return_value=None):
            with pytest.raises(LicenseNotFoundError):
                LicenseManager()
    
    def test_cli_license_enforcement(self):
        """Test that CLI tools enforce licensing."""
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            mock_validate.side_effect = LicenseError("No license")
            
            # CLI import should fail
            with pytest.raises(SystemExit):
                from eai.cli.main_cli import main


class TestLicenseFeatures:
    """Test license feature validation."""
    
    def test_academic_license_features(self):
        """Test academic license feature restrictions."""
        academic_features = [
            'core',
            'basic_networks',
            'basic_optimization', 
            'visualization',
            'tutorials'
        ]
        
        # Should allow academic features
        for feature in academic_features:
            with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
                mock_validate.return_value = True
                assert requires_license([feature]) == True
        
        # Should block professional features
        professional_features = ['molecule_evolution', 'circuit_evolution']
        for feature in professional_features:
            with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
                mock_validate.side_effect = FeatureNotLicensedError(f"Feature {feature} not licensed")
                assert requires_license([feature]) == False
    
    def test_professional_license_features(self):
        """Test professional license includes academic + professional features."""
        professional_features = [
            'core',
            'basic_networks',
            'advanced_networks',
            'molecule_evolution',
            'circuit_evolution',
            'api_access'
        ]
        
        for feature in professional_features:
            with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
                mock_validate.return_value = True
                assert requires_license([feature]) == True
    
    def test_enterprise_license_features(self):
        """Test enterprise license includes all features."""
        enterprise_features = [
            'core',
            'advanced_networks',
            'theory_discovery',
            'custom_applications',
            'distributed_computing',
            'priority_support'
        ]
        
        for feature in enterprise_features:
            with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
                mock_validate.return_value = True
                assert requires_license([feature]) == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
