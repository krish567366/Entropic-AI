"""
Comprehensive License Enforcement Test

Tests that ABSOLUTELY NO class or function can be accessed without proper licensing.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestAbsoluteLicenseEnforcement:
    """Test that NOTHING can be accessed without proper licensing."""
    
    def test_package_import_blocked_without_license(self):
        """Test that the entire package import is blocked without license."""
        
        # Mock no license found
        with patch('eai.licensing.license_manager.LicenseManager._find_license_file', return_value=None):
            with pytest.raises(SystemExit):
                # This should completely fail
                import eai
    
    def test_direct_class_import_blocked(self):
        """Test that direct class imports are blocked without license."""
        
        # Mock license failure
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            mock_validate.side_effect = Exception("No license")
            
            with pytest.raises(SystemExit):
                # Should fail on package-level license check
                from eai.core.thermodynamic_network import ThermodynamicNetwork
    
    def test_core_class_instantiation_blocked(self):
        """Test that core classes cannot be instantiated without license."""
        
        # Mock license validation to pass import but fail instantiation
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            # Allow import
            mock_validate.return_value = True
            
            # Import the module
            from eai.licensing.enforcement import BasicNetworkLicenseEnforcedBase
            
            # Now make license fail for instantiation
            mock_validate.side_effect = Exception("License expired")
            
            # Try to create a test class
            class TestNetwork(BasicNetworkLicenseEnforcedBase):
                _required_features = ['core', 'basic_networks']
            
            # Should fail on instantiation
            with pytest.raises(RuntimeError, match="LICENSE REQUIRED"):
                TestNetwork()
    
    def test_method_access_blocked_without_license(self):
        """Test that individual methods are blocked without proper license."""
        
        from eai.licensing.decorators import licensed_method
        from eai.licensing.validation import validate_or_fail
        
        class TestClass:
            @licensed_method(['premium_feature'])
            def premium_method(self):
                return "premium access"
        
        obj = TestClass()
        
        # Mock license failure
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            mock_validate.side_effect = Exception("Feature not licensed")
            
            with pytest.raises(Exception):
                obj.premium_method()
    
    def test_function_access_blocked_without_license(self):
        """Test that individual functions are blocked without proper license."""
        
        from eai.licensing.decorators import licensed_function
        
        @licensed_function(['advanced_feature'])
        def advanced_function():
            return "advanced functionality"
        
        # Mock license failure
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            mock_validate.side_effect = Exception("Feature not licensed")
            
            with pytest.raises(Exception):
                advanced_function()
    
    def test_module_level_enforcement(self):
        """Test that entire modules can be license-protected."""
        
        # Mock license failure at module level
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            mock_validate.side_effect = Exception("Module access denied")
            
            # Try to access utility functions
            try:
                import eai
                # These should either not exist or raise license errors
                if hasattr(eai, 'plot_entropy_evolution'):
                    with pytest.raises(RuntimeError, match="license"):
                        eai.plot_entropy_evolution()
            except (SystemExit, ImportError):
                # Expected if import is blocked
                pass
    
    def test_application_classes_blocked_without_tier_license(self):
        """Test that application classes require specific license tiers."""
        
        # Mock basic license (should block application classes)
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            def mock_validation(features):
                if 'molecule_evolution' in features:
                    raise Exception("Molecule evolution requires Professional license")
                return True
            
            mock_validate.side_effect = mock_validation
            
            # Try to import molecule evolution (should fail or be None)
            try:
                import eai
                if hasattr(eai, 'MoleculeEvolution') and eai.MoleculeEvolution is not None:
                    with pytest.raises(Exception):
                        eai.MoleculeEvolution()
            except (SystemExit, ImportError):
                # Expected if import is blocked
                pass
    
    def test_cli_tools_blocked_without_license(self):
        """Test that CLI tools are blocked without license."""
        
        # Mock license failure
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            mock_validate.side_effect = Exception("No license for CLI")
            
            with pytest.raises(SystemExit):
                # CLI should fail on import due to license check
                from eai.cli.main_cli import main
    
    def test_no_bypass_through_direct_module_access(self):
        """Test that you can't bypass licensing by accessing modules directly."""
        
        # Mock license failure
        with patch('eai.licensing.license_manager.LicenseManager._find_license_file', return_value=None):
            
            # Try various bypass attempts - all should fail
            bypass_attempts = [
                "from eai.core import thermodynamic_network",
                "import eai.core.thermodynamic_network",
                "from eai.applications import molecule_evolution",
                "import eai.utils.visualization",
            ]
            
            for attempt in bypass_attempts:
                with pytest.raises((SystemExit, ImportError, Exception)):
                    exec(attempt)
    
    def test_metaclass_enforcement(self):
        """Test that metaclass prevents ALL class operations without license."""
        
        from eai.licensing.enforcement import LicenseEnforcedMetaclass
        
        # Create a test class with license enforcement
        class TestClass(metaclass=LicenseEnforcedMetaclass):
            _required_features = ['test_feature']
            
            def test_method(self):
                return "test"
        
        # Mock license failure
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            mock_validate.side_effect = Exception("No license")
            
            # Should fail on instantiation
            with pytest.raises(RuntimeError, match="LICENSE REQUIRED"):
                TestClass()
    
    def test_property_access_enforcement(self):
        """Test that even property access is license-enforced."""
        
        from eai.licensing.enforcement import LicenseEnforcedProperty
        
        class TestClass:
            def __init__(self):
                self._value = "test"
            
            @property
            def licensed_property(self):
                return LicenseEnforcedProperty(['premium'], lambda self: self._value).__get__(self)
        
        obj = TestClass()
        
        # Mock license failure
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            mock_validate.side_effect = Exception("Property not licensed")
            
            with pytest.raises(RuntimeError, match="LICENSE REQUIRED"):
                _ = obj.licensed_property


class TestLicenseEnforcementBypass:
    """Test potential bypass methods to ensure they're blocked."""
    
    def test_cannot_bypass_with_getattr(self):
        """Test that getattr cannot bypass license enforcement."""
        
        # Mock license failure
        with patch('eai.licensing.validation.validate_or_fail') as mock_validate:
            mock_validate.side_effect = Exception("No license")
            
            try:
                import eai
                # Try to bypass with getattr
                with pytest.raises((AttributeError, RuntimeError)):
                    getattr(eai, 'ThermodynamicNetwork')
            except SystemExit:
                # Expected if import is blocked
                pass
    
    def test_cannot_bypass_with_importlib(self):
        """Test that importlib cannot bypass license enforcement."""
        
        import importlib
        
        # Mock license failure
        with patch('eai.licensing.license_manager.LicenseManager._find_license_file', return_value=None):
            
            with pytest.raises((SystemExit, ImportError)):
                importlib.import_module('eai.core.thermodynamic_network')
    
    def test_cannot_bypass_with_exec(self):
        """Test that exec cannot bypass license enforcement."""
        
        # Mock license failure
        with patch('eai.licensing.license_manager.LicenseManager._find_license_file', return_value=None):
            
            with pytest.raises((SystemExit, ImportError, NameError)):
                exec("from eai import ThermodynamicNetwork; net = ThermodynamicNetwork(10, [20], 5)")
    
    def test_cannot_bypass_with_eval(self):
        """Test that eval cannot bypass license enforcement."""
        
        # Mock license failure  
        with patch('eai.licensing.license_manager.LicenseManager._find_license_file', return_value=None):
            
            with pytest.raises((SystemExit, ImportError, NameError)):
                # First try to import
                exec("import eai")
                # Then try to access via eval
                eval("eai.ThermodynamicNetwork")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
