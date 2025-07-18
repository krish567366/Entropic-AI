"""
Entropic AI (E-AI): Generative Intelligence through Thermodynamic Self-Organization

This package implements a revolutionary approach to artificial intelligence based on
thermodynamics, entropy, and emergent order. Unlike traditional AI systems that 
interpolate within latent spaces, E-AI evolves solutions through thermodynamic 
principles, creating order from chaos.

Core Components:
- ThermodynamicNetwork: Neural networks with thermodynamic properties
- ComplexityOptimizer: Optimizers that maximize emergent complexity
- GenerativeDiffuser: Chaos-to-order transformation systems
- Applications: Molecule evolution, circuit design, theory discovery

⚠️  LICENSING NOTICE:
This software requires a valid license for all functionality.
No grace period or free usage is provided.

Author: Krishna Bajpai <bajpaikrishna715@gmail.com>
License: Proprietary - Patent Pending
Commercial inquiries: bajpaikrishna715@gmail.com
"""

__version__ = "0.1.0"
__author__ = "Krishna Bajpai"
__email__ = "bajpaikrishna715@gmail.com"
__license__ = "Proprietary - Patent Pending"

# License validation on import - NO GRACE PERIOD
from .licensing import validate_or_fail, show_license_info
from .licensing.license_manager import LicenseError

def _enforce_license():
    """Enforce license validation on package import."""
    try:
        validate_or_fail(['core'])
        print("✅ Entropic AI licensed - Welcome to Physics-Native Intelligence!")
    except LicenseError:
        print("\n🚫 ENTROPIC AI: LICENSE REQUIRED")
        print("This software requires a valid license for all functionality.")
        print("Contact bajpaikrishna715@gmail.com for licensing information.")
        raise SystemExit("License validation failed - access denied")

# Mandatory license check on import
_enforce_license()

# Core imports for easy access (license-protected)
# NO CLASS CAN BE IMPORTED WITHOUT VALID LICENSE
from .licensing.decorators import licensed_function, licensed_class

# License-protected imports - ALL CLASSES REQUIRE VALIDATION
try:
    # Validate core license before ANY imports
    from .licensing.validation import validate_or_fail
    validate_or_fail(['core'])
    
    # Only import after license validation succeeds
    from .core.thermodynamic_network import ThermodynamicNetwork, EntropicNetwork
    from .core.complexity_optimizer import ComplexityOptimizer, KolmogorovOptimizer
    from .core.generative_diffuser import GenerativeDiffuser, OrderEvolver
    from .utils.entropy_utils import (
        shannon_entropy,
        kolmogorov_complexity,
        thermodynamic_entropy,
        fisher_information
    )
    
    # Application modules (require higher-tier licenses)
    # These will fail if user doesn't have proper license tier
    try:
        validate_or_fail(['molecule_evolution'])
        from .applications.molecule_evolution import MoleculeEvolution
    except Exception:
        # Don't import if not licensed
        MoleculeEvolution = None
    
    try:
        validate_or_fail(['circuit_evolution'])
        from .applications.circuit_evolution import CircuitEvolution
    except Exception:
        CircuitEvolution = None
    
    try:
        validate_or_fail(['theory_discovery'])
        from .applications.theory_discovery import TheoryDiscovery
    except Exception:
        TheoryDiscovery = None
    
    # Utility functions (require visualization license)
    try:
        validate_or_fail(['visualization'])
        from .utils.visualization import plot_entropy_evolution, plot_energy_landscape
        from .utils.metrics import complexity_score, stability_measure, emergence_index
    except Exception:
        # Provide dummy functions that show license requirement
        def plot_entropy_evolution(*args, **kwargs):
            raise RuntimeError("Visualization requires 'visualization' license feature. Contact bajpaikrishna715@gmail.com")
        def plot_energy_landscape(*args, **kwargs):
            raise RuntimeError("Visualization requires 'visualization' license feature. Contact bajpaikrishna715@gmail.com")
        def complexity_score(*args, **kwargs):
            raise RuntimeError("Metrics require 'visualization' license feature. Contact bajpaikrishna715@gmail.com")
        def stability_measure(*args, **kwargs):
            raise RuntimeError("Metrics require 'visualization' license feature. Contact bajpaikrishna715@gmail.com")
        def emergence_index(*args, **kwargs):
            raise RuntimeError("Metrics require 'visualization' license feature. Contact bajpaikrishna715@gmail.com")
    
except Exception as e:
    print(f"❌ ENTROPIC AI: License validation failed during import")
    print(f"Error: {e}")
    print("Contact bajpaikrishna715@gmail.com for licensing information.")
    raise SystemExit("License validation failed - no classes available for import")

__all__ = [
    # Core classes
    "ThermodynamicNetwork",
    "EntropicNetwork", 
    "ComplexityOptimizer",
    "KolmogorovOptimizer",
    "GenerativeDiffuser",
    "OrderEvolver",
    
    # Entropy utilities
    "shannon_entropy",
    "kolmogorov_complexity", 
    "thermodynamic_entropy",
    "fisher_information",
    
    # Applications
    "MoleculeEvolution",
    "CircuitEvolution", 
    "TheoryDiscovery",
    
    # Utilities
    "plot_entropy_evolution",
    "plot_energy_landscape",
    "complexity_score",
    "stability_measure", 
    "emergence_index",
]
