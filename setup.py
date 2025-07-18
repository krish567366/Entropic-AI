#!/usr/bin/env python3
"""
Setup script for Entropic AI with license enforcement.

This setup enforces licensing requirements during installation.
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import sys


class LicenseEnforcedInstall(install):
    """Custom install command that enforces licensing."""
    
    def run(self):
        print("\n" + "="*60)
        print("🌌 ENTROPIC AI - PHYSICS-NATIVE INTELLIGENCE")
        print("="*60)
        print("\n⚠️  LICENSING NOTICE:")
        print("This software requires a valid license for all functionality.")
        print("No free usage or grace period is provided.")
        print("\n📧 For licensing information:")
        print("   Email: bajpaikrishna715@gmail.com")
        print("   Web: https://entropicai.com/licensing")
        print("\n💼 License Types Available:")
        print("   • Academic: Research and education")
        print("   • Professional: Commercial development")
        print("   • Enterprise: Full platform access")
        print("\n" + "="*60)
        
        # Proceed with installation
        super().run()
        
        print("\n✅ Installation complete!")
        print("📋 Next steps:")
        print("   1. Obtain a license: bajpaikrishna715@gmail.com")
        print("   2. Activate license: eai-license activate <license_file>")
        print("   3. Check status: eai-license status")


class LicenseEnforcedDevelop(develop):
    """Custom develop command that enforces licensing."""
    
    def run(self):
        print("\n" + "="*60)
        print("🌌 ENTROPIC AI - DEVELOPMENT INSTALLATION")
        print("="*60)
        print("\n⚠️  LICENSING NOTICE:")
        print("Development installation requires valid licensing.")
        print("Contact bajpaikrishna715@gmail.com for developer licenses.")
        print("="*60)
        
        super().run()


# Read version from package
def read_version():
    version_file = os.path.join(os.path.dirname(__file__), 'eai', '__init__.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return "0.1.0"


# Read long description
def read_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()


setup(
    name="entropic-ai",
    version=read_version(),
    author="Krishna Bajpai",
    author_email="bajpaikrishna715@gmail.com",
    description="Physics-Native Intelligence through Thermodynamic Self-Organization",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/krish567366/Entropic-AI",
    project_urls={
        "Documentation": "https://krish567366.github.io/Entropic-AI/",
        "Source Code": "https://github.com/krish567366/Entropic-AI",
        "Bug Tracker": "https://github.com/krish567366/Entropic-AI/issues",
        "Licensing": "https://entropicai.com/licensing",
        "Patent Information": "https://krish567366.github.io/Entropic-AI/about/patent/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "scipy>=1.8.0",
        "networkx>=2.8.0",
        "click>=8.0.0",
        "cryptography>=3.4.0",
        "pydantic>=1.10.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.0.0",
            "mkdocs-jupyter>=0.21.0",
            "mkdocstrings>=0.19.0",
        ],
        "viz": [
            "plotly>=5.10.0",
            "seaborn>=0.11.0",
            "dash>=2.6.0",
        ],
        "molecular": [
            "rdkit-pypi>=2022.3.0",
            "openmm>=7.7.0",
        ],
        "quantum": [
            "qiskit>=0.39.0",
            "cirq>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "eai-license=eai.cli.license_cli:cli",
            "entropic-ai=eai.cli.main_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "eai": [
            "data/*.json",
            "configs/*.yaml",
            "examples/*.py",
            "demos/*.ipynb",
        ],
    },
    cmdclass={
        'install': LicenseEnforcedInstall,
        'develop': LicenseEnforcedDevelop,
    },
    license="Proprietary - Patent Pending",
    license_files=["LICENSE"],
    keywords=[
        "artificial intelligence",
        "thermodynamics",
        "entropy",
        "physics-native",
        "generative AI",
        "self-organization",
        "emergent intelligence",
        "complexity science",
    ],
    zip_safe=False,
)
