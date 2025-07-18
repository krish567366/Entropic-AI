# Contributing to Entropic AI

Welcome to the Entropic AI project! We're excited to have you contribute to the development of the world's first thermodynamic-based generative intelligence system. This guide will help you get started with contributing to the project.

## 🌟 Ways to Contribute

- **Code Development**: Core algorithms, applications, utilities
- **Documentation**: Tutorials, API docs, examples
- **Testing**: Unit tests, integration tests, benchmarks
- **Research**: New thermodynamic methods, theoretical foundations
- **Applications**: Domain-specific implementations
- **Bug Reports**: Issues, edge cases, performance problems
- **Feature Requests**: New capabilities and enhancements

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git for version control
- Basic understanding of thermodynamics and optimization
- Familiarity with PyTorch and NumPy

> **Note**: The package is published on PyPI as `entropic-ai` but imported in Python as `eai`. This is intentional - install with `pip install entropic-ai` but use `import eai` in your code.

### Development Environment Setup

1. **Fork and Clone Repository**

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Entropic-AI.git
cd Entropic-AI

# Add upstream remote
git remote add upstream https://github.com/krish567366/Entropic-AI.git
```

2. **Set Up Development Environment**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,docs,test]"

# Install pre-commit hooks
pre-commit install
```

3. **Verify Installation**

```bash
# Run tests to ensure everything works
pytest tests/

# Check code style
flake8 entropic-ai/
black --check entropic-ai/

# Build documentation locally
cd docs/
mkdocs serve
```

## 📋 Development Workflow

### Branch Strategy

We use a feature branch workflow:

```bash
# Create feature branch from main
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add thermodynamic feature X"

# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### Commit Message Convention

We follow conventional commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**

```bash
feat(core): add adaptive temperature control
fix(applications): resolve convergence issue in circuit evolution
docs(tutorials): add molecular evolution tutorial
test(optimization): add unit tests for complexity measures
```

### Code Review Process

1. **Self-Review**: Review your own code before submitting
2. **Automated Checks**: Ensure all CI checks pass
3. **Peer Review**: At least one maintainer review required
4. **Testing**: New features must include tests
5. **Documentation**: Update docs for user-facing changes

## 🧪 Testing Guidelines

### Test Structure

```bash
tests/
├── unit/                 # Unit tests for individual components
│   ├── core/            # Core module tests
│   ├── applications/    # Application tests
│   └── utilities/       # Utility tests
├── integration/         # Integration tests
├── benchmarks/          # Performance benchmarks
└── fixtures/           # Test data and configurations
```

### Writing Tests

```python
# Example unit test
import pytest
import torch
from entropic-ai.core import ThermodynamicNetwork

class TestThermodynamicNetwork:
    """Test thermodynamic network functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.network = ThermodynamicNetwork(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=5,
            temperature=1.0
        )
    
    def test_network_initialization(self):
        """Test network initializes correctly."""
        assert self.network.input_dim == 10
        assert self.network.output_dim == 5
        assert self.network.temperature == 1.0
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        input_tensor = torch.randn(32, 10)
        output = self.network(input_tensor)
        
        assert output.shape == (32, 5)
        assert not torch.isnan(output).any()
    
    def test_energy_computation(self):
        """Test energy computation."""
        state = torch.randn(10)
        energy = self.network.compute_energy(state)
        
        assert isinstance(energy, torch.Tensor)
        assert energy.dim() == 0  # Scalar
        assert energy >= 0  # Energy should be non-negative
    
    @pytest.mark.parametrize("temperature", [0.1, 1.0, 10.0])
    def test_temperature_scaling(self, temperature):
        """Test network behavior at different temperatures."""
        self.network.set_temperature(temperature)
        
        state = torch.randn(10)
        energy = self.network.compute_energy(state)
        entropy = self.network.compute_entropy(state)
        
        # Basic sanity checks
        assert energy.item() > 0
        assert entropy.item() > 0
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/unit/core/test_thermodynamic_network.py

# Run with coverage
pytest --cov=entropic-ai --cov-report=html

# Run performance benchmarks
pytest tests/benchmarks/ --benchmark-only

# Run tests with specific markers
pytest -m "slow"  # Run slow tests
pytest -m "not slow"  # Skip slow tests
```

### Test Markers

We use pytest markers to categorize tests:

```python
@pytest.mark.slow
def test_large_scale_evolution():
    """Test that requires significant computational time."""
    pass

@pytest.mark.gpu
def test_gpu_acceleration():
    """Test that requires GPU."""
    pass

@pytest.mark.integration
def test_full_optimization_pipeline():
    """Integration test for complete workflow."""
    pass
```

## 📚 Documentation Guidelines

### Documentation Structure

- **API Documentation**: Comprehensive docstrings for all public APIs
- **Tutorials**: Step-by-step guides for common use cases
- **Theory**: Mathematical and scientific foundations
- **Examples**: Complete working examples
- **Architecture**: System design and implementation details

### Docstring Style

We use Google-style docstrings:

```python
def compute_free_energy(energy: torch.Tensor,
                       entropy: torch.Tensor,
                       temperature: float) -> torch.Tensor:
    """Compute Helmholtz free energy F = U - TS.
    
    This function computes the Helmholtz free energy for a thermodynamic
    system given its internal energy, entropy, and temperature.
    
    Args:
        energy: Internal energy tensor of shape (batch_size,) or scalar
        entropy: Entropy tensor of shape (batch_size,) or scalar  
        temperature: System temperature (positive scalar)
        
    Returns:
        Free energy tensor with same shape as input tensors
        
    Raises:
        ValueError: If temperature is not positive
        TypeError: If inputs are not tensors or numeric types
        
    Example:
        >>> energy = torch.tensor([1.0, 2.0, 3.0])
        >>> entropy = torch.tensor([0.5, 1.0, 1.5])
        >>> temperature = 2.0
        >>> free_energy = compute_free_energy(energy, entropy, temperature)
        >>> print(free_energy)  # tensor([0.0, 0.0, 0.0])
        
    Note:
        The Helmholtz free energy is a fundamental quantity in thermodynamics
        that represents the useful work obtainable from a closed system at
        constant temperature and volume.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    return energy - temperature * entropy
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build and serve documentation locally
cd docs/
mkdocs serve

# Build static documentation
mkdocs build

# Deploy to GitHub Pages (maintainers only)
mkdocs gh-deploy
```

## 🏗️ Code Standards

### Code Style

We use Black for code formatting and Flake8 for linting:

```bash
# Format code with Black
black entropic-ai/ tests/

# Check formatting
black --check entropic-ai/ tests/

# Lint with Flake8
flake8 entropic-ai/ tests/

# Type checking with mypy
mypy entropic-ai/
```

### Code Organization

```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
import torch.nn as nn

# Local imports
from entropic-ai.core.base import BaseThermodynamicSystem
from entropic-ai.utilities.math import compute_entropy
```

### Performance Guidelines

- **Vectorization**: Use vectorized operations instead of loops
- **Memory Management**: Be mindful of memory usage for large tensors
- **GPU Support**: Ensure code works on both CPU and GPU
- **Numerical Stability**: Handle edge cases and numerical issues

```python
# Good: Vectorized operation
def compute_pairwise_distances(points: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances between points."""
    # Use broadcasting for efficient computation
    diff = points.unsqueeze(1) - points.unsqueeze(0)
    distances = torch.norm(diff, dim=2)
    return distances

# Bad: Nested loops
def compute_pairwise_distances_slow(points: torch.Tensor) -> torch.Tensor:
    """Slow implementation with nested loops."""
    n = points.shape[0]
    distances = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            distances[i, j] = torch.norm(points[i] - points[j])
    return distances
```

## 🧬 Core Development Areas

### Thermodynamic Algorithms

When contributing to core thermodynamic algorithms:

1. **Physical Consistency**: Ensure algorithms respect thermodynamic laws
2. **Mathematical Rigor**: Provide mathematical foundations
3. **Computational Efficiency**: Optimize for performance
4. **Generalizability**: Design for broad applicability

```python
class NewThermodynamicMethod:
    """Template for new thermodynamic methods."""
    
    def __init__(self, **kwargs):
        """Initialize method with validated parameters."""
        self._validate_parameters(kwargs)
        
    def _validate_parameters(self, params: Dict) -> None:
        """Validate physical consistency of parameters."""
        # Check thermodynamic constraints
        pass
    
    def evolve_step(self, state: torch.Tensor) -> torch.Tensor:
        """Perform one evolution step."""
        # Implement thermodynamic evolution
        pass
    
    def compute_thermodynamic_quantities(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute energy, entropy, and other quantities."""
        pass
```

### Applications Development

For new applications:

1. **Domain Expertise**: Understand the target domain
2. **Problem Mapping**: Map domain concepts to thermodynamic variables
3. **Validation**: Provide domain-specific validation
4. **Examples**: Include complete working examples

### Performance Optimization

Areas for performance contributions:

- **GPU Acceleration**: CUDA kernels for thermodynamic operations
- **Parallel Processing**: Multi-core and distributed computing
- **Memory Optimization**: Efficient memory usage patterns
- **Numerical Methods**: Optimized numerical algorithms

## 🐛 Bug Reports

### Bug Report Template

When reporting bugs, include:

```markdown
**Bug Description**
Clear description of the issue

**Reproduction Steps**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python Version: [e.g., 3.9.7]
- Entropic AI Version: [e.g., 0.1.0]
- PyTorch Version: [e.g., 1.12.0]
- CUDA Version (if applicable): [e.g., 11.3]

**Additional Context**
- Error messages
- Stack traces
- Minimal code example
- Any relevant configuration
```

### Debugging Guidelines

1. **Minimal Example**: Create minimal reproduction case
2. **Error Analysis**: Analyze error messages and stack traces
3. **Edge Cases**: Consider boundary conditions
4. **Documentation**: Check if behavior is documented

## 🎯 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the requested feature

**Use Case**
Why is this feature needed? What problem does it solve?

**Proposed Implementation**
How might this feature be implemented?

**Alternatives Considered**
What alternatives have you considered?

**Additional Context**
Any other relevant information
```

### Feature Development Process

1. **Discussion**: Discuss feature in GitHub issues
2. **Design**: Create detailed design document
3. **Implementation**: Implement with tests and documentation
4. **Review**: Code review and feedback
5. **Integration**: Merge and announce

## 🧑‍🔬 Research Contributions

### Theoretical Contributions

We welcome contributions to the theoretical foundations:

- **New Thermodynamic Methods**: Novel evolution algorithms
- **Complexity Measures**: New complexity quantification methods
- **Convergence Analysis**: Theoretical convergence guarantees
- **Physical Interpretations**: Connections to statistical mechanics

### Experimental Contributions

- **Benchmarking**: Performance comparisons with other methods
- **Applications**: New domain applications
- **Case Studies**: Detailed analysis of specific problems
- **Validation**: Experimental validation of theoretical results

## 🏆 Recognition

### Contributor Recognition

- **Contributors List**: All contributors are listed in the repository
- **Changelog**: Significant contributions mentioned in releases
- **Academic Credit**: Research contributions eligible for co-authorship

### Contribution Guidelines

- **Quality**: Maintain high code and documentation quality
- **Originality**: Ensure contributions are original work
- **Licensing**: All contributions must be compatible with project license
- **Ethics**: Follow ethical guidelines for research and development

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: [bajpaikrishna715@gmail.com](mailto:bajpaikrishna715@gmail.com) for sensitive issues

### Development Questions

- **Architecture Questions**: Ask about system design decisions
- **Implementation Help**: Get help with specific implementation challenges
- **Performance Issues**: Discuss optimization strategies
- **Testing Support**: Get help with test development

## 📜 Code of Conduct

We are committed to fostering an open and welcoming environment. All contributors are expected to:

- **Be Respectful**: Treat all community members with respect
- **Be Inclusive**: Welcome contributors from all backgrounds
- **Be Professional**: Maintain professional standards in all interactions
- **Be Constructive**: Provide constructive feedback and criticism

### Enforcement

Unacceptable behavior may result in:
- Warning
- Temporary suspension
- Permanent ban

Report issues to [bajpaikrishna715@gmail.com](mailto:bajpaikrishna715@gmail.com).

## 🎉 Thank You!

Thank you for your interest in contributing to Entropic AI! Your contributions help advance the field of thermodynamic intelligence and make this technology available to researchers and practitioners worldwide.

Every contribution, no matter how small, makes a difference. Whether you're fixing a typo, adding a test, implementing a new feature, or contributing research insights, you're helping build the future of AI based on fundamental physical principles.

Welcome to the Entropic AI community! 🌌
