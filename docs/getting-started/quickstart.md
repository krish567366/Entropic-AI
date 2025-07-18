# Quick Start Guide

Welcome to Entropic AI! This guide will get you up and running with thermodynamic generative intelligence in just a few minutes.

## Installation

### Prerequisites

- Python 3.9 or higher
- PyTorch (automatically installed)
- 8GB+ RAM recommended

### Install from PyPI

```bash
pip install entropic-ai
```

### Development Installation

For the latest features and development:

```bash
git clone https://github.com/krish567366/Entropic-AI.git
cd Entropic-AI
pip install -e .
```

## Your First Thermodynamic Evolution

Let's start with a simple example that demonstrates the core principle of Entropic AI: evolving order from chaos.

### Example 1: Basic Chaos-to-Order Evolution

```python
import torch
from entropic-ai import EntropicNetwork, ComplexityOptimizer, GenerativeDiffuser

# Step 1: Create a thermodynamic neural network
network = EntropicNetwork(
    nodes=64,                    # Number of thermodynamic nodes
    temperature=1.0,             # Initial temperature
    entropy_regularization=0.1   # Entropy penalty weight
)

# Step 2: Initialize complexity optimizer
optimizer = ComplexityOptimizer(
    method="kolmogorov_complexity",
    target_complexity=0.7,      # Target complexity (0-1)
    stability_weight=0.3        # Balance stability vs complexity
)

# Step 3: Set up generative diffusion
diffuser = GenerativeDiffuser(
    network=network,
    optimizer=optimizer,
    diffusion_steps=100,         # Number of evolution steps
    cooling_schedule="exponential"
)

# Step 4: Start with pure chaos
chaos = torch.randn(32, 64)    # Random thermal noise
print(f"Initial entropy: {chaos.var().item():.3f}")

# Step 5: Evolve to ordered structure
ordered_state = diffuser.evolve(chaos)
print(f"Final entropy: {ordered_state.var().item():.3f}")

# The system has self-organized!
```

Expected output:

```Plaintext
Initial entropy: 0.987
Evolving thermodynamic state...  ████████████████████ 100/100
Final entropy: 0.234
```

### Example 2: Monitoring the Thermodynamic Process

```python
from entropic-ai.utils import plot_entropy_evolution, plot_energy_landscape

# Enable detailed monitoring
diffuser.enable_monitoring()

# Run evolution with tracking
result = diffuser.evolve(chaos, return_trajectory=True)

# Visualize the thermodynamic evolution
plot_entropy_evolution(result.entropy_history)
plot_energy_landscape(result.energy_history)

# Access thermodynamic properties
print(f"Free energy change: {result.free_energy_change:.3f}")
print(f"Complexity score: {result.complexity_score:.3f}")
print(f"Stability measure: {result.stability_measure:.3f}")
```

## Command Line Interface

Entropic AI provides a powerful CLI for running experiments without writing code.

### Basic Commands

```bash
# Get help
entropic-ai --help

# Run a basic evolution experiment
entropic-ai run --type basic --steps 100 --complexity-target 0.7

# Evolve molecular structures
entropic-ai evolve --type molecule --target-properties stability:0.9,complexity:0.7

# Discover mathematical theories
entropic-ai discover --domain physics --data-file experimental_data.csv

# Analyze results
entropic-ai analyze --results-dir ./results --plot-evolution
```

### Configuration Files

Create `experiment_config.json`:

```json
{
  "network": {
    "nodes": 128,
    "temperature": 1.5,
    "entropy_regularization": 0.2
  },
  "optimizer": {
    "method": "multi_objective",
    "target_complexity": 0.8,
    "stability_weight": 0.4
  },
  "diffusion": {
    "steps": 200,
    "cooling_schedule": "linear",
    "crystallization_threshold": 0.1
  }
}
```

Run with configuration:

```bash
entropic-ai run --config experiment_config.json
```

## Core Applications

### Molecule Evolution

Design novel molecular structures:

```python
from entropic-ai.applications import MoleculeEvolution

# Initialize molecular evolver
evolver = MoleculeEvolution(
    target_properties={
        "stability": 0.9,      # Thermodynamic stability
        "complexity": 0.7,     # Structural complexity
        "functionality": 0.8   # Functional capability
    },
    atomic_constraints={
        "max_atoms": 50,
        "allowed_elements": ["C", "N", "O", "H", "S"]
    }
)

# Start from atomic chaos
initial_atoms = evolver.generate_atomic_chaos(n_atoms=30)

# Evolve molecular structure
molecule = evolver.evolve_from_atoms(initial_atoms, steps=500)

print(f"Evolved molecule: {molecule.formula}")
print(f"Stability score: {molecule.stability:.3f}")
print(f"Binding affinity: {molecule.binding_affinity:.3f}")
```

### Circuit Design

Generate thermodynamically optimal digital circuits:

```python
from entropic-ai.applications import CircuitEvolution

# Define target logic function
def target_function(inputs):
    # XOR gate with error correction
    return inputs[0] ^ inputs[1]

# Initialize circuit evolver
designer = CircuitEvolution(
    logic_gates=["AND", "OR", "NOT", "XOR", "NAND"],
    thermal_noise_level=0.05,    # Operating noise level
    power_constraint=100,        # Max power consumption (μW)
    area_constraint=500          # Max area (μm²)
)

# Evolve circuit from logic chaos
circuit = designer.evolve_logic(
    target_function=target_function,
    input_bits=2,
    evolution_steps=300
)

print(f"Circuit gates: {circuit.gate_count}")
print(f"Power consumption: {circuit.power:.2f} μW")
print(f"Thermal stability: {circuit.thermal_stability:.3f}")
```

### Theory Discovery

Find symbolic expressions that explain data:

```python
from entropic-ai.applications import TheoryDiscovery
import numpy as np

# Generate some experimental data
x = np.linspace(0, 10, 100)
y = 2 * x**2 + 3 * x + 1 + 0.1 * np.random.randn(100)  # Quadratic with noise

# Initialize theory discoverer
discoverer = TheoryDiscovery(
    domain="mathematics",
    symbolic_complexity_limit=15,
    allowed_functions=["sin", "cos", "exp", "log", "poly"],
    noise_tolerance=0.1
)

# Discover underlying theory
theory = discoverer.discover_from_data(
    x_data=x,
    y_data=y,
    evolution_steps=400
)

print(f"Discovered expression: {theory.expression}")
print(f"R² score: {theory.r_squared:.4f}")
print(f"Complexity score: {theory.complexity:.3f}")
```

## Configuration and Customization

### Thermodynamic Parameters

Fine-tune the thermodynamic behavior:

```python
network = EntropicNetwork(
    nodes=128,
    temperature=2.0,              # Higher = more exploration
    entropy_regularization=0.15,  # Entropy penalty strength
    energy_scale=1.0,             # Energy landscape scale
    thermal_coupling=0.8,         # Node interaction strength
    heat_capacity=1.2,            # Thermal inertia
    phase_transition_temp=0.5     # Critical temperature
)
```

### Optimization Strategies

Choose different complexity optimization approaches:

```python
# Kolmogorov complexity optimization
optimizer = ComplexityOptimizer(method="kolmogorov_complexity")

# Shannon entropy optimization
optimizer = ComplexityOptimizer(method="shannon_entropy")

# Multi-objective optimization
optimizer = ComplexityOptimizer(
    method="multi_objective",
    objectives=["complexity", "stability", "novelty"],
    weights=[0.4, 0.4, 0.2]
)

# Adaptive optimization
optimizer = ComplexityOptimizer(
    method="adaptive",
    adaptation_rate=0.1,
    exploration_bonus=0.05
)
```

## Monitoring and Visualization

### Real-time Monitoring

```python
from entropic-ai.utils import ThermodynamicMonitor

# Set up monitoring
monitor = ThermodynamicMonitor(
    track_energy=True,
    track_entropy=True,
    track_complexity=True,
    update_frequency=10  # Steps between updates
)

# Run with monitoring
diffuser.add_monitor(monitor)
result = diffuser.evolve(chaos)

# View live plots
monitor.show_live_plots()
```

### Visualization Tools

```python
from entropic-ai.utils.visualization import (
    plot_energy_landscape,
    plot_phase_space,
    plot_complexity_evolution,
    plot_thermodynamic_state_diagram
)

# Energy landscape
plot_energy_landscape(result.energy_history, save_path="energy.png")

# Phase space trajectory
plot_phase_space(result.state_trajectory, dimensions=[0, 1, 2])

# Complexity evolution
plot_complexity_evolution(result.complexity_history)

# Full thermodynamic state diagram
plot_thermodynamic_state_diagram(
    energy=result.energy_history,
    entropy=result.entropy_history,
    temperature=result.temperature_history
)
```

## Next Steps

Now that you're familiar with the basics, explore more advanced topics:

- **[Molecule Design Tutorial](../tutorials/molecule-design.md)**: Design drug candidates
- **[Circuit Evolution Tutorial](../tutorials/circuit-evolution.md)**: Create optimal logic circuits
- **[Theory Discovery Tutorial](../tutorials/theory-discovery.md)**: Find physical laws from data
- **[Advanced Configuration](../guides/advanced-config.md)**: Fine-tune thermodynamic parameters
- **[Custom Applications](../guides/custom-applications.md)**: Build your own evolution domains

## Troubleshooting

### Common Issues

**Slow convergence**: Increase temperature or reduce cooling rate

```python
diffuser = GenerativeDiffuser(
    temperature=2.0,
    cooling_schedule="slow_exponential"
)
```

**Unstable evolution**: Increase stability weight

```python
optimizer = ComplexityOptimizer(stability_weight=0.6)
```

**Low complexity**: Reduce entropy regularization

```python
network = EntropicNetwork(entropy_regularization=0.05)
```

### Getting Help

- Check the [FAQ](../guides/faq.md)
- Browse [Examples](examples.md)
- Ask questions on [GitHub Discussions](https://github.com/krish567366/Entropic-AI/discussions)
- Report bugs on [GitHub Issues](https://github.com/krish567366/Entropic-AI/issues)

Happy evolving! 🌌✨
