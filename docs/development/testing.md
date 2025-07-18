# Testing Guide

This guide covers the comprehensive testing framework for Entropic AI, including unit tests, integration tests, performance benchmarks, and validation procedures.

## Testing Philosophy

Entropic AI testing is based on several key principles:

- **Physical Consistency**: Tests verify thermodynamic laws are respected
- **Numerical Stability**: Tests ensure algorithms work across different scales
- **Reproducibility**: Tests provide consistent results across runs
- **Performance**: Tests validate computational efficiency
- **Domain Validity**: Tests ensure correct behavior in application domains

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── core/               # Core thermodynamic engine tests
│   │   ├── test_energy.py
│   │   ├── test_entropy.py
│   │   ├── test_network.py
│   │   └── test_diffuser.py
│   ├── applications/       # Application-specific tests
│   │   ├── test_circuit_evolution.py
│   │   ├── test_molecule_evolution.py
│   │   └── test_law_discovery.py
│   └── utilities/          # Utility function tests
│       ├── test_preprocessing.py
│       ├── test_visualization.py
│       └── test_configuration.py
├── integration/            # Integration tests
│   ├── test_full_pipeline.py
│   ├── test_multi_objective.py
│   └── test_real_time.py
├── benchmarks/             # Performance benchmarks
│   ├── test_optimization_benchmarks.py
│   ├── test_scalability.py
│   └── test_gpu_acceleration.py
├── validation/             # Scientific validation tests
│   ├── test_thermodynamic_laws.py
│   ├── test_known_solutions.py
│   └── test_convergence_analysis.py
├── fixtures/               # Test data and configurations
│   ├── data/
│   ├── configs/
│   └── expected_results/
└── conftest.py            # Pytest configuration
```

## Unit Testing

### Core Component Tests

#### Testing Thermodynamic Networks

```python
import pytest
import torch
import numpy as np
from eai.core import ThermodynamicNetwork
from eai.core.energy import HamiltonianEnergy
from eai.core.entropy import BoltzmannEntropy

class TestThermodynamicNetwork:
    """Test thermodynamic network functionality."""
    
    @pytest.fixture
    def network(self):
        """Create test network."""
        return ThermodynamicNetwork(
            input_dim=10,
            hidden_dims=[64, 32],
            output_dim=5,
            temperature=1.0
        )
    
    def test_network_initialization(self, network):
        """Test network initializes with correct parameters."""
        assert network.input_dim == 10
        assert network.output_dim == 5
        assert network.temperature == 1.0
        assert len(network.layers) == 2  # Two hidden layers
    
    def test_forward_pass_shape(self, network):
        """Test forward pass produces correct output shape."""
        batch_size = 32
        input_tensor = torch.randn(batch_size, 10)
        output = network(input_tensor)
        
        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_energy_computation(self, network):
        """Test energy computation properties."""
        state = torch.randn(10)
        energy = network.compute_energy(state)
        
        # Energy should be scalar
        assert energy.dim() == 0
        assert torch.isfinite(energy)
        
        # Energy should be positive (with appropriate energy function)
        assert energy >= 0
    
    def test_entropy_computation(self, network):
        """Test entropy computation properties."""
        state = torch.randn(10)
        entropy = network.compute_entropy(state)
        
        # Entropy should be scalar
        assert entropy.dim() == 0
        assert torch.isfinite(entropy)
        
        # Entropy should be non-negative
        assert entropy >= 0
    
    def test_temperature_scaling(self, network):
        """Test network behavior at different temperatures."""
        state = torch.randn(10)
        temperatures = [0.1, 1.0, 10.0]
        
        energies = []
        entropies = []
        
        for temp in temperatures:
            network.set_temperature(temp)
            energy = network.compute_energy(state)
            entropy = network.compute_entropy(state)
            
            energies.append(energy.item())
            entropies.append(entropy.item())
        
        # Check temperature effects
        assert all(torch.isfinite(torch.tensor(energies)))
        assert all(torch.isfinite(torch.tensor(entropies)))
    
    def test_thermodynamic_consistency(self, network):
        """Test thermodynamic consistency."""
        state = torch.randn(10)
        
        # Compute thermodynamic quantities
        energy = network.compute_energy(state)
        entropy = network.compute_entropy(state)
        temperature = network.temperature
        
        # Free energy should be well-defined
        free_energy = energy - temperature * entropy
        assert torch.isfinite(free_energy)
        
        # Test fluctuation-dissipation theorem
        # <ΔE²> = kT² * C_v (heat capacity)
        energy_samples = []
        for _ in range(100):
            perturbed_state = state + 0.01 * torch.randn_like(state)
            energy_sample = network.compute_energy(perturbed_state)
            energy_samples.append(energy_sample.item())
        
        energy_variance = np.var(energy_samples)
        expected_variance = temperature**2  # Simplified heat capacity
        
        # Should be within reasonable range
        assert energy_variance > 0
        assert energy_variance < 10 * expected_variance
```

#### Testing Energy Functions

```python
class TestEnergyFunctions:
    """Test energy function implementations."""
    
    def test_hamiltonian_energy(self):
        """Test Hamiltonian energy computation."""
        dim = 5
        kinetic_operator = torch.eye(dim)
        potential_function = lambda x: 0.5 * torch.sum(x**2)
        
        hamiltonian = HamiltonianEnergy(kinetic_operator, potential_function)
        
        # Test energy computation
        state = torch.randn(dim)
        energy = hamiltonian.compute_energy(state)
        
        assert torch.isfinite(energy)
        assert energy >= 0  # For this potential
    
    def test_energy_gradient(self):
        """Test energy gradient computation."""
        potential_function = lambda x: torch.sum(x**2)
        
        state = torch.randn(10)
        state.requires_grad_(True)
        
        energy = potential_function(state)
        gradient = torch.autograd.grad(energy, state)[0]
        
        # Gradient should have same shape as state
        assert gradient.shape == state.shape
        
        # For quadratic potential, gradient should be 2*x
        expected_gradient = 2 * state
        torch.testing.assert_close(gradient, expected_gradient, rtol=1e-5)
    
    def test_energy_conservation(self):
        """Test energy conservation in isolated system."""
        # Create conservative system
        def conservative_force(x):
            return -2 * x  # F = -dV/dx for V = x²
        
        # Simulate system evolution
        dt = 0.01
        x = torch.tensor([1.0])
        v = torch.tensor([0.0])
        
        initial_energy = 0.5 * v**2 + x**2  # KE + PE
        
        # Evolve system for several steps
        for _ in range(100):
            # Velocity Verlet integration
            a = conservative_force(x)
            x = x + v * dt + 0.5 * a * dt**2
            a_new = conservative_force(x)
            v = v + 0.5 * (a + a_new) * dt
        
        final_energy = 0.5 * v**2 + x**2
        
        # Energy should be conserved (within numerical precision)
        torch.testing.assert_close(initial_energy, final_energy, rtol=1e-3)
```

#### Testing Entropy Measures

```python
class TestEntropyMeasures:
    """Test entropy measure implementations."""
    
    def test_shannon_entropy(self):
        """Test Shannon entropy computation."""
        from eai.core.entropy import ShannonEntropy
        
        shannon = ShannonEntropy()
        
        # Test uniform distribution
        uniform_probs = torch.ones(4) / 4
        entropy = shannon.compute_entropy(uniform_probs)
        expected_entropy = torch.log(torch.tensor(4.0))  # log(n) for uniform
        
        torch.testing.assert_close(entropy, expected_entropy, rtol=1e-5)
        
        # Test deterministic distribution
        deterministic_probs = torch.tensor([1.0, 0.0, 0.0, 0.0])
        entropy = shannon.compute_entropy(deterministic_probs)
        
        assert entropy < 1e-6  # Should be nearly zero
    
    def test_boltzmann_entropy(self):
        """Test Boltzmann entropy computation."""
        from eai.core.entropy import BoltzmannEntropy
        
        boltzmann = BoltzmannEntropy()
        
        # Create microstate ensemble
        microstates = torch.randn(1000, 10)  # 1000 microstates in 10D
        
        entropy = boltzmann.compute_entropy(microstates)
        
        # Entropy should be positive
        assert entropy > 0
        assert torch.isfinite(entropy)
    
    def test_entropy_extensivity(self):
        """Test that entropy is extensive."""
        from eai.core.entropy import ShannonEntropy
        
        shannon = ShannonEntropy()
        
        # Two independent systems
        probs1 = torch.tensor([0.5, 0.5])
        probs2 = torch.tensor([0.25, 0.25, 0.25, 0.25])
        
        entropy1 = shannon.compute_entropy(probs1)
        entropy2 = shannon.compute_entropy(probs2)
        
        # Combined system
        combined_probs = torch.kron(probs1, probs2)  # Tensor product
        combined_entropy = shannon.compute_entropy(combined_probs)
        
        # Should be additive for independent systems
        expected_combined = entropy1 + entropy2
        torch.testing.assert_close(combined_entropy, expected_combined, rtol=1e-5)
```

### Application Tests

#### Testing Circuit Evolution

```python
class TestCircuitEvolution:
    """Test circuit evolution functionality."""
    
    @pytest.fixture
    def simple_truth_table(self):
        """Simple AND gate truth table."""
        return {
            'inputs': ['A', 'B'],
            'outputs': ['Y'],
            'logic': [
                (0, 0, 0),
                (0, 1, 0),
                (1, 0, 0),
                (1, 1, 1)
            ]
        }
    
    def test_circuit_specification_parsing(self, simple_truth_table):
        """Test circuit specification parsing."""
        from eai.applications import CircuitEvolution
        
        evolver = CircuitEvolution()
        spec = evolver.parse_truth_table(simple_truth_table)
        
        assert spec.num_inputs == 2
        assert spec.num_outputs == 1
        assert len(spec.logic_rows) == 4
    
    def test_circuit_generation(self):
        """Test random circuit generation."""
        from eai.applications import CircuitEvolution
        
        evolver = CircuitEvolution(
            component_library=['AND', 'OR', 'NOT'],
            max_gates=10
        )
        
        circuit = evolver.generate_random_circuit()
        
        assert circuit is not None
        assert len(circuit.gates) <= 10
        assert all(gate.type in ['AND', 'OR', 'NOT'] for gate in circuit.gates)
    
    def test_circuit_simulation(self, simple_truth_table):
        """Test circuit simulation correctness."""
        from eai.applications import CircuitEvolution
        from eai.circuits import Circuit, ANDGate
        
        # Create simple AND circuit
        circuit = Circuit()
        and_gate = ANDGate('and1')
        circuit.add_gate(and_gate)
        circuit.connect_input('A', and_gate, 0)
        circuit.connect_input('B', and_gate, 1)
        circuit.connect_output(and_gate, 0, 'Y')
        
        # Test simulation
        test_cases = [
            ([0, 0], [0]),
            ([0, 1], [0]),
            ([1, 0], [0]),
            ([1, 1], [1])
        ]
        
        for inputs, expected_outputs in test_cases:
            outputs = circuit.simulate(inputs)
            assert outputs == expected_outputs
    
    @pytest.mark.slow
    def test_circuit_evolution_convergence(self, simple_truth_table):
        """Test that circuit evolution converges to correct solution."""
        from eai.applications import CircuitEvolution
        
        evolver = CircuitEvolution(
            component_library=['AND', 'OR', 'NOT'],
            thermal_parameters={'initial_temperature': 2.0, 'cooling_rate': 0.9}
        )
        
        evolver.set_target_specification(simple_truth_table)
        
        result = evolver.evolve(max_iterations=500)
        
        # Should find a working circuit
        assert result.final_error < 0.1
        assert result.best_circuit is not None
        
        # Verify circuit correctness
        circuit = result.best_circuit
        for inputs, expected_outputs in simple_truth_table['logic']:
            outputs = circuit.simulate(list(inputs))
            assert outputs == list(expected_outputs)
```

## Integration Testing

### Full Pipeline Tests

```python
class TestFullPipeline:
    """Test complete optimization pipelines."""
    
    def test_optimization_pipeline(self):
        """Test full optimization workflow."""
        import torch
        from eai import optimize
        
        # Define test function (Rosenbrock)
        def rosenbrock(x):
            return torch.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        # Set up optimization
        bounds = (torch.tensor([-2.0, -2.0]), torch.tensor([2.0, 2.0]))
        
        result = optimize(
            objective_function=rosenbrock,
            bounds=bounds,
            method='thermodynamic',
            max_iterations=1000
        )
        
        # Should converge to global minimum at (1, 1)
        expected_optimum = torch.tensor([1.0, 1.0])
        torch.testing.assert_close(result.best_solution, expected_optimum, rtol=0.1)
        assert result.final_energy < 0.1
    
    def test_discovery_pipeline(self):
        """Test scientific discovery workflow."""
        import pandas as pd
        from eai import discover
        
        # Generate synthetic pendulum data
        lengths = torch.linspace(0.1, 1.0, 20)
        periods = 2 * torch.pi * torch.sqrt(lengths / 9.81)
        
        # Add small amount of noise
        periods += 0.01 * torch.randn_like(periods)
        
        data = pd.DataFrame({
            'length': lengths.numpy(),
            'period': periods.numpy(),
            'gravity': [9.81] * len(lengths)
        })
        
        # Discover relationship
        result = discover(
            data=data,
            target_variable='period',
            method='law_discovery',
            operators=['add', 'mul', 'div', 'sqrt'],
            max_complexity=10
        )
        
        # Should discover relationship close to T = 2π√(L/g)
        discovered_law = result.best_expression
        assert discovered_law is not None
        
        # Test law accuracy
        predicted_periods = discovered_law.evaluate(data)
        mse = torch.mean((torch.tensor(predicted_periods) - periods)**2)
        assert mse < 0.01
```

### Multi-Objective Integration

```python
class TestMultiObjectiveIntegration:
    """Test multi-objective optimization integration."""
    
    def test_pareto_front_generation(self):
        """Test Pareto front generation."""
        from eai.optimization import MultiObjectiveOptimizer
        
        # Define conflicting objectives
        def objective1(x):
            return torch.sum(x**2)  # Minimize distance from origin
        
        def objective2(x):
            return torch.sum((x - 1)**2)  # Minimize distance from (1,1)
        
        optimizer = MultiObjectiveOptimizer(
            objectives=[objective1, objective2],
            bounds=(torch.tensor([-1.0, -1.0]), torch.tensor([2.0, 2.0]))
        )
        
        result = optimizer.evolve(max_iterations=500)
        
        # Should generate valid Pareto front
        assert len(result.pareto_front) > 0
        
        # All solutions should be non-dominated
        for i, sol1 in enumerate(result.pareto_front):
            for j, sol2 in enumerate(result.pareto_front):
                if i != j:
                    # Check that neither dominates the other
                    obj1_values = [obj(sol1['solution']) for obj in [objective1, objective2]]
                    obj2_values = [obj(sol2['solution']) for obj in [objective1, objective2]]
                    
                    dominates = all(v1 <= v2 for v1, v2 in zip(obj1_values, obj2_values)) and \
                               any(v1 < v2 for v1, v2 in zip(obj1_values, obj2_values))
                    
                    assert not dominates
```

## Performance Benchmarks

### Optimization Benchmarks

```python
class TestOptimizationBenchmarks:
    """Benchmark optimization performance."""
    
    @pytest.mark.benchmark
    def test_sphere_function_benchmark(self, benchmark):
        """Benchmark sphere function optimization."""
        from eai.optimization import ThermodynamicOptimizer
        
        def sphere_function(x):
            return torch.sum(x**2)
        
        optimizer = ThermodynamicOptimizer(
            thermal_parameters={'initial_temperature': 1.0, 'cooling_rate': 0.95}
        )
        
        bounds = (torch.tensor([-5.0] * 10), torch.tensor([5.0] * 10))
        
        def run_optimization():
            return optimizer.optimize(sphere_function, bounds, max_iterations=1000)
        
        result = benchmark(run_optimization)
        
        # Performance assertions
        assert result.final_energy < 1e-6
        assert result.convergence_iteration < 800
    
    @pytest.mark.benchmark
    def test_scalability_benchmark(self, benchmark):
        """Benchmark performance scaling with problem size."""
        from eai.optimization import ThermodynamicOptimizer
        
        results = {}
        
        for dim in [10, 50, 100, 200]:
            def sphere_function(x):
                return torch.sum(x**2)
            
            optimizer = ThermodynamicOptimizer()
            bounds = (torch.tensor([-5.0] * dim), torch.tensor([5.0] * dim))
            
            def run_optimization():
                return optimizer.optimize(sphere_function, bounds, max_iterations=500)
            
            timing_result = benchmark(run_optimization)
            results[dim] = timing_result.stats['mean']
        
        # Check that scaling is reasonable (should be roughly linear or quadratic)
        time_10 = results[10]
        time_100 = results[100]
        
        # 10x increase in dimension should not be more than 100x increase in time
        assert time_100 / time_10 < 100
```

### Memory Benchmarks

```python
class TestMemoryBenchmarks:
    """Benchmark memory usage."""
    
    @pytest.mark.benchmark
    def test_memory_usage_scaling(self):
        """Test memory usage with problem size."""
        import psutil
        import os
        from eai.core import ThermodynamicNetwork
        
        process = psutil.Process(os.getpid())
        memory_usage = {}
        
        for network_size in [100, 500, 1000, 2000]:
            # Measure memory before
            memory_before = process.memory_info().rss
            
            # Create network
            network = ThermodynamicNetwork(
                input_dim=network_size,
                hidden_dims=[network_size, network_size],
                output_dim=network_size
            )
            
            # Do some operations
            input_data = torch.randn(32, network_size)
            output = network(input_data)
            
            # Measure memory after
            memory_after = process.memory_info().rss
            memory_usage[network_size] = memory_after - memory_before
            
            # Clean up
            del network, input_data, output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Memory usage should scale reasonably
        for size in memory_usage:
            # Should not use more than 10GB
            assert memory_usage[size] < 10 * 1024**3
```

## Validation Testing

### Thermodynamic Law Validation

```python
class TestThermodynamicLaws:
    """Validate adherence to thermodynamic laws."""
    
    def test_energy_conservation(self):
        """Test first law of thermodynamics."""
        from eai.core import ThermodynamicNetwork
        
        network = ThermodynamicNetwork(10, [20], 5)
        
        # Create isolated system evolution
        initial_state = torch.randn(10)
        
        # Track energy over time
        energies = []
        states = [initial_state]
        
        current_state = initial_state
        for _ in range(100):
            energy = network.compute_energy(current_state)
            energies.append(energy.item())
            
            # Evolve state (in isolation, energy should be conserved)
            # Use microcanonical evolution
            current_state = network.evolve_microcanonical(current_state, dt=0.01)
            states.append(current_state)
        
        # Energy should be approximately conserved
        energy_variance = np.var(energies)
        assert energy_variance < 0.1 * np.mean(energies)
    
    def test_entropy_increase(self):
        """Test second law of thermodynamics."""
        from eai.core import ThermodynamicNetwork
        
        network = ThermodynamicNetwork(10, [20], 5)
        
        # Start with low-entropy (ordered) state
        initial_state = torch.zeros(10)
        initial_state[0] = 1.0  # Concentrated state
        
        # Evolve in thermal contact (canonical ensemble)
        entropies = []
        current_state = initial_state
        
        for _ in range(100):
            entropy = network.compute_entropy(current_state)
            entropies.append(entropy.item())
            
            # Thermal evolution at fixed temperature
            current_state = network.evolve_canonical(current_state, temperature=1.0, dt=0.01)
        
        # Entropy should generally increase or remain constant
        entropy_changes = np.diff(entropies)
        
        # Allow for small fluctuations due to finite size
        negative_changes = entropy_changes[entropy_changes < -0.01]
        assert len(negative_changes) < 0.1 * len(entropy_changes)
    
    def test_fluctuation_dissipation_theorem(self):
        """Test fluctuation-dissipation relationship."""
        from eai.core import ThermodynamicNetwork
        
        network = ThermodynamicNetwork(5, [10], 3)
        temperature = 1.0
        
        # Measure equilibrium fluctuations
        equilibrium_energies = []
        for _ in range(1000):
            state = network.sample_thermal_state(temperature)
            energy = network.compute_energy(state)
            equilibrium_energies.append(energy.item())
        
        energy_variance = np.var(equilibrium_energies)
        
        # Measure response to small perturbation
        reference_state = torch.randn(5)
        reference_energy = network.compute_energy(reference_state)
        
        perturbation = 0.01 * torch.randn(5)
        perturbed_state = reference_state + perturbation
        perturbed_energy = network.compute_energy(perturbed_state)
        
        response = (perturbed_energy - reference_energy) / 0.01
        
        # Fluctuation-dissipation: <ΔE²> ∝ T * response
        expected_variance = temperature * abs(response.item())
        
        # Should be within order of magnitude
        assert 0.1 * expected_variance < energy_variance < 10 * expected_variance
```

### Known Solution Validation

```python
class TestKnownSolutions:
    """Test against problems with known solutions."""
    
    def test_harmonic_oscillator(self):
        """Test harmonic oscillator energy levels."""
        from eai.applications import QuantumSystemEvolution
        
        # Set up 1D harmonic oscillator
        def harmonic_potential(x):
            return 0.5 * x**2
        
        quantum_evolver = QuantumSystemEvolution(
            potential_function=harmonic_potential,
            dimensions=1
        )
        
        # Find ground state
        ground_state = quantum_evolver.find_ground_state()
        
        # Should find E₀ = ℏω/2 = 0.5 (with ℏ=ω=1)
        expected_energy = 0.5
        assert abs(ground_state.energy - expected_energy) < 0.01
        
        # Ground state should be Gaussian
        x = torch.linspace(-3, 3, 100)
        psi = ground_state.wavefunction(x)
        expected_psi = torch.exp(-x**2 / 2) / (torch.pi**0.25)
        
        # Compare shapes (allow for normalization differences)
        correlation = torch.corrcoef(torch.stack([psi, expected_psi]))[0, 1]
        assert correlation > 0.99
    
    def test_hydrogen_atom_ground_state(self):
        """Test hydrogen atom ground state."""
        from eai.applications import AtomicSystemEvolution
        
        # Set up hydrogen atom (simplified)
        def coulomb_potential(r):
            return -1.0 / (r + 1e-8)  # Avoid singularity
        
        atomic_evolver = AtomicSystemEvolution(
            potential_function=coulomb_potential,
            nuclear_charge=1
        )
        
        # Find ground state
        ground_state = atomic_evolver.find_ground_state()
        
        # Should find E₀ = -13.6 eV = -0.5 Hartree
        expected_energy = -0.5
        assert abs(ground_state.energy - expected_energy) < 0.05
    
    def test_traveling_salesman_optimal(self):
        """Test TSP with known optimal solution."""
        from eai.applications import CombinatorialOptimization
        import networkx as nx
        
        # Create small TSP with known solution
        # 4-city square: optimal tour length = 4
        graph = nx.Graph()
        cities = [(0, 0), (1, 0), (1, 1), (0, 1)]
        
        for i, pos1 in enumerate(cities):
            for j, pos2 in enumerate(cities):
                if i < j:
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    graph.add_edge(i, j, weight=distance)
        
        tsp_optimizer = CombinatorialOptimization(
            problem_graph=graph,
            problem_type='tsp'
        )
        
        result = tsp_optimizer.evolve(max_iterations=1000)
        
        # Should find optimal tour length = 4
        expected_length = 4.0
        assert abs(result.best_objective - expected_length) < 0.1
```

## Test Configuration

### Pytest Configuration

```python
# conftest.py
import pytest
import torch
import numpy as np
import os

def pytest_configure(config):
    """Configure pytest environment."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configure torch settings
    torch.set_default_dtype(torch.float32)
    
    # Set environment variables
    os.environ['ENTROPIC_AI_TEST_MODE'] = '1'

@pytest.fixture(scope="session")
def device():
    """Get device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def temp_directory(tmp_path):
    """Create temporary directory for test files."""
    return tmp_path

@pytest.fixture
def sample_data():
    """Generate sample test data."""
    torch.manual_seed(123)
    return {
        'input_data': torch.randn(100, 10),
        'target_data': torch.randn(100, 5),
        'labels': torch.randint(0, 3, (100,))
    }

# Markers for different test types
pytest.mark.slow = pytest.mark.skipif(
    not pytest.config.getoption("--run-slow"),
    reason="need --run-slow option to run"
)

pytest.mark.gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available"
)

pytest.mark.benchmark = pytest.mark.skipif(
    not pytest.config.getoption("--benchmark-only"),
    reason="need --benchmark-only option to run"
)
```

### Test Execution

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=eai --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m "gpu"       # Run only GPU tests
pytest -m "benchmark" # Run only benchmarks

# Run tests in parallel
pytest -n auto

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/core/test_thermodynamic_network.py

# Run specific test function
pytest tests/unit/core/test_thermodynamic_network.py::TestThermodynamicNetwork::test_energy_computation
```

### Continuous Integration

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[test]"
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ --cov=eai --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml
```

## Testing Best Practices

### Test Design Principles

1. **Independence**: Tests should not depend on each other
2. **Reproducibility**: Tests should give consistent results
3. **Fast Execution**: Unit tests should run quickly
4. **Clear Assertions**: Test failures should be easy to diagnose
5. **Edge Cases**: Test boundary conditions and error cases

### Performance Testing

- **Benchmarking**: Use consistent hardware and conditions
- **Profiling**: Identify performance bottlenecks
- **Scaling**: Test performance across different problem sizes
- **Memory**: Monitor memory usage and leaks

### Scientific Validation

- **Physical Laws**: Verify thermodynamic consistency
- **Known Solutions**: Test against analytical solutions
- **Convergence**: Verify algorithm convergence properties
- **Numerical Stability**: Test across different numerical scales

This comprehensive testing framework ensures that Entropic AI maintains high quality, performance, and scientific validity across all components and applications.
