# API Reference - Core Module

This section provides comprehensive API documentation for the core Entropic AI components.

## eai.core.thermodynamic_network

### ThermodynamicNode

The fundamental computational unit with thermodynamic properties.

```python
class ThermodynamicNode(nn.Module):
    """A neural network node with thermodynamic state variables."""
```

#### Constructor

```python
def __init__(
    self,
    input_dim: int,
    output_dim: int,
    temperature: float = 1.0,
    activation: str = "boltzmann",
    entropy_regularization: float = 0.1
):
```

**Parameters:**

- `input_dim` (int): Input dimension
- `output_dim` (int): Output dimension  
- `temperature` (float, optional): Initial temperature. Default: 1.0
- `activation` (str, optional): Activation function type. Options: "boltzmann", "fermi_dirac", "thermal_relu". Default: "boltzmann"
- `entropy_regularization` (float, optional): Entropy penalty weight. Default: 0.1

#### Properties

```python
@property
def energy(self) -> float:
    """Current internal energy U."""

@property  
def entropy(self) -> float:
    """Current entropy S."""

@property
def free_energy(self) -> float:
    """Current Helmholtz free energy F = U - TS."""

@property
def temperature(self) -> float:
    """Current temperature T."""
```

#### Methods

##### thermodynamic_forward

```python
def thermodynamic_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass with thermodynamic state evolution.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Output with thermodynamic effects
    """
```

##### update_temperature

```python
def update_temperature(self, new_temperature: float) -> None:
    """Update node temperature.
    
    Args:
        new_temperature (float): New temperature value
    """
```

##### compute_thermodynamic_loss

```python
def compute_thermodynamic_loss(self) -> torch.Tensor:
    """Compute thermodynamic consistency loss.
    
    Returns:
        torch.Tensor: Thermodynamic loss term
    """
```

### ThermodynamicLayer

A layer containing multiple thermodynamic nodes.

```python
class ThermodynamicLayer(nn.Module):
    """Layer of thermodynamic nodes with collective behavior."""
```

#### Constructor

```python
def __init__(
    self,
    input_dim: int,
    output_dim: int,
    n_nodes: int = None,
    temperature: float = 1.0,
    thermal_coupling: float = 0.1
):
```

**Parameters:**

- `input_dim` (int): Input dimension
- `output_dim` (int): Output dimension
- `n_nodes` (int, optional): Number of nodes. If None, equals output_dim
- `temperature` (float, optional): Initial temperature. Default: 1.0
- `thermal_coupling` (float, optional): Inter-node coupling strength. Default: 0.1

#### Methods

##### forward

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Layer forward pass.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Layer output
    """
```

##### compute_layer_energy

```python
def compute_layer_energy(self) -> float:
    """Compute total layer energy.
    
    Returns:
        float: Sum of all node energies
    """
```

##### compute_layer_entropy

```python
def compute_layer_entropy(self) -> float:
    """Compute total layer entropy.
    
    Returns:
        float: Sum of all node entropies
    """
```

### ThermodynamicNetwork

Complete thermodynamic neural network.

```python
class ThermodynamicNetwork(nn.Module):
    """Multi-layer thermodynamic neural network."""
```

#### Constructor

```python
def __init__(
    self,
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    temperature: float = 1.0,
    entropy_regularization: float = 0.1,
    cooling_schedule: str = "exponential"
):
```

**Parameters:**

- `input_dim` (int): Input dimension
- `hidden_dims` (List[int]): Hidden layer dimensions
- `output_dim` (int): Output dimension
- `temperature` (float, optional): Initial temperature. Default: 1.0
- `entropy_regularization` (float, optional): Entropy regularization weight. Default: 0.1
- `cooling_schedule` (str, optional): Temperature cooling schedule. Default: "exponential"

#### Methods

##### forward

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Network forward pass.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_dim)
    """
```

##### compute_total_energy

```python
def compute_total_energy(self) -> float:
    """Compute total network energy.
    
    Returns:
        float: Sum of all layer energies
    """
```

##### compute_total_entropy

```python
def compute_total_entropy(self) -> float:
    """Compute total network entropy.
    
    Returns:
        float: Sum of all layer entropies
    """
```

##### compute_free_energy

```python
def compute_free_energy(self) -> float:
    """Compute total Helmholtz free energy.
    
    Returns:
        float: F = U - TS for the entire network
    """
```

##### update_temperature

```python
def update_temperature(
    self,
    step: int,
    total_steps: int,
    schedule: str = None
) -> None:
    """Update network temperature according to cooling schedule.
    
    Args:
        step (int): Current evolution step
        total_steps (int): Total evolution steps
        schedule (str, optional): Cooling schedule override
    """
```

### EntropicNetwork

Specialized network for entropy maximization.

```python
class EntropicNetwork(ThermodynamicNetwork):
    """Thermodynamic network optimized for entropy production."""
```

#### Constructor

```python
def __init__(
    self,
    nodes: int,
    temperature: float = 1.0,
    entropy_regularization: float = 0.1,
    max_entropy_rate: float = 1.0
):
```

**Parameters:**

- `nodes` (int): Number of nodes in the network
- `temperature` (float, optional): Initial temperature. Default: 1.0
- `entropy_regularization` (float, optional): Entropy regularization weight. Default: 0.1
- `max_entropy_rate` (float, optional): Maximum entropy production rate. Default: 1.0

#### Methods

##### compute_entropy_production_rate

```python
def compute_entropy_production_rate(self) -> float:
    """Compute current entropy production rate.
    
    Returns:
        float: Rate of entropy change dS/dt
    """
```

##### maximize_entropy_production

```python
def maximize_entropy_production(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass optimized for entropy production.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Output optimized for entropy production
    """
```

## eai.core.complexity_optimizer

### ComplexityOptimizer

Base class for complexity optimization strategies.

```python
class ComplexityOptimizer:
    """Optimizes system complexity using various measures."""
```

#### Constructor

```python
def __init__(
    self,
    method: str = "kolmogorov_complexity",
    target_complexity: float = 0.7,
    stability_weight: float = 0.3,
    exploration_bonus: float = 0.1
):
```

**Parameters:**

- `method` (str, optional): Complexity measure method. Options: "kolmogorov_complexity", "shannon_entropy", "fisher_information", "multi_objective". Default: "kolmogorov_complexity"
- `target_complexity` (float, optional): Target complexity score (0-1). Default: 0.7
- `stability_weight` (float, optional): Weight for stability vs complexity. Default: 0.3
- `exploration_bonus` (float, optional): Bonus for exploring new regions. Default: 0.1

#### Methods

##### compute_complexity_score

```python
def compute_complexity_score(self, state: torch.Tensor) -> float:
    """Compute complexity score for given state.
    
    Args:
        state (torch.Tensor): System state tensor
        
    Returns:
        float: Complexity score (0-1)
    """
```

##### optimize_step

```python
def optimize_step(self, state: torch.Tensor) -> torch.Tensor:
    """Perform one optimization step.
    
    Args:
        state (torch.Tensor): Current state
        
    Returns:
        torch.Tensor: Optimized state
    """
```

##### compute_stability

```python
def compute_stability(self, state: torch.Tensor) -> float:
    """Compute stability measure for state.
    
    Args:
        state (torch.Tensor): System state
        
    Returns:
        float: Stability score (0-1)
    """
```

### KolmogorovOptimizer

Optimizer based on Kolmogorov complexity.

```python
class KolmogorovOptimizer(ComplexityOptimizer):
    """Complexity optimizer using Kolmogorov complexity estimation."""
```

#### Constructor

```python
def __init__(
    self,
    compression_algorithms: List[str] = None,
    complexity_threshold: float = 0.5,
    **kwargs
):
```

**Parameters:**

- `compression_algorithms` (List[str], optional): Compression algorithms to use. Default: ["zlib", "bz2", "lzma"]
- `complexity_threshold` (float, optional): Minimum complexity threshold. Default: 0.5

#### Methods

##### estimate_kolmogorov_complexity

```python
def estimate_kolmogorov_complexity(self, data: torch.Tensor) -> float:
    """Estimate Kolmogorov complexity using compression.
    
    Args:
        data (torch.Tensor): Input data
        
    Returns:
        float: Estimated Kolmogorov complexity
    """
```

### MultiObjectiveOptimizer

Multi-objective complexity optimizer.

```python
class MultiObjectiveOptimizer(ComplexityOptimizer):
    """Multi-objective optimizer with Pareto optimization."""
```

#### Constructor

```python
def __init__(
    self,
    objectives: Dict[str, Dict[str, float]],
    pareto_optimization: bool = True,
    constraint_handling: str = "penalty"
):
```

**Parameters:**

- `objectives` (Dict): Objective definitions with weights and targets
- `pareto_optimization` (bool, optional): Use Pareto optimization. Default: True
- `constraint_handling` (str, optional): Constraint handling method. Default: "penalty"

#### Methods

##### compute_pareto_front

```python
def compute_pareto_front(
    self,
    population: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Compute Pareto-optimal front.
    
    Args:
        population (List[torch.Tensor]): Population of solutions
        
    Returns:
        List[torch.Tensor]: Pareto-optimal solutions
    """
```

## eai.core.generative_diffuser

### GenerativeDiffuser

Main class orchestrating chaos-to-order evolution.

```python
class GenerativeDiffuser:
    """Orchestrates thermodynamic evolution from chaos to order."""
```

#### Constructor

```python
def __init__(
    self,
    network: ThermodynamicNetwork,
    optimizer: ComplexityOptimizer,
    diffusion_steps: int = 100,
    crystallization_threshold: float = 0.1,
    cooling_schedule: str = "exponential"
):
```

**Parameters:**

- `network` (ThermodynamicNetwork): Thermodynamic neural network
- `optimizer` (ComplexityOptimizer): Complexity optimization strategy
- `diffusion_steps` (int, optional): Number of evolution steps. Default: 100
- `crystallization_threshold` (float, optional): Threshold for crystallization detection. Default: 0.1
- `cooling_schedule` (str, optional): Temperature cooling schedule. Default: "exponential"

#### Methods

##### evolve

```python
def evolve(
    self,
    initial_state: torch.Tensor,
    return_trajectory: bool = False
) -> Union[torch.Tensor, EvolutionResult]:
    """Main evolution method: chaos → order.
    
    Args:
        initial_state (torch.Tensor): Initial chaotic state
        return_trajectory (bool, optional): Return full evolution trajectory. Default: False
        
    Returns:
        Union[torch.Tensor, EvolutionResult]: Final evolved state or complete results
    """
```

##### crystallization_step

```python
def crystallization_step(
    self,
    state: torch.Tensor,
    step: int
) -> torch.Tensor:
    """Perform one crystallization step.
    
    Args:
        state (torch.Tensor): Current state
        step (int): Current evolution step
        
    Returns:
        torch.Tensor: State after crystallization step
    """
```

##### check_convergence

```python
def check_convergence(
    self,
    state: torch.Tensor,
    tolerance: float = 1e-6
) -> bool:
    """Check if evolution has converged.
    
    Args:
        state (torch.Tensor): Current state
        tolerance (float, optional): Convergence tolerance. Default: 1e-6
        
    Returns:
        bool: True if converged
    """
```

### OrderEvolver

Specialized evolver for ordered phase discovery.

```python
class OrderEvolver(GenerativeDiffuser):
    """Specialized diffuser for discovering ordered phases."""
```

#### Methods

##### evolve_with_phase_tracking

```python
def evolve_with_phase_tracking(
    self,
    initial_state: torch.Tensor
) -> PhaseEvolutionResult:
    """Evolution with phase transition detection.
    
    Args:
        initial_state (torch.Tensor): Initial state
        
    Returns:
        PhaseEvolutionResult: Results with phase transition information
    """
```

##### detect_phase_transition

```python
def detect_phase_transition(
    self,
    state_history: List[torch.Tensor]
) -> List[PhaseTransition]:
    """Detect phase transitions in evolution history.
    
    Args:
        state_history (List[torch.Tensor]): Evolution trajectory
        
    Returns:
        List[PhaseTransition]: Detected phase transitions
    """
```

### AdaptiveOrderEvolver

Evolver with adaptive parameters.

```python
class AdaptiveOrderEvolver(OrderEvolver):
    """Order evolver with adaptive temperature and complexity control."""
```

#### Constructor

```python
def __init__(
    self,
    *args,
    adaptation_rate: float = 0.1,
    target_acceptance_rate: float = 0.5,
    **kwargs
):
```

#### Methods

##### adaptive_evolve

```python
def adaptive_evolve(
    self,
    initial_state: torch.Tensor,
    adaptation_frequency: int = 10
) -> AdaptiveEvolutionResult:
    """Evolution with adaptive parameter tuning.
    
    Args:
        initial_state (torch.Tensor): Initial state
        adaptation_frequency (int, optional): Steps between adaptations. Default: 10
        
    Returns:
        AdaptiveEvolutionResult: Results with adaptation history
    """
```

## Return Types

### EvolutionResult

```python
@dataclass
class EvolutionResult:
    """Results from evolution process."""
    final_state: torch.Tensor
    trajectory: List[torch.Tensor]
    energy_history: List[float]
    entropy_history: List[float]
    complexity_history: List[float]
    convergence_step: int
    final_free_energy: float
```

### PhaseTransition

```python
@dataclass
class PhaseTransition:
    """Information about detected phase transition."""
    step: int
    temperature: float
    order_parameter_change: float
    free_energy_change: float
    transition_type: str
```

### PhaseEvolutionResult

```python
@dataclass
class PhaseEvolutionResult(EvolutionResult):
    """Evolution results with phase information."""
    phase_transitions: List[PhaseTransition]
    final_phase: str
    order_parameter_history: List[float]
```

## Usage Examples

### Basic Usage

```python
from eai.core import ThermodynamicNetwork, ComplexityOptimizer, GenerativeDiffuser
import torch

# Create components
network = ThermodynamicNetwork(
    input_dim=64,
    hidden_dims=[128, 128],
    output_dim=64,
    temperature=1.0
)

optimizer = ComplexityOptimizer(
    method="kolmogorov_complexity",
    target_complexity=0.7
)

diffuser = GenerativeDiffuser(
    network=network,
    optimizer=optimizer,
    diffusion_steps=200
)

# Evolve from chaos
chaos = torch.randn(32, 64)
order = diffuser.evolve(chaos)
```

### Advanced Usage with Monitoring

```python
# Evolution with full trajectory tracking
result = diffuser.evolve(
    initial_state=chaos,
    return_trajectory=True
)

print(f"Convergence step: {result.convergence_step}")
print(f"Final free energy: {result.final_free_energy:.3f}")
print(f"Energy change: {result.energy_history[-1] - result.energy_history[0]:.3f}")
```

## Error Handling

All core classes raise specific exceptions:

- `ThermodynamicError`: Thermodynamic consistency violations
- `ConvergenceError`: Failed to converge within specified steps
- `TemperatureError`: Invalid temperature values
- `ComplexityError`: Complexity computation failures

```python
from eai.core.exceptions import ThermodynamicError

try:
    result = diffuser.evolve(initial_state)
except ThermodynamicError as e:
    print(f"Thermodynamic violation: {e}")
except ConvergenceError as e:
    print(f"Failed to converge: {e}")
```
