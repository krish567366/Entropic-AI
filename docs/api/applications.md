# Applications API Reference

This module provides high-level application interfaces for common use cases of Entropic AI. The applications package includes pre-built solvers for optimization, design, discovery, and generation tasks across various domains.

## Core Application Classes

### BaseApplication

The foundation class for all Entropic AI applications.

```python
class BaseApplication:
    """Base class for all Entropic AI applications.
    
    Provides common functionality and interfaces for thermodynamic
    evolution-based problem solving.
    """
    
    def __init__(self, config: ApplicationConfig):
        """Initialize base application.
        
        Args:
            config: Application configuration containing:
                - thermal_parameters: Temperature and cooling settings
                - complexity_constraints: Complexity optimization settings  
                - domain_specific: Domain-specific parameters
        """
        self.config = config
        self.thermal_network = None
        self.complexity_optimizer = None
        self.evolution_state = None
        
    def setup(self, problem_definition: ProblemDefinition) -> None:
        """Setup application for specific problem.
        
        Args:
            problem_definition: Problem-specific configuration including:
                - objective_function: Function to optimize
                - constraints: Problem constraints
                - variable_bounds: Variable boundaries
                - evaluation_metrics: Success metrics
        """
        
    def evolve(self, 
               initial_state: Optional[torch.Tensor] = None,
               max_iterations: int = 1000,
               convergence_threshold: float = 1e-6) -> EvolutionResult:
        """Run thermodynamic evolution.
        
        Args:
            initial_state: Starting point for evolution (random if None)
            max_iterations: Maximum number of evolution steps
            convergence_threshold: Convergence criteria
            
        Returns:
            EvolutionResult containing:
                - best_solution: Optimal solution found
                - final_energy: Final energy value
                - evolution_history: Complete evolution trace
                - convergence_info: Convergence statistics
        """
        
    def evaluate_solution(self, solution: torch.Tensor) -> Dict[str, float]:
        """Evaluate solution quality.
        
        Args:
            solution: Solution to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
```

## Optimization Applications

### ContinuousOptimization

Solve continuous optimization problems using thermodynamic evolution.

```python
class ContinuousOptimization(BaseApplication):
    """Continuous optimization using thermodynamic principles.
    
    Suitable for:
    - Non-linear optimization
    - Multi-modal landscapes
    - Constrained optimization
    - Global optimization
    """
    
    def __init__(self, 
                 objective_function: Callable[[torch.Tensor], torch.Tensor],
                 bounds: Tuple[torch.Tensor, torch.Tensor],
                 constraints: Optional[List[Constraint]] = None):
        """Initialize continuous optimizer.
        
        Args:
            objective_function: Function to minimize f(x) -> scalar
            bounds: (lower_bounds, upper_bounds) for variables
            constraints: List of constraint objects
        """
        
    def add_constraint(self, constraint: Constraint) -> None:
        """Add optimization constraint.
        
        Args:
            constraint: Constraint object with methods:
                - evaluate(x): Returns constraint violation
                - gradient(x): Returns constraint gradient
        """
        
    def set_cooling_schedule(self, schedule: CoolingSchedule) -> None:
        """Set temperature cooling schedule.
        
        Args:
            schedule: Cooling schedule object with:
                - initial_temperature: Starting temperature
                - final_temperature: Ending temperature  
                - cooling_rate: Rate of temperature decrease
                - schedule_type: 'exponential', 'linear', 'adaptive'
        """

# Example usage
optimizer = ContinuousOptimization(
    objective_function=lambda x: torch.sum(x**2),  # Sphere function
    bounds=(torch.tensor([-5.0] * 10), torch.tensor([5.0] * 10))
)

result = optimizer.evolve(max_iterations=2000)
print(f"Optimal solution: {result.best_solution}")
print(f"Optimal value: {result.final_energy}")
```

### CombinatorialOptimization

Solve discrete optimization problems.

```python
class CombinatorialOptimization(BaseApplication):
    """Combinatorial optimization using thermodynamic evolution.
    
    Suitable for:
    - Traveling salesman problems
    - Graph coloring
    - Scheduling problems
    - Assignment problems
    """
    
    def __init__(self, 
                 problem_graph: nx.Graph,
                 objective_type: str = 'minimize',
                 neighborhood_function: Optional[Callable] = None):
        """Initialize combinatorial optimizer.
        
        Args:
            problem_graph: NetworkX graph representing problem structure
            objective_type: 'minimize' or 'maximize'
            neighborhood_function: Function defining solution neighborhoods
        """
        
    def set_encoding(self, encoding: DiscreteEncoding) -> None:
        """Set solution encoding scheme.
        
        Args:
            encoding: Encoding object that handles:
                - encode(solution): Convert to thermodynamic representation
                - decode(state): Convert from thermodynamic representation
                - validate(solution): Check solution validity
        """
        
    def add_local_search(self, local_search: LocalSearch) -> None:
        """Add local search component.
        
        Args:
            local_search: Local search object for solution refinement
        """

# Example: Traveling Salesman Problem
tsp_graph = nx.complete_graph(20)  # 20-city TSP
for (u, v) in tsp_graph.edges():
    tsp_graph[u][v]['weight'] = np.random.uniform(1, 10)

tsp_optimizer = CombinatorialOptimization(
    problem_graph=tsp_graph,
    objective_type='minimize'
)

tsp_result = tsp_optimizer.evolve()
```

## Design Applications

### CircuitEvolution

Evolve digital circuit designs using thermodynamic principles.

```python
class CircuitEvolution(BaseApplication):
    """Digital circuit synthesis and optimization.
    
    Features:
    - Logic synthesis from truth tables
    - Multi-objective optimization (area, power, delay)
    - Technology mapping
    - Noise resilience optimization
    """
    
    def __init__(self, 
                 component_library: ComponentLibrary,
                 technology_node: str = '14nm',
                 optimization_objectives: List[str] = ['area', 'power', 'delay']):
        """Initialize circuit evolution.
        
        Args:
            component_library: Available logic gates and components
            technology_node: Target technology node
            optimization_objectives: List of objectives to optimize
        """
        
    def set_specification(self, spec: CircuitSpecification) -> None:
        """Set circuit design specification.
        
        Args:
            spec: Circuit specification containing:
                - truth_table: Desired logic function
                - timing_constraints: Performance requirements
                - power_budget: Power consumption limits
                - area_constraints: Size limitations
        """
        
    def add_noise_model(self, noise_model: NoiseModel) -> None:
        """Add noise model for robust design.
        
        Args:
            noise_model: Thermal and process noise model
        """
        
    def synthesize(self) -> CircuitDesign:
        """Synthesize circuit design.
        
        Returns:
            CircuitDesign object containing:
                - netlist: Circuit netlist
                - performance_metrics: Area, power, delay
                - verification_results: Correctness verification
        """

# Example: 4-bit adder synthesis
adder_spec = CircuitSpecification(
    truth_table=generate_adder_truth_table(4),
    timing_constraint=TimeConstraint(max_delay=2.0),  # ns
    power_budget=PowerBudget(max_power=10.0),  # mW
    area_constraint=AreaConstraint(max_area=1000.0)  # μm²
)

circuit_evolver = CircuitEvolution(
    component_library=StandardCellLibrary('14nm'),
    optimization_objectives=['area', 'power', 'delay']
)

circuit_evolver.set_specification(adder_spec)
design = circuit_evolver.synthesize()
```

### ArchitecturalDesign

Optimize architectural and structural designs.

```python
class ArchitecturalDesign(BaseApplication):
    """Architectural design optimization.
    
    Applications:
    - Building layout optimization
    - Structural design
    - Network topology design
    - System architecture optimization
    """
    
    def __init__(self, 
                 design_space: DesignSpace,
                 structural_constraints: List[StructuralConstraint],
                 performance_objectives: List[str]):
        """Initialize architectural design.
        
        Args:
            design_space: Definition of design variable space
            structural_constraints: Physical and code constraints
            performance_objectives: Performance metrics to optimize
        """
        
    def add_building_codes(self, codes: BuildingCodes) -> None:
        """Add building code constraints.
        
        Args:
            codes: Building code specifications
        """
        
    def set_environmental_conditions(self, conditions: EnvironmentalConditions) -> None:
        """Set environmental design conditions.
        
        Args:
            conditions: Environmental parameters (wind, seismic, thermal)
        """
        
    def optimize_design(self) -> ArchitecturalSolution:
        """Optimize architectural design.
        
        Returns:
            ArchitecturalSolution with optimized parameters
        """
```

## Discovery Applications

### LawDiscovery

Discover scientific laws and mathematical relationships from data.

```python
class LawDiscovery(BaseApplication):
    """Scientific law discovery from experimental data.
    
    Capabilities:
    - Symbolic regression
    - Physical law discovery
    - Mathematical relationship extraction
    - Multi-scale law discovery
    """
    
    def __init__(self, 
                 operator_library: List[str],
                 complexity_weights: Dict[str, float],
                 dimensional_analysis: bool = True):
        """Initialize law discovery.
        
        Args:
            operator_library: Available mathematical operators
            complexity_weights: Weights for different complexity measures
            dimensional_analysis: Enable dimensional consistency checking
        """
        
    def set_data(self, 
                 data: pd.DataFrame,
                 variable_context: VariableContext) -> None:
        """Set experimental data and variable context.
        
        Args:
            data: Experimental data with variables as columns
            variable_context: Variable metadata including:
                - units: Physical units for each variable
                - types: Variable types (independent/dependent)
                - constraints: Variable constraints
        """
        
    def add_physics_constraints(self, constraints: List[PhysicsConstraint]) -> None:
        """Add physics-based constraints.
        
        Args:
            constraints: List of physical principles to enforce
        """
        
    def discover_laws(self, 
                     target_variables: List[str],
                     max_complexity: int = 20) -> List[SymbolicExpression]:
        """Discover laws for target variables.
        
        Args:
            target_variables: Variables to find laws for
            max_complexity: Maximum allowed expression complexity
            
        Returns:
            List of discovered symbolic expressions
        """

# Example: Discover pendulum law
pendulum_data = pd.DataFrame({
    'length': [0.1, 0.2, 0.3, 0.4, 0.5],
    'period': [0.63, 0.89, 1.10, 1.27, 1.42],
    'mass': [0.1] * 5,
    'gravity': [9.81] * 5
})

variable_context = VariableContext(
    units={'length': 'm', 'period': 's', 'mass': 'kg', 'gravity': 'm/s²'},
    independent=['length', 'mass', 'gravity'],
    dependent=['period']
)

law_discoverer = LawDiscovery(
    operator_library=['add', 'mul', 'div', 'pow', 'sqrt'],
    dimensional_analysis=True
)

law_discoverer.set_data(pendulum_data, variable_context)
discovered_laws = law_discoverer.discover_laws(['period'])
```

### PatternDiscovery

Discover patterns in complex datasets.

```python
class PatternDiscovery(BaseApplication):
    """Pattern discovery in high-dimensional data.
    
    Techniques:
    - Anomaly detection
    - Clustering with thermodynamic principles
    - Feature selection
    - Causal relationship discovery
    """
    
    def __init__(self, 
                 pattern_types: List[str],
                 complexity_regularization: float = 0.1):
        """Initialize pattern discovery.
        
        Args:
            pattern_types: Types of patterns to discover
            complexity_regularization: Regularization strength
        """
        
    def discover_clusters(self, 
                         data: torch.Tensor,
                         num_clusters: Optional[int] = None) -> ClusteringResult:
        """Discover clusters using thermodynamic principles.
        
        Args:
            data: Input data tensor
            num_clusters: Number of clusters (auto-detected if None)
            
        Returns:
            ClusteringResult with cluster assignments and centers
        """
        
    def detect_anomalies(self, data: torch.Tensor) -> AnomalyResult:
        """Detect anomalies using entropy-based methods.
        
        Args:
            data: Input data tensor
            
        Returns:
            AnomalyResult with anomaly scores and detections
        """
```

## Generation Applications

### MoleculeEvolution

Generate and optimize molecular structures.

```python
class MoleculeEvolution(BaseApplication):
    """Molecular structure generation and optimization.
    
    Applications:
    - Drug discovery
    - Material design
    - Catalyst optimization
    - Protein folding prediction
    """
    
    def __init__(self, 
                 element_library: List[str],
                 bond_constraints: BondConstraints,
                 target_properties: Dict[str, float]):
        """Initialize molecule evolution.
        
        Args:
            element_library: Available chemical elements
            bond_constraints: Chemical bonding rules
            target_properties: Desired molecular properties
        """
        
    def set_property_predictors(self, predictors: Dict[str, PropertyPredictor]) -> None:
        """Set molecular property prediction models.
        
        Args:
            predictors: Dictionary of property prediction models
        """
        
    def evolve_molecule(self, 
                       starting_structure: Optional[Molecule] = None) -> MoleculeResult:
        """Evolve molecular structure.
        
        Args:
            starting_structure: Initial molecular structure
            
        Returns:
            MoleculeResult with optimized structure and properties
        """

# Example: Drug-like molecule generation
molecule_evolver = MoleculeEvolution(
    element_library=['C', 'N', 'O', 'H', 'S', 'P'],
    bond_constraints=DrugLikeBondConstraints(),
    target_properties={
        'molecular_weight': (200, 500),  # Da
        'logP': (0, 5),  # Lipophilicity
        'TPSA': (0, 140),  # Topological polar surface area
        'binding_affinity': 'maximize'
    }
)

drug_result = molecule_evolver.evolve_molecule()
```

### ContentGeneration

Generate creative content using thermodynamic evolution.

```python
class ContentGeneration(BaseApplication):
    """Creative content generation.
    
    Content types:
    - Text generation
    - Image generation  
    - Music composition
    - Code generation
    """
    
    def __init__(self, 
                 content_type: str,
                 style_constraints: StyleConstraints,
                 quality_metrics: List[str]):
        """Initialize content generation.
        
        Args:
            content_type: Type of content to generate
            style_constraints: Style and format constraints
            quality_metrics: Quality assessment metrics
        """
        
    def generate_content(self, 
                        prompt: str,
                        length_target: int,
                        creativity_level: float = 0.7) -> GeneratedContent:
        """Generate content from prompt.
        
        Args:
            prompt: Input prompt or seed
            length_target: Target content length
            creativity_level: Balance between coherence and novelty
            
        Returns:
            GeneratedContent with text/image/music and quality scores
        """
```

## Utility Functions

### Evolution Monitoring

```python
class EvolutionMonitor:
    """Monitor and visualize evolution progress."""
    
    def __init__(self, metrics: List[str]):
        """Initialize evolution monitor.
        
        Args:
            metrics: List of metrics to track
        """
        
    def start_monitoring(self, evolution_process: BaseApplication) -> None:
        """Start monitoring evolution process."""
        
    def plot_evolution_trace(self, 
                           save_path: Optional[str] = None) -> matplotlib.figure.Figure:
        """Plot evolution progress."""
        
    def export_evolution_data(self, format: str = 'csv') -> str:
        """Export evolution data."""

def visualize_energy_landscape(application: BaseApplication,
                              variable_ranges: Dict[str, Tuple[float, float]],
                              resolution: int = 50) -> matplotlib.figure.Figure:
    """Visualize energy landscape for 2D problems.
    
    Args:
        application: Configured application instance
        variable_ranges: Ranges for visualization variables
        resolution: Grid resolution for visualization
        
    Returns:
        Matplotlib figure with energy landscape
    """

def benchmark_performance(applications: List[BaseApplication],
                         test_problems: List[TestProblem],
                         metrics: List[str]) -> pd.DataFrame:
    """Benchmark application performance.
    
    Args:
        applications: List of applications to benchmark
        test_problems: List of test problems
        metrics: Performance metrics to evaluate
        
    Returns:
        DataFrame with benchmark results
    """
```

### Configuration Helpers

```python
def create_optimization_config(problem_type: str,
                              difficulty_level: str = 'medium') -> ApplicationConfig:
    """Create standard optimization configuration.
    
    Args:
        problem_type: Type of optimization problem
        difficulty_level: 'easy', 'medium', 'hard'
        
    Returns:
        Pre-configured ApplicationConfig
    """

def create_discovery_config(domain: str,
                           data_size: int) -> ApplicationConfig:
    """Create standard discovery configuration.
    
    Args:
        domain: Scientific domain (physics, biology, etc.)
        data_size: Size of dataset
        
    Returns:
        Pre-configured ApplicationConfig
    """

def load_application_from_config(config_path: str) -> BaseApplication:
    """Load application from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured application instance
    """
```

## Error Handling

All application classes include comprehensive error handling:

```python
class EntropicAIError(Exception):
    """Base exception for Entropic AI applications."""
    pass

class ConvergenceError(EntropicAIError):
    """Raised when evolution fails to converge."""
    pass

class ConfigurationError(EntropicAIError):
    """Raised for invalid configurations."""
    pass

class DimensionalityError(EntropicAIError):
    """Raised for dimensional consistency violations."""
    pass
```

## Performance Considerations

- **GPU Acceleration**: All applications support CUDA acceleration
- **Parallel Processing**: Multi-core processing for population-based methods
- **Memory Management**: Efficient memory usage for large-scale problems
- **Checkpointing**: Save and resume long-running evolution processes

## Examples Repository

Complete examples for each application are available in the `examples/` directory:

- `examples/optimization/` - Optimization problem examples
- `examples/design/` - Design optimization examples  
- `examples/discovery/` - Scientific discovery examples
- `examples/generation/` - Generative application examples

Each example includes:
- Problem setup code
- Configuration files
- Expected results
- Performance benchmarks
- Visualization code
