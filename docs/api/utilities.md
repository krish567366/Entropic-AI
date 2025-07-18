# Utilities API Reference

This module provides utility functions and helper classes for Entropic AI applications. The utilities package includes data processing, visualization, analysis tools, and performance optimization functions.

## Data Processing Utilities

### DataPreprocessor

Preprocess data for thermodynamic evolution.

```python
class DataPreprocessor:
    """Data preprocessing for thermodynamic applications.
    
    Handles data cleaning, normalization, feature engineering,
    and preparation for thermodynamic evolution.
    """
    
    def __init__(self, 
                 preprocessing_config: PreprocessingConfig):
        """Initialize data preprocessor.
        
        Args:
            preprocessing_config: Configuration for preprocessing pipeline
        """
        self.config = preprocessing_config
        self.scalers = {}
        self.feature_selectors = {}
        
    def fit_transform(self, 
                     data: Union[pd.DataFrame, torch.Tensor],
                     target: Optional[Union[pd.Series, torch.Tensor]] = None) -> ProcessedData:
        """Fit preprocessor and transform data.
        
        Args:
            data: Input data to preprocess
            target: Target variable (for supervised tasks)
            
        Returns:
            ProcessedData object containing:
                - transformed_data: Preprocessed data
                - feature_names: Names of selected features
                - preprocessing_info: Metadata about transformations
        """
        
    def transform(self, data: Union[pd.DataFrame, torch.Tensor]) -> torch.Tensor:
        """Transform new data using fitted preprocessor.
        
        Args:
            data: New data to transform
            
        Returns:
            Transformed data tensor
        """
        
    def inverse_transform(self, 
                         transformed_data: torch.Tensor) -> Union[pd.DataFrame, torch.Tensor]:
        """Inverse transform data back to original scale.
        
        Args:
            transformed_data: Data to inverse transform
            
        Returns:
            Data in original scale
        """

# Example usage
preprocessor = DataPreprocessor(
    preprocessing_config=PreprocessingConfig(
        normalization='standard',
        feature_selection='variance_threshold',
        outlier_detection='isolation_forest',
        missing_value_strategy='iterative_imputer'
    )
)

processed_data = preprocessor.fit_transform(raw_data)
```

### FeatureEngineering

Generate features for thermodynamic representations.

```python
class FeatureEngineering:
    """Feature engineering for thermodynamic systems.
    
    Creates thermodynamic-aware features that capture
    energy, entropy, and complexity relationships.
    """
    
    def __init__(self, feature_types: List[str]):
        """Initialize feature engineering.
        
        Args:
            feature_types: Types of features to generate:
                - 'energy_features': Energy-related features
                - 'entropy_features': Entropy and disorder features
                - 'interaction_features': Feature interactions
                - 'complexity_features': Complexity measures
        """
        
    def generate_energy_features(self, data: torch.Tensor) -> torch.Tensor:
        """Generate energy-related features.
        
        Args:
            data: Input data tensor
            
        Returns:
            Tensor with energy features:
                - Local energy density
                - Energy gradients
                - Potential energy estimates
        """
        
    def generate_entropy_features(self, data: torch.Tensor) -> torch.Tensor:
        """Generate entropy-related features.
        
        Args:
            data: Input data tensor
            
        Returns:
            Tensor with entropy features:
                - Local entropy estimates
                - Information content
                - Disorder measures
        """
        
    def generate_complexity_features(self, data: torch.Tensor) -> torch.Tensor:
        """Generate complexity features.
        
        Args:
            data: Input data tensor
            
        Returns:
            Tensor with complexity features:
                - Kolmogorov complexity estimates
                - Fractal dimensions
                - Topological features
        """

def create_thermodynamic_features(data: torch.Tensor,
                                 temperature: float = 1.0) -> Dict[str, torch.Tensor]:
    """Create comprehensive thermodynamic feature set.
    
    Args:
        data: Input data
        temperature: System temperature
        
    Returns:
        Dictionary of thermodynamic features
    """
    features = {}
    
    # Energy features
    features['kinetic_energy'] = 0.5 * torch.sum(data**2, dim=-1)
    features['potential_energy'] = compute_potential_energy(data)
    
    # Entropy features  
    features['local_entropy'] = compute_local_entropy(data)
    features['relative_entropy'] = compute_relative_entropy(data, temperature)
    
    # Force features
    features['energy_gradient'] = compute_energy_gradient(data)
    features['entropy_gradient'] = compute_entropy_gradient(data)
    
    return features
```

## Visualization Utilities

### EvolutionVisualizer

Visualize thermodynamic evolution processes.

```python
class EvolutionVisualizer:
    """Visualization tools for thermodynamic evolution.
    
    Provides interactive and static visualizations of
    evolution progress, energy landscapes, and system dynamics.
    """
    
    def __init__(self, style: str = 'scientific'):
        """Initialize visualizer.
        
        Args:
            style: Visualization style ('scientific', 'minimal', 'publication')
        """
        self.style = style
        self.figure_configs = self._load_style_config()
        
    def plot_evolution_trace(self, 
                           evolution_history: List[Dict],
                           metrics: List[str] = ['energy', 'entropy', 'temperature']) -> plt.Figure:
        """Plot evolution trace over time.
        
        Args:
            evolution_history: History of evolution states
            metrics: Metrics to plot
            
        Returns:
            Matplotlib figure with evolution traces
        """
        
    def plot_energy_landscape(self, 
                             energy_function: Callable,
                             bounds: Tuple[torch.Tensor, torch.Tensor],
                             resolution: int = 100,
                             show_trajectory: bool = True) -> plt.Figure:
        """Plot 2D energy landscape.
        
        Args:
            energy_function: Energy function to visualize
            bounds: Variable bounds for plotting
            resolution: Grid resolution
            show_trajectory: Whether to show evolution trajectory
            
        Returns:
            Matplotlib figure with energy landscape
        """
        
    def plot_phase_space(self, 
                        states: torch.Tensor,
                        energies: torch.Tensor,
                        entropies: torch.Tensor) -> plt.Figure:
        """Plot thermodynamic phase space.
        
        Args:
            states: System states
            energies: Corresponding energies
            entropies: Corresponding entropies
            
        Returns:
            3D phase space plot
        """
        
    def create_interactive_dashboard(self, 
                                   evolution_data: EvolutionData) -> InteractiveDashboard:
        """Create interactive evolution dashboard.
        
        Args:
            evolution_data: Complete evolution dataset
            
        Returns:
            Interactive dashboard for exploration
        """

# Example usage
visualizer = EvolutionVisualizer(style='publication')

# Plot evolution trace
fig = visualizer.plot_evolution_trace(
    evolution_history=evolution_results.history,
    metrics=['energy', 'entropy', 'complexity']
)
fig.savefig('evolution_trace.pdf', dpi=300)
```

### LandscapeAnalyzer

Analyze optimization landscapes.

```python
class LandscapeAnalyzer:
    """Analyze optimization landscape properties.
    
    Provides tools for understanding landscape difficulty,
    multimodality, and thermodynamic properties.
    """
    
    def __init__(self):
        self.landscape_metrics = {}
        
    def analyze_landscape(self, 
                         objective_function: Callable,
                         bounds: Tuple[torch.Tensor, torch.Tensor],
                         num_samples: int = 10000) -> LandscapeAnalysis:
        """Comprehensive landscape analysis.
        
        Args:
            objective_function: Function to analyze
            bounds: Variable bounds
            num_samples: Number of samples for analysis
            
        Returns:
            LandscapeAnalysis with:
                - modality: Number of local optima
                - ruggedness: Landscape roughness measure
                - neutrality: Proportion of neutral moves
                - deceptiveness: Gradient reliability
        """
        
    def compute_fitness_distance_correlation(self, 
                                           samples: torch.Tensor,
                                           fitness_values: torch.Tensor,
                                           global_optimum: torch.Tensor) -> float:
        """Compute fitness-distance correlation.
        
        Args:
            samples: Sample points
            fitness_values: Fitness at sample points
            global_optimum: Known global optimum
            
        Returns:
            Fitness-distance correlation coefficient
        """
        
    def estimate_thermodynamic_properties(self, 
                                        objective_function: Callable,
                                        temperature: float) -> Dict[str, float]:
        """Estimate thermodynamic landscape properties.
        
        Args:
            objective_function: Objective function
            temperature: System temperature
            
        Returns:
            Dictionary of thermodynamic properties:
                - heat_capacity: System heat capacity
                - free_energy: Free energy estimate
                - entropy_production: Entropy production rate
        """

def visualize_landscape_slice(objective_function: Callable,
                            center_point: torch.Tensor,
                            slice_direction: torch.Tensor,
                            slice_range: float = 2.0) -> plt.Figure:
    """Visualize 1D slice through landscape.
    
    Args:
        objective_function: Function to slice
        center_point: Center of slice
        slice_direction: Direction of slice
        slice_range: Range of slice
        
    Returns:
        Plot of landscape slice
    """
```

## Analysis Utilities

### ConvergenceAnalyzer

Analyze evolution convergence properties.

```python
class ConvergenceAnalyzer:
    """Analyze convergence behavior of thermodynamic evolution.
    
    Provides tools for understanding convergence rates,
    identifying convergence issues, and optimizing parameters.
    """
    
    def __init__(self):
        self.convergence_tests = [
            'energy_convergence',
            'parameter_convergence', 
            'distribution_convergence'
        ]
        
    def analyze_convergence(self, 
                          evolution_history: List[Dict]) -> ConvergenceAnalysis:
        """Analyze evolution convergence.
        
        Args:
            evolution_history: Complete evolution history
            
        Returns:
            ConvergenceAnalysis with:
                - convergence_rate: Rate of convergence
                - convergence_point: Estimated convergence iteration
                - confidence_interval: Convergence confidence bounds
                - convergence_quality: Quality assessment
        """
        
    def detect_convergence_issues(self, 
                                evolution_history: List[Dict]) -> List[str]:
        """Detect convergence problems.
        
        Args:
            evolution_history: Evolution history to analyze
            
        Returns:
            List of detected issues:
                - 'premature_convergence'
                - 'slow_convergence'
                - 'oscillating_convergence'
                - 'stagnation'
        """
        
    def suggest_parameter_adjustments(self, 
                                    convergence_issues: List[str]) -> Dict[str, float]:
        """Suggest parameter adjustments for convergence issues.
        
        Args:
            convergence_issues: List of detected issues
            
        Returns:
            Dictionary of suggested parameter changes
        """

def plot_convergence_diagnostics(evolution_history: List[Dict]) -> plt.Figure:
    """Create comprehensive convergence diagnostic plots.
    
    Args:
        evolution_history: Evolution history
        
    Returns:
        Multi-panel figure with convergence diagnostics
    """
```

### PerformanceProfiler

Profile performance of thermodynamic algorithms.

```python
class PerformanceProfiler:
    """Profile performance of thermodynamic evolution algorithms.
    
    Measures computational performance, memory usage,
    and algorithmic efficiency.
    """
    
    def __init__(self, 
                 profiling_mode: str = 'comprehensive'):
        """Initialize performance profiler.
        
        Args:
            profiling_mode: 'basic', 'comprehensive', 'memory_focused'
        """
        self.profiling_mode = profiling_mode
        self.performance_data = {}
        
    def profile_evolution(self, 
                         evolution_function: Callable,
                         *args, **kwargs) -> PerformanceReport:
        """Profile evolution function performance.
        
        Args:
            evolution_function: Function to profile
            *args, **kwargs: Arguments for evolution function
            
        Returns:
            PerformanceReport with:
                - execution_time: Total execution time
                - memory_usage: Peak memory usage
                - cpu_utilization: CPU usage statistics
                - gpu_utilization: GPU usage (if applicable)
                - bottlenecks: Identified performance bottlenecks
        """
        
    def profile_memory_usage(self, 
                           evolution_process: Any) -> MemoryReport:
        """Profile memory usage during evolution.
        
        Args:
            evolution_process: Evolution process to monitor
            
        Returns:
            MemoryReport with memory usage over time
        """
        
    def benchmark_against_baseline(self, 
                                  algorithm: Any,
                                  baseline_algorithm: Any,
                                  test_problems: List[Any]) -> BenchmarkReport:
        """Benchmark algorithm against baseline.
        
        Args:
            algorithm: Algorithm to benchmark
            baseline_algorithm: Baseline for comparison
            test_problems: Set of test problems
            
        Returns:
            BenchmarkReport with comparative performance
        """

# Example usage
profiler = PerformanceProfiler(profiling_mode='comprehensive')

performance_report = profiler.profile_evolution(
    evolution_function=optimizer.evolve,
    initial_state=initial_state,
    max_iterations=1000
)

print(f"Execution time: {performance_report.execution_time:.2f}s")
print(f"Peak memory: {performance_report.peak_memory:.1f}MB")
```

## Configuration Utilities

### ConfigurationManager

Manage application configurations.

```python
class ConfigurationManager:
    """Manage Entropic AI application configurations.
    
    Handles loading, validation, and management of
    configuration files and parameters.
    """
    
    def __init__(self, config_directory: str = "configs/"):
        """Initialize configuration manager.
        
        Args:
            config_directory: Directory containing configuration files
        """
        self.config_directory = config_directory
        self.loaded_configs = {}
        
    def load_config(self, 
                   config_name: str,
                   config_type: str = 'yaml') -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_name: Name of configuration file
            config_type: Configuration file type ('yaml', 'json', 'toml')
            
        Returns:
            Configuration dictionary
        """
        
    def validate_config(self, 
                       config: Dict[str, Any],
                       schema: Dict[str, Any]) -> ValidationResult:
        """Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            schema: Validation schema
            
        Returns:
            ValidationResult with validation status and errors
        """
        
    def create_config_template(self, 
                             application_type: str) -> Dict[str, Any]:
        """Create configuration template for application type.
        
        Args:
            application_type: Type of application
            
        Returns:
            Template configuration dictionary
        """
        
    def merge_configs(self, 
                     base_config: Dict[str, Any],
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration files.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """

# Predefined configuration schemas
OPTIMIZATION_CONFIG_SCHEMA = {
    "thermal_parameters": {
        "initial_temperature": {"type": "float", "min": 0.01, "max": 100.0},
        "final_temperature": {"type": "float", "min": 0.001, "max": 10.0},
        "cooling_rate": {"type": "float", "min": 0.8, "max": 0.999}
    },
    "evolution_parameters": {
        "max_iterations": {"type": "int", "min": 100, "max": 100000},
        "convergence_threshold": {"type": "float", "min": 1e-10, "max": 1e-2}
    }
}

def create_default_config(application_type: str,
                         problem_difficulty: str = 'medium') -> Dict[str, Any]:
    """Create default configuration for application type.
    
    Args:
        application_type: Type of application
        problem_difficulty: Expected problem difficulty
        
    Returns:
        Default configuration dictionary
    """
```

### ParameterTuning

Automated parameter tuning utilities.

```python
class ParameterTuner:
    """Automated parameter tuning for thermodynamic algorithms.
    
    Uses meta-optimization to find optimal algorithm parameters
    for specific problem classes.
    """
    
    def __init__(self, 
                 tuning_strategy: str = 'bayesian_optimization'):
        """Initialize parameter tuner.
        
        Args:
            tuning_strategy: Strategy for parameter tuning:
                - 'grid_search': Exhaustive grid search
                - 'random_search': Random parameter sampling
                - 'bayesian_optimization': Bayesian optimization
                - 'evolutionary': Evolutionary parameter optimization
        """
        self.tuning_strategy = tuning_strategy
        self.parameter_space = {}
        self.tuning_history = []
        
    def define_parameter_space(self, 
                             parameter_ranges: Dict[str, Tuple]) -> None:
        """Define parameter search space.
        
        Args:
            parameter_ranges: Dictionary mapping parameter names to ranges
        """
        
    def tune_parameters(self, 
                       objective_function: Callable,
                       validation_problems: List[Any],
                       num_evaluations: int = 100) -> TuningResult:
        """Tune algorithm parameters.
        
        Args:
            objective_function: Function to optimize (algorithm performance)
            validation_problems: Set of validation problems
            num_evaluations: Number of parameter evaluations
            
        Returns:
            TuningResult with:
                - best_parameters: Optimal parameter values
                - parameter_importance: Parameter sensitivity analysis
                - tuning_curve: Performance vs. iteration
        """
        
    def cross_validate_parameters(self, 
                                 parameters: Dict[str, Any],
                                 problems: List[Any],
                                 k_folds: int = 5) -> float:
        """Cross-validate parameter configuration.
        
        Args:
            parameters: Parameter configuration to validate
            problems: Test problems
            k_folds: Number of cross-validation folds
            
        Returns:
            Cross-validated performance score
        """

# Example parameter tuning
tuner = ParameterTuner(tuning_strategy='bayesian_optimization')

tuner.define_parameter_space({
    'initial_temperature': (0.1, 10.0),
    'cooling_rate': (0.9, 0.999),
    'complexity_weight': (0.01, 1.0)
})

tuning_result = tuner.tune_parameters(
    objective_function=evaluate_algorithm_performance,
    validation_problems=benchmark_problems,
    num_evaluations=50
)
```

## Mathematical Utilities

### ThermodynamicMath

Mathematical functions for thermodynamic calculations.

```python
class ThermodynamicMath:
    """Mathematical utilities for thermodynamic calculations.
    
    Provides numerical methods for computing thermodynamic
    quantities and solving thermodynamic equations.
    """
    
    @staticmethod
    def compute_free_energy(energy: torch.Tensor,
                           entropy: torch.Tensor,
                           temperature: float) -> torch.Tensor:
        """Compute Helmholtz free energy F = U - TS.
        
        Args:
            energy: Internal energy
            entropy: Entropy  
            temperature: Temperature
            
        Returns:
            Free energy tensor
        """
        return energy - temperature * entropy
    
    @staticmethod
    def compute_heat_capacity(energy_samples: torch.Tensor,
                            temperature: float) -> float:
        """Compute heat capacity from energy fluctuations.
        
        Args:
            energy_samples: Sample of energy values
            temperature: System temperature
            
        Returns:
            Heat capacity estimate
        """
        energy_variance = torch.var(energy_samples)
        return energy_variance / (temperature ** 2)
    
    @staticmethod
    def compute_entropy_production(states: torch.Tensor,
                                 timesteps: torch.Tensor) -> torch.Tensor:
        """Compute entropy production rate.
        
        Args:
            states: Time series of system states
            timesteps: Corresponding time points
            
        Returns:
            Entropy production rate
        """
        
    @staticmethod
    def solve_thermodynamic_equilibrium(energy_function: Callable,
                                      initial_state: torch.Tensor,
                                      temperature: float) -> torch.Tensor:
        """Solve for thermodynamic equilibrium state.
        
        Args:
            energy_function: System energy function
            initial_state: Initial guess for equilibrium
            temperature: System temperature
            
        Returns:
            Equilibrium state
        """

def compute_partition_function(energy_levels: torch.Tensor,
                             temperature: float) -> float:
    """Compute partition function Z = Σ exp(-E/kT).
    
    Args:
        energy_levels: Available energy levels
        temperature: System temperature
        
    Returns:
        Partition function value
    """
    beta = 1.0 / temperature
    return torch.sum(torch.exp(-beta * energy_levels))

def compute_boltzmann_probability(energy: torch.Tensor,
                                temperature: float,
                                partition_function: float) -> torch.Tensor:
    """Compute Boltzmann probability distribution.
    
    Args:
        energy: Energy values
        temperature: System temperature
        partition_function: Partition function
        
    Returns:
        Probability distribution
    """
    beta = 1.0 / temperature
    return torch.exp(-beta * energy) / partition_function
```

### ComplexityMeasures

Complexity measurement utilities.

```python
class ComplexityMeasures:
    """Computational complexity measurement utilities.
    
    Provides various complexity measures for evaluating
    solution complexity in thermodynamic evolution.
    """
    
    @staticmethod
    def kolmogorov_complexity_estimate(data: torch.Tensor,
                                     method: str = 'compression') -> float:
        """Estimate Kolmogorov complexity.
        
        Args:
            data: Data to analyze
            method: Estimation method ('compression', 'entropy')
            
        Returns:
            Complexity estimate
        """
        
    @staticmethod
    def lempel_ziv_complexity(sequence: torch.Tensor) -> float:
        """Compute Lempel-Ziv complexity.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Lempel-Ziv complexity measure
        """
        
    @staticmethod
    def logical_depth(data: torch.Tensor,
                     computation_model: str = 'turing_machine') -> float:
        """Compute logical depth measure.
        
        Args:
            data: Data to analyze
            computation_model: Model of computation
            
        Returns:
            Logical depth estimate
        """
        
    @staticmethod
    def effective_complexity(data: torch.Tensor,
                           noise_threshold: float = 0.01) -> float:
        """Compute effective complexity (structure vs. randomness).
        
        Args:
            data: Data to analyze
            noise_threshold: Threshold for noise filtering
            
        Returns:
            Effective complexity measure
        """

def analyze_complexity_scaling(algorithm: Any,
                             problem_sizes: List[int]) -> Dict[str, Any]:
    """Analyze algorithmic complexity scaling.
    
    Args:
        algorithm: Algorithm to analyze
        problem_sizes: List of problem sizes to test
        
    Returns:
        Complexity scaling analysis
    """
```

## Export and Import Utilities

### ResultExporter

Export evolution results and analyses.

```python
class ResultExporter:
    """Export evolution results in various formats.
    
    Supports exporting to common scientific and engineering
    formats for further analysis and publication.
    """
    
    def __init__(self, export_format: str = 'comprehensive'):
        """Initialize result exporter.
        
        Args:
            export_format: Export format ('minimal', 'comprehensive', 'publication')
        """
        
    def export_evolution_results(self, 
                               results: EvolutionResult,
                               output_path: str,
                               format: str = 'hdf5') -> None:
        """Export complete evolution results.
        
        Args:
            results: Evolution results to export
            output_path: Output file path
            format: Export format ('hdf5', 'pickle', 'json', 'csv')
        """
        
    def export_for_publication(self, 
                             results: EvolutionResult,
                             figures: List[plt.Figure],
                             output_directory: str) -> None:
        """Export results formatted for publication.
        
        Args:
            results: Evolution results
            figures: Generated figures
            output_directory: Output directory
        """
        
    def create_summary_report(self, 
                            results: EvolutionResult,
                            template: str = 'standard') -> str:
        """Create formatted summary report.
        
        Args:
            results: Evolution results
            template: Report template
            
        Returns:
            Formatted report string
        """

def export_to_matlab(results: EvolutionResult,
                    filename: str) -> None:
    """Export results to MATLAB format.
    
    Args:
        results: Evolution results
        filename: Output filename (.mat)
    """

def export_to_r(results: EvolutionResult,
               filename: str) -> None:
    """Export results to R format.
    
    Args:
        results: Evolution results  
        filename: Output filename (.rds)
    """
```

## Testing Utilities

### TestProblemSuite

Standard test problems for benchmarking.

```python
class TestProblemSuite:
    """Suite of standard test problems for benchmarking.
    
    Provides well-known optimization problems for testing
    and comparing thermodynamic evolution algorithms.
    """
    
    def __init__(self):
        self.optimization_problems = {}
        self.discovery_problems = {}
        self.design_problems = {}
        
    def get_optimization_problems(self, 
                                difficulty: str = 'all') -> Dict[str, TestProblem]:
        """Get optimization test problems.
        
        Args:
            difficulty: Problem difficulty ('easy', 'medium', 'hard', 'all')
            
        Returns:
            Dictionary of test problems
        """
        
    def get_discovery_problems(self, 
                             domain: str = 'all') -> Dict[str, TestProblem]:
        """Get scientific discovery test problems.
        
        Args:
            domain: Problem domain ('physics', 'biology', 'chemistry', 'all')
            
        Returns:
            Dictionary of test problems
        """
        
    def benchmark_algorithm(self, 
                          algorithm: Any,
                          problem_set: str = 'optimization',
                          metrics: List[str] = ['accuracy', 'efficiency']) -> BenchmarkResult:
        """Benchmark algorithm on test problems.
        
        Args:
            algorithm: Algorithm to benchmark
            problem_set: Set of problems to use
            metrics: Metrics to evaluate
            
        Returns:
            Benchmark results
        """

# Standard test problems
OPTIMIZATION_PROBLEMS = {
    'sphere': SphereFunction(dimensions=10),
    'rosenbrock': RosenbrockFunction(dimensions=10),
    'rastrigin': RastriginFunction(dimensions=10),
    'ackley': AckleyFunction(dimensions=10),
    'griewank': GriewankFunction(dimensions=10)
}

DISCOVERY_PROBLEMS = {
    'pendulum_law': PendulumLawDiscovery(),
    'spring_dynamics': SpringDynamicsDiscovery(),
    'planetary_motion': PlanetaryMotionDiscovery()
}
```

This comprehensive utilities API provides all the essential tools for working with Entropic AI applications, from data preprocessing to result analysis and export. The utilities are designed to be modular and extensible, allowing users to customize and extend functionality as needed.
