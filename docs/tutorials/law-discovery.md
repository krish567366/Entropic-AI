# Scientific Law Discovery through Thermodynamic Evolution

This tutorial demonstrates how to use Entropic AI to discover fundamental scientific laws and mathematical relationships from experimental data. The approach treats symbolic expressions as thermodynamic entities that evolve toward configurations that maximize explanatory power while minimizing complexity.

## Overview

Scientific law discovery in Entropic AI operates through:

1. **Symbolic Thermodynamics**: Representing mathematical expressions as energy states
2. **Data-Driven Evolution**: Using experimental data to guide thermodynamic forces
3. **Complexity Minimization**: Applying Occam's razor through entropy constraints
4. **Emergent Simplicity**: Discovering parsimonious laws through free energy minimization

## Prerequisites

```python
import numpy as np
import pandas as pd
import sympy as sp
from sklearn.metrics import r2_score, mean_squared_error
from entropic-ai.core import ThermodynamicNetwork, ComplexityOptimizer
from entropic-ai.applications import LawDiscovery
from entropic-ai.symbolic import SymbolicExpression, OperatorLibrary
from entropic-ai.optimization import ParsimonyCriterion
```

## Basic Law Discovery

### Step 1: Prepare Experimental Data

Start with clean, well-structured experimental data:

```python
# Example: Pendulum motion data
pendulum_data = pd.DataFrame({
    'length': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # meters
    'period': [0.634, 0.897, 1.099, 1.269, 1.419, 1.554, 1.679, 1.795, 1.904, 2.006],  # seconds
    'mass': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # kg (constant)
    'gravity': [9.81] * 10  # m/s² (constant)
})

# Define variable context
variable_context = {
    'independent': ['length', 'mass', 'gravity'],
    'dependent': ['period'],
    'constants': ['pi'],
    'units': {
        'length': 'm',
        'period': 's',
        'mass': 'kg',
        'gravity': 'm/s²'
    }
}
```

### Step 2: Initialize Law Discovery System

Set up the symbolic evolution environment:

```python
# Create law discovery system
law_discoverer = LawDiscovery(
    operator_library=['add', 'mul', 'div', 'pow', 'sqrt', 'sin', 'cos', 'log', 'exp'],
    complexity_weights={
        'expression_length': 0.4,
        'operator_complexity': 0.3,
        'parameter_count': 0.3
    },
    thermal_parameters={
        'initial_temperature': 5.0,
        'final_temperature': 0.01,
        'cooling_rate': 0.95,
        'equilibration_steps': 100
    }
)

# Set target data and context
law_discoverer.set_data(pendulum_data, variable_context)
```

### Step 3: Run Symbolic Evolution

Execute the discovery process:

```python
# Initialize with random symbolic expressions
initial_population = law_discoverer.generate_random_expressions(
    population_size=50,
    max_depth=4,
    variable_probability=0.6
)

# Evolve symbolic expressions
discovery_results = law_discoverer.evolve(
    initial_population=initial_population,
    max_generations=2000,
    convergence_threshold=1e-8,
    diversity_maintenance=True
)

# Extract discovered laws
discovered_laws = discovery_results.best_expressions
performance_metrics = discovery_results.final_metrics
evolution_history = discovery_results.evolution_trace
```

## Advanced Law Discovery

### Multi-Variable Relationships

Discover laws involving multiple variables:

```python
class MultiVariableLawDiscovery:
    def __init__(self, data, target_variable):
        self.data = data
        self.target_variable = target_variable
        self.independent_vars = [col for col in data.columns if col != target_variable]
        
    def discovery_energy_function(self, expression):
        """Energy function for multi-variable law discovery."""
        
        # Prediction accuracy energy
        prediction_error = self.compute_prediction_error(expression)
        
        # Complexity penalty
        complexity_penalty = self.compute_complexity_penalty(expression)
        
        # Dimensional consistency penalty
        dimensional_penalty = self.check_dimensional_consistency(expression)
        
        # Physical plausibility penalty
        physics_penalty = self.check_physical_plausibility(expression)
        
        total_energy = (
            prediction_error +
            0.1 * complexity_penalty +
            10.0 * dimensional_penalty +
            5.0 * physics_penalty
        )
        
        return total_energy
    
    def compute_prediction_error(self, expression):
        """Compute prediction error for expression."""
        try:
            # Evaluate expression on data
            predictions = self.evaluate_expression(expression, self.data)
            targets = self.data[self.target_variable]
            
            # Mean squared error
            mse = np.mean((predictions - targets) ** 2)
            
            # Normalize by target variance
            target_variance = np.var(targets)
            normalized_mse = mse / target_variance
            
            return normalized_mse
            
        except Exception:
            # Invalid expression gets high energy
            return 1000.0
    
    def compute_complexity_penalty(self, expression):
        """Compute complexity penalty for expression."""
        
        # Expression tree size
        tree_size = expression.count_nodes()
        
        # Operator complexity
        operator_complexity = sum(
            self.operator_weights.get(op, 1.0)
            for op in expression.get_operators()
        )
        
        # Parameter count
        parameter_count = len(expression.get_parameters())
        
        return tree_size + operator_complexity + parameter_count
    
    def check_dimensional_consistency(self, expression):
        """Check dimensional consistency of expression."""
        try:
            # Perform dimensional analysis
            expr_units = expression.compute_units(self.variable_units)
            target_units = self.variable_units[self.target_variable]
            
            if expr_units == target_units:
                return 0.0
            else:
                return 1.0  # Dimensional inconsistency penalty
                
        except Exception:
            return 1.0  # Invalid dimensional analysis
```

### Physics-Informed Discovery

Incorporate physical principles into discovery:

```python
class PhysicsInformedDiscovery:
    def __init__(self, physical_principles):
        self.physical_principles = physical_principles
        
    def add_physics_constraints(self, expression):
        """Add physics-based constraints to expressions."""
        
        constraints = []
        
        # Conservation laws
        if 'energy_conservation' in self.physical_principles:
            constraints.append(self.check_energy_conservation(expression))
        
        # Symmetry principles
        if 'rotational_symmetry' in self.physical_principles:
            constraints.append(self.check_rotational_symmetry(expression))
        
        # Scale invariance
        if 'scale_invariance' in self.physical_principles:
            constraints.append(self.check_scale_invariance(expression))
        
        return constraints
    
    def check_energy_conservation(self, expression):
        """Check if expression respects energy conservation."""
        # Implement energy conservation check
        pass
    
    def check_rotational_symmetry(self, expression):
        """Check if expression has appropriate rotational symmetry."""
        # Implement symmetry check
        pass
    
    def discover_with_physics(self, data, physics_principles):
        """Discover laws incorporating physics principles."""
        
        # Enhanced energy function with physics constraints
        def physics_aware_energy(expression):
            # Basic data fitting energy
            fitting_error = self.compute_fitting_error(expression, data)
            
            # Physics constraint violations
            physics_violations = sum(
                self.evaluate_physics_constraint(expression, constraint)
                for constraint in physics_principles
            )
            
            return fitting_error + 10.0 * physics_violations
        
        # Run discovery with physics-aware energy
        discoverer = LawDiscovery(energy_function=physics_aware_energy)
        return discoverer.evolve()
```

### Hierarchical Law Discovery

Discover laws at multiple scales:

```python
class HierarchicalLawDiscovery:
    def __init__(self):
        self.scale_hierarchy = ['microscopic', 'mesoscopic', 'macroscopic']
        self.scale_laws = {}
        
    def discover_hierarchical_laws(self, multi_scale_data):
        """Discover laws across multiple scales."""
        
        # Discover laws at each scale
        for scale in self.scale_hierarchy:
            print(f"Discovering laws at {scale} scale...")
            
            scale_data = multi_scale_data[scale]
            scale_discoverer = self.create_scale_discoverer(scale)
            
            scale_laws = scale_discoverer.discover(scale_data)
            self.scale_laws[scale] = scale_laws
        
        # Find relationships between scales
        inter_scale_relations = self.discover_scale_relations()
        
        return self.scale_laws, inter_scale_relations
    
    def create_scale_discoverer(self, scale):
        """Create scale-specific law discoverer."""
        
        scale_configs = {
            'microscopic': {
                'operators': ['add', 'mul', 'div', 'exp', 'log'],
                'complexity_limit': 20,
                'precision_requirement': 1e-6
            },
            'mesoscopic': {
                'operators': ['add', 'mul', 'div', 'pow', 'sqrt'],
                'complexity_limit': 15,
                'precision_requirement': 1e-4
            },
            'macroscopic': {
                'operators': ['add', 'mul', 'div', 'pow'],
                'complexity_limit': 10,
                'precision_requirement': 1e-2
            }
        }
        
        config = scale_configs[scale]
        
        return LawDiscovery(
            operator_library=config['operators'],
            complexity_limit=config['complexity_limit'],
            precision_requirement=config['precision_requirement']
        )
    
    def discover_scale_relations(self):
        """Discover relationships between different scales."""
        
        relations = {}
        
        for i, scale1 in enumerate(self.scale_hierarchy[:-1]):
            scale2 = self.scale_hierarchy[i+1]
            
            # Look for emergence patterns
            emergence_law = self.find_emergence_pattern(
                self.scale_laws[scale1],
                self.scale_laws[scale2]
            )
            
            relations[f"{scale1}_to_{scale2}"] = emergence_law
        
        return relations
```

## Domain-Specific Discovery

### Biological Systems

Discover laws in biological data:

```python
class BiologicalLawDiscovery:
    def __init__(self):
        self.biological_operators = [
            'sigmoid', 'hill_function', 'michaelis_menten',
            'exponential_decay', 'logistic_growth'
        ]
        
    def discover_biological_laws(self, biological_data):
        """Discover laws specific to biological systems."""
        
        # Biological energy function
        def biological_energy(expression):
            # Standard fitting error
            fitting_error = self.compute_fitting_error(expression, biological_data)
            
            # Biological plausibility
            bio_plausibility = self.assess_biological_plausibility(expression)
            
            # Monotonicity constraints (many biological relationships are monotonic)
            monotonicity_violation = self.check_monotonicity(expression)
            
            return fitting_error + bio_plausibility + monotonicity_violation
        
        # Enhanced discoverer for biology
        bio_discoverer = LawDiscovery(
            operator_library=self.biological_operators,
            energy_function=biological_energy,
            constraints=['positive_values', 'bounded_growth']
        )
        
        return bio_discoverer.discover(biological_data)
    
    def assess_biological_plausibility(self, expression):
        """Assess biological plausibility of expression."""
        
        plausibility_score = 0.0
        
        # Check for unrealistic parameter values
        parameters = expression.get_parameters()
        for param_name, param_value in parameters.items():
            if param_value < 0 and param_name in ['rate_constant', 'concentration']:
                plausibility_score += 1.0
        
        # Check for appropriate functional forms
        operators = expression.get_operators()
        if 'exp' in operators and expression.has_unbounded_growth():
            plausibility_score += 0.5  # Unbounded growth is often unrealistic
        
        return plausibility_score
```

### Economics and Finance

Discover economic relationships:

```python
class EconomicLawDiscovery:
    def __init__(self):
        self.economic_operators = [
            'elasticity', 'logarithmic', 'exponential',
            'power_law', 'cobb_douglas'
        ]
        
    def discover_economic_laws(self, economic_data):
        """Discover economic relationships from data."""
        
        # Economic-specific energy function
        def economic_energy(expression):
            # Prediction accuracy
            prediction_error = self.compute_prediction_error(expression, economic_data)
            
            # Economic rationality constraints
            rationality_violations = self.check_economic_rationality(expression)
            
            # Stability requirements
            stability_violations = self.check_economic_stability(expression)
            
            return prediction_error + rationality_violations + stability_violations
        
        # Economic law discoverer
        econ_discoverer = LawDiscovery(
            operator_library=self.economic_operators,
            energy_function=economic_energy,
            constraints=['monotonicity', 'concavity', 'homogeneity']
        )
        
        return econ_discoverer.discover(economic_data)
    
    def check_economic_rationality(self, expression):
        """Check if expression satisfies economic rationality."""
        
        violations = 0.0
        
        # Demand curves should be downward sloping
        if self.is_demand_function(expression):
            if not self.is_downward_sloping(expression):
                violations += 1.0
        
        # Supply curves should be upward sloping
        if self.is_supply_function(expression):
            if not self.is_upward_sloping(expression):
                violations += 1.0
        
        return violations
```

## Verification and Validation

### Statistical Validation

Validate discovered laws statistically:

```python
class LawValidator:
    def __init__(self):
        self.validation_methods = [
            'cross_validation',
            'bootstrap_confidence',
            'information_criteria',
            'residual_analysis'
        ]
    
    def validate_discovered_law(self, law, data):
        """Comprehensive validation of discovered law."""
        
        validation_results = {}
        
        # Cross-validation
        cv_results = self.cross_validate_law(law, data)
        validation_results['cross_validation'] = cv_results
        
        # Bootstrap confidence intervals
        bootstrap_results = self.bootstrap_confidence_intervals(law, data)
        validation_results['bootstrap'] = bootstrap_results
        
        # Information criteria
        ic_results = self.compute_information_criteria(law, data)
        validation_results['information_criteria'] = ic_results
        
        # Residual analysis
        residual_results = self.analyze_residuals(law, data)
        validation_results['residuals'] = residual_results
        
        # Overall validation score
        validation_results['overall_score'] = self.compute_overall_score(validation_results)
        
        return validation_results
    
    def cross_validate_law(self, law, data, k_folds=5):
        """K-fold cross-validation of discovered law."""
        
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, test_idx in kf.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Fit law parameters on training data
            fitted_law = self.fit_law_parameters(law, train_data)
            
            # Evaluate on test data
            test_score = self.evaluate_law_performance(fitted_law, test_data)
            cv_scores.append(test_score)
        
        return {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'individual_scores': cv_scores
        }
    
    def bootstrap_confidence_intervals(self, law, data, n_bootstrap=1000):
        """Bootstrap confidence intervals for law parameters."""
        
        n_samples = len(data)
        bootstrap_params = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_data = data.iloc[bootstrap_indices]
            
            # Fit law to bootstrap sample
            try:
                fitted_law = self.fit_law_parameters(law, bootstrap_data)
                bootstrap_params.append(fitted_law.get_parameters())
            except:
                continue  # Skip failed fits
        
        # Compute confidence intervals
        confidence_intervals = {}
        for param_name in bootstrap_params[0].keys():
            param_values = [params[param_name] for params in bootstrap_params]
            ci_lower = np.percentile(param_values, 2.5)
            ci_upper = np.percentile(param_values, 97.5)
            confidence_intervals[param_name] = (ci_lower, ci_upper)
        
        return confidence_intervals
```

### Physical Validation

Validate against known physical principles:

```python
class PhysicalValidator:
    def __init__(self, known_principles):
        self.known_principles = known_principles
        
    def validate_against_physics(self, discovered_law):
        """Validate discovered law against known physics."""
        
        validation_results = {}
        
        # Dimensional analysis
        dimensional_check = self.check_dimensional_consistency(discovered_law)
        validation_results['dimensional_consistency'] = dimensional_check
        
        # Symmetry properties
        symmetry_check = self.check_symmetry_properties(discovered_law)
        validation_results['symmetry_properties'] = symmetry_check
        
        # Conservation laws
        conservation_check = self.check_conservation_laws(discovered_law)
        validation_results['conservation_laws'] = conservation_check
        
        # Limiting behavior
        limits_check = self.check_limiting_behavior(discovered_law)
        validation_results['limiting_behavior'] = limits_check
        
        return validation_results
    
    def check_dimensional_consistency(self, law):
        """Check dimensional consistency of discovered law."""
        
        # Extract all terms in the expression
        terms = law.get_terms()
        
        # Check that all terms have same dimensions
        term_dimensions = [self.compute_term_dimensions(term) for term in terms]
        
        consistent = all(dim == term_dimensions[0] for dim in term_dimensions)
        
        return {
            'consistent': consistent,
            'term_dimensions': term_dimensions
        }
```

## Real-World Example: Kepler's Laws Discovery

Complete example discovering Kepler's laws from planetary data:

```python
def discover_keplers_laws():
    """Discover Kepler's laws from planetary motion data."""
    
    # Planetary data (simplified)
    planetary_data = pd.DataFrame({
        'planet': ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn'],
        'semi_major_axis': [0.387, 0.723, 1.000, 1.524, 5.204, 9.582],  # AU
        'orbital_period': [0.241, 0.615, 1.000, 1.881, 11.862, 29.457],  # years
        'eccentricity': [0.206, 0.007, 0.017, 0.094, 0.049, 0.057]
    })
    
    # First Law Discovery: Orbital shape
    print("Discovering First Law (orbital shape)...")
    shape_discoverer = LawDiscovery(
        operator_library=['add', 'mul', 'div', 'pow', 'sqrt', 'cos'],
        target_variables=['radius'],
        independent_variables=['true_anomaly', 'eccentricity', 'semi_major_axis']
    )
    
    first_law = shape_discoverer.discover(planetary_data)
    print(f"First Law: {first_law.best_expression}")
    
    # Second Law Discovery: Equal areas
    print("Discovering Second Law (equal areas)...")
    area_discoverer = LawDiscovery(
        operator_library=['add', 'mul', 'div', 'sqrt'],
        target_variables=['angular_velocity'],
        independent_variables=['radius', 'eccentricity']
    )
    
    second_law = area_discoverer.discover(planetary_data)
    print(f"Second Law: {second_law.best_expression}")
    
    # Third Law Discovery: Period-distance relationship
    print("Discovering Third Law (period-distance relationship)...")
    period_discoverer = LawDiscovery(
        operator_library=['add', 'mul', 'div', 'pow'],
        target_variables=['orbital_period'],
        independent_variables=['semi_major_axis'],
        complexity_preference='minimal'
    )
    
    third_law = period_discoverer.discover(planetary_data)
    print(f"Third Law: {third_law.best_expression}")
    
    # Validate discoveries
    validator = LawValidator()
    
    for i, law in enumerate([first_law, second_law, third_law], 1):
        validation = validator.validate_discovered_law(law, planetary_data)
        print(f"Law {i} validation score: {validation['overall_score']:.3f}")
    
    return first_law, second_law, third_law

# Run Kepler's laws discovery
keplers_laws = discover_keplers_laws()
```

## Best Practices

### 1. Data Preparation

- **Quality Control**: Clean data, handle outliers, check for systematic errors
- **Variable Selection**: Include relevant variables, avoid redundant measurements
- **Dimensionality**: Ensure proper units and dimensional consistency

### 2. Expression Search

- **Operator Selection**: Choose operators appropriate for the domain
- **Complexity Control**: Balance expressiveness with parsimony
- **Search Strategy**: Use appropriate cooling schedules and population sizes

### 3. Validation Strategy

- **Multiple Methods**: Use cross-validation, bootstrap, and theoretical validation
- **Physical Constraints**: Incorporate known physical principles
- **Out-of-Sample Testing**: Test on independent datasets when available

### 4. Interpretation

- **Physical Meaning**: Ensure discovered laws have physical interpretation
- **Parameter Significance**: Validate that parameters are meaningful
- **Limiting Cases**: Check behavior in extreme conditions

This tutorial demonstrates how thermodynamic evolution can discover fundamental laws that capture the essential relationships in complex datasets while maintaining simplicity and physical plausibility.
