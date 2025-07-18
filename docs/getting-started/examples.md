# Examples and Use Cases

This section provides comprehensive examples showcasing the versatility and power of Entropic AI across different domains. Each example demonstrates how thermodynamic principles can be applied to solve real-world problems.

## Basic Examples

### 1. Simple Pattern Formation

Let's start with a fundamental example: spontaneous pattern formation from thermal noise.

```python
import torch
import numpy as np
from entropic-ai import EntropicNetwork, GenerativeDiffuser
from entropic-ai.utils.visualization import plot_pattern_formation

# Create a simple thermodynamic network
network = EntropicNetwork(
    nodes=64,
    temperature=2.0,
    entropy_regularization=0.1
)

# Initialize diffuser
diffuser = GenerativeDiffuser(
    network=network,
    diffusion_steps=200,
    crystallization_threshold=0.05
)

# Start with pure thermal noise
noise = torch.randn(1, 64) * 2.0
print(f"Initial entropy: {torch.var(noise).item():.3f}")

# Evolve pattern
pattern = diffuser.evolve(noise)
print(f"Final entropy: {torch.var(pattern).item():.3f}")

# Visualize the evolution
plot_pattern_formation(diffuser.evolution_history)
```

### 2. Energy Landscape Exploration

Explore how the system navigates complex energy landscapes:

```python
from entropic-ai.utils.visualization import plot_energy_landscape
from entropic-ai.core import ComplexityOptimizer

# Create a complex energy landscape
def custom_energy_function(x):
    """Multi-modal energy function with several minima."""
    return (torch.sin(x * 3) * torch.cos(x * 2) + 
            0.1 * x**2 - 
            0.05 * torch.sin(x * 10))

# Set up system with custom energy
optimizer = ComplexityOptimizer(
    method="custom_energy",
    energy_function=custom_energy_function
)

network = EntropicNetwork(nodes=32, temperature=1.5)
diffuser = GenerativeDiffuser(network, optimizer, diffusion_steps=300)

# Explore landscape from different starting points
starting_points = [
    torch.randn(1, 32) * 0.5,  # Small perturbation
    torch.randn(1, 32) * 2.0,  # Medium chaos
    torch.randn(1, 32) * 5.0   # High chaos
]

results = []
for i, start in enumerate(starting_points):
    result = diffuser.evolve(start, return_trajectory=True)
    results.append(result)
    print(f"Starting point {i+1}: Final energy = {result.final_energy:.3f}")

# Plot energy landscapes
plot_energy_landscape(results, save_path="energy_exploration.png")
```

## Scientific Applications

### 3. Crystal Structure Prediction

Predict crystal structures using thermodynamic principles:

```python
from entropic-ai.applications import CrystalEvolution
import numpy as np

# Define crystal parameters
crystal_params = {
    "lattice_type": "cubic",
    "space_group": "Pm3m",
    "unit_cell_size": (5.0, 5.0, 5.0),
    "atom_types": ["Si", "O"],
    "composition": {"Si": 1, "O": 2}  # SiO2
}

# Initialize crystal evolver
crystal_evolver = CrystalEvolution(
    crystal_params=crystal_params,
    thermodynamic_constraints={
        "pressure": 1.0,      # 1 atm
        "temperature": 298.15  # Room temperature
    }
)

# Start from random atomic positions
random_structure = crystal_evolver.generate_random_structure(
    n_unit_cells=(2, 2, 2)
)

# Evolve to stable crystal structure
stable_crystal = crystal_evolver.evolve_structure(
    initial_structure=random_structure,
    evolution_steps=500,
    include_phonons=True  # Include vibrational effects
)

print(f"Final crystal energy: {stable_crystal.formation_energy:.3f} eV/atom")
print(f"Space group: {stable_crystal.space_group}")
print(f"Lattice parameters: {stable_crystal.lattice_parameters}")

# Export to CIF format
stable_crystal.save_cif("evolved_crystal.cif")
```

### 4. Protein Folding Simulation

Apply thermodynamic evolution to protein folding:

```python
from entropic-ai.applications import ProteinFolding
from entropic-ai.utils.molecular import load_protein_sequence

# Load protein sequence
sequence = "MKALIVLGLVLLAALVTIITVPVVLLAIVMWSDLGSLC"  # Simplified sequence
protein_folder = ProteinFolding(
    sequence=sequence,
    force_field="charmm36",
    solvent_model="implicit_water"
)

# Set folding parameters
folding_params = {
    "temperature": 310.0,  # Physiological temperature
    "ph": 7.4,            # Physiological pH
    "ionic_strength": 0.15  # Physiological salt concentration
}

# Start from extended conformation
extended_protein = protein_folder.generate_extended_conformation()

# Fold using thermodynamic evolution
folded_protein = protein_folder.fold_protein(
    initial_conformation=extended_protein,
    folding_steps=1000,
    include_solvation=True,
    track_secondary_structure=True
)

print(f"Folding energy: {folded_protein.energy:.2f} kcal/mol")
print(f"Radius of gyration: {folded_protein.radius_of_gyration:.2f} Å")
print(f"Secondary structure: {folded_protein.secondary_structure}")

# Validate fold
ramachandran_score = folded_protein.ramachandran_analysis()
print(f"Ramachandran score: {ramachandran_score:.3f}")
```

### 5. Climate Model Optimization

Optimize climate model parameters using thermodynamic principles:

```python
from entropic-ai.applications import ClimateModelOptimization
import xarray as xr

# Load climate data
temperature_data = xr.open_dataset("global_temperature_anomalies.nc")
precipitation_data = xr.open_dataset("global_precipitation.nc")

# Initialize climate model optimizer
climate_optimizer = ClimateModelOptimization(
    model_type="energy_balance_model",
    observational_data={
        "temperature": temperature_data,
        "precipitation": precipitation_data
    },
    optimization_targets=[
        "temperature_trend",
        "precipitation_patterns", 
        "extreme_events"
    ]
)

# Define parameter space
parameter_space = {
    "climate_sensitivity": (1.5, 4.5),     # °C per CO2 doubling
    "ocean_heat_capacity": (10, 50),       # Heat capacity factor
    "cloud_feedback": (-0.5, 0.5),        # Cloud feedback parameter
    "aerosol_forcing": (-2.0, 0.0)        # Aerosol radiative forcing
}

# Optimize using thermodynamic evolution
optimal_params = climate_optimizer.optimize_parameters(
    parameter_space=parameter_space,
    evolution_steps=300,
    ensemble_size=20
)

print("Optimized climate parameters:")
for param, value in optimal_params.items():
    print(f"{param}: {value:.3f}")

# Validate against observations
validation_score = climate_optimizer.validate_model(optimal_params)
print(f"Model validation score: {validation_score:.3f}")
```

## Engineering Applications

### 6. Antenna Design Optimization

Design optimal antenna configurations:

```python
from entropic-ai.applications import AntennaDesign
from entropic-ai.utils.electromagnetic import calculate_radiation_pattern

# Define antenna design requirements
requirements = {
    "frequency": 2.4e9,        # 2.4 GHz (WiFi band)
    "gain": ">= 10 dBi",       # Minimum gain
    "bandwidth": ">= 100 MHz", # Minimum bandwidth
    "vswr": "<= 2.0",         # Maximum VSWR
    "size_constraint": (0.1, 0.1, 0.02)  # Max dimensions (m)
}

# Initialize antenna designer
antenna_designer = AntennaDesign(
    requirements=requirements,
    design_space="microstrip_patch",
    substrate_properties={
        "dielectric_constant": 4.4,
        "loss_tangent": 0.02,
        "thickness": 1.6e-3  # 1.6 mm
    }
)

# Start with random geometry
random_geometry = antenna_designer.generate_random_geometry()

# Evolve antenna design
optimal_antenna = antenna_designer.evolve_design(
    initial_geometry=random_geometry,
    evolution_steps=200,
    electromagnetic_simulation=True
)

print(f"Optimal antenna dimensions: {optimal_antenna.dimensions}")
print(f"Achieved gain: {optimal_antenna.gain:.2f} dBi")
print(f"Bandwidth: {optimal_antenna.bandwidth/1e6:.1f} MHz")
print(f"VSWR: {optimal_antenna.vswr:.2f}")

# Generate radiation pattern
radiation_pattern = calculate_radiation_pattern(optimal_antenna)
radiation_pattern.plot_3d(save_path="antenna_pattern.png")
```

### 7. Supply Chain Optimization

Optimize supply chain networks using thermodynamic principles:

```python
from entropic-ai.applications import SupplyChainOptimization
import pandas as pd

# Load supply chain data
suppliers = pd.read_csv("suppliers.csv")
warehouses = pd.read_csv("warehouses.csv") 
customers = pd.read_csv("customers.csv")
demand_forecast = pd.read_csv("demand_forecast.csv")

# Initialize supply chain optimizer
supply_optimizer = SupplyChainOptimization(
    suppliers=suppliers,
    warehouses=warehouses,
    customers=customers,
    constraints={
        "capacity_limits": True,
        "lead_times": True,
        "sustainability_goals": 0.7  # 70% sustainable sourcing
    }
)

# Define optimization objectives
objectives = {
    "cost_minimization": 0.4,
    "delivery_time": 0.3,
    "sustainability": 0.2,
    "resilience": 0.1
}

# Thermodynamic supply chain evolution
optimal_network = supply_optimizer.evolve_network(
    demand_forecast=demand_forecast,
    objectives=objectives,
    evolution_steps=150,
    include_disruption_scenarios=True
)

print("Optimized supply chain metrics:")
print(f"Total cost: ${optimal_network.total_cost:,.2f}")
print(f"Average delivery time: {optimal_network.avg_delivery_time:.1f} days")
print(f"Sustainability score: {optimal_network.sustainability_score:.3f}")
print(f"Resilience index: {optimal_network.resilience_index:.3f}")

# Export network configuration
optimal_network.export_to_json("optimized_supply_chain.json")
```

## Financial Applications

### 8. Portfolio Optimization

Apply thermodynamic principles to financial portfolio optimization:

```python
from entropic-ai.applications import PortfolioOptimization
import yfinance as yf

# Download stock data
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
stock_data = yf.download(symbols, start="2020-01-01", end="2023-12-31")["Adj Close"]

# Initialize portfolio optimizer
portfolio_optimizer = PortfolioOptimization(
    asset_data=stock_data,
    risk_model="black_litterman",
    constraints={
        "max_weight_per_asset": 0.15,
        "min_weight_per_asset": 0.01,
        "max_sector_exposure": 0.3,
        "target_return": 0.12  # 12% annual return
    }
)

# Define thermodynamic portfolio evolution
portfolio_params = {
    "temperature": 0.1,        # Low temperature for stability
    "risk_aversion": 3.0,      # Moderate risk aversion
    "rebalancing_frequency": "monthly"
}

# Evolve optimal portfolio
optimal_weights = portfolio_optimizer.evolve_portfolio(
    portfolio_params=portfolio_params,
    evolution_steps=100,
    include_transaction_costs=True
)

print("Optimal portfolio allocation:")
for symbol, weight in zip(symbols, optimal_weights):
    print(f"{symbol}: {weight*100:.2f}%")

# Backtest performance
backtest_results = portfolio_optimizer.backtest(
    weights=optimal_weights,
    start_date="2024-01-01",
    end_date="2024-12-31"
)

print(f"\nBacktest results:")
print(f"Annual return: {backtest_results.annual_return:.2%}")
print(f"Volatility: {backtest_results.volatility:.2%}")
print(f"Sharpe ratio: {backtest_results.sharpe_ratio:.3f}")
print(f"Max drawdown: {backtest_results.max_drawdown:.2%}")
```

### 9. Cryptocurrency Trading Strategy

Develop trading strategies using thermodynamic market analysis:

```python
from entropic-ai.applications import CryptoTradingStrategy
from entropic-ai.utils.market import get_crypto_data

# Get cryptocurrency data
crypto_data = get_crypto_data(
    symbols=["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD"],
    timeframe="1h",
    period="90d"
)

# Initialize trading strategy
trading_strategy = CryptoTradingStrategy(
    market_data=crypto_data,
    strategy_type="thermodynamic_momentum",
    risk_parameters={
        "max_position_size": 0.1,      # 10% max position
        "stop_loss": 0.05,             # 5% stop loss
        "take_profit": 0.15,           # 15% take profit
        "max_daily_trades": 5
    }
)

# Define thermodynamic indicators
indicators = {
    "market_temperature": {
        "window": 24,                   # 24-hour window
        "method": "volatility_based"
    },
    "momentum_entropy": {
        "window": 12,                   # 12-hour window
        "threshold": 0.7
    },
    "liquidity_pressure": {
        "depth_levels": 10,
        "update_frequency": "5min"
    }
}

# Evolve trading strategy
evolved_strategy = trading_strategy.evolve_strategy(
    indicators=indicators,
    evolution_steps=50,
    fitness_function="risk_adjusted_return"
)

print("Evolved trading strategy:")
print(f"Strategy parameters: {evolved_strategy.parameters}")
print(f"Expected return: {evolved_strategy.expected_return:.2%}")
print(f"Risk score: {evolved_strategy.risk_score:.3f}")

# Simulate trading
simulation_results = evolved_strategy.simulate_trading(
    initial_capital=10000,
    simulation_days=30
)

print(f"\nSimulation results:")
print(f"Final capital: ${simulation_results.final_capital:,.2f}")
print(f"Total return: {simulation_results.total_return:.2%}")
print(f"Win rate: {simulation_results.win_rate:.1%}")
print(f"Profit factor: {simulation_results.profit_factor:.2f}")
```

## Creative Applications

### 10. Generative Art

Create art using thermodynamic evolution:

```python
from entropic-ai.applications import GenerativeArt
from entropic-ai.utils.visualization import save_artistic_evolution
import matplotlib.pyplot as plt

# Initialize generative art system
art_generator = GenerativeArt(
    canvas_size=(512, 512),
    color_space="HSV",
    artistic_style="abstract_expressionism"
)

# Define artistic parameters
art_params = {
    "chaos_level": 0.8,              # High initial chaos
    "color_harmony": "complementary", # Color scheme
    "texture_complexity": 0.6,        # Medium texture complexity
    "composition_balance": 0.7        # Balanced composition
}

# Generate initial artistic chaos
chaos_canvas = art_generator.generate_artistic_chaos(
    noise_type="perlin",
    frequency_bands=5,
    color_randomness=0.9
)

# Evolve artistic composition
artwork = art_generator.evolve_artwork(
    initial_canvas=chaos_canvas,
    artistic_params=art_params,
    evolution_steps=300,
    save_evolution_frames=True
)

print(f"Artwork complexity: {artwork.complexity_score:.3f}")
print(f"Aesthetic score: {artwork.aesthetic_score:.3f}")
print(f"Color harmony: {artwork.color_harmony_score:.3f}")

# Save artwork and evolution
artwork.save_image("evolved_artwork.png", dpi=300)
save_artistic_evolution(
    evolution_frames=artwork.evolution_frames,
    output_path="art_evolution.mp4",
    fps=30
)
```

### 11. Music Composition

Compose music using thermodynamic principles:

```python
from entropic-ai.applications import MusicComposition
from entropic-ai.utils.audio import save_midi, play_audio

# Initialize music composer
composer = MusicComposition(
    musical_style="classical",
    time_signature=(4, 4),
    key_signature="C_major",
    tempo=120
)

# Define musical constraints
constraints = {
    "harmonic_progression": "functional_harmony",
    "melodic_range": (60, 84),  # MIDI note range (C4 to C6)
    "rhythmic_complexity": 0.6,
    "phrase_length": 8,         # 8-bar phrases
    "voice_leading": "smooth"
}

# Generate initial musical chaos
musical_noise = composer.generate_musical_chaos(
    duration=32,  # 32 bars
    voices=4,     # SATB arrangement
    randomness=0.8
)

# Evolve musical composition
composition = composer.evolve_composition(
    initial_material=musical_noise,
    constraints=constraints,
    evolution_steps=200,
    include_counterpoint=True
)

print(f"Composition analysis:")
print(f"Harmonic complexity: {composition.harmonic_complexity:.3f}")
print(f"Melodic coherence: {composition.melodic_coherence:.3f}")
print(f"Rhythmic interest: {composition.rhythmic_interest:.3f}")
print(f"Overall musicality: {composition.musicality_score:.3f}")

# Export composition
save_midi(composition, "evolved_composition.mid")
composition.generate_sheet_music("evolved_composition.pdf")
```

## Real-World Case Studies

### 12. Smart City Traffic Optimization

Optimize traffic flow in a smart city using thermodynamic principles:

```python
from entropic-ai.applications import SmartCityTraffic
from entropic-ai.utils.gis import load_city_network
import networkx as nx

# Load city street network
city_network = load_city_network("manhattan.osm")  # OpenStreetMap data
traffic_data = pd.read_csv("real_time_traffic.csv")

# Initialize traffic optimizer
traffic_optimizer = SmartCityTraffic(
    street_network=city_network,
    real_time_data=traffic_data,
    optimization_objectives={
        "minimize_travel_time": 0.4,
        "reduce_emissions": 0.3,
        "improve_safety": 0.2,
        "enhance_equity": 0.1
    }
)

# Define traffic control parameters
control_params = {
    "traffic_lights": {
        "adaptive_timing": True,
        "coordination_radius": 500  # meters
    },
    "route_guidance": {
        "dynamic_routing": True,
        "congestion_awareness": 0.8
    },
    "lane_management": {
        "reversible_lanes": True,
        "dynamic_pricing": True
    }
}

# Thermodynamic traffic evolution
optimized_traffic = traffic_optimizer.evolve_traffic_system(
    control_params=control_params,
    simulation_time=24,  # 24 hours
    evolution_steps=100
)

print("Traffic optimization results:")
print(f"Average travel time reduction: {optimized_traffic.travel_time_improvement:.1%}")
print(f"Emission reduction: {optimized_traffic.emission_reduction:.1%}")
print(f"Accident reduction: {optimized_traffic.safety_improvement:.1%}")
print(f"System efficiency: {optimized_traffic.efficiency_score:.3f}")

# Visualize traffic flow
optimized_traffic.visualize_traffic_flow(
    time_period="rush_hour",
    save_path="optimized_traffic_flow.html"
)
```

### 13. Drug Discovery Pipeline

Accelerate drug discovery using thermodynamic molecular evolution:

```python
from entropic-ai.applications import DrugDiscoveryPipeline
from entropic-ai.utils.cheminformatics import load_protein_target

# Load target protein
target_protein = load_protein_target("EGFR_kinase.pdb")
binding_site = target_protein.identify_binding_site("ATP_pocket")

# Initialize drug discovery pipeline
drug_pipeline = DrugDiscoveryPipeline(
    target_protein=target_protein,
    binding_site=binding_site,
    drug_requirements={
        "potency": ">= 10 nM",           # IC50 requirement
        "selectivity": ">= 100",         # vs other kinases
        "oral_bioavailability": ">= 30%",
        "half_life": ">= 4 hours",
        "toxicity_risk": "<= 0.1"
    }
)

# Multi-stage evolution process
discovery_results = drug_pipeline.run_discovery_pipeline(
    stages={
        "hit_identification": {
            "library_size": 10000,
            "evolution_steps": 100,
            "diversity_target": 0.8
        },
        "lead_optimization": {
            "population_size": 50,
            "optimization_rounds": 20,
            "admet_weight": 0.4
        },
        "candidate_selection": {
            "final_candidates": 5,
            "validation_assays": ["enzymatic", "cellular", "toxicity"]
        }
    }
)

print("Drug discovery results:")
print(f"Hit compounds identified: {len(discovery_results.hits)}")
print(f"Lead compounds: {len(discovery_results.leads)}")
print(f"Final candidates: {len(discovery_results.candidates)}")

# Analyze best candidate
best_candidate = discovery_results.candidates[0]
print(f"\nBest drug candidate:")
print(f"Structure: {best_candidate.smiles}")
print(f"Predicted IC50: {best_candidate.ic50:.2f} nM")
print(f"Selectivity: {best_candidate.selectivity:.1f}x")
print(f"Bioavailability: {best_candidate.bioavailability:.1%}")
print(f"Development probability: {best_candidate.development_probability:.1%}")
```

## Performance Benchmarks

### 14. Benchmark Comparison

Compare Entropic AI with traditional optimization methods:

```python
from entropic-ai.benchmarks import OptimizationBenchmark
from entropic-ai.utils.comparison import compare_methods
import time

# Define benchmark problems
problems = [
    "rastrigin_function",      # Multi-modal optimization
    "rosenbrock_function",     # Valley-shaped function  
    "ackley_function",         # Multi-modal with noise
    "schwefel_function",       # Deceptive global optimum
    "griewank_function"        # Rotated coordinates
]

# Initialize benchmark
benchmark = OptimizationBenchmark(
    problems=problems,
    dimensions=[10, 20, 50, 100],
    runs_per_problem=30
)

# Compare methods
methods = {
    "entropic_ai": {
        "class": GenerativeDiffuser,
        "params": {"diffusion_steps": 1000, "temperature": 1.0}
    },
    "genetic_algorithm": {
        "class": "GA",
        "params": {"population_size": 100, "generations": 1000}
    },
    "particle_swarm": {
        "class": "PSO", 
        "params": {"swarm_size": 50, "iterations": 1000}
    },
    "differential_evolution": {
        "class": "DE",
        "params": {"population_size": 100, "generations": 1000}
    }
}

# Run benchmarks
results = benchmark.run_comparison(methods)

print("Benchmark Results Summary:")
print("=" * 50)
for method, scores in results.items():
    print(f"{method}:")
    print(f"  Average best fitness: {scores['avg_fitness']:.6f}")
    print(f"  Success rate: {scores['success_rate']:.1%}")
    print(f"  Average time: {scores['avg_time']:.3f} seconds")
    print(f"  Convergence speed: {scores['convergence_speed']:.1f} generations")
    print()

# Statistical significance testing
significance_test = benchmark.statistical_analysis(results)
print("Statistical Significance (p-values):")
for comparison, p_value in significance_test.items():
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"{comparison}: p = {p_value:.6f} {significance}")
```

## Tips for Success

### General Guidelines

1. **Start Simple**: Begin with basic examples before attempting complex applications
2. **Monitor Evolution**: Always track the thermodynamic evolution process
3. **Validate Results**: Verify that evolved solutions meet physical constraints
4. **Tune Parameters**: Adjust temperature, cooling schedules, and complexity targets
5. **Use Appropriate Scales**: Ensure input data is properly normalized

### Common Patterns

```python
# Template for new applications
def create_custom_application():
    # 1. Define the problem domain
    domain_constraints = {...}
    
    # 2. Set up thermodynamic network
    network = EntropicNetwork(
        nodes=appropriate_size,
        temperature=domain_specific_temp
    )
    
    # 3. Configure complexity optimizer
    optimizer = ComplexityOptimizer(
        method="appropriate_method",
        target_complexity=domain_target
    )
    
    # 4. Initialize diffuser
    diffuser = GenerativeDiffuser(
        network=network,
        optimizer=optimizer,
        diffusion_steps=sufficient_steps
    )
    
    # 5. Run evolution with monitoring
    result = diffuser.evolve(
        initial_chaos,
        return_trajectory=True
    )
    
    # 6. Validate and analyze results
    validate_solution(result)
    
    return result
```

### Debugging Common Issues

```python
# Check thermodynamic consistency
def debug_thermodynamics(diffuser, state):
    # Energy conservation
    energy_before = diffuser.network.compute_total_energy()
    _ = diffuser.network(state)
    energy_after = diffuser.network.compute_total_energy()
    
    print(f"Energy change: {energy_after - energy_before:.6f}")
    
    # Entropy evolution
    entropy = diffuser.network.compute_total_entropy()
    print(f"Current entropy: {entropy:.6f}")
    
    # Free energy trend
    free_energy = diffuser.network.compute_free_energy()
    print(f"Free energy: {free_energy:.6f}")
    
    # Temperature profile
    temperature = diffuser.network.temperature
    print(f"Temperature: {temperature:.6f}")
```

These examples demonstrate the versatility and power of Entropic AI across diverse domains. The key is to understand how thermodynamic principles can be adapted to your specific problem domain while maintaining the fundamental chaos-to-order evolution paradigm.
