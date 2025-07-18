# Molecule Design Tutorial

In this tutorial, you'll learn how to use Entropic AI to design novel molecular structures through thermodynamic evolution. We'll start with atomic chaos and evolve stable, functional molecules using the principles of thermodynamics and complexity theory.

## Overview

Molecular design with Entropic AI follows nature's approach to molecular evolution:

1. **Start with atomic chaos** — Random collection of atoms
2. **Apply thermodynamic pressure** — Minimize free energy
3. **Evolve through intermediate states** — Metastable configurations
4. **Crystallize into stable molecules** — Thermodynamically favorable structures

## Prerequisites

```bash
pip install eai[molecules]  # Includes molecular modeling dependencies
```

Optional for advanced visualization:

```bash
pip install py3Dmol rdkit-pypi  # 3D molecular visualization
```

## Basic Molecule Evolution

Let's start with a simple example: evolving a stable organic molecule from carbon, nitrogen, oxygen, and hydrogen atoms.

### Step 1: Initialize the Molecular Evolver

```python
from eai.applications import MoleculeEvolution
from eai.utils.visualization import plot_molecular_evolution
import numpy as np

# Create molecular evolution system
evolver = MoleculeEvolution(
    target_properties={
        "stability": 0.85,      # Thermodynamic stability (0-1)
        "complexity": 0.6,      # Structural complexity (0-1)  
        "functionality": 0.7,   # Functional group diversity (0-1)
        "druglike": 0.8        # Drug-like properties (0-1)
    },
    atomic_constraints={
        "max_atoms": 30,                               # Maximum molecule size
        "allowed_elements": ["C", "N", "O", "H", "S"], # Available elements
        "charge_constraints": {"min": -2, "max": 2},   # Formal charge range
        "valence_constraints": True                    # Respect valence rules
    },
    thermodynamic_params={
        "temperature": 300.0,    # Room temperature (K)
        "pressure": 1.0,         # Standard pressure (atm)
        "ph": 7.4               # Physiological pH
    }
)

print("Molecular evolver initialized!")
print(f"Target stability: {evolver.target_properties['stability']}")
print(f"Max atoms: {evolver.atomic_constraints['max_atoms']}")
```

### Step 2: Generate Initial Atomic Chaos

```python
# Create random initial state - pure atomic chaos
initial_atoms = evolver.generate_atomic_chaos(
    n_atoms=20,
    element_probabilities={
        "C": 0.5,   # 50% carbon atoms
        "N": 0.2,   # 20% nitrogen atoms  
        "O": 0.2,   # 20% oxygen atoms
        "H": 0.1    # 10% hydrogen atoms (will be added automatically)
    },
    spatial_distribution="thermal_gas"  # Random thermal positions
)

print(f"Generated {len(initial_atoms)} atoms in chaotic state")
print(f"Initial energy: {initial_atoms.total_energy:.2f} kcal/mol")
print(f"Initial entropy: {initial_atoms.configurational_entropy:.3f}")
```

### Step 3: Evolve the Molecular Structure

```python
# Run thermodynamic evolution
molecule = evolver.evolve_from_atoms(
    initial_atoms=initial_atoms,
    evolution_steps=500,
    cooling_schedule="exponential",
    crystallization_threshold=0.05,
    monitor_evolution=True
)

print(f"\nEvolution complete!")
print(f"Final molecule: {molecule.formula}")
print(f"Stability score: {molecule.stability:.3f}")
print(f"Complexity score: {molecule.complexity:.3f}")
print(f"Final energy: {molecule.energy:.2f} kcal/mol")
```

Expected output:

```plaintext
Evolution complete!
Final molecule: C12H15N3O2
Stability score: 0.847
Complexity score: 0.623
Final energy: -234.56 kcal/mol
```

### Step 4: Analyze the Evolved Molecule

```python
# Get detailed molecular properties
properties = molecule.analyze_properties()

print("\nMolecular Analysis:")
print(f"Molecular weight: {properties['molecular_weight']:.1f} g/mol")
print(f"LogP (lipophilicity): {properties['logP']:.2f}")
print(f"H-bond donors: {properties['hbd']}")
print(f"H-bond acceptors: {properties['hba']}")
print(f"Rotatable bonds: {properties['rotatable_bonds']}")
print(f"TPSA: {properties['tpsa']:.1f} Ų")

# Check drug-likeness (Lipinski's Rule of Five)
druglike = molecule.check_druglikeness()
print(f"\nDrug-like properties:")
for rule, passes in druglike.items():
    status = "✓" if passes else "✗"
    print(f"{status} {rule}")
```

### Step 5: Visualize the Evolution Process

```python
# Plot the thermodynamic evolution
plot_molecular_evolution(
    molecule.evolution_history,
    properties_to_plot=["energy", "stability", "complexity"],
    save_path="molecule_evolution.png"
)

# 3D molecular structure (if py3Dmol is installed)
if molecule.has_3d_coordinates:
    molecule.show_3d_structure(style="ball_and_stick")
```

## Advanced Drug Design

Now let's design a molecule with specific drug-like properties, such as a potential kinase inhibitor.

### Target-Specific Evolution

```python
# Design a kinase inhibitor with specific properties
drug_evolver = MoleculeEvolution(
    target_properties={
        "stability": 0.9,
        "complexity": 0.8,
        "druglike": 0.95,
        "kinase_affinity": 0.8,     # Target-specific property
        "selectivity": 0.7,         # Selectivity over other proteins
        "bioavailability": 0.8      # Oral bioavailability
    },
    atomic_constraints={
        "max_atoms": 50,
        "allowed_elements": ["C", "N", "O", "H", "S", "F", "Cl"],
        "required_motifs": [
            "aromatic_ring",        # Essential for kinase binding
            "hydrogen_bond_donor",  # Hinge region interaction
            "hydrogen_bond_acceptor"
        ]
    },
    target_protein={
        "pdb_id": "1ATP",          # Kinase structure
        "binding_site": "ATP_pocket",
        "key_residues": ["ASP166", "GLU129", "LYS72"]
    }
)

# Start with pharmacophore-guided chaos
initial_state = drug_evolver.generate_pharmacophore_chaos(
    n_atoms=35,
    pharmacophore_model="kinase_inhibitor",
    diversity_factor=0.7
)

# Evolve with protein-ligand interactions
drug_molecule = drug_evolver.evolve_from_atoms(
    initial_atoms=initial_state,
    evolution_steps=800,
    include_protein_interactions=True,
    binding_affinity_weight=0.4,
    druglike_weight=0.3,
    stability_weight=0.3
)

print(f"Drug candidate: {drug_molecule.formula}")
print(f"Predicted binding affinity: {drug_molecule.binding_affinity:.2f} nM")
print(f"Drug-likeness score: {drug_molecule.druglike_score:.3f}")
```

### Multi-Objective Optimization

```python
# Optimize multiple properties simultaneously
from eai.core import ComplexityOptimizer

# Define multiple objectives
objectives = {
    "potency": {"target": 0.9, "weight": 0.3},      # High binding affinity
    "selectivity": {"target": 0.8, "weight": 0.25}, # Selective binding
    "admet": {"target": 0.85, "weight": 0.25},      # Good ADMET properties
    "novelty": {"target": 0.7, "weight": 0.2}       # Structural novelty
}

# Create multi-objective optimizer
multi_optimizer = ComplexityOptimizer(
    method="multi_objective",
    objectives=objectives,
    pareto_optimization=True,
    constraint_handling="penalty"
)

# Run Pareto-optimal evolution
drug_evolver.set_optimizer(multi_optimizer)
pareto_molecules = drug_evolver.evolve_pareto_set(
    initial_atoms=initial_state,
    population_size=50,
    generations=200
)

print(f"Generated {len(pareto_molecules)} Pareto-optimal drug candidates")
for i, mol in enumerate(pareto_molecules[:5]):
    print(f"Candidate {i+1}: {mol.formula} (score: {mol.total_score:.3f})")
```

## Molecular Property Prediction

### Thermodynamic Properties

```python
# Calculate thermodynamic properties
thermo_props = molecule.calculate_thermodynamics(
    temperature=298.15,  # Room temperature
    pressure=1.0,        # Standard pressure
    solvent="water"      # Aqueous solution
)

print("Thermodynamic Properties:")
print(f"Enthalpy of formation: {thermo_props['delta_hf']:.2f} kcal/mol")
print(f"Entropy: {thermo_props['entropy']:.2f} cal/(mol·K)")
print(f"Free energy: {thermo_props['gibbs_free_energy']:.2f} kcal/mol")
print(f"Heat capacity: {thermo_props['heat_capacity']:.2f} cal/(mol·K)")
```

### ADMET Predictions

```python
# Predict ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity)
admet = molecule.predict_admet()

print("ADMET Predictions:")
print(f"Absorption: {admet['absorption']:.1%}")
print(f"BBB permeability: {admet['bbb_permeability']:.1%}")
print(f"CYP3A4 inhibition: {admet['cyp3a4_inhibition']:.1%}")
print(f"hERG toxicity risk: {admet['herg_risk']:.1%}")
print(f"Plasma protein binding: {admet['ppb']:.1%}")
```

## Custom Molecular Constraints

### Defining Custom Objectives

```python
def custom_fluorescence_objective(molecule):
    """Custom objective for fluorescent molecules."""
    # Check for conjugated pi system
    conjugation_score = molecule.calculate_conjugation_length() / 20.0
    
    # Favor certain functional groups
    fluorophore_groups = molecule.count_groups(["benzene", "pyridine", "quinoline"])
    group_score = min(fluorophore_groups / 3.0, 1.0)
    
    # Penalize quenching groups
    quenchers = molecule.count_groups(["nitro", "carbonyl"])
    quench_penalty = max(0, 1.0 - quenchers * 0.3)
    
    return conjugation_score * group_score * quench_penalty

# Add custom objective to evolver
evolver.add_custom_objective("fluorescence", custom_fluorescence_objective, weight=0.3)
```

### Synthetic Accessibility

```python
# Consider synthetic accessibility during evolution
evolver.add_synthetic_constraints(
    max_synthetic_steps=8,          # Maximum synthesis steps
    commercial_availability=True,   # Prefer commercially available starting materials
    reaction_feasibility=0.8,       # Minimum reaction feasibility score
    cost_constraint=1000.0          # Maximum synthesis cost ($/g)
)

# Evolve with synthetic accessibility
synthetic_molecule = evolver.evolve_from_atoms(
    initial_atoms=initial_state,
    evolution_steps=600,
    synthetic_accessibility_weight=0.2
)

print(f"Synthetic accessibility score: {synthetic_molecule.sa_score:.3f}")
print(f"Estimated synthesis steps: {synthetic_molecule.synthesis_steps}")
print(f"Estimated cost: ${synthetic_molecule.synthesis_cost:.2f}/g")
```

## Batch Molecular Evolution

### Generating Molecular Libraries

```python
# Generate a library of related molecules
library = evolver.evolve_molecular_library(
    scaffold="c1ccc(cc1)N",         # Aniline scaffold
    n_molecules=100,                # Library size
    diversity_threshold=0.6,        # Minimum structural diversity
    property_constraints={
        "molecular_weight": (150, 500),
        "logP": (-2, 5),
        "tpsa": (20, 140)
    }
)

print(f"Generated library of {len(library)} molecules")

# Analyze library diversity
diversity_matrix = library.calculate_diversity_matrix()
print(f"Average Tanimoto distance: {diversity_matrix.mean():.3f}")
```

### Iterative Optimization

```python
# Iterative improvement of lead compounds
current_best = drug_molecule
optimization_history = []

for iteration in range(10):
    # Generate variations around current best
    variants = evolver.generate_analogs(
        parent_molecule=current_best,
        n_analogs=20,
        modification_types=["substituent", "ring_replacement", "linker_modification"]
    )
    
    # Evaluate all variants
    for variant in variants:
        variant.evaluate_properties()
    
    # Select best variant
    new_best = max(variants, key=lambda m: m.total_score)
    
    if new_best.total_score > current_best.total_score:
        current_best = new_best
        print(f"Iteration {iteration+1}: Improved score to {current_best.total_score:.3f}")
        
    optimization_history.append(current_best.total_score)

print(f"Final optimized molecule: {current_best.formula}")
print(f"Final score: {current_best.total_score:.3f}")
```

## Experimental Validation

### Virtual Screening Integration

```python
# Integrate with virtual screening workflows
from eai.interfaces import ChemblInterface, PubchemInterface

# Screen against ChEMBL bioactivity data
chembl = ChemblInterface()
bioactivity_predictions = chembl.predict_bioactivity(
    molecule=drug_molecule,
    target_types=["kinase", "gpcr", "ion_channel"]
)

print("Bioactivity Predictions:")
for target, activity in bioactivity_predictions.items():
    print(f"{target}: {activity['probability']:.3f} (IC50: {activity['predicted_ic50']:.1f} nM)")
```

### Molecular Dynamics Validation

```python
# Validate stability with molecular dynamics
md_result = molecule.run_molecular_dynamics(
    simulation_time=100,  # nanoseconds
    temperature=310,      # body temperature
    solvent="water_tip3p",
    force_field="amber99sb"
)

print("MD Simulation Results:")
print(f"RMSD: {md_result['rmsd']:.2f} Å")
print(f"Radius of gyration: {md_result['rg']:.2f} Å")
print(f"Conformational stability: {md_result['stability']:.3f}")
```

## Visualization and Analysis

### Energy Landscape Visualization

```python
from eai.utils.visualization import plot_molecular_energy_landscape

# Plot the molecular energy landscape
plot_molecular_energy_landscape(
    molecule=drug_molecule,
    conformer_range=(-180, 180),
    resolution=10,
    energy_colormap="viridis"
)
```

### Structure-Activity Relationships

```python
# Analyze structure-activity relationships
sar_analysis = library.analyze_sar(
    activity_property="kinase_affinity",
    structural_descriptors=["fingerprint", "pharmacophore", "3d_shape"]
)

# Plot SAR trends
sar_analysis.plot_activity_cliffs()
sar_analysis.plot_matched_pairs()
```

## Best Practices

### 1. Start Simple

Begin with small molecules and basic constraints before attempting complex drug design:

```python
# Good starting point
simple_evolver = MoleculeEvolution(
    target_properties={"stability": 0.8},
    atomic_constraints={"max_atoms": 15, "allowed_elements": ["C", "N", "O", "H"]}
)
```

### 2. Balance Objectives

Don't optimize too many properties simultaneously:

```python
# Recommended: 2-4 main objectives
balanced_objectives = {
    "potency": 0.4,      # Primary objective
    "druglike": 0.3,     # Secondary objective  
    "stability": 0.2,    # Tertiary objective
    "novelty": 0.1       # Exploration bonus
}
```

### 3. Monitor Evolution

Always track the evolution process:

```python
# Enable comprehensive monitoring
evolver.set_monitoring(
    track_energy=True,
    track_properties=True,
    save_intermediates=True,
    plot_realtime=True
)
```

### 4. Validate Results

Always validate evolved molecules:

```python
# Multi-level validation
molecule.validate_chemistry()    # Chemical validity
molecule.validate_druglike()     # Drug-likeness
molecule.validate_synthesis()    # Synthetic feasibility
```

## Troubleshooting

### Common Issues

**Evolution gets stuck in local minima:**

- Increase temperature: `temperature=400.0`
- Add noise: `thermal_noise=0.05`
- Use simulated annealing: `cooling_schedule="slow_exponential"`

**Unrealistic molecules generated:**

- Strengthen chemical constraints: `enforce_valence=True`
- Add stability filter: `min_stability=0.7`
- Include synthetic accessibility: `synthetic_accessibility_weight=0.3`

**Poor drug-likeness:**

- Include Lipinski filters: `enforce_rule_of_five=True`
- Add ADMET constraints: `admet_weight=0.4`
- Use drug-like starting materials: `drug_like_seeds=True`

## Next Steps

- **[Circuit Evolution Tutorial](circuit-evolution.md)**: Design logic circuits
- **[Theory Discovery Tutorial](theory-discovery.md)**: Find mathematical laws
- **[Advanced Configuration](../guides/advanced-config.md)**: Fine-tune evolution parameters
- **[Custom Applications](../guides/custom-applications.md)**: Build domain-specific evolvers

## Resources

- **Molecular Descriptors**: [RDKit Documentation](https://www.rdkit.org/docs/)
- **Drug Design**: [Medicinal Chemistry Guidelines](https://pubs.acs.org/jmc)
- **ADMET Prediction**: [SwissADME](http://www.swissadme.ch/)
- **Synthetic Accessibility**: [SAScore Paper](https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8)

Happy molecular evolution! 🧬⚗️
