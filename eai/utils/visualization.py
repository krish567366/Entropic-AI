"""
Visualization utilities for Entropic AI

This module provides visualization functions for understanding and analyzing
the evolution processes, entropy dynamics, and emergent structures in Entropic AI.

Key visualizations:
- Entropy evolution over time
- Energy landscapes and attractors
- Complexity dynamics
- Phase space trajectories
- Thermodynamic state diagrams
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_entropy_evolution(
    trajectory: List[Dict[str, torch.Tensor]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot the evolution of entropy, energy, and complexity over time.
    
    Args:
        trajectory: Evolution trajectory from GenerativeDiffuser
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    """
    
    if not trajectory:
        print("No trajectory data to plot.")
        return
    
    # Extract data from trajectory
    steps = [item.get("step", i) for i, item in enumerate(trajectory)]
    entropies = [item.get("entropy", torch.tensor(0.0)).item() for item in trajectory]
    energies = [item.get("energy", torch.tensor(0.0)).item() for item in trajectory]
    free_energies = [item.get("free_energy", torch.tensor(0.0)).item() for item in trajectory]
    complexities = [item.get("complexity", torch.tensor(0.0)).item() for item in trajectory]
    temperatures = [item.get("temperature", 1.0) for item in trajectory]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Entropic AI Evolution Dynamics", fontsize=16, fontweight='bold')
    
    # Plot 1: Entropy vs Energy
    ax1.plot(steps, entropies, 'b-', linewidth=2, label='Entropy', alpha=0.8)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(steps, energies, 'r-', linewidth=2, label='Energy', alpha=0.8)
    
    ax1.set_xlabel('Evolution Step')
    ax1.set_ylabel('Entropy', color='b')
    ax1_twin.set_ylabel('Energy', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Entropy vs Energy Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Free Energy
    ax2.plot(steps, free_energies, 'g-', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Evolution Step')
    ax2.set_ylabel('Free Energy (F = U - TS)')
    ax2.set_title('Free Energy Minimization')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Complexity Evolution
    ax3.plot(steps, complexities, 'purple', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Evolution Step')
    ax3.set_ylabel('Complexity')
    ax3.set_title('Complexity Evolution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Temperature Schedule
    ax4.plot(steps, temperatures, 'orange', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Evolution Step')
    ax4.set_ylabel('Temperature')
    ax4.set_title('Cooling Schedule')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_energy_landscape(
    network: 'ThermodynamicNetwork',
    state_range: Tuple[float, float] = (-3, 3),
    resolution: int = 50,
    save_path: Optional[str] = None
) -> None:
    """
    Plot the energy landscape of the thermodynamic network.
    
    Args:
        network: ThermodynamicNetwork instance
        state_range: Range of state values to explore
        resolution: Grid resolution for landscape
        save_path: Optional path to save the figure
    """
    
    # Create 2D grid for visualization (using first 2 dimensions)
    x = np.linspace(state_range[0], state_range[1], resolution)
    y = np.linspace(state_range[0], state_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute energy landscape
    energies = np.zeros_like(X)
    
    network.eval()
    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                # Create state vector
                state = torch.zeros(1, network.input_dim)
                state[0, 0] = X[i, j]
                state[0, 1] = Y[i, j] if network.input_dim > 1 else 0
                
                # Forward pass
                _, thermo_info = network(state)
                energies[i, j] = thermo_info["energy"].item()
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, energies, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('State Dimension 1')
    ax.set_ylabel('State Dimension 2')
    ax.set_zlabel('Energy')
    ax.set_title('Thermodynamic Energy Landscape')
    
    plt.colorbar(surf, shrink=0.5, aspect=5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Energy landscape saved to {save_path}")
    
    plt.show()


def plot_phase_space_trajectory(
    trajectory: List[Dict[str, torch.Tensor]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot the phase space trajectory (entropy vs energy).
    
    Args:
        trajectory: Evolution trajectory
        save_path: Optional path to save the figure
    """
    
    if not trajectory:
        print("No trajectory data to plot.")
        return
    
    # Extract entropy and energy
    entropies = [item.get("entropy", torch.tensor(0.0)).item() for item in trajectory]
    energies = [item.get("energy", torch.tensor(0.0)).item() for item in trajectory]
    steps = range(len(trajectory))
    
    # Create phase space plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Phase space trajectory
    scatter = ax1.scatter(entropies, energies, c=steps, cmap='plasma', s=30, alpha=0.7)
    
    # Add arrows to show direction
    for i in range(0, len(entropies)-1, max(1, len(entropies)//10)):
        ax1.annotate('', xy=(entropies[i+1], energies[i+1]), 
                    xytext=(entropies[i], energies[i]),
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
    
    ax1.set_xlabel('Entropy')
    ax1.set_ylabel('Energy')
    ax1.set_title('Phase Space Trajectory (Entropy vs Energy)')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Evolution Step')
    
    # Plot 2: Free energy vs complexity
    if all('complexity' in item for item in trajectory):
        complexities = [item.get("complexity", torch.tensor(0.0)).item() for item in trajectory]
        free_energies = [item.get("free_energy", torch.tensor(0.0)).item() for item in trajectory]
        
        scatter2 = ax2.scatter(complexities, free_energies, c=steps, cmap='plasma', s=30, alpha=0.7)
        ax2.set_xlabel('Complexity')
        ax2.set_ylabel('Free Energy')
        ax2.set_title('Complexity vs Free Energy')
        ax2.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Evolution Step')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phase space plot saved to {save_path}")
    
    plt.show()


def plot_molecular_structure(
    atoms: List[str],
    positions: torch.Tensor,
    bonds: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """
    Plot 3D molecular structure.
    
    Args:
        atoms: List of atom symbols
        positions: 3D positions tensor [n_atoms, 3]
        bonds: Bond matrix [n_atoms, n_atoms]
        save_path: Optional path to save the figure
    """
    
    if positions.dim() > 2:
        positions = positions.squeeze()
    
    # Element colors
    element_colors = {
        'H': 'white', 'C': 'black', 'N': 'blue', 'O': 'red',
        'S': 'yellow', 'P': 'orange', 'F': 'green', 'Cl': 'green'
    }
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot atoms
    for i, atom in enumerate(atoms):
        if i < len(positions):
            color = element_colors.get(atom, 'gray')
            ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], 
                      c=color, s=100, alpha=0.8, edgecolors='black')
            ax.text(positions[i, 0], positions[i, 1], positions[i, 2], 
                   f'{atom}{i}', fontsize=8)
    
    # Plot bonds
    if bonds.dim() > 2:
        bonds = bonds.squeeze()
    
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            if i < len(positions) and j < len(positions) and bonds[i, j] > 0:
                ax.plot([positions[i, 0], positions[j, 0]],
                       [positions[i, 1], positions[j, 1]],
                       [positions[i, 2], positions[j, 2]], 'k-', alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Molecular Structure')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Molecular structure saved to {save_path}")
    
    plt.show()


def plot_circuit_evolution(
    circuit: Dict,
    trajectory: List[Dict[str, torch.Tensor]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot circuit evolution metrics.
    
    Args:
        circuit: Final evolved circuit
        trajectory: Evolution trajectory
        save_path: Optional path to save the figure
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Circuit Evolution Analysis", fontsize=16, fontweight='bold')
    
    if trajectory:
        steps = range(len(trajectory))
        
        # Plot 1: Energy evolution
        if all('energy' in item for item in trajectory):
            energies = [item.get("energy", torch.tensor(0.0)).item() for item in trajectory]
            ax1.plot(steps, energies, 'b-', linewidth=2)
            ax1.set_xlabel('Evolution Step')
            ax1.set_ylabel('Circuit Energy')
            ax1.set_title('Energy Evolution')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Complexity evolution
        if all('complexity' in item for item in trajectory):
            complexities = [item.get("complexity", torch.tensor(0.0)).item() for item in trajectory]
            ax2.plot(steps, complexities, 'r-', linewidth=2)
            ax2.set_xlabel('Evolution Step')
            ax2.set_ylabel('Circuit Complexity')
            ax2.set_title('Complexity Evolution')
            ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gate distribution
    if 'gates' in circuit:
        gate_types = circuit['gates']
        unique_gates, counts = np.unique(gate_types, return_counts=True)
        
        ax3.bar(unique_gates, counts)
        ax3.set_xlabel('Gate Type')
        ax3.set_ylabel('Count')
        ax3.set_title('Gate Distribution')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 4: Performance metrics
    if 'performance' in circuit:
        perf = circuit['performance']
        metrics = list(perf.keys())
        values = [perf[metric] for metric in metrics]
        
        bars = ax4.bar(metrics, values)
        ax4.set_ylabel('Score')
        ax4.set_title('Circuit Performance')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # Color bars based on performance
        for bar, value in zip(bars, values):
            if value > 0.8:
                bar.set_color('green')
            elif value > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Circuit analysis saved to {save_path}")
    
    plt.show()


def plot_theory_discovery(
    discovered_theory: Dict,
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """
    Plot theory discovery results.
    
    Args:
        discovered_theory: Result from TheoryDiscovery
        data_x: Original input data
        data_y: Original output data
        save_path: Optional path to save the figure
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Theory Discovery Analysis", fontsize=16, fontweight='bold')
    
    # Plot 1: Data fit
    ax1.scatter(data_x.numpy(), data_y.numpy(), alpha=0.6, label='Data', s=30)
    
    # Generate predictions if expression is valid
    if 'expression' in discovered_theory:
        x_range = torch.linspace(data_x.min(), data_x.max(), 100)
        try:
            # Simple evaluation for plotting
            expr_str = discovered_theory['expression']
            variables = discovered_theory.get('variables', ['x'])
            
            if variables and len(variables) > 0:
                var_name = variables[0]
                # Replace variable with numpy array for evaluation
                plot_expr = expr_str.replace(var_name, 'x_vals')
                
                predictions = []
                for x_val in x_range:
                    try:
                        # Very basic evaluation - replace with proper symbolic evaluation
                        x_vals = x_val.item()
                        pred = eval(plot_expr)  # Note: eval is unsafe, use with caution
                        predictions.append(pred)
                    except:
                        predictions.append(0.0)
                
                ax1.plot(x_range.numpy(), predictions, 'r-', linewidth=2, 
                        label=f'Theory: {discovered_theory.get("simplified_expression", "?")}')
        except:
            pass
    
    ax1.set_xlabel('Input')
    ax1.set_ylabel('Output')
    ax1.set_title('Theory Fit to Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Theory metrics
    if 'theory_fit' in discovered_theory:
        fit_metrics = discovered_theory['theory_fit']
        metrics = list(fit_metrics.keys())
        values = [fit_metrics[metric] for metric in metrics]
        
        bars = ax2.bar(metrics, values)
        ax2.set_ylabel('Score')
        ax2.set_title('Theory Quality Metrics')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Color bars based on performance
        for bar, value in zip(bars, values):
            if value > 0.8:
                bar.set_color('green')
            elif value > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
    
    # Plot 3: Evolution trajectory
    if 'evolution_trajectory' in discovered_theory:
        trajectory = discovered_theory['evolution_trajectory']
        if trajectory:
            steps = range(len(trajectory))
            complexities = [item.get("complexity", torch.tensor(0.0)).item() for item in trajectory]
            accuracies = [item.get("objective", torch.tensor(0.0)).item() for item in trajectory]
            
            ax3.plot(steps, complexities, 'b-', label='Complexity', linewidth=2)
            ax3_twin = ax3.twinx()
            ax3_twin.plot(steps, accuracies, 'r-', label='Objective', linewidth=2)
            
            ax3.set_xlabel('Evolution Step')
            ax3.set_ylabel('Complexity', color='b')
            ax3_twin.set_ylabel('Objective', color='r')
            ax3.set_title('Theory Evolution')
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Expression complexity
    if 'symbolic_complexity' in discovered_theory:
        complexity = discovered_theory['symbolic_complexity']
        ax4.bar(['Symbolic Complexity'], [complexity])
        ax4.set_ylabel('Complexity Score')
        ax4.set_title(f'Expression Complexity: {complexity:.2f}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Theory discovery analysis saved to {save_path}")
    
    plt.show()


def plot_thermodynamic_state_diagram(
    trajectory: List[Dict[str, torch.Tensor]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot thermodynamic state diagram showing the system's path through
    temperature-entropy space.
    
    Args:
        trajectory: Evolution trajectory
        save_path: Optional path to save the figure
    """
    
    if not trajectory:
        print("No trajectory data to plot.")
        return
    
    # Extract thermodynamic variables
    temperatures = [item.get("temperature", 1.0) for item in trajectory]
    entropies = [item.get("entropy", torch.tensor(0.0)).item() for item in trajectory]
    energies = [item.get("energy", torch.tensor(0.0)).item() for item in trajectory]
    free_energies = [item.get("free_energy", torch.tensor(0.0)).item() for item in trajectory]
    steps = range(len(trajectory))
    
    # Create thermodynamic state diagram
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Thermodynamic State Diagram", fontsize=16, fontweight='bold')
    
    # Plot 1: T-S diagram
    scatter1 = ax1.scatter(entropies, temperatures, c=steps, cmap='viridis', s=30, alpha=0.7)
    ax1.set_xlabel('Entropy')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Temperature-Entropy Diagram')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Evolution Step')
    
    # Plot 2: P-V like diagram (Energy vs Volume proxy)
    # Use complexity as volume proxy
    if all('complexity' in item for item in trajectory):
        complexities = [item.get("complexity", torch.tensor(0.0)).item() for item in trajectory]
        scatter2 = ax2.scatter(complexities, energies, c=steps, cmap='plasma', s=30, alpha=0.7)
        ax2.set_xlabel('Complexity (Volume proxy)')
        ax2.set_ylabel('Energy (Pressure proxy)')
        ax2.set_title('Energy-Complexity Diagram')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Evolution Step')
    
    # Plot 3: Gibbs free energy evolution
    ax3.plot(steps, free_energies, 'g-', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Evolution Step')
    ax3.set_ylabel('Free Energy')
    ax3.set_title('Gibbs Free Energy Evolution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Heat capacity proxy (temperature derivative)
    temp_diff = np.diff(temperatures)
    ax4.plot(steps[1:], temp_diff, 'orange', linewidth=2, alpha=0.8)
    ax4.set_xlabel('Evolution Step')
    ax4.set_ylabel('Temperature Change Rate')
    ax4.set_title('Cooling Rate (Heat Capacity proxy)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Thermodynamic state diagram saved to {save_path}")
    
    plt.show()
