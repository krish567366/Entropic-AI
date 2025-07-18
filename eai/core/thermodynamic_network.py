"""
Thermodynamic Neural Network Implementation

Licensed module - requires valid E-AI license for access.

This module implements neural networks where each node behaves as a thermodynamic
system with internal energy, entropy, and temperature. The network evolves according
to thermodynamic principles rather than traditional gradient descent.

Key Concepts:
- Each node has thermodynamic properties (U, S, T)
- Free energy F = U - TS guides evolution
- Energy flows between nodes follow thermodynamic laws
- Emergent behavior arises from local thermodynamic optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from abc import ABC, abstractmethod

# License enforcement
from ..licensing.decorators import licensed_class, licensed_method
from ..licensing.enforcement import (
    BasicNetworkLicenseEnforcedBase,
    AdvancedNetworkLicenseEnforcedBase,
    LicenseEnforcedMetaclass
)

from ..utils.entropy_utils import shannon_entropy, thermodynamic_entropy


@licensed_class(['core', 'basic_networks'])
class ThermodynamicNode(BasicNetworkLicenseEnforcedBase, nn.Module, metaclass=LicenseEnforcedMetaclass):
    """
    A single thermodynamic node with energy, entropy, and temperature.
    
    LICENSE REQUIRED: Requires 'core' and 'basic_networks' features.
    
    Each node maintains:
    - Internal energy U
    - Entropy S
    - Temperature T
    - Free energy F = U - TS
    """
    
    _required_features = ['core', 'basic_networks']
    
    def __init__(
        self, 
        dim: int,
        initial_temperature: float = 1.0,
        energy_capacity: float = 10.0,
        entropy_regularization: float = 0.1
    ):
        # License validation happens in base class __new__
        super().__init__()
        self.dim = dim
        self.initial_temperature = initial_temperature
        self.energy_capacity = energy_capacity
        self.entropy_reg = entropy_regularization
        
        # Thermodynamic state variables
        self.register_buffer("temperature", torch.tensor(initial_temperature))
        self.register_parameter("internal_energy", nn.Parameter(torch.randn(dim)))
        self.register_parameter("entropy_weights", nn.Parameter(torch.randn(dim)))
        
        # Energy transformation parameters
        self.energy_transform = nn.Linear(dim, dim)
        self.entropy_transform = nn.Linear(dim, dim)
        
    def compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute entropy of the current state."""
        # Use weighted combination of Shannon and thermodynamic entropy
        shannon_h = shannon_entropy(F.softmax(x, dim=-1))
        thermo_h = thermodynamic_entropy(x, self.temperature)
        
        # Combine entropies with learned weights
        entropy_weights = F.softmax(self.entropy_weights, dim=-1)
        total_entropy = entropy_weights[0] * shannon_h + entropy_weights[1] * thermo_h
        
        return total_entropy
    
    def compute_free_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Helmholtz free energy F = U - TS."""
        # Internal energy (learned)
        U = torch.sum(self.internal_energy * x, dim=-1)
        
        # Entropy
        S = self.compute_entropy(x)
        
        # Free energy
        F = U - self.temperature * S
        
        return F
    
    def thermodynamic_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass following thermodynamic principles."""
        batch_size = x.shape[0]
        
        # Energy transformation
        energy_state = self.energy_transform(x)
        
        # Entropy transformation  
        entropy_state = self.entropy_transform(x)
        
        # Compute free energy for each state
        free_energy = self.compute_free_energy(energy_state)
        
        # System evolves to minimize free energy
        # Use negative free energy as activation (lower F = higher activation)
        activation = -free_energy.unsqueeze(-1) * entropy_state
        
        # Add thermal noise proportional to temperature
        if self.training:
            thermal_noise = torch.randn_like(activation) * torch.sqrt(self.temperature)
            activation = activation + thermal_noise
        
        return activation
    
    def update_temperature(self, step: int, cooling_schedule: str = "exponential"):
        """Update temperature according to annealing schedule."""
        if cooling_schedule == "exponential":
            self.temperature = self.initial_temperature * torch.exp(-step * 0.01)
        elif cooling_schedule == "linear":
            self.temperature = max(0.1, self.initial_temperature - step * 0.01)
        elif cooling_schedule == "inverse":
            self.temperature = self.initial_temperature / (1 + step * 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.thermodynamic_forward(x)


@licensed_class(['core', 'basic_networks'])
class ThermodynamicNetwork(BasicNetworkLicenseEnforcedBase, nn.Module, metaclass=LicenseEnforcedMetaclass):
    """
    A neural network where each layer consists of thermodynamic nodes.
    The network evolves according to thermodynamic principles.
    
    LICENSE REQUIRED: Requires 'core' and 'basic_networks' features.
    """
    
    _required_features = ['core', 'basic_networks']
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        temperature: float = 1.0,
        energy_capacity: float = 10.0,
        entropy_regularization: float = 0.1,
        cooling_schedule: str = "exponential"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.cooling_schedule = cooling_schedule
        self.step_count = 0
        
        # Build thermodynamic layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            layer = ThermodynamicNode(
                dim=dims[i+1],
                initial_temperature=temperature,
                energy_capacity=energy_capacity,
                entropy_regularization=entropy_regularization
            )
            self.layers.append(layer)
            
        # Connection weights between layers
        self.connections = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.connections.append(nn.Linear(dims[i], dims[i+1]))
    
    def compute_system_entropy(self, activations: List[torch.Tensor]) -> torch.Tensor:
        """Compute total system entropy across all layers."""
        total_entropy = 0.0
        for activation in activations:
            layer_entropy = shannon_entropy(F.softmax(activation, dim=-1))
            total_entropy += layer_entropy.mean()
        return total_entropy
    
    def compute_system_energy(self, activations: List[torch.Tensor]) -> torch.Tensor:
        """Compute total system energy."""
        total_energy = 0.0
        for i, activation in enumerate(activations):
            # Energy is sum of squared activations (kinetic-like energy)
            layer_energy = torch.sum(activation ** 2, dim=-1).mean()
            total_energy += layer_energy
        return total_energy
    
    def compute_system_free_energy(self, activations: List[torch.Tensor]) -> torch.Tensor:
        """Compute total system free energy F = U - TS."""
        U = self.compute_system_energy(activations)
        S = self.compute_system_entropy(activations)
        
        # Average temperature across layers
        avg_temp = torch.mean(torch.stack([layer.temperature for layer in self.layers]))
        
        F = U - avg_temp * S
        return F
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with thermodynamic evolution.
        
        Returns:
            output: Final network output
            thermodynamic_info: Dictionary with energy, entropy, free energy
        """
        activations = []
        current = x
        
        # Forward through each thermodynamic layer
        for i, (layer, connection) in enumerate(zip(self.layers, self.connections)):
            # Linear transformation
            current = connection(current)
            
            # Thermodynamic evolution
            current = layer(current)
            activations.append(current)
            
            # Update temperature (cooling)
            layer.update_temperature(self.step_count, self.cooling_schedule)
        
        # Increment step counter
        self.step_count += 1
        
        # Compute thermodynamic properties
        system_energy = self.compute_system_energy(activations)
        system_entropy = self.compute_system_entropy(activations)
        system_free_energy = self.compute_system_free_energy(activations)
        
        thermodynamic_info = {
            "energy": system_energy,
            "entropy": system_entropy, 
            "free_energy": system_free_energy,
            "temperature": torch.mean(torch.stack([layer.temperature for layer in self.layers])),
            "activations": activations
        }
        
        return current, thermodynamic_info


class EntropicNetwork(ThermodynamicNetwork):
    """
    Simplified alias for ThermodynamicNetwork optimized for entropic processes.
    """
    
    def __init__(
        self,
        nodes: int,
        temperature: float = 1.0,
        entropy_regularization: float = 0.1,
        **kwargs
    ):
        # Create a simple single-layer network for basic usage
        super().__init__(
            input_dim=nodes,
            hidden_dims=[],
            output_dim=nodes,
            temperature=temperature,
            entropy_regularization=entropy_regularization,
            **kwargs
        )
    
    def evolve(self, x: torch.Tensor, steps: int = 100) -> torch.Tensor:
        """
        Evolve the input through multiple thermodynamic steps.
        """
        current = x
        for step in range(steps):
            current, _ = self.forward(current)
        return current


class AdaptiveThermodynamicNetwork(ThermodynamicNetwork):
    """
    Advanced thermodynamic network with adaptive temperature and energy flow.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Adaptive mechanisms
        self.energy_flow_controller = nn.ModuleList([
            nn.Linear(dim, dim) for dim in [self.input_dim] + self.hidden_dims + [self.output_dim]
        ])
        
        # Temperature adaptation network
        self.temp_adapter = nn.Sequential(
            nn.Linear(len(self.layers), 64),
            nn.ReLU(),
            nn.Linear(64, len(self.layers)),
            nn.Sigmoid()
        )
    
    def adapt_temperatures(self, thermodynamic_state: Dict[str, torch.Tensor]):
        """Adaptively adjust temperatures based on current system state."""
        # Extract features for temperature adaptation
        features = torch.stack([
            thermodynamic_state["energy"],
            thermodynamic_state["entropy"], 
            thermodynamic_state["free_energy"]
        ])
        
        # Compute temperature adjustments
        temp_adjustments = self.temp_adapter(features)
        
        # Update layer temperatures
        for i, layer in enumerate(self.layers):
            adjustment = temp_adjustments[i] * 2.0  # Scale factor
            layer.temperature = layer.temperature * adjustment
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward with adaptive temperature control."""
        output, thermo_info = super().forward(x)
        
        # Adaptive temperature adjustment
        if self.training:
            self.adapt_temperatures(thermo_info)
        
        return output, thermo_info
