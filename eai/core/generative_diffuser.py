"""
Generative Diffusion of Order - The Heart of Entropic AI

This module implements the core chaos-to-order transformation that defines
Entropic AI. Unlike traditional diffusion models that denoise, this system
crystallizes structure from pure entropy through thermodynamic evolution.

Key Concepts:
- Starts with maximum entropy (chaos)
- Each step reduces entropy while increasing meaningful structure
- Final states are thermodynamic attractors, not samples
- Evolution follows free energy minimization: ΔF = ΔU - TΔS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from abc import ABC, abstractmethod
import math

from .thermodynamic_network import ThermodynamicNetwork, ThermodynamicNode
from .complexity_optimizer import ComplexityOptimizer, KolmogorovOptimizer
from ..utils.entropy_utils import shannon_entropy, thermodynamic_entropy


class GenerativeDiffuser(nn.Module):
    """
    Core generative diffusion system that evolves chaos into ordered structures.
    
    This is NOT traditional denoising diffusion. Instead, it's a thermodynamic
    evolution process that crystallizes meaningful structure from pure entropy.
    """
    
    def __init__(
        self,
        network: ThermodynamicNetwork,
        optimizer: ComplexityOptimizer,
        diffusion_steps: int = 100,
        initial_temperature: float = 10.0,
        final_temperature: float = 0.1,
        cooling_schedule: str = "exponential",
        structure_emergence_threshold: float = 0.5
    ):
        super().__init__()
        
        self.network = network
        self.optimizer = optimizer
        self.diffusion_steps = diffusion_steps
        self.initial_temp = initial_temperature
        self.final_temp = final_temperature
        self.cooling_schedule = cooling_schedule
        self.emergence_threshold = structure_emergence_threshold
        
        # Evolution tracking
        self.step_count = 0
        self.evolution_history = []
        
        # Structure detector network
        self.structure_detector = nn.Sequential(
            nn.Linear(network.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def get_temperature(self, step: int) -> float:
        """Get temperature for current diffusion step."""
        progress = step / self.diffusion_steps
        
        if self.cooling_schedule == "exponential":
            temp = self.initial_temp * torch.exp(-progress * 3.0)
        elif self.cooling_schedule == "linear":
            temp = self.initial_temp * (1.0 - progress) + self.final_temp * progress
        elif self.cooling_schedule == "cosine":
            temp = self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * (
                1 + torch.cos(math.pi * progress)
            )
        elif self.cooling_schedule == "adaptive":
            # Adaptive cooling based on structure emergence
            if len(self.evolution_history) > 0:
                recent_structure = self.evolution_history[-1].get("structure_score", 0.0)
                if recent_structure > self.emergence_threshold:
                    # Slow cooling when structure emerges
                    temp = self.initial_temp * torch.exp(-progress * 1.5)
                else:
                    # Fast cooling to encourage crystallization
                    temp = self.initial_temp * torch.exp(-progress * 4.0)
            else:
                temp = self.initial_temp * torch.exp(-progress * 3.0)
        else:
            temp = self.initial_temp
        
        return max(temp, self.final_temp)
    
    def detect_structure_emergence(self, state: torch.Tensor) -> torch.Tensor:
        """Detect when meaningful structure begins to emerge."""
        # Use learned structure detector
        structure_score = self.structure_detector(state.mean(dim=0, keepdim=True))
        
        # Also use entropy-based detection
        entropy_score = 1.0 - shannon_entropy(F.softmax(state.flatten(), dim=0)) / torch.log(torch.tensor(float(state.numel())))
        
        # Combine scores
        combined_score = 0.7 * structure_score.squeeze() + 0.3 * entropy_score
        
        return combined_score
    
    def crystallization_step(
        self, 
        state: torch.Tensor, 
        step: int,
        temperature: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Single step of crystallization process.
        
        This is where chaos transforms into order through thermodynamic evolution.
        """
        batch_size = state.shape[0]
        
        # Forward through thermodynamic network
        evolved_state, thermo_info = self.network(state)
        
        # Update network temperatures to match cooling schedule
        for layer in self.network.layers:
            layer.temperature = torch.tensor(temperature)
        
        # Compute complexity metrics
        complexity_info = self.optimizer.optimization_step(self.network, evolved_state)
        
        # Detect structure emergence
        structure_score = self.detect_structure_emergence(evolved_state)
        
        # Free energy gradient descent
        # State evolves to minimize free energy while maximizing complexity
        free_energy = thermo_info["free_energy"]
        complexity = complexity_info["complexity"]
        
        # Multi-objective evolution
        # Minimize: free_energy - α * complexity (α balances exploration vs exploitation)
        alpha = min(step / self.diffusion_steps, 0.8)  # Increase complexity weight over time
        
        # Energy landscape navigation
        with torch.enable_grad():
            evolved_state.requires_grad_(True)
            
            # Recompute for gradient calculation
            gradient_state, gradient_thermo = self.network(evolved_state)
            gradient_complexity = self.optimizer.compute_complexity(gradient_state)
            
            # Combined objective for gradient descent
            objective = gradient_thermo["free_energy"] - alpha * gradient_complexity
            
            # Compute gradients
            gradients = torch.autograd.grad(
                objective.mean(), evolved_state, create_graph=True
            )[0]
            
            # Thermodynamic Langevin dynamics
            # dx = -∇F dt + √(2T) dW
            dt = 0.01  # Time step
            noise_scale = torch.sqrt(2 * temperature * dt)
            thermal_noise = torch.randn_like(evolved_state) * noise_scale
            
            # Evolution step
            next_state = evolved_state - dt * gradients + thermal_noise
        
        # Stability constraint - prevent runaway evolution
        state_change = torch.norm(next_state - state, dim=-1)
        max_change = 1.0  # Maximum allowed change per step
        
        if state_change.max() > max_change:
            # Scale down changes that are too large
            scale_factor = max_change / state_change.max()
            next_state = state + scale_factor * (next_state - state)
        
        # Update evolution history
        step_info = {
            "step": step,
            "temperature": temperature,
            "free_energy": thermo_info["free_energy"],
            "entropy": thermo_info["entropy"],
            "energy": thermo_info["energy"],
            "complexity": complexity_info["complexity"],
            "structure_score": structure_score,
            "state_change": state_change.mean(),
            "objective": objective.mean() if 'objective' in locals() else torch.tensor(0.0)
        }
        
        return next_state, step_info
    
    def evolve(
        self, 
        initial_chaos: torch.Tensor,
        target_structure: Optional[torch.Tensor] = None,
        guidance_strength: float = 0.1
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Main evolution function: Transform chaos into ordered structure.
        
        Args:
            initial_chaos: Random/chaotic initial state
            target_structure: Optional target to guide evolution toward
            guidance_strength: How strongly to guide toward target
            
        Returns:
            final_structure: Evolved ordered state
            evolution_trajectory: History of evolution process
        """
        current_state = initial_chaos.clone()
        trajectory = []
        
        self.evolution_history = []
        
        # Evolution loop
        for step in range(self.diffusion_steps):
            # Get current temperature
            temperature = self.get_temperature(step)
            
            # Crystallization step
            next_state, step_info = self.crystallization_step(
                current_state, step, temperature
            )
            
            # Optional guidance toward target structure
            if target_structure is not None:
                guidance_direction = target_structure - next_state
                guidance_force = guidance_strength * guidance_direction
                next_state = next_state + guidance_force
            
            # Store evolution information
            trajectory.append(step_info)
            self.evolution_history.append(step_info)
            
            # Update state
            current_state = next_state
            
            # Early stopping if structure has crystallized
            if step_info["structure_score"] > 0.9 and step > self.diffusion_steps // 4:
                print(f"Structure crystallized at step {step}")
                break
        
        self.step_count += len(trajectory)
        
        return current_state, trajectory
    
    def batch_evolve(
        self,
        chaos_batch: torch.Tensor,
        parallel: bool = True
    ) -> Tuple[torch.Tensor, List[List[Dict[str, torch.Tensor]]]]:
        """Evolve multiple chaotic states in parallel."""
        batch_size = chaos_batch.shape[0]
        
        if parallel:
            # Parallel evolution for entire batch
            return self.evolve(chaos_batch)
        else:
            # Sequential evolution for each sample
            final_structures = []
            all_trajectories = []
            
            for i in range(batch_size):
                structure, trajectory = self.evolve(chaos_batch[i:i+1])
                final_structures.append(structure)
                all_trajectories.append(trajectory)
            
            return torch.cat(final_structures, dim=0), all_trajectories
    
    def reverse_evolve(
        self,
        ordered_structure: torch.Tensor,
        chaos_steps: int = 50
    ) -> torch.Tensor:
        """
        Reverse evolution: Take ordered structure back to chaos.
        Useful for understanding the structure-formation process.
        """
        current_state = ordered_structure.clone()
        
        for step in range(chaos_steps):
            # Increase temperature (reverse cooling)
            temperature = self.final_temp + (self.initial_temp - self.final_temp) * (step / chaos_steps)
            
            # Add increasing thermal noise
            noise_scale = temperature * 0.1
            thermal_noise = torch.randn_like(current_state) * noise_scale
            
            # Reverse network evolution (approximate)
            with torch.no_grad():
                # Simple reversal - add noise and slight randomization
                current_state = current_state + thermal_noise
                
                # Random perturbations
                if torch.rand(1) < 0.3:  # 30% chance of random perturbation
                    perturbation = torch.randn_like(current_state) * 0.05
                    current_state = current_state + perturbation
        
        return current_state


class OrderEvolver(GenerativeDiffuser):
    """
    Simplified interface for order evolution optimized for common use cases.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: Optional[int] = None,
        evolution_steps: int = 50,
        **kwargs
    ):
        output_dim = output_dim or input_dim
        
        # Create default network and optimizer
        network = ThermodynamicNetwork(
            input_dim=input_dim,
            hidden_dims=[hidden_dim],
            output_dim=output_dim,
            temperature=1.0
        )
        
        optimizer = KolmogorovOptimizer(
            target_complexity=0.7,
            method="entropy"
        )
        
        super().__init__(
            network=network,
            optimizer=optimizer,
            diffusion_steps=evolution_steps,
            **kwargs
        )
    
    def create_chaos(self, batch_size: int = 1, chaos_type: str = "gaussian") -> torch.Tensor:
        """Generate initial chaotic state."""
        if chaos_type == "gaussian":
            return torch.randn(batch_size, self.network.input_dim)
        elif chaos_type == "uniform":
            return torch.rand(batch_size, self.network.input_dim) * 2 - 1
        elif chaos_type == "heavy_tail":
            # Heavy-tailed distribution for more extreme chaos
            return torch.distributions.StudentT(df=1.0).sample((batch_size, self.network.input_dim))
        else:
            return torch.randn(batch_size, self.network.input_dim)
    
    def evolve_to_order(
        self,
        batch_size: int = 1,
        chaos_type: str = "gaussian",
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """Simple interface to evolve chaos into order."""
        chaos = self.create_chaos(batch_size, chaos_type)
        structure, trajectory = self.evolve(chaos)
        
        if return_trajectory:
            return structure, trajectory
        else:
            return structure


class AdaptiveOrderEvolver(OrderEvolver):
    """
    Order evolver with adaptive parameters based on evolution success.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.success_history = []
        self.adaptation_rate = 0.1
    
    def evaluate_evolution_success(self, trajectory: List[Dict[str, torch.Tensor]]) -> float:
        """Evaluate how successful the evolution was."""
        if not trajectory:
            return 0.0
        
        final_step = trajectory[-1]
        
        # Success metrics
        structure_score = final_step["structure_score"].item()
        complexity = final_step["complexity"].item()
        stability = 1.0 / (1.0 + final_step["state_change"].item())
        
        # Combined success score
        success = 0.4 * structure_score + 0.4 * complexity + 0.2 * stability
        
        return success
    
    def adapt_parameters(self):
        """Adapt evolution parameters based on recent success."""
        if len(self.success_history) < 5:
            return
        
        recent_success = np.mean(self.success_history[-5:])
        
        if recent_success < 0.5:
            # Poor success - increase exploration
            self.diffusion_steps = min(self.diffusion_steps + 10, 200)
            self.initial_temp *= 1.1
        elif recent_success > 0.8:
            # High success - can be more efficient
            self.diffusion_steps = max(self.diffusion_steps - 5, 20)
            self.initial_temp *= 0.95
    
    def evolve_to_order(
        self,
        batch_size: int = 1,
        chaos_type: str = "gaussian",
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List]]:
        """Adaptive evolution with parameter adjustment."""
        
        structure, trajectory = super().evolve_to_order(
            batch_size, chaos_type, return_trajectory=True
        )
        
        # Evaluate and adapt
        success = self.evaluate_evolution_success(trajectory)
        self.success_history.append(success)
        
        if len(self.success_history) > 20:
            self.success_history.pop(0)
        
        # Adapt parameters periodically
        if len(self.success_history) % 5 == 0:
            self.adapt_parameters()
        
        if return_trajectory:
            return structure, trajectory
        else:
            return structure
