"""
Complexity-Maximizing Optimizers for Entropic AI

This module implements optimizers that drive systems toward states of maximum
emergent complexity while maintaining stability. Unlike traditional loss 
minimization, these optimizers seek to maximize meaningful complexity measures.

Key Features:
- Kolmogorov complexity optimization
- Multi-objective complexity-stability tradeoffs
- Entropy-guided exploration
- Emergence detection and amplification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from abc import ABC, abstractmethod

from ..utils.entropy_utils import (
    shannon_entropy, 
    kolmogorov_complexity,
    complexity_entropy_tradeoff,
    fisher_information
)


class ComplexityOptimizer(ABC):
    """
    Abstract base class for complexity-maximizing optimizers.
    
    These optimizers seek to evolve systems toward states of high emergent
    complexity while maintaining thermodynamic stability.
    """
    
    def __init__(
        self,
        target_complexity: float = 0.7,
        stability_weight: float = 0.3,
        exploration_rate: float = 0.1
    ):
        self.target_complexity = target_complexity
        self.stability_weight = stability_weight  
        self.exploration_rate = exploration_rate
        self.step_count = 0
        
    @abstractmethod
    def compute_complexity(self, state: torch.Tensor) -> torch.Tensor:
        """Compute complexity measure for a given state."""
        pass
    
    @abstractmethod  
    def compute_stability(self, state: torch.Tensor, prev_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute stability measure for a state."""
        pass
    
    @abstractmethod
    def optimization_step(self, model: nn.Module, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform one optimization step toward higher complexity."""
        pass


class KolmogorovOptimizer(ComplexityOptimizer):
    """
    Optimizer that maximizes Kolmogorov complexity (approximated).
    
    Drives systems toward states that require long descriptions but are
    still stable and meaningful.
    """
    
    def __init__(
        self,
        method: str = "entropy",
        normalize: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.method = method
        self.normalize = normalize
        self.state_history = []
        
    def compute_complexity(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Kolmogorov complexity approximation."""
        return kolmogorov_complexity(state, self.method, self.normalize)
    
    def compute_stability(self, state: torch.Tensor, prev_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Measure stability as consistency with previous states."""
        if prev_state is None or len(self.state_history) == 0:
            # For first state, use variance as stability measure
            return 1.0 / (1.0 + torch.var(state))
        
        # Stability as negative change magnitude
        change = torch.norm(state - prev_state)
        stability = torch.exp(-change)
        
        return stability
    
    def compute_emergence(self, state: torch.Tensor) -> torch.Tensor:
        """Detect emergent properties in the state."""
        if len(self.state_history) < 3:
            return torch.tensor(0.0)
        
        # Look for non-linear patterns in state evolution
        recent_states = self.state_history[-3:]
        
        # Compute second derivatives (acceleration in state space)
        if len(recent_states) >= 3:
            accel = recent_states[-1] - 2*recent_states[-2] + recent_states[-3]
            emergence_score = torch.norm(accel)
        else:
            emergence_score = torch.tensor(0.0)
            
        return emergence_score
    
    def optimization_step(self, model: nn.Module, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Optimize toward maximum Kolmogorov complexity."""
        
        # Store state history
        self.state_history.append(state.clone().detach())
        if len(self.state_history) > 10:  # Keep limited history
            self.state_history.pop(0)
        
        prev_state = self.state_history[-2] if len(self.state_history) > 1 else None
        
        # Compute complexity and stability
        complexity = self.compute_complexity(state)
        stability = self.compute_stability(state, prev_state)
        emergence = self.compute_emergence(state)
        
        # Multi-objective score
        complexity_target_diff = torch.abs(complexity - self.target_complexity)
        complexity_score = torch.exp(-complexity_target_diff)  # Closer to target = higher score
        
        # Combined objective (maximize complexity while maintaining stability)
        objective = (
            complexity_score + 
            self.stability_weight * stability +
            0.1 * emergence  # Small emergence bonus
        )
        
        # Exploration bonus (encourage diversity)
        if torch.rand(1) < self.exploration_rate:
            exploration_bonus = torch.randn_like(objective) * 0.1
            objective = objective + exploration_bonus
        
        self.step_count += 1
        
        return {
            "complexity": complexity,
            "stability": stability,
            "emergence": emergence,
            "objective": objective,
            "complexity_score": complexity_score
        }


class EntropyGuidedOptimizer(ComplexityOptimizer):
    """
    Optimizer that uses entropy gradients to explore complexity landscapes.
    """
    
    def __init__(
        self,
        entropy_type: str = "shannon",
        temperature_schedule: str = "adaptive",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.entropy_type = entropy_type
        self.temperature_schedule = temperature_schedule
        self.temperature = 1.0
        
    def compute_complexity(self, state: torch.Tensor) -> torch.Tensor:
        """Use entropy as complexity measure."""
        if self.entropy_type == "shannon":
            probs = torch.softmax(state.flatten(), dim=0)
            return shannon_entropy(probs)
        elif self.entropy_type == "differential":
            # Differential entropy approximation
            return 0.5 * torch.log(2 * np.pi * np.e * torch.var(state))
        else:
            return complexity_entropy_tradeoff(state)
    
    def compute_stability(self, state: torch.Tensor, prev_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Stability based on entropy consistency."""
        if prev_state is None:
            return torch.tensor(1.0)
        
        curr_entropy = self.compute_complexity(state)
        prev_entropy = self.compute_complexity(prev_state)
        
        # Stable if entropy doesn't change too rapidly
        entropy_change = torch.abs(curr_entropy - prev_entropy)
        stability = torch.exp(-entropy_change / self.temperature)
        
        return stability
    
    def update_temperature(self):
        """Update temperature according to schedule."""
        if self.temperature_schedule == "adaptive":
            # Decrease temperature as complexity approaches target
            # (simulated annealing)
            self.temperature *= 0.995
            self.temperature = max(0.1, self.temperature)
        elif self.temperature_schedule == "cyclic":
            # Cyclical temperature for exploration
            cycle_length = 100
            phase = (self.step_count % cycle_length) / cycle_length
            self.temperature = 0.5 + 0.5 * np.sin(2 * np.pi * phase)
    
    def optimization_step(self, model: nn.Module, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Entropy-guided optimization step."""
        
        # Compute current metrics
        complexity = self.compute_complexity(state)
        
        # Estimate entropy gradient direction
        state.requires_grad_(True)
        entropy_loss = -self.compute_complexity(state)  # Negative for maximization
        entropy_grad = torch.autograd.grad(entropy_loss, state, create_graph=True)[0]
        
        # Update temperature
        self.update_temperature()
        
        # Stability through gradient magnitude control
        grad_norm = torch.norm(entropy_grad)
        stability = torch.exp(-grad_norm / self.temperature)
        
        # Objective combines entropy maximization with stability
        objective = complexity + self.stability_weight * stability
        
        self.step_count += 1
        
        return {
            "complexity": complexity,
            "stability": stability,
            "objective": objective,
            "entropy_gradient": entropy_grad,
            "temperature": torch.tensor(self.temperature)
        }


class MultiObjectiveComplexityOptimizer(ComplexityOptimizer):
    """
    Optimizer that balances multiple complexity measures simultaneously.
    """
    
    def __init__(
        self,
        complexity_measures: List[str] = ["kolmogorov", "shannon", "fisher"],
        weights: Optional[List[float]] = None,
        pareto_frontier: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.complexity_measures = complexity_measures
        self.weights = weights or [1.0] * len(complexity_measures)
        self.pareto_frontier = pareto_frontier
        self.pareto_archive = []
        
    def compute_complexity(self, state: torch.Tensor) -> torch.Tensor:
        """Compute weighted combination of complexity measures."""
        complexities = []
        
        for measure in self.complexity_measures:
            if measure == "kolmogorov":
                comp = kolmogorov_complexity(state, normalize=True)
            elif measure == "shannon":
                probs = torch.softmax(state.flatten(), dim=0)
                comp = shannon_entropy(probs)
            elif measure == "fisher":
                # Approximate Fisher information
                state_grad = torch.autograd.grad(
                    state.sum(), state, create_graph=True, allow_unused=True
                )[0]
                if state_grad is not None:
                    comp = torch.norm(state_grad)
                else:
                    comp = torch.tensor(0.0)
            else:
                comp = torch.tensor(0.0)
            
            complexities.append(comp)
        
        # Weighted combination
        total_complexity = sum(w * c for w, c in zip(self.weights, complexities))
        
        return total_complexity
    
    def compute_stability(self, state: torch.Tensor, prev_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Multi-measure stability."""
        if prev_state is None:
            return torch.tensor(1.0)
        
        # Stability across all complexity measures
        stabilities = []
        
        curr_comp = self.compute_complexity(state)
        prev_comp = self.compute_complexity(prev_state)
        
        change = torch.abs(curr_comp - prev_comp)
        stability = torch.exp(-change)
        
        return stability
    
    def update_pareto_archive(self, complexities: List[torch.Tensor], stability: torch.Tensor):
        """Maintain Pareto frontier of complexity-stability solutions."""
        if not self.pareto_frontier:
            return
        
        current_solution = complexities + [stability]
        
        # Check if current solution dominates any in archive
        dominated_indices = []
        is_dominated = False
        
        for i, archived_solution in enumerate(self.pareto_archive):
            # Check dominance
            dominates = all(c >= a for c, a in zip(current_solution, archived_solution))
            is_dominated_by = all(a >= c for c, a in zip(current_solution, archived_solution))
            
            if dominates:
                dominated_indices.append(i)
            elif is_dominated_by:
                is_dominated = True
                break
        
        # Add to archive if not dominated
        if not is_dominated:
            # Remove dominated solutions
            for i in reversed(dominated_indices):
                self.pareto_archive.pop(i)
            
            # Add current solution
            self.pareto_archive.append(current_solution)
            
        # Keep archive size manageable
        if len(self.pareto_archive) > 50:
            self.pareto_archive = self.pareto_archive[-50:]
    
    def optimization_step(self, model: nn.Module, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Multi-objective optimization step."""
        
        # Compute all complexity measures
        individual_complexities = []
        
        for measure in self.complexity_measures:
            if measure == "kolmogorov":
                comp = kolmogorov_complexity(state, normalize=True)
            elif measure == "shannon":
                probs = torch.softmax(state.flatten(), dim=0)
                comp = shannon_entropy(probs)
            elif measure == "fisher":
                state_copy = state.clone().requires_grad_(True)
                loss = state_copy.sum()
                grad = torch.autograd.grad(loss, state_copy, create_graph=True)[0]
                comp = torch.norm(grad)
            else:
                comp = torch.tensor(0.0)
            
            individual_complexities.append(comp)
        
        # Combined complexity
        total_complexity = sum(w * c for w, c in zip(self.weights, individual_complexities))
        
        # Stability
        stability = torch.tensor(1.0)  # Simplified for now
        
        # Update Pareto archive
        self.update_pareto_archive(individual_complexities, stability)
        
        # Objective
        objective = total_complexity + self.stability_weight * stability
        
        self.step_count += 1
        
        result = {
            "complexity": total_complexity,
            "stability": stability,
            "objective": objective,
        }
        
        # Add individual complexity measures
        for i, measure in enumerate(self.complexity_measures):
            result[f"complexity_{measure}"] = individual_complexities[i]
        
        return result


class AdaptiveComplexityOptimizer(ComplexityOptimizer):
    """
    Optimizer that adapts its complexity targets and methods based on 
    system evolution dynamics.
    """
    
    def __init__(
        self,
        adaptation_rate: float = 0.01,
        complexity_window: int = 20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.adaptation_rate = adaptation_rate
        self.complexity_window = complexity_window
        self.complexity_history = []
        self.target_history = []
        
    def adapt_target_complexity(self):
        """Adapt target complexity based on recent evolution."""
        if len(self.complexity_history) < self.complexity_window:
            return
        
        recent_complexities = self.complexity_history[-self.complexity_window:]
        
        # Compute complexity trend
        complexities_tensor = torch.stack(recent_complexities)
        trend = torch.mean(complexities_tensor[-5:]) - torch.mean(complexities_tensor[:5])
        
        # Adapt target based on trend
        if trend > 0:  # Complexity increasing
            self.target_complexity += self.adaptation_rate
        else:  # Complexity decreasing
            self.target_complexity -= self.adaptation_rate
        
        # Keep target in reasonable bounds
        self.target_complexity = torch.clamp(torch.tensor(self.target_complexity), 0.1, 0.9).item()
        
        self.target_history.append(self.target_complexity)
    
    def compute_complexity(self, state: torch.Tensor) -> torch.Tensor:
        """Adaptive complexity computation."""
        # Use different measures based on system state
        basic_complexity = kolmogorov_complexity(state, normalize=True)
        
        # Store in history
        self.complexity_history.append(basic_complexity)
        if len(self.complexity_history) > self.complexity_window * 2:
            self.complexity_history.pop(0)
        
        return basic_complexity
    
    def compute_stability(self, state: torch.Tensor, prev_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Adaptive stability based on complexity dynamics."""
        if prev_state is None or len(self.complexity_history) < 2:
            return torch.tensor(1.0)
        
        # Stability as predictability of complexity evolution
        recent_changes = []
        for i in range(1, min(len(self.complexity_history), 10)):
            change = self.complexity_history[-i] - self.complexity_history[-i-1]
            recent_changes.append(change)
        
        if len(recent_changes) > 0:
            change_variance = torch.var(torch.stack(recent_changes))
            stability = torch.exp(-change_variance)
        else:
            stability = torch.tensor(1.0)
        
        return stability
    
    def optimization_step(self, model: nn.Module, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Adaptive optimization step."""
        
        # Compute current metrics
        complexity = self.compute_complexity(state)
        stability = self.compute_stability(state)
        
        # Adapt target complexity
        if self.step_count % 10 == 0:  # Adapt every 10 steps
            self.adapt_target_complexity()
        
        # Objective with adaptive target
        target_distance = torch.abs(complexity - self.target_complexity)
        complexity_score = torch.exp(-target_distance)
        
        objective = complexity_score + self.stability_weight * stability
        
        self.step_count += 1
        
        return {
            "complexity": complexity,
            "stability": stability,
            "objective": objective,
            "target_complexity": torch.tensor(self.target_complexity),
            "complexity_score": complexity_score
        }
