"""
Metrics and evaluation utilities for Entropic AI

This module provides comprehensive metrics for evaluating the performance
and properties of Entropic AI systems, including complexity measures,
stability indicators, and emergence detection.

Key metrics:
- Complexity scoring (Kolmogorov, Shannon, etc.)
- Stability measures (thermal, structural, functional)
- Emergence detection and quantification
- Performance benchmarking
- Thermodynamic property validation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_mutual_info_score
import networkx as nx

from .entropy_utils import (
    shannon_entropy, 
    kolmogorov_complexity,
    mutual_information,
    fisher_information
)


def complexity_score(
    data: torch.Tensor,
    method: str = "combined",
    normalize: bool = True
) -> float:
    """
    Compute comprehensive complexity score for data or system state.
    
    Args:
        data: Input data tensor
        method: Complexity method ("kolmogorov", "shannon", "lmc", "combined")
        normalize: Whether to normalize score to [0, 1]
        
    Returns:
        Complexity score
    """
    
    if data.numel() == 0:
        return 0.0
    
    if method == "kolmogorov":
        score = kolmogorov_complexity(data, normalize=normalize).item()
    
    elif method == "shannon":
        # Shannon entropy of the distribution
        if data.dtype in [torch.bool, torch.int, torch.long]:
            # Discrete case
            unique_vals, counts = torch.unique(data.flatten(), return_counts=True)
            probs = counts.float() / counts.sum()
            score = shannon_entropy(probs).item()
        else:
            # Continuous case - discretize first
            data_flat = data.flatten()
            hist, _ = torch.histogram(data_flat, bins=50)
            probs = hist.float() / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities
            score = shannon_entropy(probs).item()
        
        if normalize:
            max_entropy = np.log2(len(torch.unique(data.flatten())))
            score = score / max_entropy if max_entropy > 0 else 0.0
    
    elif method == "lmc":
        # Logical depth / computational complexity approximation
        score = _logical_depth_approximation(data)
        
    elif method == "combined":
        # Weighted combination of multiple measures
        kolm_score = kolmogorov_complexity(data, normalize=True).item()
        
        # Shannon component
        if data.dtype in [torch.bool, torch.int, torch.long]:
            unique_vals, counts = torch.unique(data.flatten(), return_counts=True)
            probs = counts.float() / counts.sum()
            shannon_score = shannon_entropy(probs).item()
            max_entropy = np.log2(len(unique_vals))
            shannon_score = shannon_score / max_entropy if max_entropy > 0 else 0.0
        else:
            data_flat = data.flatten()
            hist, _ = torch.histogram(data_flat, bins=50)
            probs = hist.float() / hist.sum()
            probs = probs[probs > 0]
            shannon_score = shannon_entropy(probs).item()
            shannon_score = shannon_score / np.log2(50)  # Normalize by max possible
        
        # Logical depth component
        logical_score = _logical_depth_approximation(data)
        
        # Weighted combination
        score = 0.4 * kolm_score + 0.4 * shannon_score + 0.2 * logical_score
    
    else:
        raise ValueError(f"Unknown complexity method: {method}")
    
    return float(score)


def _logical_depth_approximation(data: torch.Tensor) -> float:
    """Approximate logical depth through iterative compression."""
    
    if data.numel() == 0:
        return 0.0
    
    # Convert to binary representation for compression simulation
    data_flat = data.flatten()
    
    # Simple iterative compression approximation
    current = data_flat.clone()
    compression_steps = 0
    max_steps = 10
    
    for step in range(max_steps):
        # Simple pattern detection and compression
        if len(current) <= 2:
            break
            
        # Look for repeated patterns
        compressed = []
        i = 0
        while i < len(current):
            # Check for simple repetitions
            if i < len(current) - 1 and current[i] == current[i + 1]:
                # Count repetitions
                count = 1
                j = i + 1
                while j < len(current) and current[j] == current[i]:
                    count += 1
                    j += 1
                
                # Represent as (value, count) - simplified
                compressed.append(current[i])
                if count > 2:  # Only compress if worthwhile
                    compression_steps += 1
                i = j
            else:
                compressed.append(current[i])
                i += 1
        
        # Check if compression occurred
        if len(compressed) >= len(current):
            break
        
        current = torch.tensor(compressed)
        compression_steps += 1
    
    # Logical depth is related to compression steps needed
    logical_depth = compression_steps / max_steps
    
    return logical_depth


def stability_measure(
    state_sequence: List[torch.Tensor],
    metric: str = "variance",
    window_size: int = 5
) -> float:
    """
    Measure stability of a sequence of system states.
    
    Args:
        state_sequence: List of state tensors over time
        metric: Stability metric ("variance", "lyapunov", "correlation")
        window_size: Window size for local stability analysis
        
    Returns:
        Stability score (higher = more stable)
    """
    
    if len(state_sequence) < 2:
        return 1.0
    
    if metric == "variance":
        # Stability as inverse of state variance
        state_stack = torch.stack(state_sequence)
        state_var = torch.var(state_stack, dim=0).mean()
        stability = 1.0 / (1.0 + state_var.item())
        
    elif metric == "lyapunov":
        # Approximate Lyapunov exponent
        stability = _approximate_lyapunov_exponent(state_sequence)
        
    elif metric == "correlation":
        # Stability through autocorrelation
        stability = _temporal_correlation_stability(state_sequence, window_size)
        
    else:
        raise ValueError(f"Unknown stability metric: {metric}")
    
    return float(stability)


def _approximate_lyapunov_exponent(state_sequence: List[torch.Tensor]) -> float:
    """Approximate largest Lyapunov exponent for stability assessment."""
    
    if len(state_sequence) < 3:
        return 1.0
    
    # Compute successive differences
    diff_norms = []
    for i in range(1, len(state_sequence)):
        diff = state_sequence[i] - state_sequence[i-1]
        norm = torch.norm(diff).item()
        if norm > 0:
            diff_norms.append(norm)
    
    if len(diff_norms) < 2:
        return 1.0
    
    # Estimate growth rate of differences
    log_norms = [np.log(norm) for norm in diff_norms]
    
    # Linear fit to estimate exponent
    x = np.arange(len(log_norms))
    if len(x) > 1:
        slope, _ = np.polyfit(x, log_norms, 1)
        lyapunov = -slope  # Negative for stability
    else:
        lyapunov = 0.0
    
    # Convert to stability score (0 to 1)
    stability = 1.0 / (1.0 + np.exp(lyapunov))
    
    return stability


def _temporal_correlation_stability(
    state_sequence: List[torch.Tensor], 
    window_size: int
) -> float:
    """Measure stability through temporal correlation."""
    
    if len(state_sequence) < window_size:
        return 1.0
    
    correlations = []
    
    for i in range(window_size, len(state_sequence)):
        # Compare current state with past states in window
        current = state_sequence[i].flatten()
        
        window_correlations = []
        for j in range(i - window_size, i):
            past = state_sequence[j].flatten()
            
            # Compute correlation
            if len(current) == len(past):
                correlation = torch.corrcoef(torch.stack([current, past]))[0, 1]
                if not torch.isnan(correlation):
                    window_correlations.append(correlation.item())
        
        if window_correlations:
            avg_correlation = np.mean(window_correlations)
            correlations.append(avg_correlation)
    
    if correlations:
        # High correlation = high stability
        stability = np.mean(correlations)
        # Ensure in [0, 1] range
        stability = (stability + 1) / 2
    else:
        stability = 0.5
    
    return stability


def emergence_index(
    state_sequence: List[torch.Tensor],
    complexity_threshold: float = 0.1,
    novelty_threshold: float = 0.3
) -> float:
    """
    Detect and quantify emergent behavior in system evolution.
    
    Args:
        state_sequence: Sequence of system states
        complexity_threshold: Minimum complexity increase for emergence
        novelty_threshold: Minimum novelty for emergence detection
        
    Returns:
        Emergence index (0 = no emergence, 1 = strong emergence)
    """
    
    if len(state_sequence) < 3:
        return 0.0
    
    # Compute complexity trajectory
    complexities = []
    for state in state_sequence:
        comp = complexity_score(state, method="combined")
        complexities.append(comp)
    
    # Detect sudden complexity increases
    complexity_increases = []
    for i in range(1, len(complexities)):
        increase = complexities[i] - complexities[i-1]
        if increase > complexity_threshold:
            complexity_increases.append(increase)
    
    # Compute novelty (distance from previous states)
    novelties = []
    for i in range(2, len(state_sequence)):
        current = state_sequence[i].flatten()
        
        # Compare with all previous states
        distances = []
        for j in range(i):
            past = state_sequence[j].flatten()
            if len(current) == len(past):
                dist = torch.norm(current - past).item()
                distances.append(dist)
        
        if distances:
            min_distance = min(distances)
            novelties.append(min_distance)
    
    # Detect novelty spikes
    if novelties:
        novelty_mean = np.mean(novelties)
        novelty_std = np.std(novelties)
        novelty_spikes = [n for n in novelties 
                         if n > novelty_mean + novelty_threshold * novelty_std]
    else:
        novelty_spikes = []
    
    # Combine complexity and novelty for emergence index
    emergence_score = 0.0
    
    # Complexity component
    if complexity_increases:
        complexity_component = min(1.0, np.sum(complexity_increases) / len(complexities))
        emergence_score += 0.6 * complexity_component
    
    # Novelty component
    if novelty_spikes:
        novelty_component = min(1.0, len(novelty_spikes) / len(novelties))
        emergence_score += 0.4 * novelty_component
    
    return float(emergence_score)


def thermodynamic_consistency(
    energy_sequence: List[float],
    entropy_sequence: List[float],
    temperature_sequence: List[float]
) -> Dict[str, float]:
    """
    Check consistency of thermodynamic properties with physical laws.
    
    Args:
        energy_sequence: Internal energy over time
        entropy_sequence: Entropy over time  
        temperature_sequence: Temperature over time
        
    Returns:
        Dictionary of consistency scores
    """
    
    consistency_scores = {}
    
    # Check energy conservation (should be approximately conserved in isolated system)
    if len(energy_sequence) > 1:
        energy_variance = np.var(energy_sequence)
        energy_mean = np.mean(energy_sequence)
        if energy_mean != 0:
            energy_consistency = 1.0 / (1.0 + energy_variance / abs(energy_mean))
        else:
            energy_consistency = 1.0 if energy_variance < 1e-6 else 0.0
        consistency_scores["energy_conservation"] = energy_consistency
    
    # Check second law (entropy should not decrease in isolated system)
    if len(entropy_sequence) > 1:
        entropy_decreases = 0
        for i in range(1, len(entropy_sequence)):
            if entropy_sequence[i] < entropy_sequence[i-1]:
                entropy_decreases += 1
        
        second_law_consistency = 1.0 - (entropy_decreases / (len(entropy_sequence) - 1))
        consistency_scores["second_law"] = second_law_consistency
    
    # Check temperature behavior (should decrease in cooling)
    if len(temperature_sequence) > 1:
        temp_decreases = 0
        for i in range(1, len(temperature_sequence)):
            if temperature_sequence[i] <= temperature_sequence[i-1]:
                temp_decreases += 1
        
        cooling_consistency = temp_decreases / (len(temperature_sequence) - 1)
        consistency_scores["cooling_behavior"] = cooling_consistency
    
    # Check Maxwell relations (simplified)
    if len(energy_sequence) > 2 and len(entropy_sequence) > 2:
        # dU/dS should be related to temperature
        du_ds = np.gradient(energy_sequence) / (np.gradient(entropy_sequence) + 1e-8)
        temp_correlation = np.corrcoef(du_ds, temperature_sequence[:len(du_ds)])[0, 1]
        
        if not np.isnan(temp_correlation):
            maxwell_consistency = abs(temp_correlation)
        else:
            maxwell_consistency = 0.0
        
        consistency_scores["maxwell_relations"] = maxwell_consistency
    
    # Overall consistency
    if consistency_scores:
        overall_consistency = np.mean(list(consistency_scores.values()))
        consistency_scores["overall"] = overall_consistency
    
    return consistency_scores


def performance_benchmark(
    system_results: Dict[str, Any],
    benchmark_targets: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Benchmark system performance against target metrics.
    
    Args:
        system_results: Dictionary of system performance metrics
        benchmark_targets: Target values for each metric
        weights: Optional weights for different metrics
        
    Returns:
        Benchmark scores and overall performance
    """
    
    if weights is None:
        weights = {key: 1.0 for key in benchmark_targets.keys()}
    
    benchmark_scores = {}
    weighted_scores = []
    
    for metric, target in benchmark_targets.items():
        if metric in system_results:
            actual = system_results[metric]
            
            # Compute normalized score based on metric type
            if metric in ["accuracy", "correctness", "stability", "complexity"]:
                # Higher is better
                diff = abs(actual - target)
                score = max(0.0, 1.0 - diff / target) if target > 0 else 0.0
            elif metric in ["energy", "delay", "error"]:
                # Lower is better
                if actual <= target:
                    score = 1.0
                else:
                    score = max(0.0, target / actual)
            else:
                # General case - closer to target is better
                diff = abs(actual - target)
                score = max(0.0, 1.0 - diff / max(abs(target), 1.0))
            
            benchmark_scores[metric] = score
            
            # Add to weighted average
            weight = weights.get(metric, 1.0)
            weighted_scores.append(weight * score)
    
    # Compute overall benchmark score
    if weighted_scores and weights:
        total_weight = sum(weights.get(metric, 1.0) for metric in benchmark_scores.keys())
        overall_score = sum(weighted_scores) / total_weight
    else:
        overall_score = 0.0
    
    benchmark_scores["overall_benchmark"] = overall_score
    
    return benchmark_scores


def entropy_production_rate(
    entropy_sequence: List[float],
    time_steps: Optional[List[float]] = None
) -> float:
    """
    Compute entropy production rate (important for non-equilibrium systems).
    
    Args:
        entropy_sequence: Entropy values over time
        time_steps: Optional time step values (default: uniform spacing)
        
    Returns:
        Average entropy production rate
    """
    
    if len(entropy_sequence) < 2:
        return 0.0
    
    if time_steps is None:
        time_steps = list(range(len(entropy_sequence)))
    
    # Compute entropy gradients
    entropy_gradients = []
    for i in range(1, len(entropy_sequence)):
        dt = time_steps[i] - time_steps[i-1]
        if dt > 0:
            gradient = (entropy_sequence[i] - entropy_sequence[i-1]) / dt
            entropy_gradients.append(gradient)
    
    if entropy_gradients:
        # Average production rate
        avg_production_rate = np.mean([max(0, g) for g in entropy_gradients])
    else:
        avg_production_rate = 0.0
    
    return float(avg_production_rate)


def structural_order_parameter(
    state: torch.Tensor,
    order_type: str = "crystalline"
) -> float:
    """
    Compute order parameter for detecting structural organization.
    
    Args:
        state: System state tensor
        order_type: Type of order to detect ("crystalline", "topological", "magnetic")
        
    Returns:
        Order parameter (0 = disordered, 1 = perfectly ordered)
    """
    
    if state.numel() == 0:
        return 0.0
    
    state_flat = state.flatten()
    
    if order_type == "crystalline":
        # Detect periodic patterns
        order_param = _detect_crystalline_order(state_flat)
        
    elif order_type == "topological":
        # Detect topological features
        order_param = _detect_topological_order(state)
        
    elif order_type == "magnetic":
        # Detect alignment (for spin-like systems)
        order_param = _detect_magnetic_order(state_flat)
        
    else:
        raise ValueError(f"Unknown order type: {order_type}")
    
    return float(order_param)


def _detect_crystalline_order(state_flat: torch.Tensor) -> float:
    """Detect crystalline-like periodic order."""
    
    if len(state_flat) < 4:
        return 0.0
    
    # Look for periodicities using autocorrelation
    state_np = state_flat.numpy()
    
    # Compute autocorrelation
    autocorr = np.correlate(state_np, state_np, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    
    # Find peaks in autocorrelation (indicating periodicity)
    if len(autocorr) > 3:
        # Look for secondary peaks
        peak_threshold = 0.5 * autocorr[0]  # 50% of zero-lag correlation
        peaks = []
        
        for i in range(2, min(len(autocorr), len(state_flat) // 2)):
            if autocorr[i] > peak_threshold:
                peaks.append(autocorr[i])
        
        if peaks:
            # Order parameter based on strongest secondary peak
            order_param = max(peaks) / autocorr[0]
        else:
            order_param = 0.0
    else:
        order_param = 0.0
    
    return order_param


def _detect_topological_order(state: torch.Tensor) -> float:
    """Detect topological order through connectivity patterns."""
    
    if state.numel() < 9:  # Need at least 3x3 for topology
        return 0.0
    
    # For 2D case, reshape to square if possible
    sqrt_n = int(np.sqrt(state.numel()))
    if sqrt_n * sqrt_n == state.numel():
        state_2d = state.view(sqrt_n, sqrt_n)
        
        # Compute local topology measures
        topology_score = 0.0
        count = 0
        
        for i in range(1, sqrt_n - 1):
            for j in range(1, sqrt_n - 1):
                # Look at local neighborhood
                center = state_2d[i, j]
                neighbors = [
                    state_2d[i-1, j], state_2d[i+1, j],
                    state_2d[i, j-1], state_2d[i, j+1]
                ]
                
                # Simple topological measure: local correlation
                neighbor_tensor = torch.tensor(neighbors)
                local_var = torch.var(neighbor_tensor)
                local_order = 1.0 / (1.0 + local_var)
                
                topology_score += local_order
                count += 1
        
        if count > 0:
            order_param = topology_score / count
        else:
            order_param = 0.0
    else:
        # 1D case - use local correlations
        order_param = 0.0
        count = 0
        
        for i in range(1, len(state) - 1):
            local_corr = torch.corrcoef(torch.stack([
                state[i-1:i+2], 
                torch.tensor([0.0, 1.0, 0.0])  # Reference pattern
            ]))[0, 1]
            
            if not torch.isnan(local_corr):
                order_param += abs(local_corr)
                count += 1
        
        if count > 0:
            order_param = order_param / count
    
    return order_param.item() if isinstance(order_param, torch.Tensor) else order_param


def _detect_magnetic_order(state_flat: torch.Tensor) -> float:
    """Detect magnetic-like alignment order."""
    
    if len(state_flat) < 2:
        return 0.0
    
    # Normalize to [-1, 1] range (like spins)
    state_normalized = 2 * (state_flat - state_flat.min()) / (state_flat.max() - state_flat.min() + 1e-8) - 1
    
    # Magnetization (net alignment)
    magnetization = torch.abs(torch.mean(state_normalized))
    
    return magnetization.item()
