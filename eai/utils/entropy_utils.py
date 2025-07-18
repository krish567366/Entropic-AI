"""
Entropy and Information Theory Utilities for Entropic AI

This module provides various entropy measures and information-theoretic functions
used throughout the Entropic AI system. These form the foundation for measuring
and optimizing complexity, uncertainty, and information content.

Functions include:
- Shannon entropy
- Thermodynamic entropy  
- Kolmogorov complexity (approximation)
- Fisher information
- Mutual information
- Relative entropy (KL divergence)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple
import warnings


def shannon_entropy(
    probabilities: torch.Tensor, 
    dim: int = -1, 
    base: float = 2.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Shannon entropy H(X) = -Σ p(x) log p(x).
    
    Args:
        probabilities: Probability distribution tensor
        dim: Dimension along which to compute entropy
        base: Logarithm base (2 for bits, e for nats)
        eps: Small constant to avoid log(0)
        
    Returns:
        Shannon entropy values
    """
    # Ensure probabilities are normalized
    probabilities = probabilities + eps
    probabilities = probabilities / probabilities.sum(dim=dim, keepdim=True)
    
    # Compute entropy
    if base == 2.0:
        log_probs = torch.log2(probabilities)
    elif base == np.e:
        log_probs = torch.log(probabilities)
    else:
        log_probs = torch.log(probabilities) / np.log(base)
    
    entropy = -torch.sum(probabilities * log_probs, dim=dim)
    
    return entropy


def thermodynamic_entropy(
    energy_states: torch.Tensor,
    temperature: Union[float, torch.Tensor],
    dim: int = -1
) -> torch.Tensor:
    """
    Compute thermodynamic entropy using Boltzmann distribution.
    
    S = k_B * ln(Ω) where Ω is the number of microstates
    
    Args:
        energy_states: Energy values for different states
        temperature: System temperature(s)
        dim: Dimension along which to compute entropy
        
    Returns:
        Thermodynamic entropy values
    """
    # Boltzmann distribution: p_i = exp(-E_i / kT) / Z
    boltzmann_factors = torch.exp(-energy_states / temperature)
    partition_function = torch.sum(boltzmann_factors, dim=dim, keepdim=True)
    probabilities = boltzmann_factors / partition_function
    
    # Entropy S = Σ p_i * E_i / T + ln(Z)
    avg_energy = torch.sum(probabilities * energy_states, dim=dim)
    log_partition = torch.log(partition_function.squeeze(dim))
    
    entropy = avg_energy / temperature + log_partition
    
    return entropy


def kolmogorov_complexity(
    data: torch.Tensor,
    approximation_method: str = "lzw",
    normalize: bool = True
) -> torch.Tensor:
    """
    Approximate Kolmogorov complexity using compression-based methods.
    
    Note: True Kolmogorov complexity is uncomputable, so we use practical
    approximations based on compression ratios.
    
    Args:
        data: Input data tensor
        approximation_method: Method for approximation ("lzw", "entropy", "neural")
        normalize: Whether to normalize by data length
        
    Returns:
        Approximate Kolmogorov complexity values
    """
    if approximation_method == "entropy":
        # Use Shannon entropy as complexity approximation
        if data.dtype == torch.bool or data.dtype == torch.int:
            # Discrete case
            unique_vals, counts = torch.unique(data.flatten(), return_counts=True)
            probs = counts.float() / counts.sum()
            complexity = shannon_entropy(probs)
        else:
            # Continuous case - discretize first
            discretized = torch.quantile(data.flatten(), torch.linspace(0, 1, 256))
            digitized = torch.searchsorted(discretized, data.flatten())
            unique_vals, counts = torch.unique(digitized, return_counts=True)
            probs = counts.float() / counts.sum()
            complexity = shannon_entropy(probs)
            
    elif approximation_method == "lzw":
        # Simplified LZW-like compression ratio
        # Convert to string representation for compression simulation
        data_flat = data.flatten()
        
        # Simple repetition-based complexity measure
        if len(data_flat) == 0:
            return torch.tensor(0.0)
            
        # Count unique subsequences
        unique_patterns = set()
        for i in range(len(data_flat)):
            for j in range(i+1, min(i+10, len(data_flat)+1)):  # Max pattern length 10
                pattern = tuple(data_flat[i:j].tolist())
                unique_patterns.add(pattern)
        
        # Complexity as ratio of unique patterns to total possible
        max_patterns = min(len(data_flat) * 9 // 2, 1000)  # Approximate max patterns
        complexity = torch.tensor(len(unique_patterns) / max_patterns)
        
    elif approximation_method == "neural":
        # Use a simple autoencoder compression ratio
        # This is a placeholder - in practice would use a trained model
        warnings.warn("Neural approximation not fully implemented, falling back to entropy")
        return kolmogorov_complexity(data, "entropy", normalize)
    
    else:
        raise ValueError(f"Unknown approximation method: {approximation_method}")
    
    if normalize and data.numel() > 0:
        complexity = complexity / torch.log2(torch.tensor(float(data.numel())))
    
    return complexity


def fisher_information(
    probability_distribution: torch.Tensor,
    parameter_gradient: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Fisher Information Matrix.
    
    I(θ) = E[(∇ log p(x|θ))^2]
    
    Args:
        probability_distribution: p(x|θ) 
        parameter_gradient: ∇_θ log p(x|θ)
        eps: Small constant for numerical stability
        
    Returns:
        Fisher information values
    """
    # Ensure probability distribution is normalized
    prob_dist = probability_distribution + eps
    prob_dist = prob_dist / prob_dist.sum(dim=-1, keepdim=True)
    
    # Fisher information as expected squared score
    score_squared = parameter_gradient ** 2
    fisher_info = torch.sum(prob_dist * score_squared, dim=-1)
    
    return fisher_info


def mutual_information(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: int = 50,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute mutual information I(X;Y) = H(X) + H(Y) - H(X,Y).
    
    Args:
        x: First variable
        y: Second variable  
        bins: Number of bins for discretization
        eps: Small constant for numerical stability
        
    Returns:
        Mutual information value
    """
    # Discretize continuous variables
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Create joint histogram
    x_bins = torch.linspace(x_flat.min(), x_flat.max(), bins)
    y_bins = torch.linspace(y_flat.min(), y_flat.max(), bins)
    
    x_digitized = torch.searchsorted(x_bins, x_flat)
    y_digitized = torch.searchsorted(y_bins, y_flat)
    
    # Joint distribution
    joint_hist = torch.zeros(bins, bins)
    for i in range(len(x_digitized)):
        if x_digitized[i] < bins and y_digitized[i] < bins:
            joint_hist[x_digitized[i], y_digitized[i]] += 1
    
    joint_prob = joint_hist / joint_hist.sum()
    joint_prob = joint_prob + eps
    
    # Marginal distributions
    x_prob = joint_prob.sum(dim=1)
    y_prob = joint_prob.sum(dim=0)
    
    # Compute entropies
    h_x = shannon_entropy(x_prob)
    h_y = shannon_entropy(y_prob)
    h_xy = shannon_entropy(joint_prob.flatten())
    
    # Mutual information
    mi = h_x + h_y - h_xy
    
    return mi


def relative_entropy(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute relative entropy (KL divergence) D(P||Q) = Σ p log(p/q).
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Small constant for numerical stability
        
    Returns:
        KL divergence value
    """
    # Add small epsilon and normalize
    p = p + eps
    q = q + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    # Compute KL divergence
    kl_div = torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)
    
    return kl_div


def conditional_entropy(
    x: torch.Tensor,
    y: torch.Tensor,
    bins: int = 50
) -> torch.Tensor:
    """
    Compute conditional entropy H(X|Y) = H(X,Y) - H(Y).
    
    Args:
        x: First variable
        y: Conditioning variable
        bins: Number of bins for discretization
        
    Returns:
        Conditional entropy value
    """
    # Use mutual information identity: H(X|Y) = H(X) - I(X;Y)
    h_x = shannon_entropy(F.softmax(x.flatten(), dim=0))
    mi_xy = mutual_information(x, y, bins)
    
    conditional_h = h_x - mi_xy
    
    return conditional_h


def cross_entropy(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute cross entropy H(P,Q) = -Σ p log q.
    
    Args:
        p: True distribution
        q: Predicted distribution
        eps: Small constant for numerical stability
        
    Returns:
        Cross entropy value
    """
    # Normalize distributions
    p = p + eps
    q = q + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    # Cross entropy
    cross_h = -torch.sum(p * torch.log(q), dim=-1)
    
    return cross_h


def entropy_rate(
    sequence: torch.Tensor,
    order: int = 1
) -> torch.Tensor:
    """
    Compute entropy rate of a sequence (information per symbol).
    
    Args:
        sequence: Sequential data
        order: Markov order for conditional entropy computation
        
    Returns:
        Entropy rate value
    """
    if order == 0:
        # Independent symbols
        unique_vals, counts = torch.unique(sequence, return_counts=True)
        probs = counts.float() / counts.sum()
        return shannon_entropy(probs)
    
    else:
        # Conditional entropy for Markov chains
        # H_rate = H(X_n | X_{n-1}, ..., X_{n-order})
        
        if len(sequence) <= order:
            return torch.tensor(0.0)
        
        # Create context-symbol pairs
        contexts = []
        symbols = []
        
        for i in range(order, len(sequence)):
            context = tuple(sequence[i-order:i].tolist())
            symbol = sequence[i].item()
            contexts.append(context)
            symbols.append(symbol)
        
        # Compute conditional entropy
        unique_contexts = list(set(contexts))
        total_entropy = 0.0
        total_count = len(contexts)
        
        for context in unique_contexts:
            # Find symbols following this context
            context_indices = [i for i, c in enumerate(contexts) if c == context]
            context_symbols = [symbols[i] for i in context_indices]
            
            if len(context_symbols) == 0:
                continue
                
            # Probability of this context
            context_prob = len(context_indices) / total_count
            
            # Entropy of symbols given this context
            unique_syms, sym_counts = torch.unique(torch.tensor(context_symbols), return_counts=True)
            sym_probs = sym_counts.float() / sym_counts.sum()
            context_entropy = shannon_entropy(sym_probs)
            
            total_entropy += context_prob * context_entropy
        
        return torch.tensor(total_entropy)


def complexity_entropy_tradeoff(
    data: torch.Tensor,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Compute a complexity-entropy tradeoff measure.
    
    Balances between high complexity (Kolmogorov) and high entropy (Shannon).
    
    Args:
        data: Input data
        alpha: Tradeoff parameter (0=pure entropy, 1=pure complexity)
        
    Returns:
        Combined complexity-entropy score
    """
    entropy = shannon_entropy(F.softmax(data.flatten(), dim=0))
    complexity = kolmogorov_complexity(data, normalize=True)
    
    # Combine with tradeoff parameter
    score = alpha * complexity + (1 - alpha) * entropy
    
    return score
