"""
Theory Discovery through Entropic AI

This module discovers symbolic theories and mathematical expressions by evolving
them from random symbolic chaos through thermodynamic principles. Unlike traditional
symbolic regression, theories emerge naturally through complexity optimization.

Key Features:
- Evolve mathematical expressions from symbolic noise
- Discover physical laws and mathematical relationships
- Balance accuracy, simplicity, and generalizability
- Emergent theoretical frameworks from data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import sympy as sp
from sympy import symbols, sympify, simplify, expand, factor
import re
import random

from ..core.generative_diffuser import GenerativeDiffuser
from ..core.thermodynamic_network import ThermodynamicNetwork
from ..core.complexity_optimizer import KolmogorovOptimizer
from ..utils.entropy_utils import shannon_entropy, kolmogorov_complexity


class SymbolicExpression:
    """Represents a symbolic mathematical expression with thermodynamic properties."""
    
    def __init__(self, expression: str, variables: List[str]):
        self.expression_str = expression
        self.variables = variables
        self.sympy_expr = None
        self.complexity_score = 0.0
        self.accuracy_score = 0.0
        self.generalization_score = 0.0
        
        try:
            # Convert to sympy expression
            var_symbols = {var: symbols(var) for var in variables}
            self.sympy_expr = sympify(expression, locals=var_symbols)
        except:
            # Invalid expression
            self.sympy_expr = None
    
    def evaluate(self, variable_values: Dict[str, float]) -> Optional[float]:
        """Evaluate expression with given variable values."""
        if self.sympy_expr is None:
            return None
        
        try:
            # Substitute values
            substitutions = {symbols(var): val for var, val in variable_values.items() 
                           if var in self.variables}
            result = float(self.sympy_expr.subs(substitutions))
            
            # Check for valid result
            if np.isfinite(result):
                return result
            else:
                return None
        except:
            return None
    
    def simplify(self) -> 'SymbolicExpression':
        """Simplify the expression."""
        if self.sympy_expr is None:
            return self
        
        try:
            simplified = simplify(self.sympy_expr)
            return SymbolicExpression(str(simplified), self.variables)
        except:
            return self
    
    def get_complexity(self) -> float:
        """Compute expression complexity."""
        if self.sympy_expr is None:
            return float('inf')
        
        # Count operations and symbols
        expr_str = str(self.sympy_expr)
        
        # Count operators
        operators = ['+', '-', '*', '/', '**', 'sin', 'cos', 'tan', 'exp', 'log']
        op_count = sum(expr_str.count(op) for op in operators)
        
        # Count unique symbols
        symbol_count = len(self.sympy_expr.free_symbols)
        
        # Count constants
        const_count = len([atom for atom in self.sympy_expr.atoms() if atom.is_number])
        
        # Complexity formula
        complexity = op_count + 0.5 * symbol_count + 0.3 * const_count
        
        return complexity


class TheorySpace:
    """
    Represents the space of possible theories with thermodynamic structure.
    """
    
    def __init__(
        self,
        variables: List[str],
        operations: List[str] = ['+', '-', '*', '/', '**', 'sin', 'cos', 'exp', 'log'],
        constants: List[float] = [0, 1, 2, 3.14159, 2.71828],
        max_depth: int = 5
    ):
        self.variables = variables
        self.operations = operations
        self.constants = constants
        self.max_depth = max_depth
        
        # Build vocabulary for expression generation
        self.vocabulary = self._build_vocabulary()
        self.vocab_size = len(self.vocabulary)
        
        # Expression templates for different theory types
        self.theory_templates = {
            "linear": "{a}*{x} + {b}",
            "quadratic": "{a}*{x}**2 + {b}*{x} + {c}",
            "exponential": "{a}*exp({b}*{x})",
            "power": "{a}*{x}**{b}",
            "trigonometric": "{a}*sin({b}*{x} + {c})",
            "logarithmic": "{a}*log({x}) + {b}",
            "rational": "({a}*{x} + {b})/({c}*{x} + {d})",
            "composite": "{a}*{func1}({b}*{x}) + {c}*{func2}({d}*{x})"
        }
    
    def _build_vocabulary(self) -> List[str]:
        """Build vocabulary of symbols, operations, and constants."""
        vocab = []
        
        # Add variables
        vocab.extend(self.variables)
        
        # Add operations
        vocab.extend(self.operations)
        
        # Add constants (as strings)
        vocab.extend([str(c) for c in self.constants])
        
        # Add parentheses
        vocab.extend(['(', ')'])
        
        # Add special tokens
        vocab.extend(['<START>', '<END>', '<UNK>'])
        
        return vocab
    
    def expression_to_tokens(self, expression: str) -> List[str]:
        """Convert expression string to tokens."""
        # Simple tokenization (could be improved)
        tokens = re.findall(r'\d*\.?\d+|[a-zA-Z_]\w*|[+\-*/()^]|\*\*', expression)
        return tokens
    
    def tokens_to_expression(self, tokens: List[str]) -> str:
        """Convert tokens back to expression string."""
        return ''.join(tokens)
    
    def random_expression(self, max_tokens: int = 20) -> str:
        """Generate a random expression."""
        tokens = []
        depth = 0
        
        for _ in range(max_tokens):
            if depth == 0:
                # Must start with variable or constant
                token = random.choice(self.variables + [str(c) for c in self.constants])
            elif random.random() < 0.3:  # 30% chance to end
                break
            else:
                # Choose operation or operand
                if random.random() < 0.5:
                    token = random.choice(self.operations)
                else:
                    token = random.choice(self.variables + [str(c) for c in self.constants])
            
            tokens.append(token)
            
            # Update depth based on parentheses
            if token == '(':
                depth += 1
            elif token == ')':
                depth = max(0, depth - 1)
        
        return ' '.join(tokens)


class TheoryThermodynamics(nn.Module):
    """
    Thermodynamic model for symbolic theories.
    Computes energy, entropy, and stability of theoretical expressions.
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_expression_length: int = 50,
        embedding_dim: int = 64
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_length = max_expression_length
        self.embedding_dim = embedding_dim
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Expression encoder (LSTM)
        self.expression_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Theory property predictors
        self.complexity_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        self.accuracy_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.generalization_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Stability predictor (based on sensitivity to perturbations)
        self.stability_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def encode_expression(self, token_indices: torch.Tensor) -> torch.Tensor:
        """Encode expression tokens into fixed-size representation."""
        # Get embeddings
        embeddings = self.token_embeddings(token_indices)  # [batch, seq_len, embed_dim]
        
        # Encode with LSTM
        output, (hidden, cell) = self.expression_encoder(embeddings)
        
        # Use final hidden state as expression representation
        expression_repr = hidden[-1]  # [batch, hidden_size]
        
        return expression_repr
    
    def compute_theory_energy(self, expression_repr: torch.Tensor) -> torch.Tensor:
        """Compute theoretical energy (related to complexity)."""
        complexity = self.complexity_predictor(expression_repr).squeeze(-1)
        
        # Energy increases with complexity
        energy = complexity
        
        return energy
    
    def compute_theory_entropy(self, expression_repr: torch.Tensor) -> torch.Tensor:
        """Compute theoretical entropy (uncertainty/generality)."""
        generalization = self.generalization_predictor(expression_repr).squeeze(-1)
        
        # Higher generalization = higher entropy (more general theories)
        entropy = generalization
        
        return entropy
    
    def forward(
        self,
        token_indices: torch.Tensor,
        data_fit_score: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing all theory properties."""
        
        # Encode expression
        expr_repr = self.encode_expression(token_indices)
        
        # Compute properties
        energy = self.compute_theory_energy(expr_repr)
        entropy = self.compute_theory_entropy(expr_repr)
        complexity = self.complexity_predictor(expr_repr).squeeze(-1)
        stability = self.stability_predictor(expr_repr).squeeze(-1)
        
        # Accuracy from data fitting (if provided)
        if data_fit_score is not None:
            accuracy = data_fit_score
        else:
            accuracy = self.accuracy_predictor(expr_repr).squeeze(-1)
        
        return {
            "energy": energy,
            "entropy": entropy,
            "complexity": complexity,
            "accuracy": accuracy,
            "stability": stability,
            "representation": expr_repr
        }


class TheoryDiscovery:
    """
    Main interface for discovering theories using Entropic AI.
    """
    
    def __init__(
        self,
        variables: List[str] = ["x", "y", "t"],
        domain: str = "physics",
        max_expression_length: int = 30,
        evolution_steps: int = 60,
        symbolic_complexity_limit: int = 10
    ):
        self.variables = variables
        self.domain = domain
        self.max_length = max_expression_length
        self.evolution_steps = evolution_steps
        self.complexity_limit = symbolic_complexity_limit
        
        # Theory space
        self.theory_space = TheorySpace(
            variables=variables,
            max_depth=symbolic_complexity_limit
        )
        
        # Theory thermodynamics model
        self.theory_thermo = TheoryThermodynamics(
            vocab_size=self.theory_space.vocab_size,
            max_expression_length=max_expression_length
        )
        
        # Expression representation dimension
        expr_dim = max_expression_length * self.theory_space.vocab_size
        
        # Thermodynamic network for theory evolution
        self.thermo_network = ThermodynamicNetwork(
            input_dim=expr_dim,
            hidden_dims=[256, 128],
            output_dim=expr_dim,
            temperature=1.0
        )
        
        # Complexity optimizer for theoretical complexity
        self.complexity_optimizer = KolmogorovOptimizer(
            target_complexity=0.7,
            method="entropy"
        )
        
        # Generative diffuser for evolution
        self.diffuser = GenerativeDiffuser(
            network=self.thermo_network,
            optimizer=self.complexity_optimizer,
            diffusion_steps=evolution_steps,
            initial_temperature=2.0,
            final_temperature=0.3
        )
    
    def encode_expression(self, expression: str) -> torch.Tensor:
        """Encode expression string into tensor representation."""
        # Tokenize expression
        tokens = self.theory_space.expression_to_tokens(expression)
        
        # Convert to indices
        token_indices = []
        for token in tokens[:self.max_length]:
            if token in self.theory_space.vocabulary:
                idx = self.theory_space.vocabulary.index(token)
            else:
                idx = self.theory_space.vocabulary.index('<UNK>')
            token_indices.append(idx)
        
        # Pad to max length
        while len(token_indices) < self.max_length:
            token_indices.append(self.theory_space.vocabulary.index('<END>'))
        
        # Convert to one-hot
        one_hot = torch.zeros(self.max_length, self.theory_space.vocab_size)
        for i, idx in enumerate(token_indices):
            one_hot[i, idx] = 1.0
        
        return one_hot.flatten().unsqueeze(0)  # Add batch dimension
    
    def decode_expression(self, flat_repr: torch.Tensor) -> str:
        """Decode tensor representation back to expression string."""
        flat_repr = flat_repr.squeeze(0)  # Remove batch dimension
        
        # Reshape to [max_length, vocab_size]
        token_probs = flat_repr.view(self.max_length, self.theory_space.vocab_size)
        
        # Get most likely tokens
        token_indices = torch.argmax(token_probs, dim=1)
        
        # Convert back to tokens
        tokens = []
        for idx in token_indices:
            token = self.theory_space.vocabulary[idx.item()]
            if token == '<END>':
                break
            if token not in ['<START>', '<UNK>']:
                tokens.append(token)
        
        # Join tokens into expression
        expression = ''.join(tokens)
        
        return expression
    
    def evaluate_theory_fit(
        self,
        expression: str,
        data_x: torch.Tensor,
        data_y: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate how well theory fits experimental data."""
        
        # Create symbolic expression
        sym_expr = SymbolicExpression(expression, self.variables)
        
        if sym_expr.sympy_expr is None:
            return {
                "accuracy": 0.0,
                "complexity": float('inf'),
                "generalization": 0.0,
                "stability": 0.0
            }
        
        # Evaluate on data points
        predictions = []
        for i in range(len(data_x)):
            if len(self.variables) >= 1:
                var_values = {self.variables[0]: data_x[i].item()}
                pred = sym_expr.evaluate(var_values)
                
                if pred is not None:
                    predictions.append(pred)
                else:
                    predictions.append(0.0)  # Default for invalid evaluations
        
        if not predictions:
            return {
                "accuracy": 0.0,
                "complexity": float('inf'),
                "generalization": 0.0,
                "stability": 0.0
            }
        
        predictions = torch.tensor(predictions)
        
        # Compute accuracy (R-squared)
        y_mean = torch.mean(data_y)
        ss_tot = torch.sum((data_y - y_mean) ** 2)
        ss_res = torch.sum((data_y[:len(predictions)] - predictions) ** 2)
        
        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
            accuracy = max(0.0, r_squared.item())
        else:
            accuracy = 0.0
        
        # Compute complexity
        complexity = sym_expr.get_complexity()
        complexity_score = 1.0 / (1.0 + complexity / 10.0)  # Normalize
        
        # Compute generalization (cross-validation on subsets)
        generalization = self._compute_generalization(sym_expr, data_x, data_y)
        
        # Compute stability (sensitivity to small data perturbations)
        stability = self._compute_stability(sym_expr, data_x, data_y)
        
        return {
            "accuracy": accuracy,
            "complexity": complexity_score,
            "generalization": generalization,
            "stability": stability
        }
    
    def _compute_generalization(
        self,
        sym_expr: SymbolicExpression,
        data_x: torch.Tensor,
        data_y: torch.Tensor
    ) -> float:
        """Compute generalization score using cross-validation."""
        
        if len(data_x) < 4:
            return 0.0
        
        # Simple 2-fold cross-validation
        mid = len(data_x) // 2
        
        # Train on first half, test on second half
        train_x, train_y = data_x[:mid], data_y[:mid]
        test_x, test_y = data_x[mid:], data_y[mid:]
        
        # Evaluate on test set
        test_preds = []
        for i in range(len(test_x)):
            if len(self.variables) >= 1:
                var_values = {self.variables[0]: test_x[i].item()}
                pred = sym_expr.evaluate(var_values)
                
                if pred is not None:
                    test_preds.append(pred)
                else:
                    test_preds.append(0.0)
        
        if not test_preds:
            return 0.0
        
        test_preds = torch.tensor(test_preds)
        
        # Compute test accuracy
        test_mean = torch.mean(test_y)
        ss_tot = torch.sum((test_y - test_mean) ** 2)
        ss_res = torch.sum((test_y[:len(test_preds)] - test_preds) ** 2)
        
        if ss_tot > 0:
            test_r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, test_r_squared.item())
        else:
            return 0.0
    
    def _compute_stability(
        self,
        sym_expr: SymbolicExpression,
        data_x: torch.Tensor,
        data_y: torch.Tensor
    ) -> float:
        """Compute stability score by adding noise to data."""
        
        # Add small amount of noise to data
        noise_level = 0.01
        noisy_x = data_x + torch.randn_like(data_x) * noise_level
        noisy_y = data_y + torch.randn_like(data_y) * noise_level
        
        # Evaluate on noisy data
        orig_fit = self.evaluate_theory_fit(str(sym_expr.sympy_expr), data_x, data_y)
        noisy_fit = self.evaluate_theory_fit(str(sym_expr.sympy_expr), noisy_x, noisy_y)
        
        # Stability as consistency between original and noisy fits
        accuracy_diff = abs(orig_fit["accuracy"] - noisy_fit["accuracy"])
        stability = max(0.0, 1.0 - accuracy_diff)
        
        return stability
    
    def discover_from_data(
        self,
        data_x: torch.Tensor,
        data_y: torch.Tensor,
        target_accuracy: float = 0.9
    ) -> Dict:
        """
        Discover theoretical expression that explains the data.
        
        Args:
            data_x: Input data points
            data_y: Output data points  
            target_accuracy: Desired accuracy threshold
            
        Returns:
            Dictionary with discovered theory and properties
        """
        
        # Generate initial chaotic expression
        random_expr = self.theory_space.random_expression()
        chaos = self.encode_expression(random_expr)
        
        # Add noise to make it truly chaotic
        chaos = chaos + torch.randn_like(chaos) * 1.0
        
        # Evolve through thermodynamic process
        final_expr_repr, trajectory = self.diffuser.evolve(chaos)
        
        # Decode final expression
        final_expression = self.decode_expression(final_expr_repr)
        
        # Evaluate theory fit
        theory_fit = self.evaluate_theory_fit(final_expression, data_x, data_y)
        
        # Create symbolic expression for analysis
        sym_expr = SymbolicExpression(final_expression, self.variables)
        simplified_expr = sym_expr.simplify()
        
        return {
            "expression": final_expression,
            "simplified_expression": str(simplified_expr.sympy_expr) if simplified_expr.sympy_expr else final_expression,
            "theory_fit": theory_fit,
            "symbolic_complexity": sym_expr.get_complexity(),
            "evolution_trajectory": trajectory,
            "success_score": theory_fit["accuracy"],
            "variables": self.variables
        }
    
    def discover_physical_law(
        self,
        phenomenon: str,
        data_x: torch.Tensor,
        data_y: torch.Tensor
    ) -> Dict:
        """
        Discover physical laws for specific phenomena.
        """
        
        # Set domain-specific theory templates
        if phenomenon == "oscillation":
            self.theory_space.operations.extend(['sin', 'cos'])
        elif phenomenon == "decay":
            self.theory_space.operations.extend(['exp'])
        elif phenomenon == "growth":
            self.theory_space.operations.extend(['exp', 'log'])
        
        # Discover theory
        result = self.discover_from_data(data_x, data_y, target_accuracy=0.85)
        
        # Add domain-specific analysis
        result["phenomenon"] = phenomenon
        result["physical_interpretation"] = self._interpret_physics(
            result["simplified_expression"], phenomenon
        )
        
        return result
    
    def _interpret_physics(self, expression: str, phenomenon: str) -> str:
        """Provide physical interpretation of discovered expression."""
        
        interpretations = {
            "oscillation": "Harmonic motion with possible damping or driving forces",
            "decay": "Exponential decay process, possibly radioactive or chemical",
            "growth": "Growth process, possibly exponential or logistic",
            "collision": "Conservation laws and momentum transfer",
            "thermodynamics": "Energy conservation and entropy principles"
        }
        
        base_interpretation = interpretations.get(phenomenon, "Unknown physical process")
        
        # Add expression-specific details
        if 'sin' in expression or 'cos' in expression:
            base_interpretation += " - Contains periodic/oscillatory behavior"
        if 'exp' in expression:
            base_interpretation += " - Contains exponential growth/decay"
        if '**2' in expression:
            base_interpretation += " - Contains quadratic scaling"
        
        return base_interpretation
    
    def batch_discovery(
        self,
        datasets: List[Tuple[torch.Tensor, torch.Tensor]],
        phenomena: List[str]
    ) -> List[Dict]:
        """Discover theories for multiple datasets."""
        
        results = []
        
        for i, ((data_x, data_y), phenomenon) in enumerate(zip(datasets, phenomena)):
            result = self.discover_physical_law(phenomenon, data_x, data_y)
            result["dataset_index"] = i
            results.append(result)
        
        return results
