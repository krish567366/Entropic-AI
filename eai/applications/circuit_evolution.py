"""
Circuit Evolution through Entropic AI

This module evolves digital circuits from thermal noise through thermodynamic
principles. Unlike traditional circuit synthesis, circuits emerge naturally
through energy minimization and complexity optimization.

Key Features:
- Evolve logic circuits from random gate arrangements
- Optimize for functionality, efficiency, and thermal stability
- Handle noise and fault tolerance through thermodynamic robustness
- Emergent parallel processing architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from itertools import product

from ..core.generative_diffuser import GenerativeDiffuser
from ..core.thermodynamic_network import ThermodynamicNetwork
from ..core.complexity_optimizer import KolmogorovOptimizer
from ..utils.entropy_utils import shannon_entropy


class LogicGate:
    """Represents a single logic gate with thermodynamic properties."""
    
    GATE_TYPES = {
        "AND": lambda x, y: x & y,
        "OR": lambda x, y: x | y,
        "NOT": lambda x: ~x,
        "XOR": lambda x, y: x ^ y,
        "NAND": lambda x, y: ~(x & y),
        "NOR": lambda x, y: ~(x | y),
        "BUFFER": lambda x: x
    }
    
    def __init__(self, gate_type: str, inputs: int = 2):
        self.gate_type = gate_type
        self.inputs = inputs
        self.thermal_noise_level = 0.0
        self.power_consumption = self._get_power_consumption()
        self.delay = self._get_propagation_delay()
        
    def _get_power_consumption(self) -> float:
        """Get typical power consumption for gate type."""
        power_map = {
            "AND": 1.0, "OR": 1.0, "NOT": 0.5,
            "XOR": 1.5, "NAND": 0.8, "NOR": 0.8, "BUFFER": 0.3
        }
        return power_map.get(self.gate_type, 1.0)
    
    def _get_propagation_delay(self) -> float:
        """Get propagation delay for gate type."""
        delay_map = {
            "AND": 1.0, "OR": 1.0, "NOT": 0.5,
            "XOR": 2.0, "NAND": 0.8, "NOR": 0.8, "BUFFER": 0.2
        }
        return delay_map.get(self.gate_type, 1.0)
    
    def compute(self, inputs: List[bool]) -> bool:
        """Compute gate output with thermal noise consideration."""
        if self.gate_type == "NOT" or self.gate_type == "BUFFER":
            result = self.GATE_TYPES[self.gate_type](inputs[0])
        else:
            result = self.GATE_TYPES[self.gate_type](inputs[0], inputs[1])
        
        # Add thermal noise effects
        if self.thermal_noise_level > 0:
            if np.random.random() < self.thermal_noise_level:
                result = not result  # Flip bit due to noise
        
        return result


class CircuitThermodynamics(nn.Module):
    """
    Thermodynamic model for digital circuits.
    Computes energy, entropy, and stability of circuit configurations.
    """
    
    def __init__(
        self,
        max_gates: int = 20,
        gate_types: List[str] = ["AND", "OR", "NOT", "XOR"],
        max_inputs: int = 8,
        max_outputs: int = 4
    ):
        super().__init__()
        
        self.max_gates = max_gates
        self.gate_types = gate_types
        self.max_inputs = max_inputs
        self.max_outputs = max_outputs
        self.n_gate_types = len(gate_types)
        
        # Gate type embeddings
        self.gate_embeddings = nn.Embedding(self.n_gate_types, 32)
        
        # Circuit property predictors
        self.power_predictor = nn.Sequential(
            nn.Linear(max_gates * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensure positive power
        )
        
        self.delay_predictor = nn.Sequential(
            nn.Linear(max_gates * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()  # Ensure positive delay
        )
        
        self.noise_tolerance = nn.Sequential(
            nn.Linear(max_gates * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0-1 noise tolerance
        )
        
        # Functional correctness predictor
        self.correctness_net = nn.Sequential(
            nn.Linear(max_gates * 32 + max_inputs + max_outputs, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def gate_type_to_index(self, gate_type: str) -> int:
        """Convert gate type to index."""
        return self.gate_types.index(gate_type)
    
    def compute_circuit_energy(
        self,
        gate_types: torch.Tensor,
        connections: torch.Tensor,
        activity_factor: float = 0.5
    ) -> torch.Tensor:
        """
        Compute total circuit energy (power consumption).
        
        Args:
            gate_types: Tensor of gate type indices [batch, max_gates]
            connections: Circuit connectivity matrix [batch, max_gates, max_gates]
            activity_factor: Average switching activity (0-1)
        """
        batch_size = gate_types.shape[0]
        
        # Get gate embeddings
        gate_embeds = self.gate_embeddings(gate_types)  # [batch, max_gates, 32]
        
        # Predict base power consumption
        circuit_embed = gate_embeds.view(batch_size, -1)
        base_power = self.power_predictor(circuit_embed).squeeze(-1)
        
        # Add dynamic power based on switching activity
        num_connections = connections.sum(dim=(1, 2))
        dynamic_power = activity_factor * num_connections * 0.1
        
        total_energy = base_power + dynamic_power
        
        return total_energy
    
    def compute_circuit_entropy(
        self,
        gate_types: torch.Tensor,
        connections: torch.Tensor
    ) -> torch.Tensor:
        """Compute circuit entropy (disorder/randomness)."""
        batch_size = gate_types.shape[0]
        
        # Gate type diversity entropy
        gate_entropies = []
        for b in range(batch_size):
            unique_gates, counts = torch.unique(gate_types[b], return_counts=True)
            if len(counts) > 1:
                probs = counts.float() / counts.sum()
                gate_entropy = shannon_entropy(probs)
            else:
                gate_entropy = torch.tensor(0.0)
            gate_entropies.append(gate_entropy)
        
        gate_entropy_batch = torch.stack(gate_entropies)
        
        # Connection pattern entropy
        conn_entropies = []
        for b in range(batch_size):
            conn_flat = connections[b].flatten()
            # Count connection patterns
            n_connections = conn_flat.sum()
            if n_connections > 0:
                conn_density = n_connections / len(conn_flat)
                # Simple entropy based on connection density
                if conn_density > 0 and conn_density < 1:
                    conn_entropy = -conn_density * torch.log2(conn_density) - \
                                  (1-conn_density) * torch.log2(1-conn_density)
                else:
                    conn_entropy = torch.tensor(0.0)
            else:
                conn_entropy = torch.tensor(0.0)
            conn_entropies.append(conn_entropy)
        
        conn_entropy_batch = torch.stack(conn_entropies)
        
        total_entropy = gate_entropy_batch + conn_entropy_batch
        
        return total_entropy
    
    def compute_thermal_stability(
        self,
        gate_types: torch.Tensor,
        connections: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Compute thermal stability of circuit."""
        batch_size = gate_types.shape[0]
        
        gate_embeds = self.gate_embeddings(gate_types)
        circuit_embed = gate_embeds.view(batch_size, -1)
        
        # Predict noise tolerance
        noise_tolerance = self.noise_tolerance(circuit_embed).squeeze(-1)
        
        # Stability decreases with temperature
        thermal_stability = noise_tolerance * torch.exp(-temperature)
        
        return thermal_stability
    
    def forward(
        self,
        gate_types: torch.Tensor,
        connections: torch.Tensor,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing all circuit properties."""
        
        batch_size = gate_types.shape[0]
        
        # Compute thermodynamic properties
        energy = self.compute_circuit_energy(gate_types, connections)
        entropy = self.compute_circuit_entropy(gate_types, connections)
        stability = self.compute_thermal_stability(gate_types, connections, temperature)
        
        # Compute functional properties
        gate_embeds = self.gate_embeddings(gate_types)
        circuit_embed = gate_embeds.view(batch_size, -1)
        
        # Add I/O information
        io_embed = torch.cat([inputs, outputs], dim=1)
        full_embed = torch.cat([circuit_embed, io_embed], dim=1)
        
        correctness = self.correctness_net(full_embed).squeeze(-1)
        delay = self.delay_predictor(circuit_embed).squeeze(-1)
        
        return {
            "energy": energy,
            "entropy": entropy,
            "stability": stability,
            "correctness": correctness,
            "delay": delay
        }


class CircuitEvolution:
    """
    Main interface for evolving digital circuits using Entropic AI.
    """
    
    def __init__(
        self,
        max_gates: int = 15,
        gate_types: List[str] = ["AND", "OR", "NOT", "XOR"],
        max_inputs: int = 4,
        max_outputs: int = 2,
        thermal_noise_level: float = 0.01,
        evolution_steps: int = 80
    ):
        self.max_gates = max_gates
        self.gate_types = gate_types
        self.max_inputs = max_inputs
        self.max_outputs = max_outputs
        self.thermal_noise_level = thermal_noise_level
        self.evolution_steps = evolution_steps
        
        # Circuit thermodynamics model
        self.circuit_thermo = CircuitThermodynamics(
            max_gates=max_gates,
            gate_types=gate_types,
            max_inputs=max_inputs,
            max_outputs=max_outputs
        )
        
        # Circuit representation dimension
        # Gates + connections + I/O
        circuit_dim = (
            max_gates +  # Gate types
            max_gates * max_gates +  # Connections
            max_inputs + max_outputs  # I/O
        )
        
        # Thermodynamic network for circuit evolution
        self.thermo_network = ThermodynamicNetwork(
            input_dim=circuit_dim,
            hidden_dims=[128, 64],
            output_dim=circuit_dim,
            temperature=1.5
        )
        
        # Complexity optimizer for circuit complexity
        self.complexity_optimizer = KolmogorovOptimizer(
            target_complexity=0.6,
            method="entropy"
        )
        
        # Generative diffuser for evolution
        self.diffuser = GenerativeDiffuser(
            network=self.thermo_network,
            optimizer=self.complexity_optimizer,
            diffusion_steps=evolution_steps,
            initial_temperature=3.0,
            final_temperature=0.2
        )
    
    def encode_circuit(
        self,
        gates: List[str],
        connections: List[Tuple[int, int]],
        inputs: List[int],
        outputs: List[int]
    ) -> torch.Tensor:
        """Encode circuit into flat representation."""
        
        # Encode gates (one-hot)
        gate_encoding = torch.zeros(self.max_gates, len(self.gate_types))
        for i, gate in enumerate(gates[:self.max_gates]):
            if gate in self.gate_types:
                gate_idx = self.gate_types.index(gate)
                gate_encoding[i, gate_idx] = 1.0
        
        # Encode connections (adjacency matrix)
        conn_matrix = torch.zeros(self.max_gates, self.max_gates)
        for src, dst in connections:
            if src < self.max_gates and dst < self.max_gates:
                conn_matrix[src, dst] = 1.0
        
        # Encode I/O
        input_encoding = torch.zeros(self.max_inputs)
        output_encoding = torch.zeros(self.max_outputs)
        
        for i in inputs[:self.max_inputs]:
            if i < self.max_inputs:
                input_encoding[i] = 1.0
        
        for i in outputs[:self.max_outputs]:
            if i < self.max_outputs:
                output_encoding[i] = 1.0
        
        # Flatten everything
        flat_repr = torch.cat([
            gate_encoding.flatten(),
            conn_matrix.flatten(),
            input_encoding,
            output_encoding
        ])
        
        return flat_repr.unsqueeze(0)  # Add batch dimension
    
    def decode_circuit(self, flat_repr: torch.Tensor) -> Dict:
        """Decode flat representation back to circuit structure."""
        flat_repr = flat_repr.squeeze(0)  # Remove batch dimension
        
        # Split representation
        gate_dim = self.max_gates * len(self.gate_types)
        conn_dim = self.max_gates * self.max_gates
        
        gate_part = flat_repr[:gate_dim].view(self.max_gates, len(self.gate_types))
        conn_part = flat_repr[gate_dim:gate_dim + conn_dim].view(self.max_gates, self.max_gates)
        input_part = flat_repr[gate_dim + conn_dim:gate_dim + conn_dim + self.max_inputs]
        output_part = flat_repr[gate_dim + conn_dim + self.max_inputs:]
        
        # Decode gates
        gate_indices = torch.argmax(gate_part, dim=1)
        gates = [self.gate_types[idx] for idx in gate_indices]
        
        # Decode connections (threshold and extract)
        connections = []
        conn_binary = (conn_part > 0.5).float()
        for i in range(self.max_gates):
            for j in range(self.max_gates):
                if conn_binary[i, j] > 0:
                    connections.append((i, j))
        
        # Decode I/O
        inputs = [i for i in range(self.max_inputs) if input_part[i] > 0.5]
        outputs = [i for i in range(self.max_outputs) if output_part[i] > 0.5]
        
        return {
            "gates": gates,
            "connections": connections,
            "inputs": inputs,
            "outputs": outputs,
            "gate_tensor": gate_indices,
            "connection_matrix": conn_binary
        }
    
    def simulate_circuit(
        self,
        circuit: Dict,
        test_inputs: List[List[bool]]
    ) -> List[List[bool]]:
        """Simulate circuit functionality."""
        
        gates = circuit["gates"]
        connections = circuit["connections"]
        circuit_inputs = circuit["inputs"]
        circuit_outputs = circuit["outputs"]
        
        results = []
        
        for test_case in test_inputs:
            # Initialize gate outputs
            gate_outputs = [None] * self.max_gates
            
            # Set primary inputs
            for i, inp_idx in enumerate(circuit_inputs):
                if i < len(test_case) and inp_idx < self.max_gates:
                    gate_outputs[inp_idx] = test_case[i]
            
            # Simulate gates in topological order (simplified)
            changed = True
            iterations = 0
            max_iterations = 20
            
            while changed and iterations < max_iterations:
                changed = False
                iterations += 1
                
                for gate_idx, gate_type in enumerate(gates):
                    if gate_outputs[gate_idx] is not None:
                        continue  # Already computed
                    
                    # Find inputs to this gate
                    gate_inputs = []
                    for src, dst in connections:
                        if dst == gate_idx and gate_outputs[src] is not None:
                            gate_inputs.append(gate_outputs[src])
                    
                    # Compute gate output if all inputs available
                    if len(gate_inputs) >= 1:  # At least one input
                        gate = LogicGate(gate_type)
                        gate.thermal_noise_level = self.thermal_noise_level
                        
                        if gate_type in ["NOT", "BUFFER"]:
                            if len(gate_inputs) >= 1:
                                gate_outputs[gate_idx] = gate.compute([gate_inputs[0]])
                                changed = True
                        else:
                            if len(gate_inputs) >= 2:
                                gate_outputs[gate_idx] = gate.compute(gate_inputs[:2])
                                changed = True
            
            # Extract outputs
            output_values = []
            for out_idx in circuit_outputs:
                if out_idx < len(gate_outputs) and gate_outputs[out_idx] is not None:
                    output_values.append(gate_outputs[out_idx])
                else:
                    output_values.append(False)  # Default
            
            results.append(output_values)
        
        return results
    
    def evaluate_circuit(self, circuit: Dict, truth_table: List[Tuple[List[bool], List[bool]]]) -> Dict[str, float]:
        """Evaluate circuit performance against truth table."""
        
        # Test functional correctness
        test_inputs = [inp for inp, _ in truth_table]
        expected_outputs = [out for _, out in truth_table]
        
        actual_outputs = self.simulate_circuit(circuit, test_inputs)
        
        # Compute correctness
        correct = 0
        total = len(truth_table)
        
        for expected, actual in zip(expected_outputs, actual_outputs):
            if len(expected) == len(actual):
                if all(e == a for e, a in zip(expected, actual)):
                    correct += 1
        
        correctness = correct / total if total > 0 else 0.0
        
        # Compute thermodynamic properties
        gate_tensor = circuit["gate_tensor"].unsqueeze(0)
        conn_tensor = circuit["connection_matrix"].unsqueeze(0)
        input_tensor = torch.zeros(1, self.max_inputs)
        output_tensor = torch.zeros(1, self.max_outputs)
        
        with torch.no_grad():
            thermo_props = self.circuit_thermo(
                gate_tensor, conn_tensor, input_tensor, output_tensor
            )
        
        # Compute complexity
        flat_repr = self.encode_circuit(
            circuit["gates"], circuit["connections"],
            circuit["inputs"], circuit["outputs"]
        )
        complexity = self.complexity_optimizer.compute_complexity(flat_repr).item()
        
        return {
            "correctness": correctness,
            "energy": thermo_props["energy"].item(),
            "entropy": thermo_props["entropy"].item(),
            "stability": thermo_props["stability"].item(),
            "delay": thermo_props["delay"].item(),
            "complexity": complexity
        }
    
    def evolve_logic(
        self,
        truth_table: List[Tuple[List[bool], List[bool]]],
        target_properties: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Evolve a circuit to implement the given truth table.
        
        Args:
            truth_table: List of (input, output) pairs
            target_properties: Optional target circuit properties
            
        Returns:
            Dictionary with evolved circuit and performance metrics
        """
        
        target_properties = target_properties or {
            "correctness": 0.95,
            "energy": 0.3,  # Lower is better
            "complexity": 0.6
        }
        
        # Determine I/O sizes from truth table
        if truth_table:
            input_size = len(truth_table[0][0])
            output_size = len(truth_table[0][1])
        else:
            input_size = self.max_inputs
            output_size = self.max_outputs
        
        # Create initial chaotic circuit
        chaos = self.encode_circuit(
            gates=["AND"] * self.max_gates,  # Start with simple gates
            connections=[],  # No connections initially
            inputs=list(range(min(input_size, self.max_inputs))),
            outputs=list(range(min(output_size, self.max_outputs)))
        )
        
        # Add noise to make it truly chaotic
        chaos = chaos + torch.randn_like(chaos) * 1.5
        
        # Evolve through thermodynamic process
        final_circuit_repr, trajectory = self.diffuser.evolve(chaos)
        
        # Decode final circuit
        final_circuit = self.decode_circuit(final_circuit_repr)
        
        # Evaluate performance
        performance = self.evaluate_circuit(final_circuit, truth_table)
        
        return {
            "circuit": final_circuit,
            "performance": performance,
            "evolution_trajectory": trajectory,
            "success_score": self._compute_success_score(performance, target_properties)
        }
    
    def _compute_success_score(self, performance: Dict[str, float], targets: Dict[str, float]) -> float:
        """Compute success score based on target properties."""
        score = 0.0
        count = 0
        
        for prop, target in targets.items():
            if prop in performance:
                if prop == "energy" or prop == "delay":
                    # Lower is better for these metrics
                    diff = max(0, performance[prop] - target)
                    prop_score = max(0.0, 1.0 - diff / target)
                else:
                    # Higher is better or target-based
                    diff = abs(performance[prop] - target)
                    prop_score = max(0.0, 1.0 - diff)
                
                score += prop_score
                count += 1
        
        return score / count if count > 0 else 0.0
    
    def design_adder(self, bit_width: int = 2) -> Dict:
        """Design an adder circuit using evolution."""
        
        # Generate truth table for adder
        truth_table = []
        for a in range(2**bit_width):
            for b in range(2**bit_width):
                sum_val = a + b
                
                # Convert to binary inputs
                a_bits = [(a >> i) & 1 == 1 for i in range(bit_width)]
                b_bits = [(b >> i) & 1 == 1 for i in range(bit_width)]
                inputs = a_bits + b_bits
                
                # Convert sum to binary outputs (including carry)
                output_bits = [(sum_val >> i) & 1 == 1 for i in range(bit_width + 1)]
                
                truth_table.append((inputs, output_bits))
        
        return self.evolve_logic(truth_table)
    
    def design_multiplexer(self, input_lines: int = 4) -> Dict:
        """Design a multiplexer circuit using evolution."""
        
        select_bits = int(np.ceil(np.log2(input_lines)))
        
        truth_table = []
        for sel in range(2**select_bits):
            for data_pattern in range(2**input_lines):
                # Select bits
                sel_inputs = [(sel >> i) & 1 == 1 for i in range(select_bits)]
                
                # Data inputs
                data_inputs = [(data_pattern >> i) & 1 == 1 for i in range(input_lines)]
                
                inputs = sel_inputs + data_inputs
                
                # Output is selected input
                if sel < input_lines:
                    output = [data_inputs[sel]]
                else:
                    output = [False]
                
                truth_table.append((inputs, output))
        
        return self.evolve_logic(truth_table)