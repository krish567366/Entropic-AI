"""
Molecule Evolution through Entropic AI

This module applies the Entropic AI framework to molecular design and evolution.
Unlike traditional approaches, molecules are evolved from atomic chaos through
thermodynamic principles, leading to stable folds with emergent properties.

Key Features:
- Evolve molecules from random atomic arrangements
- Optimize for stability, function, and complexity
- Thermodynamic folding simulation
- Drug-like property emergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from ..core.generative_diffuser import GenerativeDiffuser, OrderEvolver
from ..core.thermodynamic_network import ThermodynamicNetwork
from ..core.complexity_optimizer import KolmogorovOptimizer
from ..utils.entropy_utils import shannon_entropy, kolmogorov_complexity, thermodynamic_entropy, fisher_information


class MolecularThermodynamics(nn.Module):
    """
    Molecular system with thermodynamic properties for each atom/bond.
    """
    
    def __init__(
        self,
        max_atoms: int = 50,
        element_types: List[str] = ["C", "N", "O", "H", "S", "P"],
        bond_types: List[str] = ["single", "double", "triple", "aromatic"]
    ):
        super().__init__()
        
        self.max_atoms = max_atoms
        self.element_types = element_types
        self.bond_types = bond_types
        self.n_elements = len(element_types)
        self.n_bonds = len(bond_types)
        
        # Element embeddings with thermodynamic properties
        self.element_embeddings = nn.Embedding(self.n_elements, 64)
        
        # Atomic property predictors
        self.atomic_energy = nn.Linear(64, 1)  # Internal energy per atom
        self.atomic_entropy = nn.Linear(64, 1)  # Entropy contribution
        self.electronegativity = nn.Linear(64, 1)
        self.atomic_radius = nn.Linear(64, 1)
        
        # Bond energy calculator
        self.bond_energy = nn.Bilinear(64, 64, self.n_bonds)
        
        # Molecular stability predictor
        self.stability_net = nn.Sequential(
            nn.Linear(max_atoms * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Drug-likeness predictor
        self.druglike_net = nn.Sequential(
            nn.Linear(max_atoms * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def element_to_index(self, element: str) -> int:
        """Convert element symbol to index."""
        return self.element_types.index(element)
    
    def compute_molecular_energy(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        bonds: torch.Tensor
    ) -> torch.Tensor:
        """Compute total molecular energy."""
        batch_size = atom_types.shape[0]
        
        # Get element embeddings
        atom_embeddings = self.element_embeddings(atom_types)  # [batch, atoms, 64]
        
        # Atomic energies
        atomic_energies = self.atomic_energy(atom_embeddings).squeeze(-1)  # [batch, atoms]
        
        # Pairwise bond energies
        bond_energies = torch.zeros(batch_size)
        
        for b in range(batch_size):
            for i in range(self.max_atoms):
                for j in range(i+1, self.max_atoms):
                    if bonds[b, i, j] > 0:  # Bond exists
                        bond_type = bonds[b, i, j].long() - 1  # Convert to 0-indexed
                        if bond_type < self.n_bonds:
                            energy = self.bond_energy(
                                atom_embeddings[b, i], 
                                atom_embeddings[b, j]
                            )[bond_type]
                            bond_energies[b] += energy
        
        # Distance-based van der Waals interactions
        for b in range(batch_size):
            for i in range(self.max_atoms):
                for j in range(i+1, self.max_atoms):
                    dist = torch.norm(positions[b, i] - positions[b, j])
                    if dist > 0:
                        # Lennard-Jones potential (simplified)
                        vdw_energy = 4.0 * ((1.0/dist)**12 - (1.0/dist)**6)
                        bond_energies[b] += vdw_energy * 0.1  # Scale down
        
        total_energy = atomic_energies.sum(dim=1) + bond_energies
        
        return total_energy
    
    def compute_molecular_entropy(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute molecular entropy."""
        batch_size = atom_types.shape[0]
        
        # Get element embeddings
        atom_embeddings = self.element_embeddings(atom_types)
        
        # Atomic entropy contributions
        atomic_entropies = self.atomic_entropy(atom_embeddings).squeeze(-1)
        
        # Configurational entropy from positions
        config_entropy = torch.zeros(batch_size)
        
        for b in range(batch_size):
            # Entropy from positional distribution
            pos_var = torch.var(positions[b], dim=0).sum()
            config_entropy[b] = torch.log(1.0 + pos_var)
        
        total_entropy = atomic_entropies.sum(dim=1) + config_entropy
        
        return total_entropy
    
    def forward(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        bonds: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing all molecular properties."""
        
        energy = self.compute_molecular_energy(atom_types, positions, bonds)
        entropy = self.compute_molecular_entropy(atom_types, positions)
        
        # Get embeddings for other properties
        atom_embeddings = self.element_embeddings(atom_types)
        mol_embedding = atom_embeddings.view(atom_embeddings.shape[0], -1)
        
        stability = self.stability_net(mol_embedding).squeeze(-1)
        druglikeness = self.druglike_net(mol_embedding).squeeze(-1)
        
        return {
            "energy": energy,
            "entropy": entropy,
            "stability": stability,
            "druglikeness": druglikeness,
            "embeddings": atom_embeddings
        }


class MoleculeEvolution:
    """
    Main interface for evolving molecules using Entropic AI.
    """
    
    def __init__(
        self,
        max_atoms: int = 30,
        element_types: List[str] = ["C", "N", "O", "H"],
        target_properties: Optional[Dict[str, float]] = None,
        evolution_steps: int = 100
    ):
        self.max_atoms = max_atoms
        self.element_types = element_types
        self.target_properties = target_properties or {
            "stability": 0.8,
            "druglikeness": 0.7,
            "complexity": 0.6
        }
        self.evolution_steps = evolution_steps
        
        # Molecular thermodynamics model
        self.mol_thermo = MolecularThermodynamics(
            max_atoms=max_atoms,
            element_types=element_types
        )
        
        # Thermodynamic network for molecular evolution
        mol_repr_dim = max_atoms * (len(element_types) + 3 + max_atoms)  # atoms + positions + bonds
        
        self.thermo_network = ThermodynamicNetwork(
            input_dim=mol_repr_dim,
            hidden_dims=[256, 128],
            output_dim=mol_repr_dim,
            temperature=2.0
        )
        
        # Complexity optimizer for molecular complexity
        self.complexity_optimizer = KolmogorovOptimizer(
            target_complexity=target_properties.get("complexity", 0.6),
            method="entropy"
        )
        
        # Generative diffuser for evolution
        self.diffuser = GenerativeDiffuser(
            network=self.thermo_network,
            optimizer=self.complexity_optimizer,
            diffusion_steps=evolution_steps,
            initial_temperature=5.0,
            final_temperature=0.5
        )
    
    def encode_molecule(
        self,
        atom_types: List[str],
        positions: Optional[torch.Tensor] = None,
        bonds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode molecule into flat representation."""
        
        # Pad or truncate to max_atoms
        atom_indices = []
        for i in range(self.max_atoms):
            if i < len(atom_types):
                try:
                    idx = self.element_types.index(atom_types[i])
                except ValueError:
                    idx = 0  # Default to first element
            else:
                idx = 0  # Padding
            atom_indices.append(idx)
        
        atom_tensor = torch.tensor(atom_indices, dtype=torch.long)
        
        # Default positions if not provided
        if positions is None:
            positions = torch.randn(self.max_atoms, 3)
        else:
            # Pad or truncate positions
            if positions.shape[0] < self.max_atoms:
                padding = torch.randn(self.max_atoms - positions.shape[0], 3)
                positions = torch.cat([positions, padding], dim=0)
            else:
                positions = positions[:self.max_atoms]
        
        # Default bonds if not provided
        if bonds is None:
            bonds = torch.zeros(self.max_atoms, self.max_atoms)
        else:
            # Ensure correct size
            if bonds.shape[0] < self.max_atoms:
                new_bonds = torch.zeros(self.max_atoms, self.max_atoms)
                new_bonds[:bonds.shape[0], :bonds.shape[1]] = bonds
                bonds = new_bonds
            else:
                bonds = bonds[:self.max_atoms, :self.max_atoms]
        
        # Flatten everything into single vector
        flat_repr = torch.cat([
            F.one_hot(atom_tensor, len(self.element_types)).float().flatten(),
            positions.flatten(),
            bonds.flatten()
        ])
        
        return flat_repr.unsqueeze(0)  # Add batch dimension
    
    def decode_molecule(self, flat_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode flat representation back to molecular structure."""
        flat_repr = flat_repr.squeeze(0)  # Remove batch dimension
        
        # Split the representation
        atom_dim = self.max_atoms * len(self.element_types)
        pos_dim = self.max_atoms * 3
        bond_dim = self.max_atoms * self.max_atoms
        
        atom_onehot = flat_repr[:atom_dim].view(self.max_atoms, len(self.element_types))
        positions = flat_repr[atom_dim:atom_dim + pos_dim].view(self.max_atoms, 3)
        bonds = flat_repr[atom_dim + pos_dim:atom_dim + pos_dim + bond_dim].view(
            self.max_atoms, self.max_atoms
        )
        
        # Convert one-hot to atom types
        atom_types = torch.argmax(atom_onehot, dim=1)
        
        # Threshold bonds
        bonds = torch.round(torch.clamp(bonds, 0, len(self.mol_thermo.bond_types)))
        
        return atom_types, positions, bonds
    
    def evaluate_molecule(
        self,
        atom_types: torch.Tensor,
        positions: torch.Tensor,
        bonds: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate molecular properties."""
        
        # Compute thermodynamic properties
        with torch.no_grad():
            mol_props = self.mol_thermo(
                atom_types.unsqueeze(0),
                positions.unsqueeze(0),
                bonds.unsqueeze(0)
            )
        
        # Compute complexity
        flat_repr = self.encode_molecule(
            [self.element_types[i] for i in atom_types[:5]],  # First 5 atoms
            positions,
            bonds
        )
        complexity = self.complexity_optimizer.compute_complexity(flat_repr).item()
        
        return {
            "energy": mol_props["energy"].item(),
            "entropy": mol_props["entropy"].item(),
            "stability": mol_props["stability"].item(),
            "druglikeness": mol_props["druglikeness"].item(),
            "complexity": complexity
        }
    
    def evolve_from_atoms(
        self,
        elements: List[str],
        initial_structure: Optional[Dict] = None
    ) -> Dict:
        """
        Evolve a molecule from a list of elements.
        
        Args:
            elements: List of element symbols
            initial_structure: Optional initial molecular structure
            
        Returns:
            Dictionary with evolved molecule and properties
        """
        
        # Create initial chaotic state
        if initial_structure:
            chaos = self.encode_molecule(
                initial_structure.get("atoms", elements),
                initial_structure.get("positions"),
                initial_structure.get("bonds")
            )
        else:
            # Pure chaos - random arrangement
            chaos = self.encode_molecule(elements)
            # Add significant noise to make it truly chaotic
            chaos = chaos + torch.randn_like(chaos) * 2.0
        
        # Evolve through thermodynamic process
        final_structure, trajectory = self.diffuser.evolve(chaos)
        
        # Decode final structure
        atom_types, positions, bonds = self.decode_molecule(final_structure)
        
        # Evaluate properties
        properties = self.evaluate_molecule(atom_types, positions, bonds)
        
        # Convert atom types back to element symbols
        element_symbols = [self.element_types[i] for i in atom_types]
        
        return {
            "atoms": element_symbols,
            "positions": positions,
            "bonds": bonds,
            "properties": properties,
            "evolution_trajectory": trajectory,
            "success_score": self._compute_success_score(properties)
        }
    
    def _compute_success_score(self, properties: Dict[str, float]) -> float:
        """Compute how well the evolved molecule matches target properties."""
        score = 0.0
        count = 0
        
        for prop, target in self.target_properties.items():
            if prop in properties:
                # Score based on how close to target
                diff = abs(properties[prop] - target)
                prop_score = max(0.0, 1.0 - diff)
                score += prop_score
                count += 1
        
        return score / count if count > 0 else 0.0
    
    def batch_evolve(
        self,
        element_lists: List[List[str]],
        batch_size: int = 8
    ) -> List[Dict]:
        """Evolve multiple molecules in batches."""
        results = []
        
        for i in range(0, len(element_lists), batch_size):
            batch = element_lists[i:i + batch_size]
            
            for elements in batch:
                result = self.evolve_from_atoms(elements)
                results.append(result)
        
        return results
    
    def optimize_for_target(
        self,
        elements: List[str],
        target_properties: Dict[str, float],
        max_iterations: int = 10
    ) -> Dict:
        """
        Iteratively optimize molecule for specific target properties.
        """
        self.target_properties = target_properties
        
        best_molecule = None
        best_score = -1.0
        
        for iteration in range(max_iterations):
            # Evolve molecule
            result = self.evolve_from_atoms(elements)
            
            score = result["success_score"]
            
            if score > best_score:
                best_score = score
                best_molecule = result
                
            # Early stopping if target reached
            if score > 0.9:
                break
        
        return best_molecule
    
    def generate_drug_candidate(
        self,
        target_disease: str = "general",
        molecular_weight_range: Tuple[float, float] = (200, 500)
    ) -> Dict:
        """
        Generate drug candidate molecules using Entropic AI evolution.
        """
        
        # Common drug-like elements
        drug_elements = ["C", "N", "O", "H"]
        
        # Add some diversity based on target
        if target_disease == "cancer":
            drug_elements.extend(["S", "F"])
        elif target_disease == "neurological":
            drug_elements.extend(["P", "Cl"])
        
        # Generate diverse starting compositions
        compositions = [
            ["C"] * 8 + ["N"] * 2 + ["O"] * 2 + ["H"] * 10,
            ["C"] * 10 + ["N"] * 3 + ["O"] * 1 + ["H"] * 8,
            ["C"] * 6 + ["N"] * 1 + ["O"] * 3 + ["H"] * 12,
        ]
        
        best_candidate = None
        best_druglikeness = 0.0
        
        for composition in compositions:
            # Set drug-focused targets
            drug_targets = {
                "stability": 0.8,
                "druglikeness": 0.85,
                "complexity": 0.7
            }
            
            candidate = self.optimize_for_target(composition, drug_targets)
            
            if candidate and candidate["properties"]["druglikeness"] > best_druglikeness:
                best_druglikeness = candidate["properties"]["druglikeness"]
                best_candidate = candidate
        
        return best_candidate
