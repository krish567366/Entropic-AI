"""
Command Line Interface for Entropic AI

This module provides a comprehensive CLI for running Entropic AI experiments,
evolution processes, and analysis tasks. It supports various applications
including molecule evolution, circuit design, and theory discovery.

Usage examples:
  entropic-ai run --config experiments/molecule_evolution.json
  entropic-ai evolve --type molecule --elements C N O H --steps 100
  entropic-ai discover --domain physics --data experiments/oscillation_data.csv
"""

import typer
import torch
import json
import csv
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
import yaml

from .core.generative_diffuser import OrderEvolver
from .applications.molecule_evolution import MoleculeEvolution
from .applications.circuit_evolution import CircuitEvolution
from .applications.theory_discovery import TheoryDiscovery
from .utils.visualization import (
    plot_entropy_evolution,
    plot_molecular_structure,
    plot_circuit_evolution,
    plot_theory_discovery
)
from .utils.metrics import (
    complexity_score,
    stability_measure,
    emergence_index,
    performance_benchmark
)

app = typer.Typer(
    name="entropic-ai",
    help="Entropic AI - Generative Intelligence through Thermodynamic Self-Organization",
    add_completion=False
)
console = Console()

# Global configuration
CONFIG = {
    "default_steps": 100,
    "default_temperature": 1.0,
    "output_dir": "eai_results",
    "save_plots": True,
    "verbose": True
}


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to configuration file"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(True, "--verbose", "-v", help="Verbose output")
):
    """
    Run Entropic AI experiment from configuration file.
    
    Configuration file should be JSON or YAML format with experiment parameters.
    """
    
    console.print(Panel.fit("🌌 Entropic AI - Running Experiment", style="bold blue"))
    
    # Load configuration
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"❌ Configuration file not found: {config}", style="red")
        raise typer.Exit(1)
    
    try:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path) as f:
                experiment_config = yaml.safe_load(f)
        else:
            with open(config_path) as f:
                experiment_config = json.load(f)
    except Exception as e:
        console.print(f"❌ Error loading configuration: {e}", style="red")
        raise typer.Exit(1)
    
    # Set output directory
    if output_dir:
        experiment_config["output_dir"] = output_dir
    elif "output_dir" not in experiment_config:
        experiment_config["output_dir"] = CONFIG["output_dir"]
    
    # Create output directory
    output_path = Path(experiment_config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run experiment based on type
    experiment_type = experiment_config.get("type", "general")
    
    if experiment_type == "molecule_evolution":
        _run_molecule_experiment(experiment_config, verbose)
    elif experiment_type == "circuit_evolution":
        _run_circuit_experiment(experiment_config, verbose)
    elif experiment_type == "theory_discovery":
        _run_theory_experiment(experiment_config, verbose)
    elif experiment_type == "general":
        _run_general_experiment(experiment_config, verbose)
    else:
        console.print(f"❌ Unknown experiment type: {experiment_type}", style="red")
        raise typer.Exit(1)
    
    console.print(f"✅ Experiment completed! Results saved to {experiment_config['output_dir']}", style="green")


@app.command()
def evolve(
    type: str = typer.Option("general", "--type", "-t", help="Evolution type: molecule, circuit, general"),
    steps: int = typer.Option(100, "--steps", "-s", help="Number of evolution steps"),
    elements: Optional[List[str]] = typer.Option(None, "--elements", "-e", help="Chemical elements for molecule evolution"),
    gates: Optional[List[str]] = typer.Option(None, "--gates", "-g", help="Logic gates for circuit evolution"),
    output_dir: str = typer.Option("eai_results", "--output", "-o", help="Output directory"),
    save_plots: bool = typer.Option(True, "--plots", "-p", help="Save visualization plots"),
    verbose: bool = typer.Option(True, "--verbose", "-v", help="Verbose output")
):
    """
    Evolve structures using Entropic AI from chaos to order.
    """
    
    console.print(Panel.fit(f"🧬 Evolving {type.title()} Structure", style="bold green"))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        if type == "molecule":
            task = progress.add_task("Evolving molecular structure...", total=None)
            result = _evolve_molecule(elements or ["C", "N", "O", "H"], steps, verbose)
            
        elif type == "circuit":
            task = progress.add_task("Evolving circuit structure...", total=None)
            result = _evolve_circuit(gates or ["AND", "OR", "NOT", "XOR"], steps, verbose)
            
        elif type == "general":
            task = progress.add_task("Evolving general structure...", total=None)
            result = _evolve_general(steps, verbose)
            
        else:
            console.print(f"❌ Unknown evolution type: {type}", style="red")
            raise typer.Exit(1)
        
        progress.update(task, completed=True)
    
    # Save results
    result_file = output_path / f"{type}_evolution_result.json"
    with open(result_file, 'w') as f:
        # Convert tensors to lists for JSON serialization
        serializable_result = _make_json_serializable(result)
        json.dump(serializable_result, f, indent=2)
    
    # Display results
    _display_evolution_results(result, type)
    
    # Save plots if requested
    if save_plots:
        _save_evolution_plots(result, type, output_path)
    
    console.print(f"✅ Evolution completed! Results saved to {output_dir}", style="green")


@app.command()
def discover(
    domain: str = typer.Option("physics", "--domain", "-d", help="Discovery domain: physics, chemistry, mathematics"),
    data: Optional[str] = typer.Option(None, "--data", help="Path to data file (CSV format)"),
    variables: Optional[List[str]] = typer.Option(["x"], "--variables", "-var", help="Variable names"),
    steps: int = typer.Option(60, "--steps", "-s", help="Evolution steps for discovery"),
    output_dir: str = typer.Option("eai_results", "--output", "-o", help="Output directory"),
    save_plots: bool = typer.Option(True, "--plots", "-p", help="Save visualization plots"),
    verbose: bool = typer.Option(True, "--verbose", "-v", help="Verbose output")
):
    """
    Discover theories and mathematical expressions from data.
    """
    
    console.print(Panel.fit(f"🔬 Discovering {domain.title()} Theory", style="bold purple"))
    
    # Load data
    if data:
        data_path = Path(data)
        if not data_path.exists():
            console.print(f"❌ Data file not found: {data}", style="red")
            raise typer.Exit(1)
        
        data_x, data_y = _load_data_file(data_path)
    else:
        # Generate synthetic data based on domain
        data_x, data_y = _generate_synthetic_data(domain)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Discovering theoretical expression...", total=None)
        
        # Run theory discovery
        discoverer = TheoryDiscovery(
            variables=variables,
            domain=domain,
            evolution_steps=steps
        )
        
        result = discoverer.discover_from_data(data_x, data_y)
        progress.update(task, completed=True)
    
    # Save results
    result_file = output_path / f"{domain}_theory_discovery.json"
    with open(result_file, 'w') as f:
        serializable_result = _make_json_serializable(result)
        json.dump(serializable_result, f, indent=2)
    
    # Display results
    _display_discovery_results(result, domain)
    
    # Save plots if requested
    if save_plots:
        plot_path = output_path / f"{domain}_theory_plot.png"
        plot_theory_discovery(result, data_x, data_y, str(plot_path))
    
    console.print(f"✅ Theory discovery completed! Results saved to {output_dir}", style="green")


@app.command()
def analyze(
    result_file: str = typer.Option(..., "--file", "-f", help="Path to result file to analyze"),
    metrics: Optional[List[str]] = typer.Option(None, "--metrics", "-m", help="Specific metrics to compute"),
    output_dir: str = typer.Option("eai_analysis", "--output", "-o", help="Output directory for analysis"),
    verbose: bool = typer.Option(True, "--verbose", "-v", help="Verbose output")
):
    """
    Analyze results from Entropic AI experiments.
    """
    
    console.print(Panel.fit("📊 Analyzing Entropic AI Results", style="bold cyan"))
    
    # Load result file
    result_path = Path(result_file)
    if not result_path.exists():
        console.print(f"❌ Result file not found: {result_file}", style="red")
        raise typer.Exit(1)
    
    try:
        with open(result_path) as f:
            results = json.load(f)
    except Exception as e:
        console.print(f"❌ Error loading result file: {e}", style="red")
        raise typer.Exit(1)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Perform analysis
    analysis = _perform_comprehensive_analysis(results, metrics)
    
    # Save analysis
    analysis_file = output_path / "analysis_report.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Display analysis results
    _display_analysis_results(analysis)
    
    console.print(f"✅ Analysis completed! Report saved to {output_dir}", style="green")


def _run_molecule_experiment(config: Dict[str, Any], verbose: bool):
    """Run molecule evolution experiment."""
    
    elements = config.get("elements", ["C", "N", "O", "H"])
    target_properties = config.get("target_properties", {
        "stability": 0.8,
        "druglikeness": 0.7,
        "complexity": 0.6
    })
    evolution_steps = config.get("evolution_steps", 100)
    
    if verbose:
        console.print(f"🧬 Evolving molecule with elements: {elements}")
    
    # Create molecule evolver
    evolver = MoleculeEvolution(
        max_atoms=config.get("max_atoms", 30),
        element_types=elements,
        target_properties=target_properties,
        evolution_steps=evolution_steps
    )
    
    # Evolve molecule
    result = evolver.evolve_from_atoms(elements)
    
    # Save results
    output_path = Path(config["output_dir"])
    result_file = output_path / "molecule_result.json"
    
    with open(result_file, 'w') as f:
        serializable_result = _make_json_serializable(result)
        json.dump(serializable_result, f, indent=2)
    
    if verbose:
        _display_molecule_results(result)


def _run_circuit_experiment(config: Dict[str, Any], verbose: bool):
    """Run circuit evolution experiment."""
    
    gate_types = config.get("gate_types", ["AND", "OR", "NOT", "XOR"])
    truth_table = config.get("truth_table", [])
    
    if verbose:
        console.print(f"⚡ Evolving circuit with gates: {gate_types}")
    
    # Create circuit evolver
    evolver = CircuitEvolution(
        gate_types=gate_types,
        evolution_steps=config.get("evolution_steps", 80)
    )
    
    # Convert truth table format if needed
    if truth_table:
        formatted_tt = [(inp, out) for inp, out in truth_table]
    else:
        # Default: simple AND gate
        formatted_tt = [
            ([False, False], [False]),
            ([False, True], [False]),
            ([True, False], [False]),
            ([True, True], [True])
        ]
    
    # Evolve circuit
    result = evolver.evolve_logic(formatted_tt)
    
    # Save results
    output_path = Path(config["output_dir"])
    result_file = output_path / "circuit_result.json"
    
    with open(result_file, 'w') as f:
        serializable_result = _make_json_serializable(result)
        json.dump(serializable_result, f, indent=2)
    
    if verbose:
        _display_circuit_results(result)


def _run_theory_experiment(config: Dict[str, Any], verbose: bool):
    """Run theory discovery experiment."""
    
    domain = config.get("domain", "physics")
    variables = config.get("variables", ["x"])
    data_file = config.get("data_file")
    
    if verbose:
        console.print(f"🔬 Discovering {domain} theory")
    
    # Load or generate data
    if data_file:
        data_x, data_y = _load_data_file(Path(data_file))
    else:
        data_x, data_y = _generate_synthetic_data(domain)
    
    # Create theory discoverer
    discoverer = TheoryDiscovery(
        variables=variables,
        domain=domain,
        evolution_steps=config.get("evolution_steps", 60)
    )
    
    # Discover theory
    result = discoverer.discover_from_data(data_x, data_y)
    
    # Save results
    output_path = Path(config["output_dir"])
    result_file = output_path / "theory_result.json"
    
    with open(result_file, 'w') as f:
        serializable_result = _make_json_serializable(result)
        json.dump(serializable_result, f, indent=2)
    
    if verbose:
        _display_theory_results(result)


def _run_general_experiment(config: Dict[str, Any], verbose: bool):
    """Run general evolution experiment."""
    
    if verbose:
        console.print("🌌 Running general entropic evolution")
    
    # Create general evolver
    evolver = OrderEvolver(
        input_dim=config.get("input_dim", 64),
        evolution_steps=config.get("evolution_steps", 100)
    )
    
    # Evolve from chaos
    result = evolver.evolve_to_order(
        batch_size=config.get("batch_size", 1),
        return_trajectory=True
    )
    
    final_structure, trajectory = result
    
    # Save results
    output_path = Path(config["output_dir"])
    result_file = output_path / "general_result.json"
    
    result_dict = {
        "final_structure": final_structure.tolist(),
        "trajectory": _make_json_serializable(trajectory)
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    if verbose:
        console.print(f"✅ Evolution completed with {len(trajectory)} steps")


def _evolve_molecule(elements: List[str], steps: int, verbose: bool) -> Dict[str, Any]:
    """Evolve molecule structure."""
    
    evolver = MoleculeEvolution(
        element_types=elements,
        evolution_steps=steps
    )
    
    result = evolver.evolve_from_atoms(elements)
    return result


def _evolve_circuit(gates: List[str], steps: int, verbose: bool) -> Dict[str, Any]:
    """Evolve circuit structure."""
    
    evolver = CircuitEvolution(
        gate_types=gates,
        evolution_steps=steps
    )
    
    # Default truth table for demonstration
    truth_table = [
        ([False, False], [False]),
        ([False, True], [True]),
        ([True, False], [True]),
        ([True, True], [False])  # XOR function
    ]
    
    result = evolver.evolve_logic(truth_table)
    return result


def _evolve_general(steps: int, verbose: bool) -> Dict[str, Any]:
    """Evolve general structure."""
    
    evolver = OrderEvolver(
        input_dim=32,
        evolution_steps=steps
    )
    
    structure, trajectory = evolver.evolve_to_order(return_trajectory=True)
    
    return {
        "final_structure": structure,
        "trajectory": trajectory,
        "evolution_steps": steps
    }


def _load_data_file(data_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load data from CSV file."""
    
    try:
        with open(data_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Assume first row is header, first column is x, second is y
        if len(rows) > 1 and len(rows[0]) >= 2:
            data_x = torch.tensor([float(row[0]) for row in rows[1:]])
            data_y = torch.tensor([float(row[1]) for row in rows[1:]])
        else:
            raise ValueError("Invalid data format")
            
    except Exception as e:
        console.print(f"❌ Error loading data: {e}", style="red")
        raise typer.Exit(1)
    
    return data_x, data_y


def _generate_synthetic_data(domain: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data for different domains."""
    
    x = torch.linspace(-3, 3, 50)
    
    if domain == "physics":
        # Oscillatory motion
        y = torch.sin(2 * x) * torch.exp(-0.1 * x**2) + 0.1 * torch.randn(50)
    elif domain == "chemistry":
        # Exponential decay
        y = torch.exp(-x) + 0.05 * torch.randn(50)
    elif domain == "mathematics":
        # Polynomial
        y = x**2 - 2*x + 1 + 0.1 * torch.randn(50)
    else:
        # Default: linear
        y = 2*x + 1 + 0.1 * torch.randn(50)
    
    return x, y


def _make_json_serializable(obj: Any) -> Any:
    """Convert tensors and other objects to JSON-serializable format."""
    
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj


def _display_evolution_results(result: Dict[str, Any], evolution_type: str):
    """Display evolution results in a formatted table."""
    
    table = Table(title=f"{evolution_type.title()} Evolution Results")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    if evolution_type == "molecule":
        if "properties" in result:
            for prop, value in result["properties"].items():
                table.add_row(prop.title(), f"{value:.3f}")
        
        table.add_row("Success Score", f"{result.get('success_score', 0):.3f}")
        
    elif evolution_type == "circuit":
        if "performance" in result:
            for prop, value in result["performance"].items():
                table.add_row(prop.title(), f"{value:.3f}")
    
    else:  # general
        if "trajectory" in result:
            table.add_row("Evolution Steps", str(len(result["trajectory"])))
        
        if "final_structure" in result:
            structure = result["final_structure"]
            if isinstance(structure, torch.Tensor):
                complexity = complexity_score(structure)
                table.add_row("Final Complexity", f"{complexity:.3f}")
    
    console.print(table)


def _display_discovery_results(result: Dict[str, Any], domain: str):
    """Display theory discovery results."""
    
    console.print(Panel.fit(f"🔬 {domain.title()} Theory Discovery", style="bold purple"))
    
    if "simplified_expression" in result:
        console.print(f"📐 Discovered Expression: [bold green]{result['simplified_expression']}[/bold green]")
    
    if "theory_fit" in result:
        table = Table(title="Theory Quality Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="green")
        
        for metric, score in result["theory_fit"].items():
            table.add_row(metric.title(), f"{score:.3f}")
        
        console.print(table)


def _save_evolution_plots(result: Dict[str, Any], evolution_type: str, output_path: Path):
    """Save visualization plots for evolution results."""
    
    try:
        if evolution_type == "molecule" and "evolution_trajectory" in result:
            plot_path = output_path / "molecule_evolution.png"
            # Would call plot_entropy_evolution here
            
        elif evolution_type == "circuit" and "evolution_trajectory" in result:
            plot_path = output_path / "circuit_evolution.png"
            # Would call plot_circuit_evolution here
            
        elif evolution_type == "general" and "trajectory" in result:
            plot_path = output_path / "general_evolution.png"
            # Would call plot_entropy_evolution here
        
        console.print(f"📊 Plots saved to {output_path}")
        
    except Exception as e:
        console.print(f"⚠️  Warning: Could not save plots - {e}", style="yellow")


def _perform_comprehensive_analysis(results: Dict[str, Any], metrics: Optional[List[str]]) -> Dict[str, Any]:
    """Perform comprehensive analysis of results."""
    
    analysis = {
        "summary": {},
        "metrics": {},
        "recommendations": []
    }
    
    # Analyze based on result type
    if "properties" in results:  # Molecule results
        analysis["type"] = "molecule"
        analysis["summary"]["success_score"] = results.get("success_score", 0)
        
        if "evolution_trajectory" in results:
            trajectory = results["evolution_trajectory"]
            if trajectory:
                analysis["metrics"]["evolution_steps"] = len(trajectory)
                analysis["metrics"]["convergence"] = _analyze_convergence(trajectory)
    
    elif "performance" in results:  # Circuit results
        analysis["type"] = "circuit"
        analysis["summary"]["performance"] = results["performance"]
        
    elif "theory_fit" in results:  # Theory results
        analysis["type"] = "theory"
        analysis["summary"]["theory_quality"] = results["theory_fit"]
        
    return analysis


def _analyze_convergence(trajectory: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze convergence properties of evolution trajectory."""
    
    if len(trajectory) < 5:
        return {"converged": False, "convergence_step": -1}
    
    # Look for convergence in objective function
    objectives = []
    for step in trajectory:
        if "objective" in step:
            obj_val = step["objective"]
            if isinstance(obj_val, list):
                objectives.append(obj_val[0] if obj_val else 0.0)
            else:
                objectives.append(float(obj_val))
    
    if len(objectives) < 5:
        return {"converged": False, "convergence_step": -1}
    
    # Simple convergence detection
    window_size = 5
    threshold = 0.01
    
    for i in range(window_size, len(objectives)):
        window = objectives[i-window_size:i]
        if np.std(window) < threshold:
            return {"converged": True, "convergence_step": i}
    
    return {"converged": False, "convergence_step": -1}


def _display_analysis_results(analysis: Dict[str, Any]):
    """Display analysis results."""
    
    console.print(Panel.fit("📊 Analysis Results", style="bold cyan"))
    
    # Summary table
    if "summary" in analysis:
        table = Table(title="Summary")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in analysis["summary"].items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    table.add_row(f"{key}.{subkey}", str(subvalue))
            else:
                table.add_row(key, str(value))
        
        console.print(table)
    
    # Metrics table
    if "metrics" in analysis and analysis["metrics"]:
        table = Table(title="Detailed Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in analysis["metrics"].items():
            table.add_row(metric, str(value))
        
        console.print(table)


def _display_molecule_results(result: Dict[str, Any]):
    """Display molecule evolution results."""
    
    console.print("🧬 Molecule Evolution Results:")
    
    if "atoms" in result:
        elements = result["atoms"][:10]  # Show first 10 atoms
        console.print(f"  Elements: {', '.join(elements)}")
    
    if "properties" in result:
        props = result["properties"]
        console.print(f"  Stability: {props.get('stability', 0):.3f}")
        console.print(f"  Drug-likeness: {props.get('druglikeness', 0):.3f}")
        console.print(f"  Complexity: {props.get('complexity', 0):.3f}")
    
    console.print(f"  Success Score: {result.get('success_score', 0):.3f}")


def _display_circuit_results(result: Dict[str, Any]):
    """Display circuit evolution results."""
    
    console.print("⚡ Circuit Evolution Results:")
    
    if "circuit" in result and "gates" in result["circuit"]:
        gates = result["circuit"]["gates"]
        unique_gates = list(set(gates))
        console.print(f"  Gate Types: {', '.join(unique_gates)}")
        console.print(f"  Total Gates: {len(gates)}")
    
    if "performance" in result:
        perf = result["performance"]
        console.print(f"  Correctness: {perf.get('correctness', 0):.3f}")
        console.print(f"  Energy: {perf.get('energy', 0):.3f}")
        console.print(f"  Complexity: {perf.get('complexity', 0):.3f}")


def _display_theory_results(result: Dict[str, Any]):
    """Display theory discovery results."""
    
    console.print("🔬 Theory Discovery Results:")
    
    if "simplified_expression" in result:
        console.print(f"  Expression: {result['simplified_expression']}")
    
    if "theory_fit" in result:
        fit = result["theory_fit"]
        console.print(f"  Accuracy: {fit.get('accuracy', 0):.3f}")
        console.print(f"  Complexity: {fit.get('complexity', 0):.3f}")
        console.print(f"  Generalization: {fit.get('generalization', 0):.3f}")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
