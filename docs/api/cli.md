# Command Line Interface (CLI) Reference

The Entropic AI CLI provides convenient command-line access to all major functionality. The CLI is designed for both interactive exploration and automated workflows, supporting batch processing and pipeline integration.

## Installation and Setup

### CLI Installation

The CLI is automatically installed with the Entropic AI package:

```bash
pip install entropic-ai
```

### Verify Installation

```bash
entropic-ai --version
# Output: Entropic AI CLI v0.1.0

entropic-ai --help
# Display comprehensive help information
```

### Configuration

Set up global configuration:

```bash
# Initialize configuration directory
entropic-ai config init

# Set default parameters
entropic-ai config set thermal.initial_temperature 1.0
entropic-ai config set thermal.cooling_rate 0.95
entropic-ai config set evolution.max_iterations 1000

# View current configuration
entropic-ai config show
```

## Core Commands

### `entropic-ai run`

Execute thermodynamic evolution with specified configuration.

```bash
# Basic usage
entropic-ai run --config config.yaml --output results.json

# Advanced usage with custom parameters
entropic-ai run \
  --config optimization_config.yaml \
  --output experiment_results/ \
  --temperature 5.0 \
  --cooling-rate 0.98 \
  --max-iterations 2000 \
  --parallel-processes 4 \
  --gpu-acceleration \
  --verbose
```

**Options:**

- `--config, -c`: Configuration file path (YAML/JSON)
- `--output, -o`: Output directory or file
- `--temperature, -T`: Initial temperature override
- `--cooling-rate`: Temperature cooling rate override
- `--max-iterations, -i`: Maximum evolution iterations
- `--convergence-threshold`: Convergence criteria override
- `--parallel-processes, -p`: Number of parallel processes
- `--gpu-acceleration`: Enable GPU acceleration
- `--seed`: Random seed for reproducibility
- `--verbose, -v`: Verbose output
- `--quiet, -q`: Minimal output
- `--dry-run`: Validate configuration without execution

**Examples:**

```bash
# Optimize a function from configuration
entropic-ai run --config examples/sphere_optimization.yaml

# Run with custom parameters
entropic-ai run -c config.yaml -T 10.0 -i 5000 --gpu-acceleration

# Batch processing with multiple configurations
entropic-ai run --config-dir configs/ --output-dir results/ --batch
```

### `entropic-ai evolve`

Interactive evolution for specific problem types.

```bash
# Circuit evolution
entropic-ai evolve circuit \
  --truth-table examples/adder_4bit.tt \
  --objectives area,power,delay \
  --technology-node 14nm \
  --output circuit_design.v

# Molecule evolution  
entropic-ai evolve molecule \
  --target-properties examples/drug_properties.json \
  --elements C,N,O,H,S \
  --output molecule_result.sdf

# Law discovery
entropic-ai evolve law \
  --data experimental_data.csv \
  --target-variable period \
  --operators add,mul,div,pow,sqrt \
  --output discovered_laws.json
```

**Circuit Evolution Options:**

- `--truth-table`: Truth table file path
- `--objectives`: Optimization objectives (comma-separated)
- `--technology-node`: Target technology node
- `--max-gates`: Maximum number of gates
- `--noise-model`: Noise model configuration
- `--verification`: Enable formal verification

**Molecule Evolution Options:**

- `--target-properties`: Target molecular properties (JSON)
- `--elements`: Available chemical elements
- `--constraints`: Chemical constraints file
- `--property-predictors`: Property prediction models
- `--starting-molecule`: Initial molecular structure

**Law Discovery Options:**

- `--data`: Experimental data file (CSV/JSON)
- `--target-variable`: Variable to find laws for
- `--operators`: Available mathematical operators
- `--dimensional-analysis`: Enable dimensional consistency
- `--physics-constraints`: Physics principles to enforce

### `entropic-ai optimize`

General-purpose optimization interface.

```bash
# Continuous optimization
entropic-ai optimize continuous \
  --function "lambda x: sum(x**2)" \
  --bounds "(-5,5)" \
  --dimensions 10 \
  --output optimization_result.json

# Combinatorial optimization
entropic-ai optimize combinatorial \
  --problem-type tsp \
  --input cities.json \
  --encoding permutation \
  --output tour_result.json

# Multi-objective optimization
entropic-ai optimize multi-objective \
  --objectives objective1.py,objective2.py \
  --bounds bounds.json \
  --pareto-front-size 100 \
  --output pareto_solutions.json
```

**Options:**

- `--function`: Objective function (Python expression or file)
- `--bounds`: Variable bounds
- `--dimensions`: Problem dimensionality
- `--problem-type`: Combinatorial problem type
- `--encoding`: Solution encoding scheme
- `--objectives`: Multiple objective functions
- `--constraints`: Constraint specifications
- `--algorithm-variant`: Algorithm variant to use

### `entropic-ai discover`

Scientific discovery workflows.

```bash
# Discover laws from data
entropic-ai discover laws \
  --data pendulum_experiment.csv \
  --variables length,period,gravity \
  --output discovered_laws.json \
  --complexity-limit 10

# Pattern discovery
entropic-ai discover patterns \
  --data dataset.csv \
  --pattern-types clustering,anomaly \
  --output patterns.json

# Causal discovery
entropic-ai discover causality \
  --data observational_data.csv \
  --method thermodynamic-causal \
  --output causal_graph.json
```

**Options:**

- `--data`: Input dataset
- `--variables`: Variables to analyze
- `--complexity-limit`: Maximum expression complexity
- `--pattern-types`: Types of patterns to discover
- `--method`: Discovery method variant
- `--validation-split`: Data split for validation

### `entropic-ai generate`

Content and structure generation.

```bash
# Generate molecular structures
entropic-ai generate molecules \
  --properties molecular_weight:300-500,logP:2-4 \
  --count 100 \
  --output generated_molecules.sdf

# Generate text content
entropic-ai generate text \
  --prompt "Scientific abstract about thermodynamics" \
  --length 200 \
  --style academic \
  --output generated_text.txt

# Generate code
entropic-ai generate code \
  --language python \
  --specification requirements.txt \
  --output generated_code.py
```

**Options:**

- `--properties`: Target properties for generation
- `--count`: Number of items to generate
- `--prompt`: Generation prompt or seed
- `--length`: Target length/size
- `--style`: Generation style
- `--language`: Programming language (for code generation)
- `--specification`: Requirements specification

## Analysis Commands

### `entropic-ai analyze`

Analyze results and system behavior.

```bash
# Analyze evolution results
entropic-ai analyze evolution \
  --results evolution_results.json \
  --metrics convergence,efficiency,quality \
  --output analysis_report.html

# Analyze energy landscape
entropic-ai analyze landscape \
  --function objective_function.py \
  --bounds bounds.json \
  --resolution 100 \
  --output landscape_analysis.pdf

# Performance analysis
entropic-ai analyze performance \
  --algorithm-results results/ \
  --baseline-results baseline/ \
  --output performance_report.html
```

**Options:**

- `--results`: Results file or directory to analyze
- `--metrics`: Analysis metrics to compute
- `--baseline-results`: Baseline results for comparison
- `--resolution`: Analysis resolution
- `--format`: Output format (html, pdf, json)

### `entropic-ai visualize`

Create visualizations of results and processes.

```bash
# Evolution trace visualization
entropic-ai visualize evolution \
  --data evolution_trace.json \
  --metrics energy,entropy,temperature \
  --output evolution_plot.pdf

# Energy landscape visualization
entropic-ai visualize landscape \
  --function objective_function.py \
  --bounds "(-2,2),(-2,2)" \
  --trajectory evolution_trace.json \
  --output landscape_3d.pdf

# Interactive dashboard
entropic-ai visualize dashboard \
  --data experiment_results/ \
  --port 8080 \
  --host localhost
```

**Options:**

- `--data`: Data file or directory
- `--metrics`: Metrics to visualize
- `--trajectory`: Evolution trajectory to overlay
- `--port`: Port for interactive visualizations
- `--host`: Host for web interface
- `--format`: Output format
- `--style`: Visualization style

## Utility Commands

### `entropic-ai config`

Configuration management.

```bash
# Initialize configuration
entropic-ai config init [--template optimization|discovery|generation]

# Show current configuration
entropic-ai config show [--section thermal|evolution|complexity]

# Set configuration values
entropic-ai config set thermal.initial_temperature 2.0
entropic-ai config set evolution.max_iterations 5000

# Reset to defaults
entropic-ai config reset [--section section_name]

# Validate configuration
entropic-ai config validate --config custom_config.yaml
```

### `entropic-ai benchmark`

Performance benchmarking.

```bash
# Run standard benchmarks
entropic-ai benchmark run \
  --suite optimization \
  --algorithms thermodynamic,gradient,evolutionary \
  --output benchmark_results.json

# Custom benchmark
entropic-ai benchmark custom \
  --problems problem_suite.json \
  --algorithm-config algorithm_configs/ \
  --trials 10 \
  --output custom_benchmark.json

# Compare algorithms
entropic-ai benchmark compare \
  --results results1.json,results2.json \
  --metrics accuracy,efficiency,convergence \
  --output comparison_report.html
```

**Options:**

- `--suite`: Standard benchmark suite
- `--algorithms`: Algorithms to benchmark
- `--problems`: Custom problem suite
- `--trials`: Number of trial runs
- `--metrics`: Evaluation metrics
- `--statistical-tests`: Statistical significance tests

### `entropic-ai validate`

Validation and verification utilities.

```bash
# Validate configuration files
entropic-ai validate config --file config.yaml --schema optimization

# Validate results
entropic-ai validate results \
  --file results.json \
  --expected-format evolution_result \
  --check-completeness

# Validate data format
entropic-ai validate data \
  --file dataset.csv \
  --schema data_schema.json \
  --check-quality
```

### `entropic-ai convert`

Data format conversion utilities.

```bash
# Convert configuration formats
entropic-ai convert config \
  --input config.yaml \
  --output config.json \
  --format json

# Convert results format
entropic-ai convert results \
  --input results.pkl \
  --output results.hdf5 \
  --format hdf5

# Convert data format
entropic-ai convert data \
  --input data.csv \
  --output data.json \
  --format json
```

## Configuration Files

### Basic Configuration

```yaml
# config.yaml
thermal:
  initial_temperature: 1.0
  final_temperature: 0.01
  cooling_rate: 0.95
  cooling_schedule: exponential

evolution:
  max_iterations: 1000
  convergence_threshold: 1e-6
  population_size: 50
  elite_fraction: 0.1

complexity:
  target_complexity: 0.7
  complexity_weight: 0.1
  measures: [kolmogorov, logical_depth]

output:
  format: json
  verbose: true
  save_history: true
  checkpoint_interval: 100
```

### Advanced Configuration

```yaml
# advanced_config.yaml
thermal:
  adaptive_temperature: true
  temperature_schedule:
    type: adaptive
    parameters:
      adaptation_rate: 0.1
      target_acceptance: 0.44
  
evolution:
  algorithm_variant: hybrid
  hybrid_config:
    thermodynamic_weight: 0.7
    gradient_weight: 0.3
    switching_criteria: landscape_roughness
  
parallelization:
  enabled: true
  num_processes: 4
  gpu_acceleration: true
  distributed: false

optimization:
  multi_objective:
    enabled: true
    objectives: [accuracy, complexity, robustness]
    weights: [0.5, 0.3, 0.2]
    pareto_analysis: true
```

## Environment Variables

Control CLI behavior through environment variables:

```bash
# Set default configuration directory
export ENTROPIC_AI_CONFIG_DIR=/path/to/configs

# Set default output directory
export ENTROPIC_AI_OUTPUT_DIR=/path/to/outputs

# Enable GPU by default
export ENTROPIC_AI_GPU_ENABLED=true

# Set logging level
export ENTROPIC_AI_LOG_LEVEL=INFO

# Set number of parallel processes
export ENTROPIC_AI_NUM_PROCESSES=8
```

## Scripting and Automation

### Batch Processing Scripts

```bash
#!/bin/bash
# batch_optimization.sh

# Run multiple optimization experiments
for config in configs/optimization_*.yaml; do
    output_dir="results/$(basename $config .yaml)"
    entropic-ai run --config "$config" --output "$output_dir" --verbose
done

# Generate summary report
entropic-ai analyze performance \
  --algorithm-results results/ \
  --output summary_report.html
```

### Python Integration

```python
# python_integration.py
import subprocess
import json

def run_entropic_ai(config_file, output_file):
    """Run Entropic AI from Python script."""
    
    cmd = [
        'entropic-ai', 'run',
        '--config', config_file,
        '--output', output_file,
        '--format', 'json'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        with open(output_file, 'r') as f:
            return json.load(f)
    else:
        raise RuntimeError(f"CLI command failed: {result.stderr}")

# Usage
results = run_entropic_ai('config.yaml', 'results.json')
print(f"Best solution: {results['best_solution']}")
```

## Pipeline Integration

### Integration with Make

```makefile
# Makefile
.PHONY: all optimize analyze visualize clean

CONFIG_DIR = configs
RESULTS_DIR = results
PLOTS_DIR = plots

all: optimize analyze visualize

optimize:
	@mkdir -p $(RESULTS_DIR)
	@for config in $(CONFIG_DIR)/*.yaml; do \
		output=$(RESULTS_DIR)/$$(basename $$config .yaml).json; \
		entropic-ai run --config $$config --output $$output; \
	done

analyze: optimize
	entropic-ai analyze performance \
		--algorithm-results $(RESULTS_DIR) \
		--output $(RESULTS_DIR)/analysis.html

visualize: analyze
	@mkdir -p $(PLOTS_DIR)
	entropic-ai visualize evolution \
		--data $(RESULTS_DIR) \
		--output $(PLOTS_DIR)

clean:
	rm -rf $(RESULTS_DIR) $(PLOTS_DIR)
```

### Integration with CI/CD

```yaml
# .github/workflows/entropic-ai.yml
name: Entropic AI Pipeline

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install Entropic AI
      run: pip install entropic-ai
    
    - name: Run Benchmarks
      run: |
        entropic-ai benchmark run \
          --suite optimization \
          --output benchmark_results.json
    
    - name: Generate Report
      run: |
        entropic-ai analyze performance \
          --algorithm-results benchmark_results.json \
          --output benchmark_report.html
    
    - name: Upload Results
      uses: actions/upload-artifact@v2
      with:
        name: benchmark-results
        path: |
          benchmark_results.json
          benchmark_report.html
```

## Troubleshooting

### Common Issues

**1. Configuration Validation Errors**

```bash
# Check configuration syntax
entropic-ai validate config --file config.yaml

# Use verbose mode for detailed error messages
entropic-ai run --config config.yaml --verbose
```

**2. Memory Issues with Large Problems**

```bash
# Use streaming mode for large datasets
entropic-ai run --config config.yaml --streaming-mode

# Reduce memory usage
entropic-ai run --config config.yaml --memory-efficient
```

**3. GPU Acceleration Issues**

```bash
# Check GPU availability
entropic-ai config gpu-info

# Force CPU mode
entropic-ai run --config config.yaml --force-cpu
```

### Debug Mode

Enable comprehensive debugging:

```bash
# Full debug mode
entropic-ai run --config config.yaml --debug --log-level DEBUG

# Profile performance
entropic-ai run --config config.yaml --profile --output-profile profile.json
```

### Getting Help

```bash
# General help
entropic-ai --help

# Command-specific help
entropic-ai run --help
entropic-ai evolve --help

# List available examples
entropic-ai examples list

# Show example configuration
entropic-ai examples show optimization.sphere
```

The CLI provides a comprehensive interface for all Entropic AI functionality, supporting everything from simple optimization tasks to complex multi-objective evolution workflows. The modular design allows for easy integration into existing workflows and automation pipelines.
