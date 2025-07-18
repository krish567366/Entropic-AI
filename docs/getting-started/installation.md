# Installation Guide

This guide covers everything you need to install and set up Entropic AI on your system.

## Quick Installation

### From PyPI (Recommended)

The easiest way to install Entropic AI is through PyPI:

```bash
pip install entropic-ai
```

### With Optional Dependencies

For enhanced functionality, install with optional dependencies:

```bash
# For GPU acceleration
pip install entropic-ai[gpu]

# For molecular applications
pip install entropic-ai[molecules]

# For circuit design
pip install entropic-ai[circuits]

# For all features
pip install entropic-ai[full]
```

## Prerequisites

### System Requirements

**Minimum Requirements:**

- Python 3.9 or higher
- 8GB RAM
- 2GB available disk space
- CPU with AVX2 support (Intel Sandy Bridge+ or AMD Bulldozer+)

**Recommended Requirements:**

- Python 3.10 or higher
- 16GB+ RAM
- 5GB available disk space
- GPU with CUDA 11.8+ support
- SSD storage for better I/O performance

### Python Environment

We strongly recommend using a virtual environment:

```bash
# Using conda (recommended)
conda create -n entropic-ai python=3.10
conda activate entropic-ai

# Using venv
python -m venv entropic-ai-env
source entropic-ai-env/bin/activate  # Linux/Mac
# or
entropic-ai-env\Scripts\activate     # Windows
```

## Installation Methods

### Method 1: PyPI Installation

Install the latest stable release:

```bash
pip install entropic-ai
```

Verify installation:

```bash
python -c "import entropic-ai; print(entropic-ai.__version__)"
```

### Method 2: Development Installation

For the latest features and development:

```bash
git clone https://github.com/krish567366/Entropic-AI.git
cd Entropic-AI
pip install -e .
```

This installs in "editable" mode, allowing you to modify the source code.

### Method 3: Docker Installation

Run Entropic AI in a containerized environment:

```bash
docker pull krish567366/entropic-ai:latest
docker run -it krish567366/entropic-ai:latest
```

For GPU support:

```bash
docker run --gpus all -it krish567366/entropic-ai:gpu
```

## GPU Support

### CUDA Installation

For GPU acceleration, install CUDA-compatible PyTorch:

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Support

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name()}")
```

## Optional Dependencies

### Molecular Modeling

For molecular design applications:

```bash
pip install rdkit-pypi py3dmol biopython
```

### Circuit Design

For electronic circuit applications:

```bash
pip install ngspice-python schemdraw electronics
```

### Visualization

For enhanced plotting and visualization:

```bash
pip install plotly bokeh seaborn
```

### Scientific Computing

For advanced scientific applications:

```bash
pip install sympy numba cupy-cuda118
```

## Platform-Specific Instructions

### Windows

1. **Install Visual Studio Build Tools:**
   - Download from [Microsoft](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   - Install C++ build tools

2. **Install Entropic AI:**

   ```cmd
   pip install entropic-ai
   ```

3. **For GPU support:**

   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### macOS

1. **Install Xcode Command Line Tools:**

   ```bash
   xcode-select --install
   ```

2. **Install using Homebrew Python (recommended):**

   ```bash
   brew install python@3.10
   pip3.10 install entropic-ai
   ```

3. **For M1/M2 Macs:**

   ```bash
   # Use MPS backend for GPU acceleration
   pip install torch torchvision torchaudio
   ```

### Linux (Ubuntu/Debian)

1. **Install system dependencies:**

   ```bash
   sudo apt update
   sudo apt install python3-pip python3-dev build-essential
   ```

2. **Install Entropic AI:**

   ```bash
   pip3 install entropic-ai
   ```

3. **For GPU support:**

   ```bash
   # Install NVIDIA drivers and CUDA
   sudo apt install nvidia-driver-525 nvidia-cuda-toolkit
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Linux (CentOS/RHEL)

1. **Install system dependencies:**

   ```bash
   sudo yum install python3-pip python3-devel gcc gcc-c++
   ```

2. **Install Entropic AI:**

   ```bash
   pip3 install entropic-ai
   ```

## Configuration

### Environment Variables

Set up environment variables for optimal performance:

```bash
# Bash/Zsh
export entropic-ai_CACHE_DIR="$HOME/.entropic-ai/cache"
export entropic-ai_DATA_DIR="$HOME/.entropic-ai/data"
export entropic-ai_NUM_THREADS="8"

# Windows CMD
set entropic-ai_CACHE_DIR=%USERPROFILE%\.entropic-ai\cache
set entropic-ai_DATA_DIR=%USERPROFILE%\.entropic-ai\data
set entropic-ai_NUM_THREADS=8
```

### Configuration File

Create a configuration file at `~/.entropic-ai/config.yaml`:

```yaml
# Entropic AI Configuration
general:
  log_level: INFO
  cache_enabled: true
  num_threads: auto

thermodynamics:
  default_temperature: 1.0
  cooling_schedule: exponential
  entropy_regularization: 0.1

performance:
  use_gpu: auto
  memory_fraction: 0.8
  mixed_precision: true

applications:
  molecules:
    force_field: universal
    implicit_solvent: true
  circuits:
    simulator: ngspice
    optimization_level: 2
```

## Verification

### Basic Installation Test

```python
import entropic-ai
from entropic-ai import EntropicNetwork, GenerativeDiffuser
import torch

print(f"Entropic AI version: {entropic-ai.__version__}")

# Create a simple network
network = EntropicNetwork(nodes=32)
diffuser = GenerativeDiffuser(network)

# Test evolution
chaos = torch.randn(1, 32)
order = diffuser.evolve(chaos)

print("✅ Basic installation test passed!")
```

### Performance Benchmark

```python
from entropic-ai.benchmarks import installation_benchmark

# Run installation benchmark
results = installation_benchmark()
print(f"Performance score: {results.score}")
print(f"GPU acceleration: {results.gpu_available}")
print(f"All tests passed: {results.all_passed}")
```

### Application Tests

```python
# Test molecular evolution
from entropic-ai.applications import MoleculeEvolution
mol_evolver = MoleculeEvolution()
print("✅ Molecular evolution available")

# Test circuit design
from entropic-ai.applications import CircuitEvolution
circuit_evolver = CircuitEvolution()
print("✅ Circuit evolution available")

# Test theory discovery
from entropic-ai.applications import TheoryDiscovery
theory_evolver = TheoryDiscovery()
print("✅ Theory discovery available")
```

## Troubleshooting

### Common Issues

#### **ImportError: No module named 'entropic-ai'**

- Solution: Ensure you're in the correct virtual environment
- Check: `pip list | grep entropic-ai`

#### **CUDA out of memory**

- Solution: Reduce batch size or use CPU
- Set: `export CUDA_VISIBLE_DEVICES=""`

#### **Slow performance**

- Solution: Install with GPU support
- Check: `torch.cuda.is_available()`

#### **Permission denied errors**

- Solution: Use virtual environment or `--user` flag
- Command: `pip install --user entropic-ai`

### Getting Help

If you encounter issues:

1. **Check the FAQ**: [Frequently Asked Questions](../guides/faq.md)
2. **Search existing issues**: [GitHub Issues](https://github.com/krish567366/Entropic-AI/issues)
3. **Ask for help**: [GitHub Discussions](https://github.com/krish567366/Entropic-AI/discussions)

### System Information

To report issues, include system information:

```python
import entropic-ai
entropic-ai.print_system_info()
```

This will output:

- Entropic AI version
- Python version
- PyTorch version
- CUDA version (if available)
- Operating system
- Hardware details

## Next Steps

After installation, check out:

- **[Quick Start Guide](quickstart.md)**: Get up and running in minutes
- **[Basic Examples](examples.md)**: Try simple examples
- **[API Reference](../api/core.md)**: Explore the full API

## Updates

### Keeping Entropic AI Updated

Check for updates regularly:

```bash
pip list --outdated | grep entropic-ai
pip install --upgrade entropic-ai
```

### Beta Releases

To try beta features:

```bash
pip install --pre entropic-ai
```

### Development Snapshots

For the absolute latest code:

```bash
pip install git+https://github.com/krish567366/Entropic-AI.git
```

## Uninstallation

To remove Entropic AI:

```bash
pip uninstall entropic-ai

# Remove cache and data directories
rm -rf ~/.entropic-ai/  # Linux/Mac
rmdir /s %USERPROFILE%\.entropic-ai  # Windows
```

Welcome to the world of thermodynamic intelligence! 🌌
