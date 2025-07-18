# Installation Guide

This guide covers everything you need to install and set up Entropic AI on your system.

## Quick Installation

### From PyPI (Recommended)

The easiest way to install Entropic AI is through PyPI:

```bash
pip install eai
```

### With Optional Dependencies

For enhanced functionality, install with optional dependencies:

```bash
# For GPU acceleration
pip install eai[gpu]

# For molecular applications
pip install eai[molecules]

# For circuit design
pip install eai[circuits]

# For all features
pip install eai[full]
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
conda create -n eai python=3.10
conda activate eai

# Using venv
python -m venv eai-env
source eai-env/bin/activate  # Linux/Mac
# or
eai-env\Scripts\activate     # Windows
```

## Installation Methods

### Method 1: PyPI Installation

Install the latest stable release:

```bash
pip install eai
```

Verify installation:

```bash
python -c "import eai; print(eai.__version__)"
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
   pip install eai
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
   pip3.10 install eai
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
   pip3 install eai
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
   pip3 install eai
   ```

## Configuration

### Environment Variables

Set up environment variables for optimal performance:

```bash
# Bash/Zsh
export EAI_CACHE_DIR="$HOME/.eai/cache"
export EAI_DATA_DIR="$HOME/.eai/data"
export EAI_NUM_THREADS="8"

# Windows CMD
set EAI_CACHE_DIR=%USERPROFILE%\.eai\cache
set EAI_DATA_DIR=%USERPROFILE%\.eai\data
set EAI_NUM_THREADS=8
```

### Configuration File

Create a configuration file at `~/.eai/config.yaml`:

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
import eai
from eai import EntropicNetwork, GenerativeDiffuser
import torch

print(f"Entropic AI version: {eai.__version__}")

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
from eai.benchmarks import installation_benchmark

# Run installation benchmark
results = installation_benchmark()
print(f"Performance score: {results.score}")
print(f"GPU acceleration: {results.gpu_available}")
print(f"All tests passed: {results.all_passed}")
```

### Application Tests

```python
# Test molecular evolution
from eai.applications import MoleculeEvolution
mol_evolver = MoleculeEvolution()
print("✅ Molecular evolution available")

# Test circuit design
from eai.applications import CircuitEvolution
circuit_evolver = CircuitEvolution()
print("✅ Circuit evolution available")

# Test theory discovery
from eai.applications import TheoryDiscovery
theory_evolver = TheoryDiscovery()
print("✅ Theory discovery available")
```

## Troubleshooting

### Common Issues

#### **ImportError: No module named 'eai'**

- Solution: Ensure you're in the correct virtual environment
- Check: `pip list | grep eai`

#### **CUDA out of memory**

- Solution: Reduce batch size or use CPU
- Set: `export CUDA_VISIBLE_DEVICES=""`

#### **Slow performance**

- Solution: Install with GPU support
- Check: `torch.cuda.is_available()`

#### **Permission denied errors**

- Solution: Use virtual environment or `--user` flag
- Command: `pip install --user eai`

### Getting Help

If you encounter issues:

1. **Check the FAQ**: [Frequently Asked Questions](../guides/faq.md)
2. **Search existing issues**: [GitHub Issues](https://github.com/krish567366/Entropic-AI/issues)
3. **Ask for help**: [GitHub Discussions](https://github.com/krish567366/Entropic-AI/discussions)

### System Information

To report issues, include system information:

```python
import eai
eai.print_system_info()
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
pip list --outdated | grep eai
pip install --upgrade eai
```

### Beta Releases

To try beta features:

```bash
pip install --pre eai
```

### Development Snapshots

For the absolute latest code:

```bash
pip install git+https://github.com/krish567366/Entropic-AI.git
```

## Uninstallation

To remove Entropic AI:

```bash
pip uninstall eai

# Remove cache and data directories
rm -rf ~/.eai/  # Linux/Mac
rmdir /s %USERPROFILE%\.eai  # Windows
```

Welcome to the world of thermodynamic intelligence! 🌌
