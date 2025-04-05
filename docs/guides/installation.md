# Installation Guide

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.9.0
- CUDA (optional, but recommended for GPU support)

## Basic Installation

Install DiffusionLM using pip:

```bash
pip install diffusion-llm
```

## Development Installation

For development or to use the latest features:

```bash
git clone https://github.com/codewithdark-git/DiffusionLM.git
cd DiffusionLM
pip install -e .
```

## Installing with Specific Dependencies

For CPU-only installation:
```bash
pip install diffusion-llm[cpu]
```

For full installation with all features:
```bash
pip install diffusion-llm[all]
```

## Verifying Installation

```python
from diffusionLM import DiffusionLLM
from diffusionLM.utils import setup_logging

# Setup logging
setup_logging()

# Create a model instance
model = DiffusionLLM.from_pretrained("codewithdark/DiffusionLM")
```

## Common Issues

### CUDA Installation

If you encounter CUDA-related issues:

1. Ensure your NVIDIA drivers are up to date
2. Install the correct PyTorch version for your CUDA version:
   ```bash
   pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```

### Package Conflicts

If you encounter dependency conflicts:

1. Create a new virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # Linux/Mac
   env\Scripts\activate     # Windows
   ```

2. Install dependencies in the clean environment:
   ```bash
   pip install diffusion-llm
   ```