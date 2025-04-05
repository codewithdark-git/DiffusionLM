# Troubleshooting Guide

## Common Issues

### CUDA Out of Memory

**Symptoms:**
- RuntimeError: CUDA out of memory
- GPU memory exhaustion during training

**Solutions:**
```python
# 1. Reduce batch size
trainer(model, dataset, batch_size=4)  # Use smaller batch size

# 2. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 3. Use mixed precision training
from torch.cuda.amp import autocast
with autocast():
    outputs = model(**batch)
```

### Package Import Errors

**Symptoms:**
- ModuleNotFoundError
- ImportError: cannot import name

**Solutions:**
```python
# Ensure correct installation
pip install -e .  # Install in editable mode
pip install --force-reinstall diffusion-llm  # Reinstall package
```

### Training Issues

**Symptoms:**
- Loss not decreasing
- NaN values in training

**Solutions:**
```python
# 1. Check learning rate
trainer(
    model,
    dataset,
    learning_rate=5e-5,  # Reduce learning rate
    warmup_steps=1000    # Add warmup
)

# 2. Enable gradient clipping
trainer(
    model,
    dataset,
    max_grad_norm=1.0
)
```

### Generation Issues

**Symptoms:**
- Poor quality outputs
- Slow generation

**Solutions:**
```python
# 1. Adjust temperature
output = model.generate(
    prompt="test",
    temperature=0.7  # Lower for more focused generation
)

# 2. Use different strategies
output = model.generate(
    prompt="test",
    strategy="confidence",  # Try different strategies
    num_beams=5
)
```

## Debug Mode

Enable debug logging for detailed information:
```python
from diffusionLM.utils import setup_logging
import logging

setup_logging(level=logging.DEBUG)
```

## Support

If issues persist:
1. Check the [GitHub Issues](https://github.com/codewithdark-git/DiffusionLM/issues)
2. Submit a detailed bug report
3. Contact maintainers