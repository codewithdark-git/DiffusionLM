# Configuration Guide

## Model Configuration

### Basic Configuration

```python
from diffusionLM import DiffusionConfig

config = DiffusionConfig(
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=1024
)
```

### Advanced Configuration

```python
config = DiffusionConfig(
    vocab_size=50257,
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=2048,
    num_timesteps=1000,
    time_embed_dim=256
)
```

## Training Configuration

### Basic Training

```python
training_args = {
    "batch_size": 8,
    "num_epochs": 3,
    "learning_rate": 5e-5,
    "warmup_steps": 1000,
}
```

### Advanced Training

```python
training_args = {
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "warmup_steps": 2000,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "num_timesteps": 1000,
    "gradient_accumulation_steps": 4,
    "fp16": True,
    "distributed": True
}
```

## Generation Configuration

### Basic Generation

```python
generation_args = {
    "max_length": 100,
    "num_inference_steps": 50,
    "temperature": 0.7
}
```

### Advanced Generation

```python
generation_args = {
    "max_length": 200,
    "num_inference_steps": 100,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 50,
    "num_beams": 5,
    "strategy": "confidence",
    "use_streaming": True
}
```

## Logging Configuration

```python
from diffusionLM.utils import setup_logging

setup_logging(
    log_file="training.log",
    level="DEBUG",
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)