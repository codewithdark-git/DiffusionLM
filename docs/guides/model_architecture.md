# Model Architecture

## Overview

DiffusionLM combines transformer-based language models with diffusion processes through a novel architecture.

```
DiffusionLLM
├── Transformer Backbone
│   ├── Multi-head Self-attention
│   ├── Feed-forward Networks
│   └── Layer Normalization
├── Diffusion Components
│   ├── Time Embeddings
│   ├── Noise Schedule
│   └── Denoising Process
└── Generation Head
    ├── Token Prediction
    └── Confidence Estimation
```

## Components

### Transformer Backbone

- **Multi-head Self-attention**: Processes token relationships
- **Feed-forward Networks**: Token-wise transformations
- **Layer Normalization**: Stabilizes training

### Diffusion Process

- **Forward Process**: Gradually adds noise to text
- **Reverse Process**: Denoises text during generation
- **Time Conditioning**: Guides the denoising process

### Generation Strategies

- **Auto-regressive**: Traditional token-by-token
- **Parallel**: Simultaneous token generation
- **Hybrid**: Combines both approaches

## Model Sizes

| Size   | Parameters | Layers | Heads | Hidden Size |
| ------ | ---------- | ------ | ----- | ----------- |
| Small  | 117M       | 12     | 12    | 768         |
| Medium | 345M       | 24     | 16    | 1024        |
| Large  | 762M       | 36     | 20    | 1280        |

## Implementation Details

### Attention Mechanism

```python
attention_scores = (Q @ K.transpose(-2, -1)) / sqrt(head_dim)
attention_probs = softmax(attention_scores + attention_mask)
context = attention_probs @ V
```

### Time Embeddings

```python
time_embed = sin_cos_embedding(timesteps)
time_proj = MLP(time_embed)
```

### Denoising Process

```python
x_noisy = alpha_t * x + sigma_t * noise
x_pred = model(x_noisy, t)
x_denoised = (x_noisy - sigma_t * x_pred) / alpha_t
```