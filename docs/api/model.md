# Model API Reference

## DiffusionConfig

Configuration class for the DiffusionLM model.

### Parameters

- `vocab_size` (int, default=50257): Size of the vocabulary
- `hidden_size` (int, default=768): Dimensionality of the hidden layers
- `num_hidden_layers` (int, default=12): Number of transformer layers
- `num_attention_heads` (int, default=12): Number of attention heads per layer
- `intermediate_size` (int, default=3072): Dimensionality of feed-forward layers
- `hidden_dropout_prob` (float, default=0.1): Dropout probability for hidden layers
- `attention_probs_dropout_prob` (float, default=0.1): Dropout probability for attention
- `max_position_embeddings` (int, default=1024): Maximum sequence length
- `num_timesteps` (int, default=100): Number of diffusion timesteps
- `time_embed_dim` (int, default=128): Dimensionality of time embeddings

## DiffusionLLM

Main model class implementing the transformer-based diffusion language model.

### Methods

#### forward()

```python
def forward(
    input_ids=None,
    attention_mask=None,
    timesteps=None,
    labels=None,
    return_dict=True
)
```

Perform a forward pass through the model.

#### generate()

```python
def generate(
    prompt=None,
    max_length=100,
    num_inference_steps=50,
    temperature=1.0,
    strategy='random',
    top_p=0.9,
    top_k=50,
    num_beams=5,
    return_scores=False,
    use_streaming=False,
    callback_fn=None
)
```

Generate text using the reverse diffusion process.

## MultiHeadAttention

Multi-head self-attention mechanism implementation.

### Methods

#### forward()

```python
def forward(
    hidden_states,
    attention_mask=None,
    head_mask=None,
    output_attentions=False
)
```

Perform multi-head attention computation.

## TimeEmbedding

Embedding layer for diffusion timesteps.

### Methods

#### forward()

```python
def forward(timesteps)
```

Generate time embeddings for the given timesteps.