# Basic Usage

## Quick Start

```python
from diffusionLM import DiffusionLLM, setup_logging
from transformers import AutoTokenizer

# Setup logging
setup_logging()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = DiffusionLLM.from_pretrained("codewithdark/DiffusionLM")

# Generate text
output = model.generate(
    prompt="Once upon a time",
    max_length=100,
    temperature=0.7
)
print(output)
```

## Training Example

```python
from diffusionLM import trainer
from diffusionLM.utils import prepare_dataset

# Prepare dataset
train_dataset, val_dataset, tokenizer = prepare_dataset(
    dataset_name="wikitext/wikitext-103-v1",
    tokenizer_name="gpt2"
)

# Train the model
trained_model = trainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=8,
    num_epochs=3,
    learning_rate=5e-5
)
```

## Saving and Loading

```python
# Save model
model.save_pretrained("path/to/save")

# Load model
loaded_model = DiffusionLLM.from_pretrained("path/to/save")
```

## Streaming Generation

```python
def callback(token):
    print(token, end="", flush=True)

model.generate(
    prompt="Once upon a time",
    max_length=100,
    use_streaming=True,
    callback_fn=callback
)
```

## Error Handling

```python
from diffusionLM import DiffusionLMError, handle_errors

@handle_errors()
def generate_text(prompt):
    return model.generate(prompt=prompt)

try:
    output = generate_text("Your prompt here")
except DiffusionLMError as e:
    print(f"Generation failed: {e}")
```