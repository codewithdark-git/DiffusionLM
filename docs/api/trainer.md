# Trainer API Reference

## Trainer Function

Main training function for DiffusionLM models.

```python
def trainer(
    model: DiffusionLLM,
    train_dataset,
    val_dataset = None,
    batch_size: int = 8,
    num_epochs: int = 5,
    learning_rate: float = 5e-5,
    warmup_steps: int = 1000,
    max_grad_norm: float = 1.0,
    num_timesteps: int = 100,
    save_path: Optional[str] = None,
    device: torch.device = None
) -> DiffusionLLM
```

### Parameters

- `model`: The DiffusionLLM model to train
- `train_dataset`: Training dataset
- `val_dataset`: Validation dataset (optional)
- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `warmup_steps`: Number of warmup steps
- `max_grad_norm`: Maximum gradient norm
- `num_timesteps`: Number of diffusion timesteps
- `save_path`: Path to save checkpoints
- `device`: Device to train on

### Returns

- Trained DiffusionLLM model

## Evaluate Function

```python
def evaluate(
    model: DiffusionLLM,
    dataloader: DataLoader,
    device: torch.device,
    num_timesteps: int = 100,
    num_eval_steps: int = None
) -> float
```

### Parameters

- `model`: Model to evaluate
- `dataloader`: DataLoader for evaluation data
- `device`: Device to evaluate on
- `num_timesteps`: Number of diffusion timesteps
- `num_eval_steps`: Number of evaluation steps

### Returns

- Average loss on the evaluation set