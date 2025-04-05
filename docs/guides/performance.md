# Performance Optimization Guide

## Memory Optimization

### Gradient Checkpointing

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Configure specific layers
model.config.use_checkpoint = True
```

### Memory-Efficient Training

```python
# Use mixed precision
from torch.cuda.amp import autocast

with autocast():
    outputs = model(**batch)

# Use gradient accumulation
for i, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs["loss"] / gradient_accumulation_steps
    loss.backward()
    
    if (i + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Speed Optimization

### Data Loading

```python
# Optimize DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
```

### Model Inference

```python
# Use torch.compile for faster inference
model = torch.compile(model)

# Use device-specific optimizations
model = model.to(device).half()  # Use FP16
```

## Multi-GPU Training

### DDP Setup

```python
# Initialize process group
dist.init_process_group(backend="nccl")
model = DistributedDataParallel(model)

# Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Monitoring and Profiling

```python
# Use torch profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    outputs = model(**batch)
    
print(prof.key_averages().table())
```