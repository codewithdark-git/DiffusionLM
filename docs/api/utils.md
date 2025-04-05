# Utils API Reference

## Dataset Preparation

### prepare_dataset()

```python
def prepare_dataset(
    dataset_name: str = "wikitext/wikitext-2-raw-v1",
    tokenizer_name: str = "gpt2",
    max_length: int = 1024,
    cache_dir: Optional[str] = None,
    num_proc: int = 4
) -> Tuple[PYTORCH_Dataset, Optional[PYTORCH_Dataset], AutoTokenizer]
```

Prepares datasets for training DiffusionLM models.

### PYTORCH_Dataset

```python
class PYTORCH_Dataset(Dataset):
    def __init__(
        self,
        dataset: Any,
        mask_token_id: int,
        pad_token_id: int,
    )
```

Custom dataset class for DiffusionLM training.

## Error Handling

### DiffusionLMError

```python
class DiffusionLMError(Exception):
    """Base exception class for DiffusionLM package"""
    pass
```

### handle_errors()

```python
def handle_errors(
    error_class: Type[Exception] = DiffusionLMError, 
    reraise: bool = True, 
    logger: logging.Logger = logger
) -> Callable
```

Decorator for handling errors in functions.

## Logging

### setup_logging()

```python
def setup_logging(
    log_file: str = None, 
    level: int = logging.INFO,
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> None
```

Sets up logging configuration for the package.