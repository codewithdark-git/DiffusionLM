# Custom Dataset Guide

## Creating a Custom Dataset

### Basic Custom Dataset

```python
from diffusionLM import PYTORCH_Dataset
import torch

class CustomTextDataset(PYTORCH_Dataset):
    def __init__(self, texts, tokenizer):
        # Tokenize all texts
        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        super().__init__(
            dataset=encodings,
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

# Usage
texts = ["Sample text 1", "Sample text 2"]
dataset = CustomTextDataset(texts, tokenizer)
```

### Advanced Custom Dataset

```python
class StructuredDataset(PYTORCH_Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and process your data
        self.data = self._load_data(data_path)
        
        # Create encodings
        encodings = self._encode_data()
        
        super().__init__(
            dataset=encodings,
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    def _load_data(self, path):
        # Implement your data loading logic
        pass
        
    def _encode_data(self):
        # Implement your encoding logic
        pass
```

## Using Custom Datasets for Training

```python
# Create and prepare dataset
custom_dataset = CustomTextDataset(texts, tokenizer)

# Train with custom dataset
trained_model = trainer(
    model=model,
    train_dataset=custom_dataset,
    batch_size=8,
    num_epochs=3
)
```

## Data Processing Tips

### Handling Large Datasets

```python
class StreamingDataset(PYTORCH_Dataset):
    def __init__(self, data_iterator, tokenizer):
        self.data_iterator = data_iterator
        self.tokenizer = tokenizer
        
        # Initialize with first batch
        self.current_batch = next(self.data_iterator)
        
        super().__init__(
            dataset=self.current_batch,
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    def refresh_batch(self):
        """Load next batch of data"""
        self.current_batch = next(self.data_iterator)
```

### Data Augmentation

```python
class AugmentedDataset(PYTORCH_Dataset):
    def __init__(self, base_dataset, augmentation_fn):
        self.base_dataset = base_dataset
        self.augmentation_fn = augmentation_fn
        
        super().__init__(
            dataset=base_dataset.dataset,
            mask_token_id=base_dataset.mask_token_id,
            pad_token_id=base_dataset.pad_token_id
        )
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        return self.augmentation_fn(item)
```