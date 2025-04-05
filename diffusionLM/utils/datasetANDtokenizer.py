import torch
from datasets import load_dataset
import logging
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from .dataset import PYTORCH_Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreparationError(Exception):
    """Custom exception for dataset preparation errors"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

def prepare_dataset(
    dataset_name: str = "wikitext/wikitext-2-raw-v1",
    tokenizer_name: str = "gpt2",
    max_length: int = 1024,
    cache_dir: Optional[str] = None,
    num_proc: int = 4
) -> Tuple[PYTORCH_Dataset, Optional[PYTORCH_Dataset], AutoTokenizer]:
    """
    Prepare a Hugging Face dataset for training.

    Args:
        dataset_name: Name of the dataset to load
        tokenizer_name: Name of the tokenizer to use
        max_length: Maximum sequence length
        cache_dir: Directory to cache the dataset
        num_proc: Number of processes for tokenization

    Returns:
        Train and validation datasets
    """
    try:
        # Load tokenizer
        logger.info(f"Loading tokenizer {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Ensure the tokenizer has required tokens
        if tokenizer.mask_token is None:
            logger.warning("Mask token not found, adding [MASK] token")
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})

        if tokenizer.pad_token is None:
            logger.warning("Pad token not found, adding [PAD] token")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Load dataset
        logger.info(f"Loading dataset {dataset_name}")
        try:
            dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        except Exception as e:
            raise DatasetPreparationError(f"Failed to load dataset {dataset_name}: {str(e)}")

        # Tokenize the dataset
        def tokenize_function(examples):
            try:
                return tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
            except Exception as e:
                raise DatasetPreparationError(f"Tokenization failed: {str(e)}")

        logger.info("Tokenizing dataset")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=["text"]
        )

        # Convert to PyTorch datasets
        logger.info("Converting to PyTorch datasets")
        train_dataset = PYTORCH_Dataset(
            tokenized_dataset["train"],
            mask_token_id=tokenizer.mask_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        val_dataset = None
        if "validation" in tokenized_dataset:
            val_dataset = PYTORCH_Dataset(
                tokenized_dataset["validation"],
                mask_token_id=tokenizer.mask_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        return train_dataset, val_dataset, tokenizer

    except Exception as e:
        logger.error(f"Dataset preparation failed: {str(e)}")
        raise DatasetPreparationError(f"Dataset preparation failed: {str(e)}")
