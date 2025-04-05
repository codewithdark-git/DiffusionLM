import torch
from torch.utils.data import Dataset
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DatasetError(Exception):
    """
    Custom exception for dataset errors.

    This exception is raised when there is an issue with the dataset structure or access.
    """
    pass

class PYTORCH_Dataset(Dataset):
    """
    Dataset wrapper for Hugging Face datasets for training.

    This class wraps a Hugging Face dataset and provides compatibility with PyTorch's DataLoader.

    Args:
        dataset: The Hugging Face dataset to wrap.
        mask_token_id: The ID of the mask token.
        pad_token_id: The ID of the padding token.

    Raises:
        DatasetError: If the dataset is None or does not have the required structure.
    """
    def __init__(
        self,
        dataset: Any,
        mask_token_id: int,
        pad_token_id: int,
    ):
        if dataset is None:
            raise DatasetError("Dataset cannot be None")
            
        if not hasattr(dataset, '__len__') or not hasattr(dataset, '__getitem__'):
            raise DatasetError("Dataset must support len() and indexing")

        self.dataset = dataset
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        
        # Validate the first item to ensure dataset structure
        try:
            first_item = self.dataset[0]
            required_keys = ["input_ids", "attention_mask"]
            for key in required_keys:
                if key not in first_item:
                    raise DatasetError(f"Dataset items must contain '{key}'")
        except Exception as e:
            raise DatasetError(f"Failed to validate dataset structure: {str(e)}")

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            The number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing:
                - input_ids: Token IDs (1D tensor).
                - attention_mask: Attention mask (1D tensor).
                - orig_len: Original length of the sequence (non-padding tokens).

        Raises:
            DatasetError: If there is an issue retrieving the sample.
        """
        try:
            item = self.dataset[idx]

            # Get input_ids and attention_mask
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(item["attention_mask"], dtype=torch.bool)

            # Validate tensor shapes
            if input_ids.ndim != 1:
                raise DatasetError(f"input_ids must be 1D tensor, got shape {input_ids.shape}")
            if attention_mask.ndim != 1:
                raise DatasetError(f"attention_mask must be 1D tensor, got shape {attention_mask.shape}")
            if input_ids.shape != attention_mask.shape:
                raise DatasetError("input_ids and attention_mask must have same shape")

            # Calculate original length (non-padding tokens)
            orig_len = attention_mask.sum().item()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "orig_len": orig_len,
            }
        except Exception as e:
            logger.error(f"Error retrieving item {idx}: {str(e)}")
            raise DatasetError(f"Failed to get item {idx}: {str(e)}")
