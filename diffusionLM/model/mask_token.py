import torch
import torch.nn as nn
from typing import Dict

def mask_tokens_for_diffusion(
    batch: Dict[str, torch.Tensor],
    timestep: float,
    mask_token_id: int,
):
    """
    Apply forward diffusion process by masking tokens according to the timestep.

    Args:
        batch: A dictionary containing input token sequences and attention masks.
        timestep: Current timestep (between 0 and 1) for masking probability.
        mask_token_id: ID of the mask token to replace selected tokens.

    Returns:
        A dictionary containing:
            - input_ids: Masked input token IDs.
            - attention_mask: Attention mask for the input.
            - labels: Labels for the masked tokens (-100 for unmasked tokens).
            - mask_ratio: Ratio of tokens that were masked.
    """
    input_ids = batch["input_ids"].clone()
    attention_mask = batch["attention_mask"]
    batch_size, seq_length = input_ids.shape

    # Create mask for masking tokens (don't mask padding)
    valid_tokens = (attention_mask == 1)

    # Randomly select tokens to mask based on timestep
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask_indices = (rand < timestep) & valid_tokens

    # Create labels (we use -100 for unmasked tokens to ignore them in loss)
    labels = torch.full_like(input_ids, -100)
    labels[mask_indices] = input_ids[mask_indices]

    # Apply mask
    input_ids[mask_indices] = mask_token_id

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "mask_ratio": mask_indices.float().mean().item(),
    }