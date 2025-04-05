import torch
from torch.utils.data import DataLoader
from diffusionLM.model.transformers_model import DiffusionLLM
from diffusionLM.model.mask_token import mask_tokens_for_diffusion



def evaluate(
    model: DiffusionLLM,
    dataloader: DataLoader,
    device: torch.device,
    num_timesteps: int = 100,
    num_eval_steps: int = None,
):
    """
    Evaluate the Diffusion-LLM model on a validation set.

    Args:
        model: The Diffusion-LLM model to evaluate.
        dataloader: DataLoader for the validation dataset.
        device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').
        num_timesteps: Number of diffusion timesteps for evaluation.
        num_eval_steps: Number of steps to evaluate (None for full dataset).

    Returns:
        Average loss on the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():

        for batch_idx, batch in enumerate(dataloader):
            if num_eval_steps is not None and batch_idx >= num_eval_steps:
                break

            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Sample multiple timesteps for more robust evaluation
            timesteps = torch.linspace(0.1, 0.9, 3, device=device)
            batch_loss = 0.0

            for t in timesteps:
                # Apply forward diffusion (masking)
                diffusion_batch = mask_tokens_for_diffusion(
                    batch,
                    t.item(),
                    model.config.mask_token_id,
                )
                # Removed the unnecessary reshaping of attention_mask
                # diffusion_batch["attention_mask"] = diffusion_batch["attention_mask"].unsqueeze(1).unsqueeze(2)
                # Forward pass
                outputs = model(
                    input_ids=diffusion_batch["input_ids"],
                    attention_mask=diffusion_batch["attention_mask"],
                    timesteps=t.unsqueeze(0),
                    labels=diffusion_batch["labels"],
                )

                batch_loss += outputs["loss"].item()

            # Average loss across timesteps
            batch_loss /= len(timesteps)
            total_loss += batch_loss
            total_steps += 1

    return total_loss / total_steps