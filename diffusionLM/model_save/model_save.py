import torch
import logging
from pathlib import Path
from typing import Tuple, Optional

from transformers import PreTrainedModel
from model.transformers_model import DiffusionConfig, DiffusionLLM

logger = logging.getLogger(__name__)

class ModelSaveError(Exception):
    """Custom exception for model saving/loading errors"""
    pass

def save_model(
    model: DiffusionLLM,
    optimizer: torch.optim.Optimizer,
    save_path: str,
    final: bool = False,
) -> None:
    """Save model and optimizer state."""
    try:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        step = getattr(model, 'current_step', 1)
        prefix = "final" if final else f"step_{step}"
        save_name = save_dir / f"{prefix}_model.pt"

        # Save the model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "config": model.config.__dict__,
            },
            save_name,
        )
        logger.info(f"Model saved to {save_name}")
        
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise ModelSaveError(f"Failed to save model: {str(e)}")

def load_model(
    load_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[DiffusionLLM, torch.optim.Optimizer]:
    """Load saved model."""
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        if not Path(load_path).exists():
            raise ModelSaveError(f"Checkpoint not found at {load_path}")
            
        checkpoint = torch.load(load_path, map_location=device)
        
        # Create config and model
        config_dict = checkpoint.get("config", {})
        if not config_dict:
            raise ModelSaveError("No config found in checkpoint")

        # Filter out unexpected keyword arguments
        expected_keys = DiffusionConfig.__init__.__code__.co_varnames
        filtered_config_dict = {k: v for k, v in config_dict.items() if k in expected_keys}

        config = DiffusionConfig(**filtered_config_dict)

        # Create model
        model = DiffusionLLM(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return model, optimizer

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise ModelSaveError(f"Failed to load model: {str(e)}")
