import torch
import logging
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from diffusionLM.model.transformers_model import DiffusionLLM
from diffusionLM.model.mask_token import mask_tokens_for_diffusion
from .evaluate import evaluate
from diffusionLM.save_model.model_save import save_model
from diffusionLM.utils.error_handler import handle_errors, DiffusionLMError

logger = logging.getLogger(__name__)

class TrainingError(DiffusionLMError):
    """Custom exception for training errors"""
    pass

@handle_errors(TrainingError)
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
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Train the model with comprehensive error handling and logging."""
    logger.info("Starting training process")
    try:
        model.to(device)
        train_loader = _setup_data_loader(train_dataset, batch_size, True)
        val_loader = _setup_data_loader(val_dataset, batch_size, False) if val_dataset else None
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        lr_scheduler = _setup_scheduler(optimizer, warmup_steps)
        
        best_val_loss = float('inf')
        early_stopping_patience = 3
        no_improvement_count = 0
        
        for epoch in range(num_epochs):
            epoch_loss = _train_epoch(
                model, train_loader, optimizer, lr_scheduler, 
                device, max_grad_norm, num_timesteps, epoch
            )
            
            if val_loader:
                val_loss = evaluate(model, val_loader, device, num_timesteps)
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_count = 0
                    if save_path:
                        _save_checkpoint(model, optimizer, save_path, epoch, val_loss)
                else:
                    no_improvement_count += 1
                    
                if no_improvement_count >= early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
                    
        return model
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise TrainingError(f"Training process failed: {str(e)}")

@handle_errors(TrainingError)
def _train_epoch(model, train_loader, optimizer, lr_scheduler, device, max_grad_norm, num_timesteps, epoch):
    """Train for one epoch with error handling."""
    model.train()
    epoch_losses = []
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", unit="batch") as pbar:
        for batch in train_loader:
            try:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                timestep = torch.rand(len(batch["input_ids"]), device=device)
                diffusion_batch = mask_tokens_for_diffusion(
                    batch,
                    timestep.mean().item(),
                    model.config.mask_token_id,
                )
                batch_timesteps = timestep.mean().unsqueeze(0)
                outputs = model(
                    input_ids=diffusion_batch["input_ids"],
                    attention_mask=diffusion_batch["attention_mask"],
                    timesteps=batch_timesteps,
                    labels=diffusion_batch["labels"],
                )
                loss = outputs["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
            except Exception as e:
                logger.error(f"Error during batch processing: {str(e)}")
                raise TrainingError(f"Error during batch processing: {str(e)}")
    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss

def _setup_data_loader(dataset, batch_size, shuffle):
    """Set up data loader."""
    try:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )
    except Exception as e:
        logger.error(f"Error setting up data loader: {str(e)}")
        raise TrainingError(f"Error setting up data loader: {str(e)}")

def _setup_scheduler(optimizer, warmup_steps):
    """Set up learning rate scheduler."""
    try:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0,
        )
    except Exception as e:
        logger.error(f"Error setting up scheduler: {str(e)}")
        raise TrainingError(f"Error setting up scheduler: {str(e)}")

def _save_checkpoint(model, optimizer, save_path, epoch, val_loss):
    """Save model checkpoint."""
    try:
        checkpoint_path = Path(save_path) / f"checkpoint_epoch_{epoch}_val_loss_{val_loss:.4f}.pt"
        save_model(model, optimizer, str(checkpoint_path))
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error saving checkpoint: {str(e)}")
        raise TrainingError(f"Error saving checkpoint: {str(e)}")