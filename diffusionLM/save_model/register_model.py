import torch
import os
import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, Repository
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from diffusionLM.model.transformers_model import DiffusionConfig, DiffusionLLM

logger = logging.getLogger(__name__)

class ModelRegistrationError(Exception):
    """Custom exception for model registration errors"""
    pass

def registerANDpush(
    model: DiffusionLLM,
    tokenizer,
    model_type: str,
    model_name: type[DiffusionLLM],
    model_config: type[DiffusionConfig],
    repo_id: str = "codewithdark/DiffusionLM",
    private: bool = False,
) -> None:
    """Register and push model to Hugging Face Hub."""
    try:
        # Register model architecture
        AutoConfig.register(model_type, model_config)
        AutoModel.register(model_config, model_name)
        AutoModelForCausalLM.register(model_config, model_name)

        api = HfApi()
        
        # Create repo
        try:
            api.create_repo(repo_id=repo_id, private=private)
            logger.info(f"Created new repository: {repo_id}")
        except Exception as e:
            logger.warning(f"Repository creation failed (may already exist): {e}")

        # Setup local repo
        repo_local_path = Path("SaveModel/DiffusionLM")
        repo_local_path.mkdir(parents=True, exist_ok=True)
        
        repo = Repository(local_dir=str(repo_local_path), clone_from=repo_id)

        # Save model and tokenizer
        tokenizer.save_pretrained(repo_local_path)
        model.save_pretrained(repo_local_path)
        
        # Push to hub
        repo.push_to_hub(commit_message="Initial model and tokenizer commit")
        logger.info(f"Model and tokenizer pushed to {repo_id}")

    except Exception as e:
        logger.error(f"Model registration failed: {str(e)}")
        raise ModelRegistrationError(f"Failed to register model: {str(e)}")