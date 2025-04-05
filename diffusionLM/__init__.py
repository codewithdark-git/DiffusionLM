"""DiffusionLM: A Diffusion-based Language Model Package"""

from .trainer import trainer, TrainingError, evaluate
from .model import (
        DiffusionLLM, 
        DiffusionConfig, 
        mask_tokens_for_diffusion 
        )
from .save_model import (
        save_model, 
        load_model, 
        ModelSaveError, 
        registerANDpush,
        ModelRegistrationError
        )
from .utils import (
        prepare_dataset, 
        DatasetPreparationError, 
        PYTORCH_Dataset, 
        DatasetError,
        setup_logging,
        DiffusionLMError,
        handle_errors
        )

__version__ = "0.1.0"

# Setup default logging
setup_logging()


__all__ = [
    "DiffusionLLM",
    "DiffusionConfig",
    "setup_logging",
    "DiffusionLMError",
    "handle_errors",
    "trainer",
    "TrainingError",
    "evaluate",
    "save_model",
    "load_model",
    "ModelSaveError",
    "registerANDpush",
    "ModelRegistrationError",
    "PYTORCH_Dataset",
    "DatasetError",
    "prepare_dataset",
    "DatasetPreparationError",
    "mask_tokens_for_diffusion",
]
