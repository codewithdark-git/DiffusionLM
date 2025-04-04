"""DiffusionLM: A Diffusion-based Language Model Package"""

from utils.error_handler import setup_logging, DiffusionLMError, handle_errors
from trainer.trainer import trainer, TrainingError
from trainer.evaluate import evaluate
from model_save.model_save import save_model, load_model, ModelSaveError
from model_save.register_model import registerANDpush, ModelRegistrationError
from utils.dataset import PYTORCH_Dataset, DatasetError
from utils.datasetANDtokenizer import prepare_dataset, DatasetPreparationError

__version__ = "0.1.0"

# Setup default logging
setup_logging()


__all__ = [
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
]
