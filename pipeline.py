
import torch

import os
import random
import argparse
import numpy as np
from IPython import get_ipython

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

from MODEL.model import DiffusionLLM, DiffusionConfig
from trainer import trainer
from model_save import load_model
from datasetANDtokenizer import prepare_dataset
from register_model import registerANDpush


parser = argparse.ArgumentParser(description="Train LLaDA model")
parser.add_argument("--dataset", type=str, default="wikitext/wikitext-103-v1", help="Dataset name")
parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name")
parser.add_argument("--model_size", type=str, default="small", help="Model size (small, medium, large)")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
parser.add_argument("--save_dir", type=str, default="./saved_models", help="Directory to save models")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--num_timesteps", type=int, default=100, help="Number of diffusion timesteps")
parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
parser.add_argument("--load_path", type=str, default=None, help="Path to load model from")
parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")
parser.add_argument("--num_proc", type=int, default=4, help="Number of processes for tokenization")
parser.add_argument("--push_to_hub", type=bool, default=False, help="Push model to Hugging Face Hub")

if get_ipython() is not None:
    # If in Jupyter, use default arguments
    args = parser.parse_args([])  # Pass empty list to parse_args
else:
    # If in command-line, parse arguments normally
    args = parser.parse_args()


# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Create save directory if it doesn't exist
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# Prepare dataset
print(f"Loading dataset {args.dataset}...")
train_dataset, val_dataset, tokenizer = prepare_dataset(
    dataset_name=args.dataset,
    tokenizer_name=args.tokenizer,
    max_length=args.max_length,
    cache_dir=args.cache_dir,
    num_proc=args.num_proc
)
print(f"Dataset loaded. Train size: {len(train_dataset)}")
# used the portions data of training
train_dataset = torch.utils.data.Subset(train_dataset, range(int(len(train_dataset) * 0.1)))
# print(f"Train Data size: {len(train_dataset)}")
if val_dataset:
    print(f"Validation size: {len(val_dataset)}")
    val_dataset = torch.utils.data.Subset(val_dataset, range(int(len(val_dataset) * 0.1)))
    print(f"Validation Data size: {len(val_dataset)}")

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configure model size
model_configs = {
    "small": {
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
    },
    "medium": {
        "hidden_size": 512,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
    },
    "large": {
        "hidden_size": 1024,
        "num_hidden_layers": 16,
        "num_attention_heads": 16,
    }
}

config_kwargs = model_configs.get(args.model_size, model_configs["small"])

# Create or load model
if args.load_path is not None:
    print(f"Loading model from {args.load_path}...")
    model, optimizer = load_model(args.load_path, device)
else:
    print("Creating new model...")
    # Create config with appropriate vocabulary size
    config = DiffusionConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.max_length,
        num_timesteps=args.num_timesteps,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        **config_kwargs
    )
    model = DiffusionLLM(config)

# Log number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params:,} parameters")

# Train model
print("Starting training...")
train_model = trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_timesteps=args.num_timesteps,
        save_path=args.save_dir,
        device=device,
    )

print("Training completed!")


# push model to hub

if args.push_to_hub:
    registerANDpush(model, 
                    tokenizer, 
                    "diffusionLM",
                    DiffusionLLM, 
                    DiffusionConfig, 
                    repo_id="codewithdark/DiffusionLM")



# Path: utils/pipeline.py \
#   dataset_name wikitext/wikitext-103-v1 \
#   tokenizer gpt2 \
#   model_size small \
#   batch_size 8 \
#   num_epochs 1 \
#   learning_rate 5e-5 \
#   max_length 256 \
#   save_dir ./saved_models \
#   seed 42 \
#   num_timesteps 100 \
#   num_inference_steps 50 \
#   cache_dir None \
#   num_proc 4
#   push_to_hub True
