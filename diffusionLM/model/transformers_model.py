import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from .diffusionLM import LLaDAModel

class DiffusionConfig(PretrainedConfig):
    """Configuration class for Diffusion-LLM model."""
    model_type = "diffusionLM"
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 1024,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        mask_token_id: int = 50256,
        eos_token_id: int = 50256,
        num_timesteps: int = 100,
        time_embed_dim: int = 128,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads 
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id
        self.num_timesteps = num_timesteps
        self.time_embed_dim = time_embed_dim

class DiffusionLLM(PreTrainedModel):
    """Main Diffusion-LLM model class"""
    config_class = DiffusionConfig
    base_model_prefix = "diffusionLM"

    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
        self.model = LLaDAModel(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        timesteps=None,
        labels=None,
        return_dict=True,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            timesteps=timesteps,
            labels=labels,
        )
        
        return outputs

    def generate(
        self,
        prompt=None,
        max_length=100,
        num_inference_steps=50,
        temperature=1.0,
        strategy='random',
        top_p=0.9,
        top_k=50,
        num_beams=5,
        return_scores=False,
        use_streaming=False,
        callback_fn=None
    ):
        """Unified generation interface"""
        if use_streaming:
            return self.generate_stream(
                prompt=prompt,
                max_length=max_length,
                num_inference_steps=num_inference_steps,
                temperature=temperature,
                strategy=strategy,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                callback_fn=callback_fn
            )
        else:
            return self.model.generate(
                prompt=prompt,
                max_length=max_length,
                num_inference_steps=num_inference_steps,
                temperature=temperature,
                strategy=strategy,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                return_scores=return_scores
            )

    def generate_stream(self, **kwargs):
        """Streaming generation wrapper"""
        return self.model.generate_stream(**kwargs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Prepare inputs for generation compatibility"""
        return {
            "input_ids": input_ids,
            "attention_mask": kwargs.get("attention_mask", None),
            "timesteps": kwargs.get("timesteps", None),
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """Reorder cache for beam search compatibility"""
        return past
