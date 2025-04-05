import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from .diffusionLM import LLaDAModel

class DiffusionConfig(PretrainedConfig):
    """
    Configuration class for the Diffusion-LLM model.
    This class defines the model's hyperparameters and architecture details.
    Args:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of the hidden layers.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads per layer.
        intermediate_size: Dimensionality of the feed-forward layers.
        hidden_dropout_prob: Dropout probability for hidden layers.
        attention_probs_dropout_prob: Dropout probability for attention layers.
        max_position_embeddings: Maximum number of positional embeddings.
        initializer_range: Standard deviation for weight initialization.
        layer_norm_eps: Epsilon value for layer normalization.
        pad_token_id: ID of the padding token.
        mask_token_id: ID of the mask token.
        eos_token_id: ID of the end-of-sequence token.
        num_timesteps: Number of diffusion timesteps.
        time_embed_dim: Dimensionality of the time embedding.
    """
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
    """
    Main Diffusion Language Model (Diffusion-LLM) implementing a transformer-based 
    architecture with diffusion-based text generation.
    This model combines transformer architectures with diffusion models to generate
    high-quality text through an iterative denoising process. It supports both
    standard and streaming text generation with various decoding strategies.
    Args:
        config (DiffusionConfig): Configuration object containing model hyperparameters,
            architecture settings, and training parameters.
    The model can be used for:
    - Conditional text generation
    - Iterative text refinement
    - Streaming text generation
    - Beam search decoding
    Methods:
        - forward: Perform a forward pass through the model.
        - generate: Generate text using the reverse diffusion process.
        - generate_stream: Stream generated tokens for real-time applications.

    """
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
        """
        Perform a forward pass through the model.
        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask to avoid attending to padding tokens.
            timesteps: Diffusion timesteps for conditioning.
            labels: Target labels for supervised training.
            return_dict: Whether to return a dictionary of outputs.
        Returns:
            A dictionary containing:
                - loss: Loss value (if labels are provided).
                - logits: Logits for each token in the sequence.
        """
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
        """
        Generate text using the reverse diffusion process.
        Args:
            prompt: Optional text prompt to condition generation.
            max_length: Maximum sequence length.
            num_inference_steps: Number of denoising steps.
            temperature: Sampling temperature.
            strategy: Remasking strategy during generation.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            num_beams: Number of beams for beam search.
            return_scores: Whether to return confidence scores.
            use_streaming: Whether to stream generated tokens.
            callback_fn: Optional callback function for streaming.
        Returns:
            A dictionary containing generated tokens and optional scores.
        """
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
        """
        Generate text using streaming generation.
        This method allows for real-time generation of text tokens.
        Args:
            **kwargs: Additional arguments for generation.
            Returns:
                A generator yielding generated tokens one by one.
        """
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
