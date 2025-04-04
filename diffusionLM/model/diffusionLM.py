import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from model.transformer_block import TransformerBlock
from model.time_embedding import TimeEmbedding


class LLaDAModel(nn.Module):
    """
    LLaDA: Large Language Diffusion Model
    Uses a transformer backbone with timestep conditioning for diffusion-based language modeling
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer blocks
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Time step embedding
        self.time_embed = TimeEmbedding(config.time_embed_dim)

        # Project time embeddings to be added to each transformer layer
        self.time_proj = nn.ModuleList([
            nn.Linear(config.time_embed_dim, config.hidden_size)
            for _ in range(config.num_hidden_layers)
        ])

        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying with token embeddings
        # self.lm_head.weight = nn.Parameter(self.wte.weight.clone())

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model

        Args:
            input_ids: Tensor of token ids [batch_size, seq_len]
            attention_mask: Mask tensor [batch_size, seq_len]
            timesteps: Current diffusion timesteps [batch_size]
            labels: Target token ids for masked positions [batch_size, seq_len]

        Returns:
            dict with loss and logits
        """
        batch_size, seq_length = input_ids.shape

        # Get position indices
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Get embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        # Sum embeddings and apply dropout
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        # Get time embeddings
        time_emb = self.time_embed(timesteps)

        # Prepare attention mask for self-attention
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=input_ids.device)

        # Make it broadcastable to [batch, heads, seq, seq]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Pass through transformer layers with time step conditioning
        for i, block in enumerate(self.h):
            # Add timestep embedding to hidden states
            time_proj = self.time_proj[i](time_emb)
            time_proj = time_proj.unsqueeze(1).expand(-1, seq_length, -1)
            layer_hidden = hidden_states + time_proj

            # Apply transformer block
            layer_outputs = block(
                layer_hidden,
                attention_mask=extended_attention_mask,
                head_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Calculate loss only for masked tokens (where labels != -100)
            # First, get mask of valid tokens
            mask = (labels != -100)

            # Reshape logits and labels for loss calculation
            logits_flat = logits.view(-1, self.config.vocab_size)
            labels_flat = labels.view(-1)

            # Calculate masked cross entropy
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(logits_flat, labels_flat)
            masked_losses = losses * mask.view(-1).float()

            # Calculate the average over masked tokens
            n_masked = mask.sum().item()
            if n_masked > 0:
                loss = masked_losses.sum() / n_masked
            else:
                loss = masked_losses.sum() * 0.0  # No masked tokens

        return {
            "loss": loss,
            "logits": logits,
        }

    def generate(
        self,
        prompt=None,
        max_length=100,
        num_inference_steps=50,
        temperature=1.0,
        strategy='random',  # 'random', 'confidence', 'semi-autoregressive', 'top-p', 'beam'
        top_p=0.9,
        top_k=50,
        num_beams=5,
        return_scores=False
    ):
        """
        Enhanced generate text using the reverse diffusion process.

        Args:
            prompt: Optional text prompt to condition generation
            max_length: Maximum sequence length
            num_inference_steps: Number of denoising steps
            temperature: Temperature for sampling
            strategy: Remasking strategy during generation
            top_p: Nucleus sampling parameter (for top-p strategy)
            top_k: Top-k sampling parameter
            num_beams: Number of beams for beam search
            return_scores: Whether to return confidence scores

        Returns:
            dict containing:
                - tokens: Generated token ids
                - scores: Token confidence scores if return_scores=True
                - attention_weights: Attention weights for visualization
        """
        device = next(self.parameters()).device
        
        # Initialize sequence tracking
        generated_tokens = []
        token_scores = []
        attention_weights = []

        # Initialize with all masked tokens
        if prompt is None:
            # Start with all masks
            x = torch.full((1, max_length), self.config.mask_token_id, dtype=torch.long, device=device)
            # Create attention mask (all 1s for fully masked sequence)
            attention_mask = torch.ones((1, max_length), dtype=torch.long, device=device)
        else:
            # Handle prompt conditioning
            prompt_ids = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
            prompt_len = prompt_ids.size(1)

            # Concatenate prompt with masks
            x = torch.cat([
                prompt_ids,
                torch.full((1, max_length - prompt_len), self.config.mask_token_id, dtype=torch.long, device=device)
            ], dim=1)

            # Create attention mask
            attention_mask = torch.ones((1, max_length), dtype=torch.long, device=device)

        # Track original prompt tokens which should not be modified
        prompt_mask = (x != self.config.mask_token_id)

        # Reverse diffusion process
        for t_step in reversed(range(num_inference_steps)):
            # Current time step (normalized between 0 and 1)
            t = torch.tensor([t_step / num_inference_steps], dtype=torch.float, device=device)

            # Get model predictions
            with torch.no_grad():
                attention_mask_float = attention_mask.type(torch.float32)
                outputs = self.forward(x, attention_mask=attention_mask_float.unsqueeze(1).unsqueeze(2), timesteps=t)
                logits = outputs["logits"]
                
                # Temperature scaling
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)

            # Enhanced token sampling strategies
            mask_positions = (x == self.config.mask_token_id) & (~prompt_mask)
            
            if mask_positions.sum() > 0:
                if strategy == 'top-p':
                    # Nucleus sampling
                    sorted_probs, sorted_indices = torch.sort(probs[mask_positions], descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    probs[mask_positions][indices_to_remove] = 0.0
                    probs[mask_positions] = probs[mask_positions] / probs[mask_positions].sum(dim=-1, keepdim=True)
                
                elif strategy == 'beam':
                    # Simple beam search implementation
                    beam_scores = torch.zeros((num_beams,), device=device)
                    beam_sequences = x.repeat(num_beams, 1)
                    
                    for pos in mask_positions.nonzero():
                        pos_probs = probs[pos[0], pos[1]]
                        top_k_probs, top_k_indices = torch.topk(pos_probs, k=num_beams)
                        
                        for beam_idx in range(num_beams):
                            beam_sequences[beam_idx, pos[1]] = top_k_indices[beam_idx]
                            beam_scores[beam_idx] += torch.log(top_k_probs[beam_idx])
                    
                    # Select best beam
                    best_beam = beam_scores.argmax()
                    x = beam_sequences[best_beam].unsqueeze(0)
                
                else:  # Default to top-k sampling
                    top_k_probs, top_k_indices = torch.topk(probs[mask_positions], k=min(top_k, probs.size(-1)))
                    sampled_tokens = top_k_indices[
                        torch.arange(top_k_probs.size(0)),
                        torch.multinomial(top_k_probs, num_samples=1).squeeze(-1)
                    ]
                    x[mask_positions] = sampled_tokens

            # Track token confidence scores
            if return_scores:
                token_scores.append(torch.gather(probs, -1, x.unsqueeze(-1)).squeeze(-1))
            
            # Apply masking strategy for next step
            if t_step > 0:
                next_mask_prob = (t_step - 1) / num_inference_steps
                self._apply_masking_strategy(x, prompt_mask, next_mask_prob, strategy, probs)

        # Prepare return values
        result = {
            "tokens": x,
            "generated_sequence": x.squeeze().tolist()
        }
        
        if return_scores:
            result["confidence_scores"] = torch.stack(token_scores, dim=0).mean(dim=0)
        
        return result

    def _apply_masking_strategy(self, x, prompt_mask, mask_prob, strategy, probs):
        """Helper method for applying different masking strategies"""
        if strategy == 'confidence':
            token_probs = torch.gather(probs, -1, x.unsqueeze(-1)).squeeze(-1)
            non_prompt_positions = (~prompt_mask)
            num_to_mask = int(mask_prob * non_prompt_positions.sum().item())
            
            if num_to_mask > 0:
                non_prompt_probs = token_probs.clone()
                non_prompt_probs[prompt_mask] = 1.0
                _, indices = torch.topk(non_prompt_probs, k=num_to_mask, largest=False)
                x.view(-1)[indices.view(-1)] = self.config.mask_token_id
        
        elif strategy == 'semi-autoregressive':
            valid_positions = (~prompt_mask).nonzero()
            if len(valid_positions) > 0:
                cutoff = valid_positions[torch.randint(0, len(valid_positions), (1,))].item()
                mask = torch.arange(x.size(1), device=x.device) > cutoff
                x[~prompt_mask & mask] = self.config.mask_token_id
        
        else:  # random strategy
            random_mask = torch.rand(x.shape, device=x.device) < mask_prob
            x[random_mask & ~prompt_mask] = self.config.mask_token_id

    def generate_stream(
        self,
        prompt=None,
        max_length=100,
        num_inference_steps=50,
        temperature=1.0,
        strategy='random',
        top_p=0.9,
        top_k=50,
        num_beams=5,
        callback_fn=None
    ):
        """
        Streaming version of generate that yields tokens as they're generated
        
        Args:
            ...existing args...
            callback_fn: Optional callback function(step, tokens, probs) for visualization
        
        Yields:
            dict containing:
                - current_tokens: Current state of generated sequence
                - token_probs: Probabilities for current tokens
                - step: Current generation step
                - is_finished: Whether generation is complete
        """
        device = next(self.parameters()).device
        
        # Initialize sequence
        x, attention_mask, prompt_mask = self._initialize_sequence(prompt, max_length, device)
        
        # Track generation progress
        generation_metadata = {
            "step": 0,
            "strategy": strategy,
            "temperature": temperature
        }

        # Reverse diffusion process
        for t_step in reversed(range(num_inference_steps)):
            t = torch.tensor([t_step / num_inference_steps], dtype=torch.float, device=device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.forward(x, 
                    attention_mask=attention_mask.type(torch.float32).unsqueeze(1).unsqueeze(2),
                    timesteps=t)
                logits = outputs["logits"] / temperature
                probs = F.softmax(logits, dim=-1)

            # Apply different sampling strategies
            if strategy == 'auto-regressive':
                # Generate tokens left-to-right
                new_x = x.clone()
                for pos in range(prompt_mask.size(1)):
                    if not prompt_mask[0, pos]:
                        pos_probs = probs[0, pos]
                        if top_p < 1.0:
                            pos_probs = self._nucleus_sampling(pos_probs, top_p)
                        sampled_token = torch.multinomial(pos_probs, num_samples=1)
                        new_x[0, pos] = sampled_token
                        
                        # Yield current state
                        yield {
                            "current_tokens": new_x,
                            "token_probs": pos_probs[sampled_token],
                            "position": pos,
                            "step": t_step,
                            "is_finished": False,
                            "metadata": generation_metadata
                        }
                x = new_x

            elif strategy == 'parallel':
                # Sample all masked tokens simultaneously
                mask_positions = (x == self.config.mask_token_id) & (~prompt_mask)
                if mask_positions.sum() > 0:
                    masked_probs = probs[mask_positions]
                    sampled_tokens = torch.multinomial(masked_probs, num_samples=1).squeeze(-1)
                    x[mask_positions] = sampled_tokens
                    
                    yield {
                        "current_tokens": x,
                        "token_probs": torch.gather(probs, -1, x.unsqueeze(-1)).squeeze(-1),
                        "step": t_step,
                        "is_finished": False,
                        "metadata": generation_metadata
                    }

            else:  # Default to progressive masking
                next_mask_prob = (t_step - 1) / num_inference_steps
                self._apply_masking_strategy(x, prompt_mask, next_mask_prob, strategy, probs)
                
                yield {
                    "current_tokens": x,
                    "token_probs": torch.gather(probs, -1, x.unsqueeze(-1)).squeeze(-1),
                    "step": t_step,
                    "is_finished": False,
                    "metadata": generation_metadata
                }

            # Update metadata
            generation_metadata["step"] = num_inference_steps - t_step
            
            # Optional visualization callback
            if callback_fn is not None:
                callback_fn(t_step, x, probs)

        # Final yield with completed sequence
        yield {
            "current_tokens": x,
            "token_probs": torch.gather(probs, -1, x.unsqueeze(-1)).squeeze(-1),
            "step": num_inference_steps,
            "is_finished": True,
            "metadata": generation_metadata
        }

    def _nucleus_sampling(self, probs, top_p):
        """Helper for nucleus (top-p) sampling"""
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        probs[sorted_indices[sorted_indices_to_remove]] = 0
        return probs / probs.sum(dim=-1, keepdim=True)

    def _initialize_sequence(self, prompt, max_length, device):
        """Helper for sequence initialization"""
        if prompt is None:
            x = torch.full((1, max_length), self.config.mask_token_id, dtype=torch.long, device=device)
            attention_mask = torch.ones((1, max_length), dtype=torch.long, device=device)
            prompt_mask = torch.zeros((1, max_length), dtype=torch.bool, device=device)
        else:
            prompt_ids = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
            prompt_len = prompt_ids.size(1)
            x = torch.cat([
                prompt_ids,
                torch.full((1, max_length - prompt_len), self.config.mask_token_id, dtype=torch.long, device=device)
            ], dim=1)
            attention_mask = torch.ones((1, max_length), dtype=torch.long, device=device)
            prompt_mask = torch.cat([
                torch.ones((1, prompt_len), dtype=torch.bool, device=device),
                torch.zeros((1, max_length - prompt_len), dtype=torch.bool, device=device)
            ], dim=1)
        return x, attention_mask, prompt_mask