import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    This class implements the scaled dot-product attention mechanism with multiple attention heads.
    Args:
        config: Configuration object containing model hyperparameters.

    Attributes:
        q_proj: Linear layer for projecting queries.
        k_proj: Linear layer for projecting keys.
        v_proj: Linear layer for projecting values.
        out_proj: Linear layer for projecting the output.
        attn_dropout: Dropout layer for attention probabilities.
        resid_dropout: Dropout layer for residual connections.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        assert self.head_dim * config.num_attention_heads == self.hidden_size, "hidden_size must be divisible by num_attention_heads"

        # Query, key, value projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, head_dim)"""
        x = x.view(batch_size, -1, self.num_attention_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]

    def merge_heads(self, x, batch_size):
        """Merge the heads back together"""
        if x.dim() != 4:
            # If not, reduce the dimensions using squeeze
            x = x.squeeze(0).squeeze(0)

        x = x.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, heads, head_dim]
        return x.view(batch_size, -1, self.hidden_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        """
        Perform a forward pass through the multi-head attention mechanism.

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size].
            attention_mask: Optional mask to avoid attending to padding tokens.
            head_mask: Optional mask for specific attention heads.
            output_attentions: Whether to return attention probabilities.

        Returns:
            A tuple containing:
                - output: Output tensor of shape [batch_size, seq_length, hidden_size].
                - attention_probs (optional): Attention probabilities if output_attentions=True.
        """
        batch_size, seq_length = hidden_states.shape[:2]

        # Project to queries, keys, values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Split heads
        q = self.split_heads(q, batch_size)  # [batch, heads, seq_len, head_dim]
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled dot-product attention
        # (batch, heads, seq_len, head_dim) x (batch, heads, head_dim, seq_len)
        # -> (batch, heads, seq_len, seq_len)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / (self.head_dim ** 0.5)

        # Apply attention mask if provided
        if attention_mask is not None:
            # Add large negative value to masked positions
            attention_scores = attention_scores + (attention_mask * -1e9)

        # Softmax over the sequence dimension
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        # Apply head mask if provided
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # Get the weighted sum of values
        context = torch.matmul(attention_probs, v)  # [batch, heads, seq_len, head_dim]
        context = self.merge_heads(context, batch_size)  # [batch, seq_len, hidden_size]

        # Output projection and dropout
        output = self.out_proj(context)
        output = self.resid_dropout(output)

        outputs = (output,)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs