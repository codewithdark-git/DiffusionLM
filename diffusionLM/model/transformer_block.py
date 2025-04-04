import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.MLP import MLP


class TransformerBlock(nn.Module):
    """Transformer block with attention, MLP, and layer normalization"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        # Self-attention with layer norm and residual connection
        attn_outputs = self.attn(
            self.ln_1(x),
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        x = x + attn_output

        # Feed-forward with layer norm and residual connection
        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output

        outputs = (x,)
        if output_attentions:
            outputs += attn_outputs[1:]

        return outputs