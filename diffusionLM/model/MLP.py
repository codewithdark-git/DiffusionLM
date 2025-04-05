import torch 
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Feed-forward neural network (MLP) used in transformer blocks.

    This class implements a two-layer feed-forward network with GELU activation and dropout.

    Args:
        config: Configuration object containing model hyperparameters.

    Attributes:
        fc1: First linear layer.
        fc2: Second linear layer.
        dropout: Dropout layer applied after the second linear layer.
    """
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Perform a forward pass through the MLP.

        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_size].

        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size].
        """
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        h = self.dropout(h)
        return h