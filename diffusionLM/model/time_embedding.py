import math
import torch.nn as nn
import torch

class TimeEmbedding(nn.Module):
    """Embedding for diffusion timesteps."""
    def __init__(self, time_embed_dim):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim // 4, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        """
        Args:
            timesteps: [batch_size] tensor of timestep values
        Returns:
            [batch_size, time_embed_dim] tensor of embeddings
        """
        half_dim = self.time_embed_dim // 8
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return self.time_embed(emb)
