"""TransformerBlock: single emotion-aware transformer layer."""

import torch
import torch.nn as nn

from .emotion_attention import EmotionAttention


class TransformerBlock(nn.Module):
    """Single transformer layer with EmotionAttention + FFN + LayerNorm.

    forward(x: [B, S, dim], emotion: [B, S, EMOTION_DIM]) -> [B, S, dim]

    Requirements 4.2:
    - Residual connection and LayerNorm around attention: x = norm1(x + attn(x, emotion))
    - Residual connection and LayerNorm around FFN:       x = norm2(x + ff(x))
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = EmotionAttention(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       [B, S, dim]
            emotion: [B, S, EMOTION_DIM]
        Returns:
            x: [B, S, dim]
        """
        x = self.norm1(x + self.attn(x, emotion))
        x = self.norm2(x + self.ff(x))
        return x
