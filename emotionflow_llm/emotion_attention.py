"""EmotionAttention: emotion-aware scaled dot-product attention."""

import math
import torch
import torch.nn as nn

from .emotion_encoder import EMOTION_DIM

HEADS = 4


class EmotionAttention(nn.Module):
    """Attention that adds an emotion projection bias to Q and K (not V).

    forward(x: [B, S, dim], emotion: [B, S, EMOTION_DIM]) -> [B, S, dim]

    Requirements 3.1, 3.2, 3.3:
    - emotion_proj bias is added to Q and K before computing attention
    - Attention weights are computed with scaled dot-product + softmax (sum to 1 per row)
    - V is NOT modified by emotion
    """

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.emotion_proj = nn.Linear(EMOTION_DIM, dim)
        self.dim = dim
        self.heads = HEADS

    def forward(self, x: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       [B, S, dim]
            emotion: [B, S, EMOTION_DIM]
        Returns:
            out: [B, S, dim]
        """
        emotion_bias = self.emotion_proj(emotion)   # [B, S, dim]

        Q = self.q(x) + emotion_bias                # emotion bias on Q
        K = self.k(x) + emotion_bias                # emotion bias on K
        V = self.v(x)                               # V unmodified by emotion

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim)
        attn = torch.softmax(scores, dim=-1)        # rows sum to 1

        return torch.matmul(attn, V)
