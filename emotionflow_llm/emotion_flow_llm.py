"""EmotionFlowLLM: full emotion-aware language model stacking TransformerBlocks."""

import torch
import torch.nn as nn
from typing import List

from .token_embedding import TokenEmbedding, MODEL_DIM
from .transformer_block import TransformerBlock
from .emotion_encoder import VOCAB_SIZE

LAYERS = 3


class EmotionFlowLLM(nn.Module):
    """Full emotion-aware LM: TokenEmbedding → LAYERS × TransformerBlock → lm_head.

    Requirements 4.1–4.4:
    - Stacks LAYERS TransformerBlock instances
    - forward() resets and populates self.activations with per-layer x.detach()
    - forward() returns logits of shape [B, S, VOCAB_SIZE]
    - get_activations() returns self.activations
    """

    def __init__(self):
        super().__init__()
        self.embedding = TokenEmbedding()
        self.layers = nn.ModuleList([
            TransformerBlock(MODEL_DIM)
            for _ in range(LAYERS)
        ])
        self.lm_head = nn.Linear(MODEL_DIM, VOCAB_SIZE)
        self.activations: List[torch.Tensor] = []

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, S] integer token IDs
        Returns:
            logits: [B, S, VOCAB_SIZE]
        """
        self.activations = []
        x, emotion = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x, emotion)
            self.activations.append(x.detach())
        return self.lm_head(x)

    def get_activations(self) -> List[torch.Tensor]:
        """Return per-layer activations stored during the last forward pass."""
        return self.activations
