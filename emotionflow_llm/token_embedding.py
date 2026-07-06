"""TokenEmbedding: combines word embeddings with emotion vectors."""

import torch
import torch.nn as nn

from .emotion_encoder import EmotionEncoder, VOCAB_SIZE, EMOTION_DIM

WORD_DIM = 128
MODEL_DIM = WORD_DIM + EMOTION_DIM  # 136


class TokenEmbedding(nn.Module):
    """Combine word embeddings with emotion vectors into a unified representation.

    forward(tokens: [B, S]) -> (combined [B, S, 136], emotions [B, S, 8])
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, word_dim: int = WORD_DIM, emotion_dim: int = EMOTION_DIM):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_dim)
        self.emotion_embedding = EmotionEncoder(vocab_size, emotion_dim)

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: [B, S] integer tensor
        Returns:
            combined: [B, S, word_dim + emotion_dim]  (e.g. [B, S, 136])
            emotions: [B, S, emotion_dim]              (e.g. [B, S, 8])
        """
        word_vec = self.word_embedding(tokens)       # [B, S, 128]
        emotion_vec = self.emotion_embedding(tokens) # [B, S, 8]
        combined = torch.cat([word_vec, emotion_vec], dim=-1)  # [B, S, 136]
        return combined, emotion_vec
