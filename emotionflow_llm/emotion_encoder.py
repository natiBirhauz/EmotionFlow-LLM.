"""EmotionEncoder: maps token IDs to 8-dimensional emotion vectors."""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

VOCAB_SIZE = 10000
EMOTION_DIM = 8


class EmotionEncoder(nn.Module):
    """Generate 8-dimensional emotion vectors for input tokens.

    forward(tokens: [B, S]) -> [B, S, 8], values in [0, 1] via sigmoid.
    validate() clamps out-of-range values and replaces NaN/inf with zeros,
    logging a warning for each type of occurrence.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, emotion_dim: int = EMOTION_DIM):
        super().__init__()
        self.vocab_size = vocab_size
        self.emotion_dim = emotion_dim
        self.embedding = nn.Embedding(vocab_size, emotion_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, S] integer tensor
        Returns:
            emotion_vectors: [B, S, 8] float tensor in [0, 1]
        """
        emotion_vec = self.embedding(tokens)          # [B, S, 8]
        emotion_vec = torch.sigmoid(emotion_vec)      # enforce [0, 1]
        emotion_vec = self.validate(emotion_vec)
        return emotion_vec

    def validate(self, emotion_vector: torch.Tensor) -> torch.Tensor:
        """Validate and clean an emotion tensor.

        Steps:
        1. Assert last dim == 8
        2. Replace NaN values with 0, log a warning for each occurrence
        3. Replace inf values with 0, log a warning for each occurrence
        4. Clamp values to [0, 1], log a warning if any clamping occurs

        Returns the cleaned tensor.
        """
        assert emotion_vector.shape[-1] == 8, (
            f"Emotion vector must have 8 dimensions, got {emotion_vector.shape[-1]}"
        )

        # Replace NaN with 0
        nan_mask = torch.isnan(emotion_vector)
        nan_count = int(nan_mask.sum().item())
        if nan_count > 0:
            logger.warning(
                f"NaN detected in emotion vector: {nan_count} occurrence(s) replaced with 0"
            )
            emotion_vector = emotion_vector.clone()
            emotion_vector[nan_mask] = 0.0

        # Replace inf with 0
        inf_mask = torch.isinf(emotion_vector)
        inf_count = int(inf_mask.sum().item())
        if inf_count > 0:
            logger.warning(
                f"Inf detected in emotion vector: {inf_count} occurrence(s) replaced with 0"
            )
            emotion_vector = emotion_vector.clone()
            emotion_vector[inf_mask] = 0.0

        # Clamp out-of-range values to [0, 1]
        out_of_range = int(((emotion_vector < 0) | (emotion_vector > 1)).sum().item())
        if out_of_range > 0:
            logger.warning(
                f"Out-of-range values in emotion vector: {out_of_range} value(s) clamped to [0, 1]"
            )
            emotion_vector = torch.clamp(emotion_vector, 0.0, 1.0)

        return emotion_vector
