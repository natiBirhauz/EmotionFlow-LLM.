"""Utility functions for EmotionFlow-LLM."""

import torch
from typing import List

SEQ_LEN = 32

EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "trust",
    "disgust",
    "surprise",
    "anticipation",
]


def emotion_profile(model, tokens: torch.Tensor) -> str:
    """Return the dominant emotion name for the given token sequence.

    Requirement 8.1–8.2:
    - Runs model.forward(tokens) to get logits
    - Gets emotion vectors from model.embedding.emotion_embedding(tokens) → [B, S, 8]
    - Computes mean across batch and sequence: mean_emotion = emotions.mean(dim=[0,1]) → [8]
    - Returns EMOTIONS[mean_emotion.argmax().item()]

    Args:
        model: EmotionFlowLLM instance
        tokens: [B, S] integer tensor

    Returns:
        str: name of the dominant emotion from EMOTIONS
    """
    with torch.no_grad():
        model.forward(tokens)
        emotions = model.embedding.emotion_embedding(tokens)  # [B, S, 8]

    mean_emotion = emotions.mean(dim=[0, 1])  # [8]
    return EMOTIONS[mean_emotion.argmax().item()]


def synthesize(outputs: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate a list of token tensors and truncate to SEQ_LEN.

    Requirement 9.1–9.2:
    - Accepts a list of token tensors
    - Concatenates along the sequence dimension (dim=1)
    - Truncates to SEQ_LEN tokens: result[:, :SEQ_LEN]

    Args:
        outputs: list of tensors, each [B, S_i]

    Returns:
        Tensor of shape [B, min(total_len, SEQ_LEN)]
    """
    merged = torch.cat(outputs, dim=1)
    return merged[:, :SEQ_LEN]


def collect_emotion_trajectory(model) -> List[float]:
    """Return mean activation per layer as a list of scalar floats.

    Requirement 10.1–10.2:
    - Gets model.get_activations() → list of LAYERS tensors
    - Returns [layer.mean().item() for layer in activations]

    Args:
        model: EmotionFlowLLM instance (must have run forward() first)

    Returns:
        List[float] of length LAYERS
    """
    activations = model.get_activations()
    return [layer.mean().item() for layer in activations]
