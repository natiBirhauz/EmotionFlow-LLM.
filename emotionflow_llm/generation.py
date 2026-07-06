"""Text generation functions for EmotionFlow-LLM."""

import torch
from typing import List

from .emotion_flow_llm import EmotionFlowLLM


def generate(model: EmotionFlowLLM, prompt: torch.Tensor, max_len: int = 20, temperature: float = 0.9) -> torch.Tensor:
    """Autoregressively generate max_len tokens appended to prompt.

    Requirements 5.1–5.3:
    - Appends max_len tokens one at a time
    - Each token sampled from softmax distribution scaled by temperature
    - Returns full sequence including original prompt

    Args:
        model: EmotionFlowLLM instance
        prompt: 1D [S] or 2D [B, S] tensor of token IDs
        max_len: number of tokens to generate (default 20)
        temperature: softmax temperature scaling factor (default 0.9)

    Returns:
        Tensor of shape [B, prompt_len + max_len]
    """
    model.eval()

    # Ensure 2D: [B, S]
    if prompt.dim() == 1:
        tokens = prompt.unsqueeze(0).clone()
    else:
        tokens = prompt.clone()

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(tokens)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


def generate_samples(model: EmotionFlowLLM, prompt: torch.Tensor, n: int = 20) -> List[torch.Tensor]:
    """Generate n independent sequences from the given prompt.

    Requirements 6.1–6.3:
    - Returns a list of exactly n generated sequences
    - Default n is 20
    - Each sample generated independently via generate()

    Args:
        model: EmotionFlowLLM instance
        prompt: 1D [S] or 2D [B, S] tensor of token IDs
        n: number of samples to generate (default 20)

    Returns:
        List of n tensors, each of shape [B, prompt_len + max_len]
    """
    return [generate(model, prompt) for _ in range(n)]
