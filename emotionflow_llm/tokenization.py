"""
Simple tokenization utilities for EmotionFlow-LLM demo.

Note: These are placeholder functions for demonstration purposes.
In production, use a proper tokenizer like BPE or GPT-2 tokenizer.
"""

import torch
from typing import List


def simple_tokenize(text: str, vocab_size: int = 10000) -> torch.Tensor:
    """
    Simple character-based tokenization for demo purposes.
    
    Maps each character to a token ID by hashing. This is NOT a proper
    tokenizer but works for demonstrations.
    
    Args:
        text: Input text string
        vocab_size: Vocabulary size (default 10000)
        
    Returns:
        Token tensor [1, len(text)]
    """
    if not text:
        # Return a single padding token
        return torch.tensor([[0]], dtype=torch.long)
    
    # Hash each character to a token ID
    tokens = []
    for char in text:
        # Use hash to map char to [0, vocab_size)
        token_id = hash(char) % vocab_size
        tokens.append(token_id)
    
    return torch.tensor([tokens], dtype=torch.long)


def decode_tokens(tokens: torch.Tensor, max_length: int = 100) -> str:
    """
    Decode tokens back to text (placeholder implementation).
    
    For demo purposes, this just creates a readable representation
    of the token IDs. In production, use proper detokenization.
    
    Args:
        tokens: Token tensor [B, S] or [S]
        max_length: Maximum output length
        
    Returns:
        String representation
    """
    if tokens.dim() == 2:
        tokens = tokens[0]  # Take first batch
    
    # Truncate if too long
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    # Create readable representation
    # For demo: just show token IDs with some structure
    token_list = tokens.tolist()
    
    # Group into "words" (every 3-5 tokens)
    words = []
    i = 0
    while i < len(token_list):
        word_len = (hash(str(i)) % 3) + 3  # 3-5 tokens per "word"
        word_tokens = token_list[i:i+word_len]
        words.append(f"[{','.join(map(str, word_tokens))}]")
        i += word_len
    
    return " ".join(words)


def create_prompt_from_text(text: str) -> torch.Tensor:
    """
    Convenience function to create a prompt tensor from text.
    
    Args:
        text: Prompt text
        
    Returns:
        Token tensor ready for model input
    """
    return simple_tokenize(text)


def batch_tokenize(texts: List[str], vocab_size: int = 10000) -> torch.Tensor:
    """
    Tokenize multiple texts into a batched tensor.
    
    Args:
        texts: List of input texts
        vocab_size: Vocabulary size
        
    Returns:
        Batched token tensor [B, max_len] (padded)
    """
    if not texts:
        return torch.tensor([[0]], dtype=torch.long)
    
    # Tokenize each text
    token_lists = []
    max_len = 0
    
    for text in texts:
        tokens = simple_tokenize(text, vocab_size)
        token_lists.append(tokens[0])  # Remove batch dim
        max_len = max(max_len, len(tokens[0]))
    
    # Pad to same length
    padded = []
    for tokens in token_lists:
        if len(tokens) < max_len:
            # Pad with zeros
            padding = torch.zeros(max_len - len(tokens), dtype=torch.long)
            tokens = torch.cat([tokens, padding])
        padded.append(tokens)
    
    return torch.stack(padded)


# Placeholder vocabulary for demo
DEMO_VOCAB = [
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "is", "was", "are", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "when", "where", "why", "how",
    "happy", "sad", "angry", "scared", "trusted", "disgusted", "surprised", "excited",
    "hero", "villain", "scientist", "doctor", "teacher", "student",
    "saved", "destroyed", "discovered", "analyzed", "created", "wrote",
    "day", "night", "world", "life", "time", "place", "way",
    ".", ",", "!", "?", ";", ":", "'", '"', "-", "(", ")"
]


def get_demo_vocab_size() -> int:
    """Get the size of demo vocabulary."""
    return len(DEMO_VOCAB)


def tokenize_with_vocab(text: str) -> List[int]:
    """
    Tokenize using demo vocabulary (word-based).
    
    This is closer to a real tokenizer but still simplified.
    
    Args:
        text: Input text
        
    Returns:
        List of token IDs
    """
    words = text.lower().split()
    tokens = []
    
    for word in words:
        # Clean punctuation
        word = word.strip('.,!?;:\'"()-')
        if word in DEMO_VOCAB:
            tokens.append(DEMO_VOCAB.index(word))
        else:
            # Unknown word - use hash
            tokens.append(hash(word) % len(DEMO_VOCAB))
    
    return tokens if tokens else [0]


def detokenize_with_vocab(token_ids: List[int]) -> str:
    """
    Detokenize using demo vocabulary.
    
    Args:
        token_ids: List of token IDs
        
    Returns:
        Reconstructed text
    """
    words = []
    for token_id in token_ids:
        if 0 <= token_id < len(DEMO_VOCAB):
            words.append(DEMO_VOCAB[token_id])
        else:
            words.append(f"<UNK{token_id}>")
    
    return " ".join(words)
