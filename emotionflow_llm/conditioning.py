"""
Emotion Conditioning System for EmotionFlow-LLM

Enables control over the emotional profile of generated text by specifying
target emotion distributions.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

# Emotion order: joy, sadness, anger, fear, trust, disgust, surprise, anticipation
EMOTIONS = ["joy", "sadness", "anger", "fear", "trust", "disgust", "surprise", "anticipation"]


@dataclass
class EmotionConditioningConfig:
    """Configuration for emotion-conditioned text generation."""
    
    target_emotions: Dict[str, float]  # e.g., {"trust": 0.8, "joy": 0.2}
    strategy: str = "filter"  # "filter", "biased_sampling", "rl"
    num_candidates: int = 50  # For filtering strategy
    temperature: float = 0.9
    max_length: int = 30
    
    def to_vector(self) -> torch.Tensor:
        """
        Convert emotion dictionary to 8-dimensional vector.
        
        Returns:
            torch.Tensor: Shape (8,) with emotions in standard order
        """
        vec = torch.zeros(8)
        for emotion, value in self.target_emotions.items():
            if emotion not in EMOTIONS:
                raise ValueError(f"Unknown emotion: {emotion}. Must be one of {EMOTIONS}")
            idx = EMOTIONS.index(emotion)
            vec[idx] = value
        return vec
    
    def validate(self) -> bool:
        """
        Validate emotion conditioning configuration.
        
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        vec = self.to_vector()
        
        # Check range
        if (vec < 0).any() or (vec > 1).any():
            raise ValueError("Emotion values must be in [0, 1]")
        
        # Check sum
        total = vec.sum().item()
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Emotions must sum to 1.0 (got {total:.3f})")
        
        # Check at least one emotion > 0
        if total == 0:
            raise ValueError("At least one emotion must be > 0")
        
        return True


class EmotionConditioner:
    """Manages emotion-conditioned text generation."""
    
    def __init__(self, model):
        """
        Initialize emotion conditioner.
        
        Args:
            model: EmotionFlowLLM model instance
        """
        self.model = model
    
    def generate_conditioned(
        self,
        prompt: torch.Tensor,
        config: EmotionConditioningConfig
    ) -> List[torch.Tensor]:
        """
        Generate text matching target emotion profile.
        
        Args:
            prompt: Input token tensor [B, S]
            config: Emotion conditioning configuration
            
        Returns:
            List of token tensors matching target emotions
        """
        # Validate config
        config.validate()
        
        if config.strategy == "filter":
            return self._generate_with_filtering(prompt, config)
        elif config.strategy == "biased_sampling":
            return self._generate_with_biasing(prompt, config)
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")
    
    def _generate_with_filtering(
        self, 
        prompt: torch.Tensor, 
        config: EmotionConditioningConfig
    ) -> List[torch.Tensor]:
        """
        Strategy A: Generate many candidates, filter by emotion distance.
        
        Steps:
        1. Generate N candidates (e.g., 50)
        2. Compute emotion profile for each
        3. Calculate distance to target emotion
        4. Return top K closest matches
        
        Args:
            prompt: Input token tensor
            config: Conditioning configuration
            
        Returns:
            List of token tensors closest to target emotion
        """
        from emotionflow_llm.generation import generate
        
        # Generate candidates using generate() directly to control parameters
        candidates = []
        for _ in range(config.num_candidates):
            output = generate(
                self.model,
                prompt,
                max_len=config.max_length,
                temperature=config.temperature
            )
            candidates.append(output)
        
        # Score candidates by emotion distance
        scored = []
        target_vec = config.to_vector()
        
        for candidate in candidates:
            # Extract emotion vector for this candidate
            candidate_emotion = self._extract_emotion_vector(candidate)
            
            # Compute distance (MSE)
            distance = torch.nn.functional.mse_loss(
                candidate_emotion,
                target_vec,
                reduction='sum'
            )
            
            scored.append((candidate, distance.item()))
        
        # Sort by distance (ascending - closer is better)
        scored.sort(key=lambda x: x[1])
        
        # Return top 20% of candidates
        num_return = max(1, config.num_candidates // 5)
        return [cand for cand, dist in scored[:num_return]]
    
    def _generate_with_biasing(
        self,
        prompt: torch.Tensor,
        config: EmotionConditioningConfig
    ) -> List[torch.Tensor]:
        """
        Strategy B: Bias token sampling toward target emotion.
        
        More complex - adjusts logits during generation based on
        how each token would affect emotion trajectory.
        
        TODO: Implement in Phase 2
        """
        raise NotImplementedError(
            "Biased sampling strategy not yet implemented. "
            "Use strategy='filter' instead."
        )
    
    def _extract_emotion_vector(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract mean emotion vector from generated sequence.
        
        Args:
            tokens: Token tensor [B, S]
            
        Returns:
            Tensor: Mean emotion vector [8]
        """
        # Forward pass to get activations
        self.model.eval()
        with torch.no_grad():
            _ = self.model(tokens)
            activations = self.model.get_activations()
        
        # Activations: List of [B, S, 136]
        # Extract last 8 dims (emotion portion) and average
        emotions = []
        for act in activations:
            # Extract emotion dimensions (last 8)
            emo = act[:, :, -8:]  # [B, S, 8]
            emotions.append(emo)
        
        # Stack and compute mean across layers, batch, and sequence
        stacked = torch.stack(emotions, dim=0)  # [layers, B, S, 8]
        mean_emotion = stacked.mean(dim=[0, 1, 2])  # [8]
        
        return mean_emotion


def normalize_emotions(emotions: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize emotion dictionary to sum to 1.0.
    
    Args:
        emotions: Dictionary mapping emotion names to values
        
    Returns:
        Normalized emotion dictionary
        
    Raises:
        ValueError: If all emotions are 0
    """
    total = sum(emotions.values())
    if total == 0:
        raise ValueError("At least one emotion must be > 0")
    return {k: v / total for k, v in emotions.items()}


def validate_emotion_conditioning(config: EmotionConditioningConfig) -> bool:
    """
    Validate emotion conditioning configuration.
    
    Args:
        config: Configuration to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    return config.validate()


# Utility function for creating quick configs
def create_emotion_config(
    emotion_dict: Dict[str, float],
    normalize: bool = True,
    num_candidates: int = 50,
    temperature: float = 0.9,
    max_length: int = 30
) -> EmotionConditioningConfig:
    """
    Create emotion conditioning config with optional normalization.
    
    Args:
        emotion_dict: Target emotions (may not sum to 1.0)
        normalize: Whether to auto-normalize
        num_candidates: Number of candidates to generate
        temperature: Sampling temperature
        max_length: Maximum generation length
        
    Returns:
        EmotionConditioningConfig instance
    """
    if normalize:
        emotion_dict = normalize_emotions(emotion_dict)
    
    return EmotionConditioningConfig(
        target_emotions=emotion_dict,
        strategy="filter",
        num_candidates=num_candidates,
        temperature=temperature,
        max_length=max_length
    )
