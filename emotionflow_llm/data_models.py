"""Core data models for EmotionFlow-LLM."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch import Tensor


@dataclass
class EmotionVector:
    """8-dimensional emotion representation based on Plutchik's wheel."""
    
    joy: float          # [0, 1]
    sadness: float      # [0, 1]
    anger: float        # [0, 1]
    fear: float         # [0, 1]
    trust: float        # [0, 1]
    disgust: float      # [0, 1]
    surprise: float     # [0, 1]
    anticipation: float # [0, 1]
    
    def to_tensor(self) -> Tensor:
        """Convert to [8] tensor."""
        return torch.tensor([
            self.joy, self.sadness, self.anger, self.fear,
            self.trust, self.disgust, self.surprise, self.anticipation
        ], dtype=torch.float32)
    
    def validate(self) -> bool:
        """Check all values in [0, 1]."""
        values = [
            self.joy, self.sadness, self.anger, self.fear,
            self.trust, self.disgust, self.surprise, self.anticipation
        ]
        return all(0.0 <= v <= 1.0 for v in values)
    
    def dominant_emotion(self) -> str:
        """Return emotion with highest value."""
        emotions = {
            'joy': self.joy,
            'sadness': self.sadness,
            'anger': self.anger,
            'fear': self.fear,
            'trust': self.trust,
            'disgust': self.disgust,
            'surprise': self.surprise,
            'anticipation': self.anticipation
        }
        return max(emotions.items(), key=lambda x: x[1])[0]


@dataclass
class GenerationConfig:
    """Configuration for multi-sample generation."""
    
    num_samples: int = 20
    max_length: int = 100
    strategy: str = "temperature"  # "temperature", "nucleus", "top_k"
    temperature: float = 0.8       # for temperature sampling
    top_p: float = 0.9             # for nucleus sampling
    top_k: int = 50                # for top-k sampling
    
    def validate(self) -> None:
        """Validate parameter ranges."""
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        
        if self.strategy not in ["temperature", "nucleus", "top_k"]:
            raise ValueError(
                f"strategy must be 'temperature', 'nucleus', or 'top_k', got {self.strategy}"
            )
        
        if not 0.1 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be in [0.1, 2.0], got {self.temperature}"
            )
        
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(
                f"top_p must be in [0.0, 1.0], got {self.top_p}"
            )
        
        if not 1 <= self.top_k <= 100:
            raise ValueError(
                f"top_k must be in [1, 100], got {self.top_k}"
            )


@dataclass
class ModelConfig:
    """EmotionFlow-LLM configuration."""
    
    vocab_size: int = 10000
    word_dim: int = 128
    emotion_dim: int = 8
    num_layers: int = 3
    num_heads: int = 4
    hidden_dim: int = 136  # word_dim + emotion_dim
    emotion_enabled: bool = True
    lambda_emotion: float = 0.1  # emotion loss weight


@dataclass
class VisualizationConfig:
    """Emotion trajectory visualization settings."""
    
    method: str = "pca"  # "pca", "umap", "tsne"
    output_format: str = "png"  # "png", "svg"
    dpi: int = 300
    figsize: Tuple[int, int] = (10, 8)
    show_tokens: bool = False
    color_scheme: str = "viridis"


@dataclass
class PipelineResult:
    """Complete pipeline output."""
    
    synthesized_text: str
    candidates: List[str]
    emotion_trajectory: List[EmotionVector]
    visualization: Optional[Any]  # Image or None
    timing_stats: Dict[str, float]
    token_emotions: Dict[int, Dict[int, EmotionVector]]  # {layer: {position: emotion}}
