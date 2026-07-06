"""EmotionFlow-LLM: Emotion-aware language model architecture."""

from .data_models import (
    EmotionVector,
    GenerationConfig,
    ModelConfig,
    VisualizationConfig,
    PipelineResult,
)
from .emotion_encoder import EmotionEncoder, VOCAB_SIZE, EMOTION_DIM
from .token_embedding import TokenEmbedding, WORD_DIM, MODEL_DIM
from .emotion_attention import EmotionAttention, HEADS
from .transformer_block import TransformerBlock
from .emotion_flow_llm import EmotionFlowLLM, LAYERS
from .generation import generate, generate_samples
from .utils import emotion_profile, synthesize, collect_emotion_trajectory, EMOTIONS, SEQ_LEN
from .performance_monitor import PerformanceMonitor

__all__ = [
    "EmotionVector",
    "GenerationConfig",
    "ModelConfig",
    "VisualizationConfig",
    "PipelineResult",
    "EmotionEncoder",
    "VOCAB_SIZE",
    "EMOTION_DIM",
    "TokenEmbedding",
    "WORD_DIM",
    "MODEL_DIM",
    "EmotionAttention",
    "HEADS",
    "TransformerBlock",
    "EmotionFlowLLM",
    "LAYERS",
    "generate",
    "generate_samples",
    "emotion_profile",
    "synthesize",
    "collect_emotion_trajectory",
    "EMOTIONS",
    "SEQ_LEN",
    "PerformanceMonitor",
]
