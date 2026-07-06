"""EmotionFlow-LLM: An emotion-aware language model architecture.

This module implements a transformer-based language model that integrates
8-dimensional emotional representations (based on Plutchik's wheel of emotions)
into the attention mechanism. The model can:
- Generate text with emotional awareness
- Track emotional trajectories across transformer layers
- Generate multiple candidate outputs and synthesize them
- Profile the emotional content of generated text

Author: Portfolio Project
License: MIT
"""

import torch
import torch.nn as nn
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time
from contextlib import contextmanager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Global Configuration
# ============================================================================

# Model architecture hyperparameters
VOCAB_SIZE = 10000      # Size of token vocabulary
SEQ_LEN = 32            # Maximum sequence length
WORD_DIM = 128          # Dimension of word embeddings
EMOTION_DIM = 8         # Dimension of emotion vectors (8 emotions)
MODEL_DIM = WORD_DIM + EMOTION_DIM  # Combined embedding dimension
HEADS = 4               # Number of attention heads (not used in current single-head impl)
LAYERS = 3              # Number of transformer layers

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Plutchik's 8 basic emotions
EMOTIONS = [
    "joy",          # 0
    "sadness",      # 1
    "anger",        # 2
    "fear",         # 3
    "trust",        # 4
    "disgust",      # 5
    "surprise",     # 6
    "anticipation"  # 7
]

emotion_to_id = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class EmotionVector:
    """8-dimensional emotion representation based on Plutchik's wheel.
    
    Each dimension represents the strength (0.0 to 1.0) of one of the
    8 basic emotions: joy, sadness, anger, fear, trust, disgust, surprise,
    and anticipation.
    
    Attributes:
        joy: Strength of joy emotion [0, 1]
        sadness: Strength of sadness emotion [0, 1]
        anger: Strength of anger emotion [0, 1]
        fear: Strength of fear emotion [0, 1]
        trust: Strength of trust emotion [0, 1]
        disgust: Strength of disgust emotion [0, 1]
        surprise: Strength of surprise emotion [0, 1]
        anticipation: Strength of anticipation emotion [0, 1]
    """
    joy: float
    sadness: float
    anger: float
    fear: float
    trust: float
    disgust: float
    surprise: float
    anticipation: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert emotion vector to PyTorch tensor.
        
        Returns:
            Tensor of shape [8] with emotion values in order
        """
        return torch.tensor([
            self.joy, self.sadness, self.anger, self.fear,
            self.trust, self.disgust, self.surprise, self.anticipation
        ], dtype=torch.float32)
    
    def validate(self) -> bool:
        """Check if all emotion values are in valid range [0, 1].
        
        Returns:
            True if all values are in [0, 1], False otherwise
        """
        values = [
            self.joy, self.sadness, self.anger, self.fear,
            self.trust, self.disgust, self.surprise, self.anticipation
        ]
        return all(0 <= v <= 1 for v in values)
    
    def dominant_emotion(self) -> str:
        """Return the name of the emotion with the highest value.
        
        Returns:
            String name of the dominant emotion
        """
        values = [
            self.joy, self.sadness, self.anger, self.fear,
            self.trust, self.disgust, self.surprise, self.anticipation
        ]
        max_idx = values.index(max(values))
        return EMOTIONS[max_idx]
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'EmotionVector':
        """Create EmotionVector from a tensor.
        
        Args:
            tensor: Tensor of shape [8] with emotion values
            
        Returns:
            EmotionVector instance
        """
        values = tensor.tolist()
        return cls(*values)


@dataclass
class GenerationConfig:
    """Configuration parameters for multi-sample text generation.
    
    Attributes:
        num_samples: Number of candidate sequences to generate
        max_length: Maximum number of tokens to generate per sequence
        strategy: Sampling strategy ("temperature", "nucleus", or "top_k")
        temperature: Temperature scaling factor for softmax (0.1-2.0)
        top_p: Cumulative probability threshold for nucleus sampling (0.0-1.0)
        top_k: Number of top tokens to consider for top-k sampling (1-100)
    """
    num_samples: int = 20
    max_length: int = 100
    strategy: str = "temperature"
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    
    def validate(self) -> None:
        """Validate that all parameters are within acceptable ranges.
        
        Raises:
            ValueError: If any parameter is outside its valid range
        """
        if not 0.1 <= self.temperature <= 2.0:
            raise ValueError(
                f"Temperature must be in [0.1, 2.0], got {self.temperature}"
            )
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be in [0.0, 1.0], got {self.top_p}")
        if not 1 <= self.top_k <= 100:
            raise ValueError(f"top_k must be in [1, 100], got {self.top_k}")
        if self.strategy not in ["temperature", "nucleus", "top_k"]:
            raise ValueError(
                f"strategy must be one of ['temperature', 'nucleus', 'top_k'], "
                f"got {self.strategy}"
            )


@dataclass
class PipelineResult:
    """Complete output from the EmotionFlow-LLM pipeline.
    
    Attributes:
        synthesized_text: Final merged output tensor
        candidates: List of candidate output tensors
        emotion_trajectories: Emotion vectors at each layer
        timing_stats: Dict mapping component names to execution times
        token_emotions: Nested dict {layer: {position: emotion_tensor}}
    """
    synthesized_text: torch.Tensor
    candidates: List[torch.Tensor]
    emotion_trajectories: List[torch.Tensor]
    timing_stats: Dict[str, float]
    token_emotions: Dict[int, Dict[int, torch.Tensor]]


# ============================================================================
# Performance Monitoring
# ============================================================================

class PerformanceMonitor:
    """Track and report execution time statistics for pipeline components.
    
    Example:
        monitor = PerformanceMonitor()
        with monitor.measure("encoder"):
            # ... code to time
        stats = monitor.report()
    """
    
    def __init__(self):
        """Initialize empty timing storage."""
        self.timings: Dict[str, List[float]] = {}
    
    @contextmanager
    def measure(self, component_name: str):
        """Context manager that times the enclosed code block.
        
        Args:
            component_name: Name of the component being timed
            
        Yields:
            None
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            if component_name not in self.timings:
                self.timings[component_name] = []
            self.timings[component_name].append(elapsed)
    
    def report(self) -> Dict[str, Dict[str, float]]:
        """Generate statistical report of all recorded timings.
        
        Returns:
            Dict mapping component names to stats dicts containing:
                - mean: Average execution time
                - min: Minimum execution time
                - max: Maximum execution time  
                - total: Total cumulative time
        """
        stats = {}
        for component, times in self.timings.items():
            if times:
                stats[component] = {
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times)
                }
        return stats


# ============================================================================
# Neural Network Components
# ============================================================================

class EmotionEncoder(nn.Module):
    """Generates 8-dimensional emotion vectors for input tokens.
    
    This module learns to associate each token in the vocabulary with an
    emotion vector. The output is passed through sigmoid to ensure all
    values are in the range [0, 1].
    
    Args:
        vocab_size: Size of the token vocabulary
        emotion_dim: Dimension of emotion vectors (should be 8)
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, emotion_dim: int = EMOTION_DIM):
        super().__init__()
        self.vocab_size = vocab_size
        self.emotion_dim = emotion_dim
        self.embedding = nn.Embedding(vocab_size, emotion_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute emotion vectors for input tokens.
        
        Args:
            tokens: Integer tensor of shape [batch_size, seq_len]
            
        Returns:
            Emotion vectors of shape [batch_size, seq_len, 8] with values in [0, 1]
        """
        emotion_vec = self.embedding(tokens)
        emotion_vec = torch.sigmoid(emotion_vec)  # Ensure [0, 1] range
        emotion_vec = self.validate(emotion_vec)
        return emotion_vec
    
    def validate(self, emotion_vector: torch.Tensor) -> torch.Tensor:
        """Validate and clean emotion vectors.
        
        Performs the following checks and corrections:
        - Asserts last dimension is 8
        - Replaces NaN values with 0
        - Replaces inf values with 0
        - Clamps out-of-range values to [0, 1]
        
        Logs warnings for any corrections made.
        
        Args:
            emotion_vector: Emotion tensor to validate
            
        Returns:
            Cleaned emotion tensor
            
        Raises:
            AssertionError: If last dimension is not 8
        """
        assert emotion_vector.shape[-1] == 8, (
            f"Emotion vector must have 8 dimensions, got {emotion_vector.shape[-1]}"
        )

        # Replace NaN with 0
        nan_mask = torch.isnan(emotion_vector)
        nan_count = nan_mask.sum().item()
        if nan_count > 0:
            logger.warning(
                f"NaN detected in emotion vector: "
                f"{int(nan_count)} occurrence(s) replaced with 0"
            )
            emotion_vector = emotion_vector.clone()
            emotion_vector[nan_mask] = 0.0

        # Replace inf with 0
        inf_mask = torch.isinf(emotion_vector)
        inf_count = inf_mask.sum().item()
        if inf_count > 0:
            logger.warning(
                f"Inf detected in emotion vector: "
                f"{int(inf_count)} occurrence(s) replaced with 0"
            )
            emotion_vector = emotion_vector.clone()
            emotion_vector[inf_mask] = 0.0

        # Clamp to [0, 1]
        out_of_range = ((emotion_vector < 0) | (emotion_vector > 1)).sum().item()
        if out_of_range > 0:
            logger.warning(
                f"Out-of-range values in emotion vector: "
                f"{int(out_of_range)} value(s) clamped to [0, 1]"
            )
            emotion_vector = torch.clamp(emotion_vector, 0.0, 1.0)

        return emotion_vector


class TokenEmbedding(nn.Module):
    """Combines word embeddings with emotion vectors.
    
    Creates a unified token representation by concatenating standard
    word embeddings with learned emotion embeddings.
    
    Args:
        vocab_size: Size of the token vocabulary
        word_dim: Dimension of word embeddings
        emotion_dim: Dimension of emotion vectors
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        word_dim: int = WORD_DIM,
        emotion_dim: int = EMOTION_DIM
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_dim)
        self.emotion_embedding = EmotionEncoder(vocab_size, emotion_dim)

    def forward(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute combined embeddings and separate emotion vectors.
        
        Args:
            tokens: Integer tensor of shape [batch_size, seq_len]
            
        Returns:
            Tuple of:
                - combined: Concatenated embeddings [batch_size, seq_len, word_dim+emotion_dim]
                - emotions: Emotion vectors [batch_size, seq_len, emotion_dim]
        """
        word_vec = self.word_embedding(tokens)
        emotion_vec = self.emotion_embedding(tokens)
        combined = torch.cat([word_vec, emotion_vec], dim=-1)
        return combined, emotion_vec


class EmotionAttention(nn.Module):
    """Attention mechanism that incorporates emotional context.
    
    Modifies standard scaled dot-product attention by adding emotion-based
    bias to the query and key projections. This allows emotional context
    to influence which tokens attend to each other.
    
    Args:
        dim: Model dimension for Q, K, V projections
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.emotion_proj = nn.Linear(EMOTION_DIM, dim)

    def forward(
        self, x: torch.Tensor, emotion: torch.Tensor
    ) -> torch.Tensor:
        """Apply emotion-aware attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            emotion: Emotion vectors of shape [batch_size, seq_len, EMOTION_DIM]
            
        Returns:
            Attention output of shape [batch_size, seq_len, dim]
        """
        # Project input to Q, K, V
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        # Add emotion bias to queries and keys
        emotion_bias = self.emotion_proj(emotion)
        Q = Q + emotion_bias
        K = K + emotion_bias

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        
        output = torch.matmul(attn_weights, V)
        return output


class TransformerBlock(nn.Module):
    """Single transformer layer with emotion-aware attention.
    
    Applies emotion-aware self-attention followed by a feed-forward network,
    with residual connections and layer normalization around each sub-layer.
    
    Args:
        dim: Model dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = EmotionAttention(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, x: torch.Tensor, emotion: torch.Tensor
    ) -> torch.Tensor:
        """Apply transformer block with emotion context.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            emotion: Emotion vectors of shape [batch_size, seq_len, EMOTION_DIM]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, dim]
        """
        # Self-attention with residual and norm
        attn_out = self.attn(x, emotion)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual and norm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class EmotionFlowLLM(nn.Module):
    """Complete emotion-aware language model.
    
    A transformer-based language model that integrates emotional representations
    into the attention mechanism. The model tracks activations at each layer
    for trajectory visualization and analysis.
    
    Architecture:
        - Token embedding layer (word + emotion)
        - Stack of transformer blocks with emotion-aware attention
        - Language modeling head for next-token prediction
    """

    def __init__(self):
        super().__init__()
        self.embedding = TokenEmbedding()
        self.layers = nn.ModuleList([
            TransformerBlock(MODEL_DIM) for _ in range(LAYERS)
        ])
        self.lm_head = nn.Linear(MODEL_DIM, VOCAB_SIZE)
        self.activations: List[torch.Tensor] = []

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            tokens: Integer tensor of shape [batch_size, seq_len]
            
        Returns:
            Logits tensor of shape [batch_size, seq_len, VOCAB_SIZE]
        """
        # Reset activation tracking
        self.activations = []
        
        # Embed tokens with emotion
        x, emotion = self.embedding(tokens)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, emotion)
            self.activations.append(x.detach())
        
        # Project to vocabulary
        logits = self.lm_head(x)
        return logits

    def get_activations(self) -> List[torch.Tensor]:
        """Retrieve stored layer activations from the last forward pass.
        
        Returns:
            List of activation tensors, one per layer
        """
        return self.activations


# ============================================================================
# Text Generation Functions
# ============================================================================

def generate(
    model: EmotionFlowLLM,
    prompt: torch.Tensor,
    max_len: int = 20,
    temperature: float = 0.9
) -> torch.Tensor:
    """Generate text autoregressively from a prompt.
    
    Samples tokens one at a time from the model's output distribution,
    scaled by temperature, and appends them to the sequence.
    
    Args:
        model: EmotionFlowLLM instance
        prompt: Token IDs of shape [seq_len] or [batch_size, seq_len]
        max_len: Number of tokens to generate
        temperature: Softmax temperature scaling factor (higher = more random)
        
    Returns:
        Generated sequence of shape [batch_size, prompt_len + max_len]
    """
    model.eval()

    # Ensure 2D: [batch_size, seq_len]
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


def generate_samples(
    model: EmotionFlowLLM,
    prompt: torch.Tensor,
    n: int = 20
) -> List[torch.Tensor]:
    """Generate multiple independent sequences from a single prompt.
    
    Useful for exploring diverse reasoning paths and creating
    multiple candidate outputs for synthesis.
    
    Args:
        model: EmotionFlowLLM instance
        prompt: Token IDs of shape [seq_len] or [batch_size, seq_len]
        n: Number of independent samples to generate
        
    Returns:
        List of n generated sequences, each of shape [batch_size, total_len]
    """
    return [generate(model, prompt) for _ in range(n)]


# ============================================================================
# Analysis and Utility Functions
# ============================================================================

def emotion_profile(model: EmotionFlowLLM, tokens: torch.Tensor) -> str:
    """Determine the dominant emotion of a token sequence.
    
    Computes the mean emotion vector across all tokens in the sequence
    and returns the name of the highest-valued emotion.
    
    Args:
        model: EmotionFlowLLM instance
        tokens: Integer tensor of shape [batch_size, seq_len]
        
    Returns:
        Name of the dominant emotion (e.g., "joy", "fear")
    """
    with torch.no_grad():
        # Get emotion embeddings for all tokens
        emotions = model.embedding.emotion_embedding(tokens)  # [B, S, 8]
        
    # Average across batch and sequence dimensions
    mean_emotion = emotions.mean(dim=[0, 1])  # [8]
    dominant_idx = mean_emotion.argmax().item()
    return EMOTIONS[dominant_idx]


def synthesize(outputs: List[torch.Tensor]) -> torch.Tensor:
    """Merge multiple candidate outputs into a single sequence.
    
    Simple synthesis strategy: concatenate all outputs along the sequence
    dimension and truncate to the maximum sequence length.
    
    Args:
        outputs: List of token tensors, each of shape [batch_size, seq_len_i]
        
    Returns:
        Merged tensor of shape [batch_size, min(total_len, SEQ_LEN)]
    """
    merged = torch.cat(outputs, dim=1)
    return merged[:, :SEQ_LEN]


def collect_emotion_trajectory(model: EmotionFlowLLM) -> List[float]:
    """Extract emotional trajectory across transformer layers.
    
    Computes the mean activation value at each layer to create a
    simple scalar trajectory showing how representations evolve.
    
    Args:
        model: EmotionFlowLLM instance (must have run forward() first)
        
    Returns:
        List of mean activation values, one per layer
    """
    activations = model.get_activations()
    return [layer_act.mean().item() for layer_act in activations]


# ============================================================================
# Demo & Main Execution
# ============================================================================

def main():
    """Demo of EmotionFlow-LLM capabilities."""
    print("=" * 70)
    print("EmotionFlow-LLM Demo")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Model configuration:")
    print(f"  - Vocabulary size: {VOCAB_SIZE}")
    print(f"  - Word dimension: {WORD_DIM}")
    print(f"  - Emotion dimension: {EMOTION_DIM}")
    print(f"  - Transformer layers: {LAYERS}")
    print(f"  - Model dimension: {MODEL_DIM}")
    print()
    
    # Initialize model
    print("Initializing model...")
    model = EmotionFlowLLM().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print()
    
    # Create random prompt
    prompt = torch.randint(0, VOCAB_SIZE, (1, 10)).to(DEVICE)
    print(f"Prompt shape: {prompt.shape}")
    print()
    
    # Generate multiple samples
    print("Generating 5 candidate outputs...")
    outputs = generate_samples(model, prompt, n=5)
    print(f"✓ Generated {len(outputs)} samples")
    print()
    
    # Analyze emotion profile of each output
    print("Emotion analysis:")
    for i, output in enumerate(outputs):
        emotion = emotion_profile(model, output)
        print(f"  Output {i+1}: {emotion:>12} (length: {output.shape[1]})")
    print()
    
    # Synthesize outputs
    final = synthesize(outputs)
    print(f"Synthesized output shape: {final.shape}")
    print()
    
    # Collect emotion trajectory
    traj = collect_emotion_trajectory(model)
    print("Emotion trajectory (mean activation per layer):")
    for i, val in enumerate(traj):
        print(f"  Layer {i+1}: {val:>8.4f}")
    print()
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()