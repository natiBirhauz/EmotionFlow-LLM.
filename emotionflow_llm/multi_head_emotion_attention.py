"""
Multi-Head Emotion Attention

Implements attention mechanism with 8 heads, one specialized for each
emotion dimension. Heads are combined using emotion-based weighting.
"""

import torch
import torch.nn as nn


class EmotionAttentionHead(nn.Module):
    """Single attention head specialized for one emotion dimension."""
    
    def __init__(self, model_dim: int, head_dim: int, emotion_idx: int):
        """
        Initialize emotion-specific attention head.
        
        Args:
            model_dim: Full model dimension (136)
            head_dim: Dimension per head (model_dim // num_heads)
            emotion_idx: Which emotion dimension this head focuses on (0-7)
        """
        super().__init__()
        self.emotion_idx = emotion_idx
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Standard attention projections
        self.q_proj = nn.Linear(model_dim, head_dim)
        self.k_proj = nn.Linear(model_dim, head_dim)
        self.v_proj = nn.Linear(model_dim, head_dim)
        
        # Emotion bias (only for this emotion dimension)
        self.emotion_bias = nn.Linear(1, head_dim)
    
    def forward(self, x: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through emotion-specific attention head.
        
        Args:
            x: Input tensor [B, S, model_dim]
            emotion: Full emotion vector [B, S, 8]
            
        Returns:
            Attention output [B, S, head_dim]
        """
        B, S, D = x.shape
        
        # Extract this head's emotion dimension
        emo = emotion[:, :, self.emotion_idx:self.emotion_idx+1]  # [B, S, 1]
        
        # Compute Q, K, V with emotion bias on Q and K
        Q = self.q_proj(x) + self.emotion_bias(emo)  # [B, S, head_dim]
        K = self.k_proj(x) + self.emotion_bias(emo)  # [B, S, head_dim]
        V = self.v_proj(x)  # [B, S, head_dim] - no emotion bias on V
        
        # Scaled dot-product attention
        # Reshape for batch matrix multiplication
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # [B, S, S]
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(attn_weights, V)  # [B, S, head_dim]
        
        return output


class MultiHeadEmotionAttention(nn.Module):
    """
    Multi-head attention with 8 heads, one per emotion dimension.
    
    Each head specializes in one emotion (joy, sadness, anger, etc.)
    and the outputs are combined using emotion-based weighted averaging.
    """
    
    def __init__(self, model_dim: int = 136, num_heads: int = 8):
        """
        Initialize multi-head emotion attention.
        
        Args:
            model_dim: Model dimension (default 136 = 128 word + 8 emotion)
            num_heads: Number of attention heads (default 8, one per emotion)
        """
        super().__init__()
        
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})")
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        # Create 8 emotion-specific heads
        self.heads = nn.ModuleList([
            EmotionAttentionHead(model_dim, self.head_dim, emotion_idx=i)
            for i in range(num_heads)
        ])
        
        # Output projection to combine heads
        self.out_proj = nn.Linear(model_dim, model_dim)
    
    def forward(self, x: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head emotion attention.
        
        Args:
            x: Input tensor [B, S, model_dim]
            emotion: Emotion vectors [B, S, 8]
            
        Returns:
            Attention output [B, S, model_dim]
        """
        B, S, D = x.shape
        
        # Process each head
        head_outputs = []
        for head in self.heads:
            out = head(x, emotion)  # [B, S, head_dim]
            head_outputs.append(out)
        
        # Stack heads: [B, S, num_heads, head_dim]
        stacked = torch.stack(head_outputs, dim=2)
        
        # Emotion weights for combining heads: [B, S, num_heads, 1]
        weights = emotion.unsqueeze(-1)
        
        # Weighted combination: sum over heads dimension
        # [B, S, num_heads, head_dim] * [B, S, num_heads, 1] -> [B, S, num_heads, head_dim]
        weighted = stacked * weights
        combined = weighted.sum(dim=2)  # [B, S, head_dim * num_heads] = [B, S, D]
        
        # Final output projection
        output = self.out_proj(combined)
        
        return output
    
    def get_head_attention_weights(self, x: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for each head (for visualization).
        
        Args:
            x: Input tensor [B, S, model_dim]
            emotion: Emotion vectors [B, S, 8]
            
        Returns:
            Attention weights [B, num_heads, S, S]
        """
        B, S, D = x.shape
        all_weights = []
        
        for i, head in enumerate(self.heads):
            # Extract emotion for this head
            emo = emotion[:, :, i:i+1]
            
            # Compute attention weights
            Q = head.q_proj(x) + head.emotion_bias(emo)
            K = head.k_proj(x) + head.emotion_bias(emo)
            
            scores = torch.bmm(Q, K.transpose(1, 2)) * head.scale
            weights = torch.softmax(scores, dim=-1)  # [B, S, S]
            all_weights.append(weights)
        
        # Stack: [B, num_heads, S, S]
        return torch.stack(all_weights, dim=1)


def visualize_head_specialization(
    model_with_multihead,
    tokens: torch.Tensor,
    emotion_names: list = None
) -> dict:
    """
    Analyze how each attention head specializes in different emotions.
    
    Args:
        model_with_multihead: EmotionFlowLLM with multi-head attention
        tokens: Input tokens [B, S]
        emotion_names: List of 8 emotion names (default: standard order)
        
    Returns:
        Dictionary with per-head statistics
    """
    if emotion_names is None:
        emotion_names = ["joy", "sadness", "anger", "fear", 
                        "trust", "disgust", "surprise", "anticipation"]
    
    model_with_multihead.eval()
    with torch.no_grad():
        _ = model_with_multihead(tokens)
        activations = model_with_multihead.get_activations()
    
    # Extract emotion portions from each layer
    head_specializations = []
    
    for layer_idx, act in enumerate(activations):
        # Extract emotion dims: [B, S, 8]
        emotions = act[:, :, -8:]
        
        # Mean across batch and sequence
        mean_emotions = emotions.mean(dim=[0, 1])  # [8]
        
        head_specializations.append({
            'layer': layer_idx,
            'emotion_strengths': {
                emotion_names[i]: mean_emotions[i].item()
                for i in range(8)
            }
        })
    
    return {
        'per_layer': head_specializations,
        'overall_mean': {
            emotion_names[i]: sum(
                layer['emotion_strengths'][emotion_names[i]] 
                for layer in head_specializations
            ) / len(head_specializations)
            for i in range(8)
        }
    }
