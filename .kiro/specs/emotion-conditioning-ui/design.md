# Design Document: Emotion Conditioning & Interactive UI

## Overview

This design extends EmotionFlow-LLM with three major capabilities:
1. **Emotion Conditioning**: Control emotional output via target emotion vectors
2. **Multi-Head Emotion Attention**: Specialized attention heads per emotion
3. **Interactive Web UI**: Gradio-based interface for live demonstrations

## Architecture

### Component Diagram

```
User Input (UI/API)
    ↓
[Emotion Conditioning Config]
    ├─→ Target Emotion Vector [8-dim]
    └─→ Generation Parameters
         ↓
[EmotionFlowLLM with Multi-Head Attention]
    ├─→ 8 Attention Heads (one per emotion)
    ├─→ Emotion-guided sampling
    └─→ Activations tracking
         ↓
[Candidate Generation & Filtering]
    ├─→ Generate N samples
    ├─→ Score by emotion distance
    └─→ Rank and select best matches
         ↓
[Output + Visualizations]
    ├─→ Generated text
    ├─→ Emotion trajectory plot
    └─→ Emotion distribution chart
```

## Component Designs

### 1. Emotion Conditioning System

**File:** `emotionflow_llm/conditioning.py`

#### EmotionConditioner Class

```python
@dataclass
class EmotionConditioningConfig:
    """Configuration for emotion conditioning."""
    target_emotions: Dict[str, float]  # e.g., {"trust": 0.8, "joy": 0.2}
    strategy: str = "filter"  # "filter", "biased_sampling", "rl"
    num_candidates: int = 50  # For filtering strategy
    temperature: float = 0.9
    max_length: int = 30
    
    def to_vector(self) -> torch.Tensor:
        """Convert emotion dict to 8-dim vector."""
        # Order: joy, sadness, anger, fear, trust, disgust, surprise, anticipation
        pass
    
    def validate(self) -> bool:
        """Ensure emotions sum to ~1.0 and are in [0,1]."""
        pass

class EmotionConditioner:
    """Manages emotion-conditioned text generation."""
    
    def __init__(self, model: EmotionFlowLLM):
        self.model = model
    
    def generate_conditioned(
        self,
        prompt: torch.Tensor,
        config: EmotionConditioningConfig
    ) -> List[torch.Tensor]:
        """Generate text matching target emotion profile."""
        
        if config.strategy == "filter":
            return self._generate_with_filtering(prompt, config)
        elif config.strategy == "biased_sampling":
            return self._generate_with_biasing(prompt, config)
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")
    
    def _generate_with_filtering(self, prompt, config):
        """
        Strategy A: Generate many candidates, filter by emotion distance.
        
        Steps:
        1. Generate N candidates (e.g., 50)
        2. Compute emotion profile for each
        3. Calculate distance to target emotion
        4. Return top K closest matches
        """
        candidates = generate_samples(
            self.model, 
            prompt, 
            n=config.num_candidates,
            max_len=config.max_length,
            temperature=config.temperature
        )
        
        # Score candidates
        scored = []
        target_vec = config.to_vector()
        
        for candidate in candidates:
            # Get emotion profile
            self.model(candidate)
            activations = self.model.get_activations()
            # Average emotion across layers and tokens
            candidate_emotion = self._extract_emotion_vector(activations)
            
            # Compute distance (MSE or cosine)
            distance = torch.nn.functional.mse_loss(
                candidate_emotion, 
                target_vec
            )
            scored.append((candidate, distance.item()))
        
        # Sort by distance (ascending)
        scored.sort(key=lambda x: x[1])
        
        # Return top matches
        return [cand for cand, dist in scored[:config.num_candidates // 5]]
    
    def _generate_with_biasing(self, prompt, config):
        """
        Strategy B: Bias token sampling toward target emotion.
        
        More complex - adjusts logits during generation based on
        how each token would affect emotion trajectory.
        """
        # TODO: Implement in Phase 2
        raise NotImplementedError("Biased sampling not yet implemented")
    
    def _extract_emotion_vector(self, activations):
        """Extract mean emotion vector from layer activations."""
        # Activations: List of [B, S, 136]
        # Extract last 8 dims (emotion portion)
        emotions = [act[:, :, -8:] for act in activations]
        # Average across layers, batch, sequence
        mean_emotion = torch.stack(emotions).mean(dim=[0, 1, 2])
        return mean_emotion

def normalize_emotions(emotions: Dict[str, float]) -> Dict[str, float]:
    """Normalize emotion dict to sum to 1.0."""
    total = sum(emotions.values())
    if total == 0:
        raise ValueError("At least one emotion must be > 0")
    return {k: v / total for k, v in emotions.items()}

def validate_emotion_conditioning(config: EmotionConditioningConfig) -> bool:
    """Validate emotion conditioning configuration."""
    vec = config.to_vector()
    
    # Check range
    if (vec < 0).any() or (vec > 1).any():
        raise ValueError("Emotion values must be in [0, 1]")
    
    # Check sum
    total = vec.sum().item()
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Emotions must sum to 1.0, got {total}")
    
    return True
```

### 2. Multi-Head Emotion Attention

**File:** `emotionflow_llm/multi_head_emotion_attention.py`

#### Design Rationale

Current `EmotionAttention` uses a single attention mechanism biased by all 8 emotion dimensions. Multi-head version creates 8 specialized heads, each focusing on one emotion, then combines them weighted by emotion strengths.

**Benefits:**
- Each head can learn emotion-specific patterns
- Better capture of mixed emotions
- More interpretable (can visualize per-emotion attention)

```python
class EmotionAttentionHead(nn.Module):
    """Single attention head specialized for one emotion dimension."""
    
    def __init__(self, model_dim: int, head_dim: int, emotion_idx: int):
        super().__init__()
        self.emotion_idx = emotion_idx
        self.head_dim = head_dim
        
        # Standard attention projections
        self.q_proj = nn.Linear(model_dim, head_dim)
        self.k_proj = nn.Linear(model_dim, head_dim)
        self.v_proj = nn.Linear(model_dim, head_dim)
        
        # Emotion bias (only for this emotion dimension)
        self.emotion_bias = nn.Linear(1, head_dim)  # 1-dim emotion input
    
    def forward(self, x: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] input
            emotion: [B, S, 8] full emotion vector
        Returns:
            [B, S, head_dim] attention output
        """
        B, S, D = x.shape
        
        # Extract this head's emotion dimension
        emo = emotion[:, :, self.emotion_idx:self.emotion_idx+1]  # [B, S, 1]
        
        # Compute Q, K, V with emotion bias
        Q = self.q_proj(x) + self.emotion_bias(emo)  # [B, S, head_dim]
        K = self.k_proj(x) + self.emotion_bias(emo)
        V = self.v_proj(x)  # No bias on V
        
        # Scaled dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(attn_weights, V)
        
        return output


class MultiHeadEmotionAttention(nn.Module):
    """8-head attention with one head per emotion dimension."""
    
    def __init__(self, model_dim: int = 136, num_heads: int = 8):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        # Create 8 emotion-specific heads
        self.heads = nn.ModuleList([
            EmotionAttentionHead(model_dim, self.head_dim, emotion_idx=i)
            for i in range(num_heads)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(model_dim, model_dim)
    
    def forward(self, x: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] input
            emotion: [B, S, 8] emotion vectors
        Returns:
            [B, S, D] attention output
        """
        B, S, D = x.shape
        
        # Process each head
        head_outputs = []
        for head in self.heads:
            out = head(x, emotion)  # [B, S, head_dim]
            head_outputs.append(out)
        
        # Stack and compute weighted combination
        stacked = torch.stack(head_outputs, dim=2)  # [B, S, 8, head_dim]
        
        # Emotion weights: [B, S, 8] -> [B, S, 8, 1]
        weights = emotion.unsqueeze(-1)
        
        # Weighted sum: [B, S, head_dim * 8] = [B, S, D]
        combined = (stacked * weights).sum(dim=2)
        
        # Final projection
        output = self.out_proj(combined)
        
        return output
```

#### Integration with EmotionFlowLLM

**Option A:** Add as alternative architecture (new class)
```python
class EmotionFlowLLM_MultiHead(EmotionFlowLLM):
    """Variant using multi-head emotion attention."""
    
    def __init__(self, use_multi_head: bool = False):
        super().__init__()
        if use_multi_head:
            # Replace attention in transformer blocks
            for block in self.transformer_blocks:
                block.attention = MultiHeadEmotionAttention()
```

**Option B:** Make it configurable in existing class
```python
# In EmotionFlowLLM.__init__:
if config.use_multi_head_attention:
    self.attention_cls = MultiHeadEmotionAttention
else:
    self.attention_cls = EmotionAttention
```

*Recommendation: Option B for flexibility*

### 3. Interactive Web UI

**File:** `app.py`

#### Gradio Interface Design

```python
import gradio as gr
import torch
from emotionflow_llm import EmotionFlowLLM
from emotionflow_llm.conditioning import EmotionConditioner, EmotionConditioningConfig
from emotionflow_llm.visualization import plot_emotion_trajectory, plot_emotion_distribution

# Load model (cached)
@gr.cache
def load_model():
    return EmotionFlowLLM()

def generate_with_ui(
    prompt_text: str,
    joy: float, sadness: float, anger: float, fear: float,
    trust: float, disgust: float, surprise: float, anticipation: float,
    temperature: float,
    num_samples: int
):
    """Main generation function called by UI."""
    
    # Normalize emotions
    emotions = {
        "joy": joy, "sadness": sadness, "anger": anger, "fear": fear,
        "trust": trust, "disgust": disgust, "surprise": surprise, 
        "anticipation": anticipation
    }
    total = sum(emotions.values())
    if total == 0:
        return "Error: At least one emotion must be > 0", None, None
    
    emotions = {k: v/total for k, v in emotions.items()}
    
    # Create config
    config = EmotionConditioningConfig(
        target_emotions=emotions,
        strategy="filter",
        num_candidates=num_samples * 3,  # Generate 3x, filter to best
        temperature=temperature,
        max_length=30
    )
    
    # Tokenize prompt (simplified - use proper tokenizer in production)
    model = load_model()
    prompt_tokens = simple_tokenize(prompt_text)
    
    # Generate conditioned text
    conditioner = EmotionConditioner(model)
    outputs = conditioner.generate_conditioned(prompt_tokens, config)
    
    # Format outputs as text (decode tokens)
    generated_texts = [decode_tokens(out) for out in outputs[:num_samples]]
    text_output = "\n\n---\n\n".join([
        f"**Sample {i+1}:**\n{text}" 
        for i, text in enumerate(generated_texts)
    ])
    
    # Create visualizations
    # 1. Emotion distribution of generated samples
    emotions_plot = plot_emotion_distribution(model, outputs[:num_samples])
    
    # 2. Emotion trajectory (using first sample)
    model(outputs[0])
    trajectory_plot = plot_emotion_trajectory(model)
    
    return text_output, emotions_plot, trajectory_plot


# Build Gradio interface
def build_interface():
    with gr.Blocks(title="EmotionFlow-LLM Demo") as demo:
        gr.Markdown("# 🎭 EmotionFlow-LLM: Emotion-Conditioned Text Generation")
        gr.Markdown("Control the emotional tone of generated text by adjusting the emotion sliders.")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=3
                )
                
                gr.Markdown("### Target Emotion Profile")
                gr.Markdown("*Adjust sliders to set desired emotions (will be auto-normalized)*")
                
                joy = gr.Slider(0, 1, value=0.25, label="😊 Joy")
                sadness = gr.Slider(0, 1, value=0.0, label="😢 Sadness")
                anger = gr.Slider(0, 1, value=0.0, label="😠 Anger")
                fear = gr.Slider(0, 1, value=0.0, label="😨 Fear")
                trust = gr.Slider(0, 1, value=0.25, label="🤝 Trust")
                disgust = gr.Slider(0, 1, value=0.0, label="🤢 Disgust")
                surprise = gr.Slider(0, 1, value=0.25, label="😲 Surprise")
                anticipation = gr.Slider(0, 1, value=0.25, label="🔮 Anticipation")
                
                gr.Markdown("### Generation Parameters")
                temperature = gr.Slider(0.1, 2.0, value=0.9, label="Temperature")
                num_samples = gr.Slider(1, 10, value=5, step=1, label="Number of Samples")
                
                generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                output_text = gr.Markdown(label="Generated Text")
                
                with gr.Row():
                    emotion_dist_plot = gr.Plot(label="Emotion Distribution")
                    trajectory_plot = gr.Plot(label="Emotion Trajectory")
        
        # Examples
        gr.Examples(
            examples=[
                ["The hero saved the day", 0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.9, 5],  # Joy + trust
                ["The dark forest whispered", 0.0, 0.2, 0.0, 0.6, 0.0, 0.0, 0.2, 0.0, 0.9, 5],  # Fear + sadness
                ["The scientist discovered", 0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.3, 0.2, 0.9, 5],  # Surprise + anticipation
            ],
            inputs=[prompt, joy, sadness, anger, fear, trust, disgust, surprise, anticipation, temperature, num_samples]
        )
        
        # Connect button
        generate_btn.click(
            fn=generate_with_ui,
            inputs=[prompt, joy, sadness, anger, fear, trust, disgust, surprise, anticipation, temperature, num_samples],
            outputs=[output_text, emotion_dist_plot, trajectory_plot]
        )
    
    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(share=True)  # share=True creates public link
```

#### Deployment Options

**Hugging Face Spaces (Recommended for Portfolio):**
1. Create account on huggingface.co
2. Create new Space (type: Gradio)
3. Push code:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/emotionflow-llm
   git push hf main
   ```
4. Add `requirements.txt`:
   ```
   torch
   gradio
   matplotlib
   numpy
   ```

**Vercel (Alternative):**
- Requires serverless function setup
- More complex but more customizable
- Better for custom frontend

### 4. Visualization Utilities

**File:** `emotionflow_llm/visualization.py` (extend existing)

```python
def plot_emotion_distribution(model, outputs):
    """Plot emotion distribution across multiple samples."""
    import matplotlib.pyplot as plt
    from collections import Counter
    
    emotions = [emotion_profile(model, out) for out in outputs]
    counts = Counter(emotions)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(counts.keys(), counts.values())
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Emotion Distribution ({len(outputs)} samples)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_emotion_trajectory_comparison(model, outputs, target_emotion):
    """Compare emotion trajectories of multiple samples vs target."""
    # Shows how close each sample gets to target across layers
    pass
```

## Data Flow

### Emotion-Conditioned Generation Flow

```
1. User Input
   ├─→ Prompt text: "The hero"
   └─→ Target emotions: {trust: 0.8, joy: 0.2}

2. Conditioning Setup
   ├─→ Normalize emotions → [0.2, 0, 0, 0, 0.8, 0, 0, 0]
   ├─→ Create config (strategy=filter, num_candidates=50)
   └─→ Tokenize prompt → [1245, 3421]

3. Candidate Generation
   ├─→ Generate 50 samples using generate_samples()
   └─→ Each: [1245, 3421, ...] (length ~30)

4. Emotion Scoring
   ├─→ For each candidate:
   │   ├─→ Forward pass through model
   │   ├─→ Extract activation emotions (last 8 dims)
   │   ├─→ Average across layers/tokens → [8] vector
   │   └─→ Compute MSE vs target
   └─→ Results: [(candidate1, 0.12), (candidate2, 0.34), ...]

5. Filtering & Ranking
   ├─→ Sort by distance (ascending)
   └─→ Return top 5 closest matches

6. Visualization
   ├─→ Decode tokens → text strings
   ├─→ Plot emotion distribution bar chart
   └─→ Plot emotion trajectory for best match
```

## Performance Considerations

### Inference Time Estimates

| Component | Time (CPU) | Notes |
|-----------|-----------|-------|
| Single generation | ~50ms | 30 tokens |
| 50 candidates | ~2.5s | Parallelizable |
| Emotion scoring | ~0.5s | Forward pass per candidate |
| Multi-head attention | +15% | vs single-head |
| **Total (UI request)** | **~3-4s** | Acceptable for demo |

### Optimization Strategies

1. **Batch Processing**: Score candidates in batches of 10
2. **Early Stopping**: Stop generating candidates once top-5 haven't changed
3. **Caching**: Cache model on first load
4. **Quantization**: Use INT8 quantization for deployment (50% faster)

## Testing Strategy

### Unit Tests

```python
# tests/test_conditioning.py
def test_emotion_normalization():
    emotions = {"trust": 0.8, "joy": 0.4}
    normalized = normalize_emotions(emotions)
    assert abs(sum(normalized.values()) - 1.0) < 0.01

def test_emotion_vector_conversion():
    config = EmotionConditioningConfig(target_emotions={"joy": 1.0})
    vec = config.to_vector()
    assert vec.shape == (8,)
    assert vec[0] == 1.0  # joy is index 0

def test_multi_head_attention_shapes():
    attn = MultiHeadEmotionAttention(model_dim=136, num_heads=8)
    x = torch.randn(2, 10, 136)
    emotion = torch.rand(2, 10, 8)
    out = attn(x, emotion)
    assert out.shape == (2, 10, 136)
```

### Integration Tests

```python
def test_conditioned_generation():
    model = EmotionFlowLLM()
    conditioner = EmotionConditioner(model)
    prompt = torch.randint(0, 10000, (1, 5))
    
    config = EmotionConditioningConfig(
        target_emotions={"joy": 0.5, "trust": 0.5},
        num_candidates=20
    )
    
    outputs = conditioner.generate_conditioned(prompt, config)
    assert len(outputs) == 4  # Top 20% of 20
    
    # Verify outputs are closer to target than random
    # (statistical test with confidence interval)
```

## Deployment Checklist

- [ ] Implement `conditioning.py`
- [ ] Implement `multi_head_emotion_attention.py`
- [ ] Update `EmotionFlowLLM` to support multi-head
- [ ] Create `app.py` Gradio interface
- [ ] Add visualization functions
- [ ] Write unit tests
- [ ] Test UI locally
- [ ] Create `requirements.txt` for deployment
- [ ] Deploy to Hugging Face Spaces
- [ ] Update README with UI link and conditioning docs
- [ ] (Optional) Add Vercel deployment

