# Requirements Document: Emotion Conditioning & Interactive UI

## Introduction

This spec extends EmotionFlow-LLM with three major features:
1. **Emotion Conditioning Interface**: Allow users to control the emotional profile of generated text
2. **Multi-Head Emotion Attention**: Use separate attention heads for each emotion dimension
3. **Interactive Web UI**: Gradio/Streamlit interface for live demonstrations

## Glossary

- **Emotion Conditioning**: User-specified target emotion vector that guides text generation
- **Multi-Head Emotion Attention**: Attention mechanism with 8 heads, one per emotion dimension
- **Conditioning Loss**: Additional loss term that encourages generated text to match target emotions
- **Gradio**: Python library for building ML web interfaces
- **Streamlit**: Python framework for creating data/ML web apps

## Requirements

### Requirement 1: Emotion Conditioning Interface

**User Story:** As a user, I want to specify the desired emotional profile for generated text (e.g., "80% trust, 20% joy") so I can control the emotional tone.

#### Acceptance Criteria

1. `EmotionConditioner` class SHALL accept a target emotion vector or emotion name + intensity
2. `generate_conditioned()` function SHALL modify the generation process to guide toward target emotions
3. Target emotions SHALL be validated to sum to 1.0 (probability distribution)
4. Conditioning SHALL work by:
   - Adding emotion loss term during generation: `|generated_emotion - target_emotion|²`
   - Or by biasing the emotion encoder toward target values
5. API SHALL support both formats:
   - Dictionary: `{"trust": 0.8, "joy": 0.2}`
   - Vector: `torch.tensor([0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0])`  # [joy, sadness, anger, fear, trust, disgust, surprise, anticipation]

### Requirement 2: Multi-Head Emotion Attention

**User Story:** As a researcher, I want attention heads specialized for different emotions so the model can capture complex mixed emotional states.

#### Acceptance Criteria

1. `MultiHeadEmotionAttention` class SHALL implement 8 attention heads, one per emotion
2. Each head SHALL focus on its corresponding emotion dimension
3. Heads SHALL be combined via weighted sum, where weights come from the emotion vector
4. Forward pass SHALL:
   - Split input across 8 heads
   - Apply emotion-specific attention for each head
   - Combine outputs weighted by emotion strengths
5. Model SHALL remain compatible with existing `EmotionFlowLLM` architecture

### Requirement 3: Interactive Web UI

**User Story:** As a portfolio visitor, I want to interact with the model through a web interface so I can see it working without installing code locally.

#### Acceptance Criteria

1. `app.py` SHALL implement a Gradio interface with:
   - Text input for prompt
   - Sliders for 8 emotion dimensions (sum to 1.0)
   - Temperature slider
   - Number of samples slider
   - Generate button
2. Output SHALL display:
   - Generated text samples
   - Emotion distribution visualization (bar chart)
   - Emotion trajectory plot across layers
3. UI SHALL handle errors gracefully (invalid inputs, model errors)
4. Optional: SHALL be deployable to Vercel/Hugging Face Spaces for public access
5. Interface SHALL be responsive and work on mobile devices

### Requirement 4: Conditioning Validation

**User Story:** As a developer, I want the conditioning system to validate inputs and provide helpful error messages.

#### Acceptance Criteria

1. `validate_emotion_conditioning()` SHALL check:
   - All emotion values are in [0, 1]
   - Emotion vector sums to 1.0 (±0.01 tolerance)
   - At least one emotion has value > 0
2. SHALL raise `ValueError` with descriptive message on invalid input
3. SHALL provide `normalize_emotions()` utility to auto-fix minor issues

### Requirement 5: Examples and Documentation

**User Story:** As a user, I want clear examples showing how to use emotion conditioning.

#### Acceptance Criteria

1. `examples/emotion_conditioning.py` SHALL demonstrate:
   - Basic conditioning with single emotion
   - Mixed emotion conditioning
   - Comparing outputs with/without conditioning
2. README SHALL include:
   - Emotion conditioning API documentation
   - Multi-head attention explanation
   - UI usage instructions
   - Deployment guide for Vercel/HF Spaces

### Requirement 6: Performance Considerations

**User Story:** As a developer, I want multi-head attention to have acceptable performance overhead.

#### Acceptance Criteria

1. Multi-head attention SHALL add < 20% inference time vs single-head
2. Conditioning SHALL add < 5% inference time
3. UI SHALL respond within 2 seconds for typical generation (20 samples, length 30)
4. Model SHALL remain deployable on CPU (no GPU requirement for demo)

## Technical Notes

### Emotion Conditioning Approaches

**Approach A: Post-hoc Filtering (Simple)**
- Generate many samples (e.g., 50)
- Filter/rank by distance to target emotion
- Return top N closest matches

**Approach B: Biased Sampling (Medium)**
- Modify generation logits based on emotion gradient
- At each step, adjust token probabilities to favor target emotion

**Approach C: Reinforcement Learning (Complex)**
- Train with emotion distance as reward signal
- Use policy gradient to optimize for target emotions

*Recommendation: Start with Approach A, optionally add Approach B*

### Multi-Head Emotion Attention Architecture

```python
class MultiHeadEmotionAttention(nn.Module):
    def __init__(self, dim=136, emotion_dim=8, num_heads=8):
        # num_heads = emotion_dim = 8
        self.heads = nn.ModuleList([
            EmotionAttentionHead(dim, head_dim=dim // num_heads)
            for _ in range(num_heads)
        ])
        self.combine = nn.Linear(dim, dim)
    
    def forward(self, x, emotion):
        # emotion: [B, S, 8]
        head_outputs = []
        for i, head in enumerate(self.heads):
            # Extract emotion dim i
            emo_i = emotion[:, :, i:i+1]  # [B, S, 1]
            out_i = head(x, emo_i)
            head_outputs.append(out_i)
        
        # Weighted combination
        # emotion weights: [B, S, 8, 1, 1]
        # head_outputs: 8 × [B, S, D]
        stacked = torch.stack(head_outputs, dim=2)  # [B, S, 8, D]
        weights = emotion.unsqueeze(-1)  # [B, S, 8, 1]
        combined = (stacked * weights).sum(dim=2)  # [B, S, D]
        return self.combine(combined)
```

### Deployment Options

**Option 1: Gradio + Hugging Face Spaces**
- Free hosting
- Easy deployment (push to git)
- Good for demos

**Option 2: Streamlit + Vercel**
- More customization
- Better for portfolio integration
- May require serverless function setup

**Option 3: FastAPI + Vercel**
- Most flexible
- Can build custom frontend
- More development effort

*Recommendation: Start with Gradio for rapid prototyping*

