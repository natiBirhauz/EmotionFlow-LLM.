# Requirements Document

## Introduction

EmotionFlow-LLM is an experimental emotion-aware language model implemented as a single Python module (`emotionflow_llm.py`). It integrates 8-dimensional emotional representations into a transformer architecture, generates multiple candidate responses, synthesizes them, and tracks emotional trajectories across layers.

## Glossary

- **EmotionVector**: 8-dimensional dataclass with values for joy, sadness, anger, fear, trust, disgust, surprise, and anticipation
- **EmotionEncoder**: `nn.Module` that maps token IDs to emotion vectors via a learned embedding + sigmoid
- **TokenEmbedding**: Combines word embeddings and emotion vectors into a single concatenated representation
- **EmotionAttention**: Attention mechanism that adds an emotion bias to Q and K projections
- **TransformerBlock**: Single transformer layer using EmotionAttention + FFN + LayerNorm
- **EmotionFlowLLM**: Full model stacking N TransformerBlocks with a language model head
- **GenerationConfig**: Dataclass for sampling strategy parameters
- **PipelineResult**: Dataclass holding synthesized output, candidates, trajectories, and timing stats
- **PerformanceMonitor**: Context-manager-based timing utility

## Requirements

### Requirement 1: Emotion Vector

**User Story:** As a researcher, I want each token to carry an 8-dimensional emotion vector so emotional context can influence model reasoning.

#### Acceptance Criteria

1. `EmotionEncoder.forward(tokens)` SHALL return a tensor of shape `[batch, seq_len, 8]`
2. All values in the returned tensor SHALL be in `[0, 1]` (enforced via sigmoid)
3. `EmotionEncoder.validate()` SHALL clamp out-of-range values and replace NaN/inf with zeros, logging each occurrence
4. `EmotionVector.validate()` SHALL return `True` iff all 8 fields are in `[0, 1]`
5. `EmotionVector.dominant_emotion()` SHALL return the name of the highest-valued emotion

### Requirement 2: Token Embedding

**User Story:** As a developer, I want word and emotion embeddings concatenated so the transformer receives a unified representation.

#### Acceptance Criteria

1. `TokenEmbedding.forward(tokens)` SHALL return `(combined, emotions)` where `combined` has shape `[batch, seq_len, WORD_DIM + EMOTION_DIM]`
2. `emotions` SHALL be the raw output of `EmotionEncoder` with shape `[batch, seq_len, 8]`

### Requirement 3: Emotion-Aware Attention

**User Story:** As a researcher, I want attention to incorporate emotional context so the model reasons with emotional awareness.

#### Acceptance Criteria

1. `EmotionAttention.forward(x, emotion)` SHALL add an emotion projection bias to both Q and K before computing attention
2. Attention weights SHALL be computed with scaled dot-product and softmax, preserving normalization (sum to 1 per row)
3. V SHALL remain unmodified by emotion

### Requirement 4: Transformer Stack

**User Story:** As a developer, I want a configurable stack of emotion-aware transformer layers.

#### Acceptance Criteria

1. `EmotionFlowLLM` SHALL stack `LAYERS` `TransformerBlock` instances
2. Each block SHALL apply residual connections and LayerNorm around attention and FFN
3. `EmotionFlowLLM.forward(tokens)` SHALL store per-layer activations in `self.activations` and return logits of shape `[batch, seq_len, VOCAB_SIZE]`
4. `get_activations()` SHALL return the list of stored layer activations

### Requirement 5: Text Generation

**User Story:** As a user, I want to generate text autoregressively from a prompt.

#### Acceptance Criteria

1. `generate(model, prompt, max_len, temperature)` SHALL append `max_len` tokens to the prompt one at a time
2. Each token SHALL be sampled from the softmax distribution scaled by `temperature`
3. The function SHALL return the full token sequence including the original prompt

### Requirement 6: Multi-Sample Generation

**User Story:** As a researcher, I want multiple diverse candidate outputs from a single prompt.

#### Acceptance Criteria

1. `generate_samples(model, prompt, n)` SHALL return a list of exactly `n` generated sequences
2. Default `n` SHALL be 20
3. Each sample SHALL be generated independently using `generate()`

### Requirement 7: Sampling Strategy Configuration

**User Story:** As a researcher, I want to configure sampling parameters with validation.

#### Acceptance Criteria

1. `GenerationConfig` SHALL have fields: `num_samples=20`, `max_length=100`, `strategy="temperature"`, `temperature=0.8`, `top_p=0.9`, `top_k=50`
2. `GenerationConfig.validate()` SHALL raise `ValueError` with a descriptive message if `temperature` ∉ `[0.1, 2.0]`, `top_p` ∉ `[0.0, 1.0]`, `top_k` ∉ `[1, 100]`, or `strategy` is not one of `["temperature", "nucleus", "top_k"]`

### Requirement 8: Emotion Analysis

**User Story:** As a researcher, I want to extract the dominant emotion from a generated sequence.

#### Acceptance Criteria

1. `emotion_profile(model, tokens)` SHALL return the name of the dominant emotion (string) for the given token sequence
2. It SHALL compute the mean emotion vector across the sequence and return the argmax emotion name

### Requirement 9: Output Synthesis

**User Story:** As a user, I want multiple candidates merged into a single output.

#### Acceptance Criteria

1. `synthesize(outputs)` SHALL accept a list of token tensors and return a single tensor
2. The returned tensor SHALL be truncated to `SEQ_LEN` tokens

### Requirement 10: Emotion Trajectory

**User Story:** As a researcher, I want to inspect how emotional activations evolve across transformer layers.

#### Acceptance Criteria

1. `collect_emotion_trajectory(model)` SHALL return a list of scalar values, one per layer
2. Each value SHALL be the mean of that layer's stored activation tensor

### Requirement 11: Performance Monitoring

**User Story:** As a researcher, I want to time each pipeline component.

#### Acceptance Criteria

1. `PerformanceMonitor.measure(name)` SHALL be usable as a context manager that records elapsed time under `name`
2. `PerformanceMonitor.report()` SHALL return a dict mapping component names to `{mean, min, max, total}` timing stats

### Requirement 12: Pipeline Result

**User Story:** As a developer, I want a structured object holding all pipeline outputs.

#### Acceptance Criteria

1. `PipelineResult` SHALL be a dataclass with fields: `synthesized_text`, `candidates`, `emotion_trajectories`, `timing_stats`, `token_emotions`
