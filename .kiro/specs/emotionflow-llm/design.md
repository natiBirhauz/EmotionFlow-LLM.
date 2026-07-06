# Design Document: EmotionFlow-LLM

## Overview

EmotionFlow-LLM is implemented as a single self-contained Python module: `emotionflow_llm.py`. The design prioritises simplicity — all components live in one file with no external package structure. PyTorch is the only deep-learning dependency.

### Key Design Decisions

- **Single-file**: Everything in `emotionflow_llm.py`; no sub-packages
- **Global constants** drive model dimensions (`VOCAB_SIZE`, `WORD_DIM`, `EMOTION_DIM`, `MODEL_DIM`, `HEADS`, `LAYERS`)
- **Emotion as additive bias**: emotion vectors are projected and added to Q and K, keeping the attention math standard
- **Activations stored on the model**: `EmotionFlowLLM.activations` is populated each forward pass for trajectory inspection
- **Synthesis by concatenation + truncation**: simple and dependency-free

---

## Module Layout

```
emotionflow_llm.py
├── Constants & globals          (VOCAB_SIZE, WORD_DIM, EMOTION_DIM, …)
├── EMOTIONS list + emotion_to_id
├── Data models                  (EmotionVector, GenerationConfig, PipelineResult)
├── PerformanceMonitor
├── EmotionEncoder               (nn.Module)
├── TokenEmbedding               (nn.Module)
├── EmotionAttention             (nn.Module)
├── TransformerBlock             (nn.Module)
├── EmotionFlowLLM               (nn.Module)
├── generate()
├── generate_samples()
├── emotion_profile()
├── synthesize()
├── collect_emotion_trajectory()
└── __main__ demo
```

---

## Components

### Constants

```python
VOCAB_SIZE = 10000
SEQ_LEN    = 32
WORD_DIM   = 128
EMOTION_DIM = 8
MODEL_DIM  = WORD_DIM + EMOTION_DIM   # 136
HEADS      = 4
LAYERS     = 3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
```

### EmotionVector (dataclass)

Fields: `joy, sadness, anger, fear, trust, disgust, surprise, anticipation` — all `float`.

Methods:
- `to_tensor() → Tensor[8]`
- `validate() → bool` — all values in `[0, 1]`
- `dominant_emotion() → str`
- `from_tensor(tensor) → EmotionVector` (classmethod)

### GenerationConfig (dataclass)

Fields with defaults: `num_samples=20`, `max_length=100`, `strategy="temperature"`, `temperature=0.8`, `top_p=0.9`, `top_k=50`.

`validate()` raises `ValueError` for out-of-range parameters.

### PipelineResult (dataclass)

Fields: `synthesized_text`, `candidates`, `emotion_trajectories`, `timing_stats`, `token_emotions`.

### PerformanceMonitor

```python
class PerformanceMonitor:
    timings: Dict[str, List[float]]

    @contextmanager
    def measure(self, name: str): ...   # records elapsed seconds

    def report(self) -> Dict[str, Dict[str, float]]: ...  # mean/min/max/total
```

### EmotionEncoder (nn.Module)

```python
class EmotionEncoder(nn.Module):
    embedding: nn.Embedding(vocab_size, emotion_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: [B, S]  →  [B, S, 8]  via embedding + sigmoid
        # then self.validate()

    def validate(self, emotion_vector: Tensor) -> Tensor:
        # 1. assert last dim == 8
        # 2. replace NaN/inf with 0, log warning
        # 3. clamp to [0, 1], log warning
```

### TokenEmbedding (nn.Module)

```python
class TokenEmbedding(nn.Module):
    word_embedding: nn.Embedding(vocab_size, word_dim)
    emotion_embedding: EmotionEncoder

    def forward(self, tokens) -> Tuple[Tensor, Tensor]:
        # returns (cat([word, emotion], dim=-1), emotion)
        # shapes: ([B,S,136], [B,S,8])
```

### EmotionAttention (nn.Module)

```python
class EmotionAttention(nn.Module):
    q, k, v: nn.Linear(dim, dim)
    emotion_proj: nn.Linear(EMOTION_DIM, dim)

    def forward(self, x, emotion):
        emotion_bias = emotion_proj(emotion)   # [B, S, dim]
        Q = q(x) + emotion_bias
        K = k(x) + emotion_bias
        V = v(x)
        scores = Q @ K.T / sqrt(dim)
        attn   = softmax(scores, dim=-1)
        return attn @ V
```

### TransformerBlock (nn.Module)

```python
class TransformerBlock(nn.Module):
    attn:  EmotionAttention(dim)
    ff:    Linear(dim, 4*dim) → ReLU → Linear(4*dim, dim)
    norm1, norm2: LayerNorm(dim)

    def forward(self, x, emotion):
        x = norm1(x + attn(x, emotion))
        x = norm2(x + ff(x))
        return x
```

### EmotionFlowLLM (nn.Module)

```python
class EmotionFlowLLM(nn.Module):
    embedding: TokenEmbedding()
    layers:    ModuleList([TransformerBlock(MODEL_DIM)] * LAYERS)
    lm_head:   Linear(MODEL_DIM, VOCAB_SIZE)
    activations: List[Tensor]   # populated each forward pass

    def forward(self, tokens):
        x, emotion = embedding(tokens)
        for layer in layers:
            x = layer(x, emotion)
            activations.append(x.detach())
        return lm_head(x)   # [B, S, VOCAB_SIZE]

    def get_activations(self) -> List[Tensor]: ...
```

### Free Functions

```python
def generate(model, prompt, max_len=20, temperature=0.9) -> Tensor:
    # autoregressive loop: sample next token, cat to sequence

def generate_samples(model, prompt, n=20) -> List[Tensor]:
    # calls generate() n times independently

def emotion_profile(model, tokens) -> str:
    # mean emotion across sequence → argmax → EMOTIONS[idx]

def synthesize(outputs: List[Tensor]) -> Tensor:
    # cat all outputs along seq dim, truncate to SEQ_LEN

def collect_emotion_trajectory(model) -> List[float]:
    # [layer.mean().item() for layer in model.get_activations()]
```

---

## Data Flow

```
tokens [B, S]
  │
  ▼ TokenEmbedding
combined [B, S, 136],  emotion [B, S, 8]
  │
  ▼ TransformerBlock × LAYERS  (stores activations)
x [B, S, 136]
  │
  ▼ lm_head
logits [B, S, VOCAB_SIZE]
  │
  ▼ generate_samples (×20)
List[Tensor]
  │
  ▼ synthesize
Tensor [B, SEQ_LEN]
```

---

## Correctness Properties

| # | Property | Validated by |
|---|----------|-------------|
| 1 | `EmotionEncoder` output shape is `[B, S, 8]` | `test_emotion_encoder_shape` |
| 2 | All emotion values in `[0, 1]` after forward | `test_emotion_values_range` |
| 3 | NaN/inf replaced with 0 in `validate()` | `test_validate_nan_inf` |
| 4 | Out-of-range values clamped in `validate()` | `test_validate_clamp` |
| 5 | `TokenEmbedding` combined shape is `[B, S, 136]` | `test_token_embedding_shape` |
| 6 | Attention weights sum to 1 per row | `test_attention_weights_sum` |
| 7 | `EmotionFlowLLM` logits shape is `[B, S, VOCAB_SIZE]` | `test_model_output_shape` |
| 8 | `get_activations()` returns `LAYERS` tensors | `test_activations_count` |
| 9 | `generate()` returns sequence of length `prompt_len + max_len` | `test_generate_length` |
| 10 | `generate_samples(n=20)` returns exactly 20 items | `test_generate_samples_count` |
| 11 | `GenerationConfig.validate()` raises on bad params | `test_generation_config_validate` |
| 12 | `emotion_profile()` returns a string in `EMOTIONS` | `test_emotion_profile_valid` |
| 13 | `synthesize()` output length ≤ `SEQ_LEN` | `test_synthesize_length` |
| 14 | `collect_emotion_trajectory()` length == `LAYERS` | `test_trajectory_length` |
| 15 | `PerformanceMonitor.report()` keys match measured components | `test_performance_monitor` |

---

## Testing Strategy

All tests live in `tests/test_emotionflow.py` using `pytest`. Property-based tests use `hypothesis`.

```python
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=emotionflow_llm
```

Tests are grouped by component and cover both happy-path and edge cases (empty sequences, NaN inputs, boundary parameter values).
