# Implementation Plan: EmotionFlow-LLM

## Overview

All implementation lives in `emotionflow_llm.py`. Tasks below complete and harden that file, then add a matching test suite in `tests/test_emotionflow.py`.

## Tasks

- [x] 1. Core data models and constants
  - EmotionVector, GenerationConfig, PipelineResult dataclasses
  - EMOTIONS list, emotion_to_id, global constants
  - _Requirements: 1, 7, 12_

- [x] 2. Complete EmotionEncoder
  - [x] 2.1 Verify forward() returns [B, S, 8] with sigmoid
  - [x] 2.2 Harden validate(): NaN/inf → 0, clamp to [0,1], log warnings
  - _Requirements: 1_

- [x] 3. Complete TokenEmbedding
  - Verify forward() returns (combined [B,S,136], emotions [B,S,8])
  - _Requirements: 2_

- [x] 4. Complete EmotionAttention
  - Verify emotion bias added to Q and K, V unchanged
  - Verify softmax normalization
  - _Requirements: 3_

- [x] 5. Complete TransformerBlock
  - Verify residual connections and LayerNorm around attention and FFN
  - _Requirements: 4_

- [x] 6. Complete EmotionFlowLLM
  - Verify activations stored per layer in self.activations
  - Verify logits shape [B, S, VOCAB_SIZE]
  - _Requirements: 4_

- [x] 7. Complete generation functions
  - [x] 7.1 generate(): autoregressive loop, temperature scaling
  - [x] 7.2 generate_samples(): calls generate() n=20 times
  - _Requirements: 5, 6_

- [x] 8. Complete utility functions
  - [x] 8.1 emotion_profile(): mean emotion → argmax → EMOTIONS name
  - [x] 8.2 synthesize(): cat outputs, truncate to SEQ_LEN
  - [x] 8.3 collect_emotion_trajectory(): mean per layer activation
  - _Requirements: 8, 9, 10_

- [x] 9. Complete PerformanceMonitor
  - measure() context manager, report() with mean/min/max/total
  - _Requirements: 11_

- [x] 10. Write test suite in tests/test_emotionflow.py
  - [x] 10.1 Shape and range tests for EmotionEncoder
  - [x] 10.2 validate() edge cases: NaN, inf, out-of-range
  - [x] 10.3 TokenEmbedding shape test
  - [x] 10.4 Attention weight normalization test
  - [x] 10.5 EmotionFlowLLM output shape and activations count
  - [x] 10.6 generate() length test
  - [x] 10.7 generate_samples() count test
  - [x] 10.8 GenerationConfig.validate() error cases
  - [x] 10.9 emotion_profile() returns valid emotion name
  - [x] 10.10 synthesize() length constraint
  - [x] 10.11 collect_emotion_trajectory() length == LAYERS
  - [x] 10.12 PerformanceMonitor report keys
  - _Requirements: all_

- [x] 11. Final validation
  - Run pytest, confirm all tests pass
  - Verify __main__ demo runs without errors
