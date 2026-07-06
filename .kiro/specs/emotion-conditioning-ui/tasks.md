# Implementation Plan: Emotion Conditioning & Interactive UI

## Overview

Implement emotion conditioning interface, multi-head emotion attention, and Gradio web UI for EmotionFlow-LLM.

## Tasks

- [ ] 1. Emotion Conditioning Core
  - [ ] 1.1 Create `emotionflow_llm/conditioning.py`
  - [ ] 1.2 Implement `EmotionConditioningConfig` dataclass with validation
  - [ ] 1.3 Implement `EmotionConditioner` class with filtering strategy
  - [ ] 1.4 Add `normalize_emotions()` and `validate_emotion_conditioning()` utilities
  - [ ] 1.5 Implement `_extract_emotion_vector()` method
  - _Requirements: 1, 4_

- [ ] 2. Multi-Head Emotion Attention
  - [ ] 2.1 Create `emotionflow_llm/multi_head_emotion_attention.py`
  - [ ] 2.2 Implement `EmotionAttentionHead` class
  - [ ] 2.3 Implement `MultiHeadEmotionAttention` class with 8 heads
  - [ ] 2.4 Add configuration option to `EmotionFlowLLM` for multi-head vs single-head
  - [ ] 2.5 Update `TransformerBlock` to support both attention types
  - _Requirements: 2_

- [ ] 3. Visualization Extensions
  - [ ] 3.1 Add `plot_emotion_distribution()` to `visualize.py`
  - [ ] 3.2 Add `plot_emotion_trajectory_comparison()` to `visualize.py`
  - [ ] 3.3 Ensure plots work with Gradio (return matplotlib figures)
  - _Requirements: 3, 5_

- [ ] 4. Gradio Web UI
  - [ ] 4.1 Create `app.py` with Gradio interface
  - [ ] 4.2 Add prompt input and 8 emotion sliders
  - [ ] 4.3 Add temperature and num_samples controls
  - [ ] 4.4 Implement `generate_with_ui()` function
  - [ ] 4.5 Add output displays: text, emotion distribution plot, trajectory plot
  - [ ] 4.6 Add example presets (joyful, fearful, analytical)
  - [ ] 4.7 Add error handling for invalid inputs
  - [ ] 4.8 Test locally with `gradio.launch()`
  - _Requirements: 3_

- [ ] 5. Tokenization Utilities
  - [ ] 5.1 Add `simple_tokenize()` function for demo purposes
  - [ ] 5.2 Add `decode_tokens()` function for text output
  - [ ] 5.3 (Optional) Integrate proper tokenizer (e.g., BPE or GPT-2 tokenizer)
  - _Requirements: 3_

- [ ] 6. Testing
  - [ ] 6.1 Create `tests/test_conditioning.py`
  - [ ] 6.2 Test emotion normalization and validation
  - [ ] 6.3 Test `EmotionConditioningConfig.to_vector()`
  - [ ] 6.4 Test `EmotionConditioner.generate_conditioned()` output shapes
  - [ ] 6.5 Create `tests/test_multi_head_attention.py`
  - [ ] 6.6 Test `EmotionAttentionHead` forward pass shapes
  - [ ] 6.7 Test `MultiHeadEmotionAttention` output shapes and weighted combination
  - [ ] 6.8 Integration test: conditioned generation produces closer matches to target
  - _Requirements: All_

- [ ] 7. Documentation & Examples
  - [ ] 7.1 Create `examples/emotion_conditioning.py` with usage examples
  - [ ] 7.2 Update README.md with emotion conditioning section
  - [ ] 7.3 Add API documentation for conditioning classes
  - [ ] 7.4 Add multi-head attention explanation to README
  - [ ] 7.5 Add UI usage instructions
  - _Requirements: 5_

- [ ] 8. Deployment Preparation
  - [ ] 8.1 Create `requirements.txt` for deployment (torch, gradio, matplotlib, numpy)
  - [ ] 8.2 Test app.py runs without errors
  - [ ] 8.3 Optimize for CPU inference (ensure < 5s response time)
  - [ ] 8.4 Add loading message/spinner in UI
  - _Requirements: 3, 6_

- [ ] 9. Hugging Face Spaces Deployment
  - [ ] 9.1 Create Hugging Face account (if not exists)
  - [ ] 9.2 Create new Space (Gradio type)
  - [ ] 9.3 Add `README.md` for Space description
  - [ ] 9.4 Push code to Hugging Face Space repository
  - [ ] 9.5 Test deployed UI works correctly
  - [ ] 9.6 Update main README with live demo link
  - _Requirements: 3, 5_

- [ ] 10. (Optional) Advanced Features
  - [ ] 10.1 Implement biased sampling strategy (Strategy B)
  - [ ] 10.2 Add emotion trajectory comparison visualization
  - [ ] 10.3 Add "Surprise me" button that generates random emotion profiles
  - [ ] 10.4 Add download button for generated samples
  - [ ] 10.5 Add sharing functionality for interesting generations
  - _Requirements: 1, 3_

## Task Dependencies

```
1 (Conditioning Core)
  ├─→ 5 (Tokenization) [needed for demo]
  ├─→ 6.1-6.4 (Conditioning tests)
  └─→ 7.1 (Examples)

2 (Multi-Head Attention)
  ├─→ 6.5-6.7 (Multi-head tests)
  └─→ 7.4 (Documentation)

3 (Visualization Extensions)
  └─→ 4 (Gradio UI) [uses plots]

4 (Gradio UI)
  ├─→ 1 (Conditioning) [core functionality]
  ├─→ 3 (Visualizations) [for plots]
  ├─→ 5 (Tokenization) [for text I/O]
  └─→ 8 (Deployment Prep)

7 (Documentation)
  └─→ 9 (Deployment) [README needs demo link]

8 (Deployment Prep)
  └─→ 9 (HF Spaces) [must work locally first]
```

## Recommended Order

### Phase 1: Core Functionality (Tasks 1, 2, 5)
Build the emotion conditioning and multi-head attention systems.

### Phase 2: UI & Visualization (Tasks 3, 4)
Create the Gradio interface and visualization functions.

### Phase 3: Testing & Polish (Tasks 6, 7, 8)
Add comprehensive tests and documentation.

### Phase 4: Deployment (Task 9)
Deploy to Hugging Face Spaces for live demo.

### Phase 5: Enhancements (Task 10)
Optional advanced features if time permits.

## Success Criteria

- [ ] User can specify target emotions via dict or sliders
- [ ] Generated text matches target emotions better than random
- [ ] Multi-head attention model works and can be toggled on/off
- [ ] Gradio UI runs locally without errors
- [ ] UI responds in < 5 seconds on CPU
- [ ] All tests pass
- [ ] Deployed to Hugging Face Spaces with public link
- [ ] README includes conditioning API docs and demo link

