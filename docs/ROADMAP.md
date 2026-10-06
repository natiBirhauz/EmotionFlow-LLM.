# Roadmap & improvement notes

Findings from the October 2026 clean-up, grouped by area. Items marked **done** shipped with the restructuring PR; everything else is a proposal.

## 1. Web app (EmotionFlow Studio)

### Done in the clean-up
- `api/index.py` (1,800 lines, HTML inside a Python string) split into `emotionflow_studio/` package + Jinja template.
- Input validation: `creativity`, `length`, `variations` clamped; `mode`/`language` whitelisted; emotion keys filtered and values clamped to [0, 1]; request body capped at 256 KB; `/api/analyze` rejects >20k chars.
- Variation offsets used Python's `hash()`, which is salted per process → the same request produced different "variations" on every cold start. Replaced with SHA-256 based `variation_offset()`.
- Local analyser matched substrings (`"sad"` fired on *saddle*, `"mad"` on *made*) and gave every emotion a 0.05 floor, so neutral text was never neutral. Now whole-word (with Hebrew clitic prefixes ו/ה/ב/ל/מ/ש/כ) and truly 0 when no cue is found; sentences without cues are labelled `neutral`.
- LLM JSON replies are parsed by one tolerant helper (`parse_llm_json`) and emotion scores re-validated instead of trusted.
- Model output was inserted into `innerHTML` unescaped — an LLM reply containing `<img onerror=…>` would execute. Output is now HTML-escaped on the client.
- `print()` → `logging`; provider model lists / timeouts configurable by env.

### Proposed
1. **One LLM call instead of two per variation.** Today every variation = 1 generation call + 1 annotation call (+1 more for `/api/analyze`). Ask the model to return `{"draft": …, "sentences": [{"text","emotion"}]}` in a single JSON-mode call; fall back to the local annotator when the sentences list is missing. Halves latency and quota usage.
2. **Emotion fidelity loop.** After generation, score the draft (`analyze_text_emotions`) and compare against the requested profile (cosine / L1). If the distance exceeds a threshold, re-prompt once with "the previous draft was too {dominant_detected}; increase {requested}". This is the single biggest quality win and is cheap to implement.
3. **Stream the draft** (`text/event-stream`) so Hebrew 4-paragraph outputs don't look frozen for 10–20 s.
4. **Cache** identical `(prompt, mode, emotions, length, language)` requests for a few minutes (in-memory LRU is enough on Vercel; Upstash Redis if persistence is wanted). Also rate-limit per IP — the API is currently open and burns the shared Gemini quota.
5. **Secondary emotions are under-used.** The prompt only mentions the top emotion strongly and two "undertones". Plutchik's model defines *dyads* (joy+trust = love, fear+surprise = awe, anger+anticipation = aggressiveness…). Map the top-2 sliders to the named dyad and put it in the prompt — the output becomes noticeably more specific.
6. **Local analyser upgrades:** stem/lemmatise English (`-ness`, `-ful`), add negation handling ("not happy" → sadness, not joy), weight intensifiers ("very", "מאוד"), and expand the Hebrew lexicon (currently ~10 words per emotion). Longer term, a small multilingual classifier (e.g. fine-tuned `xlm-roberta-base` on GoEmotions mapped to Plutchik) run on the server replaces the lexicon entirely and works offline.
7. **Frontend:** split the 1,250-line template into `static/app.js` + `static/styles.css`; add a "requested vs detected" radar overlay (two series on the existing radar chart); persist last prompt/sliders in `localStorage`; make the slider panel accessible (labels, `aria-valuetext`).
8. **Observability:** log provider used, latency and fallback events per request (structured JSON); expose them in `/health`.
9. **Tests:** add a mocked-provider test (monkeypatch `call_free_llm`) covering the JSON-mode paths, and a Hypothesis property test that `normalize_emotions` output is always within [0, 1].

## 2. Research model (`emotionflow_llm/`)

The package is a clean, readable sketch of emotion-conditioned attention, but several things prevent it from being an actual language model:

1. **No causal mask.** `EmotionAttention` applies softmax over all positions, so during autoregressive `generate()` each token attends to tokens that don't exist yet at inference time. Add `torch.triu` masking (or `F.scaled_dot_product_attention(is_causal=True)`).
2. **"Multi-head" is single-head.** `HEADS = 4` is stored but never used; Q/K/V are full-dimension. Either reshape into `[B, H, S, d/H]` or drop the constant.
3. **Tokenizer is `hash(char) % vocab`.** Hash salting means the same text gets different IDs per process and `decode_tokens()` cannot invert it. Swap in a real BPE tokenizer (`tokenizers` library / GPT-2 vocab) — this also unlocks pre-trained embeddings.
4. **No training pipeline.** Weights are random, so every figure in `docs/images/` shows mechanics only. Minimal viable training: GoEmotions (58k Reddit comments, 27 labels → map to Plutchik 8) or `dair-ai/emotion`, LM loss + auxiliary emotion-classification loss on the pooled hidden state. Even a 2-layer model trained for an hour on CPU would make the emotion trajectory plots meaningful.
5. **Emotion is injected only as a Q/K bias.** This changes *where* the model attends but not *what it says* (V is untouched, and the FFN never sees the emotion). Stronger alternatives in the literature: FiLM-style scale/shift of the hidden state per layer, or adding the emotion embedding to the residual stream after each block (as CTRL/PPLM-style control codes do).
6. **Conditioning = generate-many-then-filter.** `_generate_with_biasing` raises `NotImplementedError`. Implement logit biasing: at each step add `λ · W_emo · e` to the vocabulary logits, where `W_emo` is a learned (vocab × 8) matrix, or use a lightweight classifier-guided decoding (PPLM). The current filter strategy costs N× compute and offers no guarantee the filter is satisfiable.
7. **Emotional trajectory ≠ mean activation.** `get_activations()` averages hidden states; an "emotion trajectory" needs a probe: train a linear classifier per layer from hidden state → emotion and plot its output. This also gives a quantitative metric (per-layer probe accuracy) instead of pictures.
8. **Evaluation.** There is no metric for "did the text carry the requested emotion?". Define one: agreement between requested profile and an external classifier on the output (the same classifier can serve item 1.2 above). Track it in CI for both the local engine and the LLM path.

## 3. Repo hygiene
- `.kiro/specs/` describes the old single-file / Gradio layout. Either refresh or delete.
- `tests/test_emotionflow.py` requires torch; mark it with `@pytest.mark.research` so the web-app suite stays fast in CI.
- Add a GitHub Actions workflow: `pip install -r requirements-dev.txt` (CPU torch) → `pytest`.
