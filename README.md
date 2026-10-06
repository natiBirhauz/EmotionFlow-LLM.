# EmotionFlow Studio

**AI-powered creative writing with emotion control**

Control the emotional tone of AI-generated text using Plutchik's 8 core emotions. Generate stories, emails, pitches and social posts with precise emotional colouring — in English and Hebrew — and analyse any text sentence-by-sentence.

**Live demo:** https://emotion-flow-llm-six.vercel.app/

---

## Features

**Emotion control**
- 8 Plutchik emotions: joy, sadness, anger, fear, trust, disgust, surprise, anticipation
- Interactive Plutchik wheel + quick presets (Happy, Sad, Dramatic, Calm, Intense)
- 1–4 variations per prompt with deterministic emotion diversity

**Generation**
- Google Gemini free tier → Groq free tier → OpenAI, with a zero-dependency local engine as the final fallback
- English & Hebrew (RTL aware)
- Formats: story, email, pitch, social post

**Analysis**
- Emotion scores with bar & radar charts (Chart.js)
- Sentence-by-sentence colouring; sentences without an emotional cue are marked `neutral`

**No login, no registration, no user API key.**

---

## Repository layout

```
api/index.py              Flask entrypoint (Vercel function) – routes + request validation only
emotionflow_studio/       Web-app logic (framework-free, unit-tested)
  emotions.py             Plutchik vocabulary: labels, prompt descriptions, keyword lexicon
  llm_providers.py        Gemini / Groq / OpenAI HTTP clients + tolerant JSON parsing
  local_engine.py         Offline fallback: template drafts + lexicon analyser
  services.py             generate / analyse / annotate orchestration, input normalisation
  templates/index.html    The single-page UI (vanilla JS + Chart.js)
emotionflow_llm/          Experimental PyTorch emotion-conditioned transformer (research only,
                          NOT used by the web app – see docs/MODEL_EXPLANATION.md)
research/                 Scripts that exercise the research model (visualisations, demo)
docs/                     Model explanation, generated figures, ROADMAP.md
tests/                    pytest suite (web app tests run offline; model tests need torch)
```

---

## Running locally

```bash
pip install -r requirements.txt          # Flask only
python api/index.py                      # http://127.0.0.1:5000
```

Without API keys the app uses the built-in local engine. To enable LLM generation copy `.env.example` and export one or more of `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENAI_API_KEY`.

Optional env vars: `GEMINI_MODELS`, `GROQ_MODELS`, `OPENAI_MODEL` (comma-separated fallback lists), `LLM_TIMEOUT_SECONDS`, `LOG_LEVEL`.

### API

| Route | Method | Body / query |
|---|---|---|
| `/api/generate` | POST, GET | `prompt`, `mode` (story/email/pitch/social), `creativity` 0–1, `length` 2–5, `variations` 1–4, `language` (english/hebrew), `emotions` `{joy: 0.8, ...}` |
| `/api/analyze` | POST | `text`, `language` |
| `/health` | GET | — |

Out-of-range values are clamped, unknown emotion keys are dropped.

### Development & research

```bash
pip install -r requirements-dev.txt      # pytest, hypothesis, torch (CPU is fine), matplotlib…
python -m pytest                         # whole suite
python -m pytest tests/test_free_llm_api.py tests/test_studio_services.py   # web app only, no torch

python research/visualize.py             # regenerates docs/images/viz_*.png
python research/neuron_emotion_viz.py    # regenerates docs/images/neuron_*.png, emotion_flow.png
python research/emotion_conditioning_demo.py
```

---

## The science

Based on **Robert Plutchik's wheel of emotions** (1980): 8 primary emotions that blend into complex feelings. The app turns slider values into explicit intensity instructions for the LLM, then re-analyses the output sentence-by-sentence so the colours reflect what was *actually* written, not just what was requested.

The `emotionflow_llm/` package explores the same idea at the architecture level — injecting an 8-d emotion vector into attention — and is documented in [docs/MODEL_EXPLANATION.md](docs/MODEL_EXPLANATION.md). Planned improvements for both the app and the model are tracked in [docs/ROADMAP.md](docs/ROADMAP.md).

## Emotion colours

🟡 Joy – gold · 🔵 Sadness – blue · 🔴 Anger – red · 🟣 Fear – purple · 🟢 Trust – green · 🟢 Disgust – lime · 🟠 Surprise – orange · 🩷 Anticipation – pink · ⚪ Neutral – grey

---

**Built by [Nati Birhauz](https://github.com/natiBirhauz)**
