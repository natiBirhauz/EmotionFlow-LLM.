# EmotionFlow-LLM 🎭

EmotionFlow-LLM is an experimental transformer-style language model that blends token generation with an 8-dimensional emotion profile. The project now also includes a polished web studio for generating stories, emails, pitches, and social posts with emotional controls and optional API-key support.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ What’s new

- A redesigned Gradio app for real drafting workflows
- Emotion-driven controls for story, email, pitch, and social writing
- Optional OpenAI API-key support for higher-quality output
- Vercel-ready API endpoint for hosted use
- Cleaner repo structure and refreshed documentation

---

## 🚀 Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
pip install gradio
```

### 2) Run the local studio

```bash
python app.py
```

Then open http://localhost:7860

### 3) Run the model demo

```bash
python emotionflow_llm.py
```

### 4) Optional: deploy to Vercel

Set an environment variable named OPENAI_API_KEY in Vercel, then deploy the project. The repository includes a Vercel configuration at [vercel.json](vercel.json) and an API endpoint at [api/generate.py](api/generate.py).

---

## 🧠 Project structure

- [app.py](app.py) — the redesigned web studio
- [emotionflow_llm/](emotionflow_llm/) — core model modules
- [tests/](tests/) — regression tests for the model and app generation flow
- [api/generate.py](api/generate.py) — Vercel-compatible API endpoint
- [vercel.json](vercel.json) — Vercel deployment config

---

## 🔬 Usage ideas

- Generate a story with a dark, suspenseful emotional profile
- Draft an email with a calm and reassuring tone
- Shape a pitch around joy, trust, and anticipation
- Create social posts with stronger surprise or fear energy

---

## 📦 Notes

This project is still an experimental research/demo project. The web app is designed to feel polished and practical, but the underlying model is intentionally lightweight and educational rather than production-trained.

## 🎯 Research Motivation

### The Problem

Traditional language models encode semantic meaning but ignore emotional context. This limits:
- Emotional coherence in generated text
- Interpretability of model reasoning
- Controllability of narrative tone
- Human-like communication

### Our Approach

EmotionFlow-LLM integrates emotion **during** generation, not as post-processing:

1. **Explicit representation**: 8-dimensional emotion vectors
2. **Attention modification**: Emotion influences token relationships
3. **Trajectory tracking**: Interpretable emotion evolution
4. **Neuron specialization**: Individual neurons develop emotional preferences
5. **Multi-sample exploration**: Diverse emotional reasoning paths

### Key Insight

> Emotion isn't just a property of text—it's a feature that guides reasoning, attention, and generation at the neuron level.

---

## 🔗 Related Work

- **Plutchik (1980)**: Wheel of Emotions - theoretical foundation
- **Mohammad & Turney (2013)**: NRC Emotion Lexicon
- **Vaswani et al. (2017)**: Transformer architecture
- **Zhou et al. (2018)**: Emotional Chatting Machine
- **Poria et al. (2019)**: Emotion recognition in conversation

### Our Contribution

EmotionFlow-LLM differs by:
- Integrating emotion **during** attention computation (not post-hoc)
- Providing neuron-level emotion analysis
- Enabling interpretable emotional trajectories
- Maintaining standard transformer compatibility

---

## 📚 Documentation

- **[EXPLANATION.md](EXPLANATION.md)**: Technical deep dive
- **[PORTFOLIO_GUIDE.md](PORTFOLIO_GUIDE.md)**: How to improve this project
- **[SUMMARY.md](SUMMARY.md)**: Quick project summary

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Training on emotion-labeled datasets
- Additional emotion theories (beyond Plutchik)
- Multi-head emotion attention
- Emotion conditioning interface
- Benchmark datasets

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Plutchik's Wheel of Emotions for the theoretical foundation
- PyTorch team for the excellent deep learning framework
- The transformer architecture (Vaswani et al.)

---

<p align="center">Made with ❤️ and 🧠</p>