# EmotionFlow-LLM

**AI-powered creative writing guided by Plutchik's 8 emotions**

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/natiBirhauz/EmotionFlow-LLM)

## ✨ Features

- 🎭 **8 Plutchik Emotions** - Control joy, sadness, anger, fear, trust, disgust, surprise, and anticipation
- 🤖 **AI-Powered** - Uses GPT-4o-mini for high-quality generation  
- 🔐 **Google OAuth** - Sign in for free usage
- 📊 **Visual Feedback** - Interactive charts show emotional balance
- 🎨 **Multiple Formats** - Stories, emails, pitches, or social posts

## 🚀 Quick Deploy

1. Click "Deploy with Vercel" button above
2. Add environment variables in Vercel:
   - `SHARED_OPENAI_API_KEY` - Your OpenAI API key
   - `GOOGLE_CLIENT_ID` - Your Google OAuth client ID
3. Deploy!

## 💻 Local Development

```bash
git clone https://github.com/natiBirhauz/EmotionFlow-LLM.git
cd EmotionFlow-LLM
pip install -r requirements.txt

# Create .env file
echo SHARED_OPENAI_API_KEY=your-key > .env
echo GOOGLE_CLIENT_ID=your-google-id >> .env

python api/index.py
# Visit http://localhost:5000
```

## 🎯 How It Works

1. **Adjust Emotions** - Set 8 emotion sliders based on Plutchik's Wheel of Emotions
2. **AI Generation** - GPT-4o-mini creates content matching your emotional tone
3. **Visualize** - Charts show your emotional balance

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **AI:** OpenAI GPT-4o-mini
- **Auth:** Google OAuth 2.0
- **Charts:** Chart.js
- **Hosting:** Vercel

## 📊 Plutchik's 8 Emotions

Based on Robert Plutchik's Wheel of Emotions (1980):
- Joy, Sadness, Anger, Fear, Trust, Disgust, Surprise, Anticipation

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

**Made with ❤️ by [Nati Birhauz](https://github.com/natiBirhauz)**

 🎭

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

## 🌐 Live Deployment

**The API is now live on Vercel:** https://emotion-flow-llm-six.vercel.app/

### Use the API

**Generate text with your OpenAI API key:**
```bash
curl -X POST https://emotion-flow-llm-six.vercel.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A hidden garden at midnight",
    "api_key": "sk-your-key-here",
    "mode": "story",
    "creativity": 0.7
  }'
```

**Check API health:**
```bash
curl https://emotion-flow-llm-six.vercel.app/health
```

Supported modes: `story`, `email`, `pitch`, `social`

---

## 🚀 Quick start

### 1) Install dependencies

**For local development with the full UI and model:**
```bash
pip install -r requirements-dev.txt
```

**For Vercel deployment (minimal):**
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes only Flask and Gradio for the API. The `requirements-dev.txt` adds PyTorch, scikit-learn, and testing tools for local development.

### 2) Run the local studio

```bash
python app.py
```

Then open http://localhost:7860

### 3) Run the model demo

```bash
python emotionflow_llm.py
```

### 4) Deploy to Vercel (already live!)

The project is already deployed and running at https://emotion-flow-llm-six.vercel.app/

To deploy your own instance:
1. Fork this repository
2. Create a Vercel project connected to your GitHub repo
3. Set environment variable `OPENAI_API_KEY` in Vercel settings
4. Deploy!

The repository includes a Vercel configuration at [vercel.json](vercel.json).

---

## 🧠 Project structure

- [app.py](app.py) — the redesigned web studio (Gradio)
- [emotionflow_llm/](emotionflow_llm/) — core model modules
- [tests/](tests/) — regression tests for the model and app generation flow
- [api/index.py](api/index.py) — Vercel-compatible Flask API endpoint
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