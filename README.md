# EmotionFlow-LLM

**Turn emotions into words with AI-powered creative writing**

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/natiBirhauz/EmotionFlow-LLM)

## ✨ Features

- 🎭 **Emotion Control** - Fine-tune 8 different emotions to shape your content
- 🤖 **AI-Powered** - Uses GPT-4o-mini for high-quality text generation  
- 🔐 **Google OAuth** - Sign in with Google for free usage
- 📊 **Visual Feedback** - Interactive charts show your emotional balance
- 🎨 **Multiple Formats** - Generate stories, emails, pitches, or social posts
- 🆓 **Free Tier** - 1 free generation per authenticated user

## 🚀 Quick Start

### Deploy to Vercel (Recommended)

1. Click the "Deploy with Vercel" button above
2. Follow the [Google OAuth Setup Guide](GOOGLE_OAUTH_SETUP.md)
3. Add environment variables in Vercel:
   - `SHARED_OPENAI_API_KEY` - Your OpenAI API key
   - `GOOGLE_CLIENT_ID` - Your Google OAuth client ID
4. Deploy!

For detailed instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

### Local Development

```bash
# Clone the repository
git clone https://github.com/natiBirhauz/EmotionFlow-LLM.git
cd EmotionFlow-LLM

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the app
python api/index.py

# Visit http://localhost:5000
```

## 📖 Documentation

- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Complete deployment instructions
- **[Google OAuth Setup](GOOGLE_OAUTH_SETUP.md)** - Step-by-step OAuth configuration  
- **[Changes Log](CHANGES.md)** - Recent updates and fixes
- **[Technical Explanation](EXPLANATION.md)** - How EmotionFlow works

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Required: Your OpenAI API key for free user generations
SHARED_OPENAI_API_KEY=sk-proj-...

# Required: Google OAuth Client ID from Google Cloud Console  
GOOGLE_CLIENT_ID=123456789-abc...xyz.apps.googleusercontent.com

# Optional: Additional API key fallbacks
OPENAI_API_KEY=sk-...
```

**Important:** Never commit your `.env` file! It's already in `.gitignore`.

## 💡 How It Works

1. **Sign In** - Users authenticate with Google OAuth
2. **Free Generation** - Each user gets 1 free AI generation using your shared API key
3. **Custom Key** - After free use, users can provide their own OpenAI API key
4. **Emotion Control** - Adjust 8 emotion sliders to shape the content tone
5. **AI Generation** - GPT-4o-mini creates polished content based on emotions
6. **Visualizations** - Charts show the emotional balance of your generation

## 🎯 Use Cases

- **Creative Writing** - Generate story drafts with specific emotional tones
- **Email Drafting** - Create professional emails with the right mood
- **Marketing Copy** - Craft pitches that resonate emotionally
- **Social Media** - Write engaging posts with intentional tone

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Frontend:** Vanilla JavaScript, Chart.js
- **AI:** OpenAI GPT-4o-mini
- **Auth:** Google OAuth 2.0
- **Hosting:** Vercel
- **Styling:** Custom CSS with gradient backgrounds

## 📊 Cost Estimates

Using GPT-4o-mini (~270 tokens per generation):
- Cost per generation: ~$0.00014 (0.014 cents)
- 1,000 generations: ~$0.14
- 10,000 generations: ~$1.40

With 1 free use per user, monitor your OpenAI usage and set billing alerts.

## 🐛 Troubleshooting

### Chart.js not loading
- **Fix:** Clear browser cache and refresh
- **Cause:** CDN caching issue
- **Details:** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md#troubleshooting)

### Google Sign-In not working  
- **Fix:** Verify `GOOGLE_CLIENT_ID` is set in environment variables
- **Cause:** Missing or incorrect OAuth configuration
- **Details:** See [GOOGLE_OAUTH_SETUP.md](GOOGLE_OAUTH_SETUP.md#troubleshooting)

### API key errors
- **Fix:** Check that `SHARED_OPENAI_API_KEY` is set correctly
- **Cause:** Missing environment variable or invalid key
- **Details:** Verify your OpenAI API key has available credits

## 🔒 Security

- ✅ API keys stored as environment variables, never in code
- ✅ `.env` files excluded from version control  
- ✅ OAuth 2.0 authentication for user verification
- ✅ No sensitive data exposed to frontend
- ✅ Error messages don't leak system details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4o-mini API
- Google for OAuth authentication
- Chart.js for beautiful visualizations
- Vercel for seamless hosting

## 📞 Support

For issues or questions:
1. Check the [Deployment Guide](DEPLOYMENT_GUIDE.md)
2. Review [Google OAuth Setup](GOOGLE_OAUTH_SETUP.md)  
3. Open an issue on GitHub
4. Check browser console for error messages

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