# EmotionFlow Studio

**AI-Powered Creative Writing with Emotion Control**

Control the emotional tone of AI-generated text using Plutchik's 8 core emotions. Generate stories, emails, pitches, and social posts with precise emotional coloring—now in English and Hebrew.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/natiBirhauz/EmotionFlow-LLM)

---

## 🎭 What Makes This Different

Every word is **color-coded by emotion** using AI analysis. Not keyword matching—actual emotional understanding.

- Set disgust to 100%? Get a genuinely disgusting story.
- Want 80% joy + 20% anticipation? The AI makes it happen.
- Works in **English** and **עברית (Hebrew)**

See the exact emotions detected in every sentence, not just what you asked for.

---

## ✨ Features

**Emotion Control**
- 🎨 8 Plutchik emotions: joy, sadness, anger, fear, trust, disgust, surprise, anticipation
- 🎭 Interactive Plutchik Wheel visualization
- 🎯 Quick presets: Happy, Sad, Dramatic, Calm, Intense

**Smart Generation**
- 🤖 GPT-4o-mini powered
- 🌍 English & Hebrew support
- 📊 Real-time emotion detection with color-coded text
- 📝 Multiple formats: Story, Email, Pitch, Social Media

**Analysis Tools**
- 🔍 Text emotion analyzer with visual graphs
- 📈 Bar & radar charts of emotional balance
- 🎨 Sentence-by-sentence emotion coloring

**User Experience**
- 🔐 Google OAuth for free usage
- 📋 One-click copy to clipboard
- 📊 Word count tracking
- 🎲 Generate 1-4 variations with emotion diversity

---

## 🚀 Quick Start

### Live Demo

Visit: https://emotion-flow-llm-six.vercel.app/

### Deploy Your Own

1. Click the Vercel button above
2. Set environment variables:
   ```
   SHARED_OPENAI_API_KEY=sk-your-key
   GOOGLE_CLIENT_ID=your-google-oauth-id
   ```
3. Deploy!

### Local Development

```bash
git clone https://github.com/natiBirhauz/EmotionFlow-LLM.git
cd EmotionFlow-LLM
pip install -r requirements.txt

# Create .env file
echo "SHARED_OPENAI_API_KEY=your-key" > .env
echo "GOOGLE_CLIENT_ID=your-google-id" >> .env

python api/index.py
# Open http://localhost:5000
```

---

## 🎯 How to Use

1. **Enter a prompt**: "A hidden library at dawn"
2. **Adjust emotions**: Move sliders or use the emotion wheel
3. **Choose format**: Story, Email, Pitch, or Social Post
4. **Generate**: Click and watch emotions color your text
5. **Analyze**: See which emotions the AI actually used

**Pro tip:** The "💡 Insight" shows the **detected** emotions in the output, not just what you requested.

---

## 📊 The Science

Based on **Robert Plutchik's Wheel of Emotions** (1980), which identifies 8 primary emotions that combine to form complex feelings.

The AI:
- Receives explicit emotion instructions with percentages
- Generates text matching the emotional profile
- Gets analyzed sentence-by-sentence for actual emotional content
- Colors each phrase by its detected emotion

**Example**: Set 100% disgust → AI generates genuinely repulsive content → Text appears in lime green

---

## 🛠️ Tech Stack

- **AI**: OpenAI GPT-4o-mini
- **Backend**: Python + Flask
- **Frontend**: Vanilla JavaScript + Chart.js
- **Auth**: Google OAuth 2.0
- **Hosting**: Vercel
- **Languages**: English & Hebrew (עברית)

---

## 🌈 Emotion Colors

- 🟡 **Joy** - Gold
- 🔵 **Sadness** - Blue
- 🔴 **Anger** - Red
- 🟣 **Fear** - Purple
- 🟢 **Trust** - Green
- 🟢 **Disgust** - Lime
- 🟠 **Surprise** - Orange
- 🩷 **Anticipation** - Pink

---

## 🔥 Coming Soon

Ideas we're exploring:
- Voice tone analysis
- Emotion intensity gradients
- More language support
- Export as styled documents
- Emotion-based story templates

---

## 📝 License

MIT License - See [LICENSE](LICENSE)

---

**Built by [Nati Birhauz](https://github.com/natiBirhauz)**

*Turn emotions into words. Watch feelings come alive.*
