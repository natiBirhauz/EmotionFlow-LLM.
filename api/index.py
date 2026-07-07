import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request

# Set up the app
app = Flask(__name__)


def analyze_text_emotions(text: str, api_key: str) -> dict:
    """Analyze text to detect emotional content using GPT."""
    if not api_key:
        return None

    request_data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an emotion analysis expert. Analyze text and rate the presence of Plutchik's 8 emotions (joy, sadness, anger, fear, trust, disgust, surprise, anticipation) on a scale of 0.0 to 1.0. Return ONLY a JSON object with emotion names as keys and decimal values.",
            },
            {
                "role": "user",
                "content": f"Analyze the emotional content of this text and return emotion scores:\n\n{text}\n\nReturn format: {{\"joy\": 0.0, \"sadness\": 0.0, \"anger\": 0.0, \"fear\": 0.0, \"trust\": 0.0, \"disgust\": 0.0, \"surprise\": 0.0, \"anticipation\": 0.0}}",
            },
        ],
        "temperature": 0.3,  # Lower temperature for consistent analysis
        "max_tokens": 150,
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(request_data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            payload = json.loads(response.read().decode("utf-8"))
            content = payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            # Parse JSON response
            emotions = json.loads(content)
            return emotions
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None


def get_api_key(user_api_key: str | None) -> str | None:
    # If user_api_key is "FREE_USE", use the shared key for authenticated free users
    if user_api_key == "FREE_USE":
        shared_key = os.getenv("SHARED_OPENAI_API_KEY", "").strip()
        if shared_key:
            return shared_key
    # If a specific user API key is provided, use it
    if user_api_key and user_api_key.strip() and user_api_key != "FREE_USE":
        return user_api_key.strip()
    # Fallback to environment variables
    for env_name in ("OPENAI_API_KEY", "DEFAULT_OPENAI_API_KEY", "FREE_OPENAI_API_KEY", "SHARED_OPENAI_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


def generate_with_openai(prompt: str, mode: str, api_key: str | None, creativity: float = 0.7, emotions: dict = None, length: int = 3, language: str = "english") -> str | None:
    if not api_key:
        return None

    # Build emotion description if provided
    emotion_instruction = ""
    if emotions:
        # Get top 2 emotions
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotions = [name for name, val in sorted_emotions[:2] if val > 0.1]
        if top_emotions:
            emotion_map = {
                "joy": "joyful, uplifting, and optimistic",
                "sadness": "melancholic, reflective, and somber",
                "anger": "intense, forceful, and confrontational",
                "fear": "tense, anxious, and foreboding",
                "trust": "reassuring, steady, and confident",
                "disgust": "critical, skeptical, and sharp",
                "surprise": "unexpected, striking, and dramatic",
                "anticipation": "eager, forward-looking, and exciting"
            }
            emotion_desc = " and ".join([emotion_map.get(e, e) for e in top_emotions])
            emotion_instruction = f" The tone should feel {emotion_desc}."

    # Adjust length instruction
    length_map = {
        2: "very brief and concise",
        3: "moderate length",
        4: "detailed and thorough",
        5: "comprehensive and elaborate"
    }
    length_instruction = length_map.get(length, "moderate length")
    
    # Adjust max_tokens based on length - increased to prevent cutoff
    max_tokens_map = {2: 200, 3: 350, 4: 500, 5: 700}
    max_tokens = max_tokens_map.get(length, 350)
    
    # Language instruction
    language_instruction = ""
    if language == "hebrew":
        language_instruction = " Write the entire response in Hebrew (עברית). Use proper Hebrew grammar and natural Hebrew expressions."

    request_data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": f"You are a creative writing assistant. Write a polished draft matching the requested format, length, and emotional tone.{language_instruction}",
            },
            {
                "role": "user",
                "content": f"Create a {length_instruction} {mode} style draft for this prompt: {prompt}.{emotion_instruction} Keep it vivid, useful, and polished.{language_instruction}",
            },
        ],
        "temperature": min(1.0, 0.6 + creativity * 0.25),
        "max_tokens": max_tokens,
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(request_data).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=25) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
        return None


@app.route("/", methods=["GET"])
def index():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EmotionFlow Studio</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script>
    <script src="https://accounts.google.com/gsi/client" async></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #e2e8f0;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 42px;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header p {
            font-size: 18px;
            opacity: 0.8;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }
        
        @media (max-width: 1024px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(10px);
        }
        
        .card h2 {
            font-size: 20px;
            margin-bottom: 20px;
            color: #a78bfa;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #cbd5e1;
            font-weight: 500;
            font-size: 14px;
        }
        
        input[type="text"],
        input[type="password"],
        select,
        textarea {
            width: 100%;
            padding: 12px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: #e2e8f0;
            font-family: inherit;
            font-size: 14px;
            transition: all 0.3s;
        }
        
        input[type="text"]:focus,
        input[type="password"]:focus,
        select:focus,
        textarea:focus {
            outline: none;
            border-color: #a78bfa;
            background: rgba(0, 0, 0, 0.5);
            box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.1);
        }
        
        textarea {
            resize: vertical;
            min-height: 80px;
        }
        
        .slider-group {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }
        
        input[type="range"] {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.1);
            outline: none;
            -webkit-appearance: none;
            appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
            cursor: pointer;
            box-shadow: 0 0 10px rgba(167, 139, 250, 0.4);
        }
        
        input[type="range"]::-moz-range-thumb {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
            cursor: pointer;
            border: none;
            box-shadow: 0 0 10px rgba(167, 139, 250, 0.4);
        }
        
        .slider-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 4px;
        }
        
        .slider-value {
            min-width: 35px;
            text-align: right;
            font-weight: 600;
            color: #60a5fa;
            font-size: 13px;
        }
        
        .emotion-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        
        .emotion-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #94a3b8;
            margin-bottom: 16px;
        }
        
        .controls-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        
        .btn-generate {
            grid-column: span 2;
            padding: 14px 24px;
            background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 10px;
        }
        
        .btn-generate:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(167, 139, 250, 0.3);
        }
        
        .btn-generate:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .output-text {
            white-space: pre-wrap;
            line-height: 1.6;
            color: #cbd5e1;
            font-size: 14px;
            max-height: 500px;
            overflow-y: auto;
        }
        
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        
        .chart-wrapper {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            padding: 16px;
            position: relative;
            min-height: 300px;
        }
        
        @media (max-width: 768px) {
            .charts-container {
                grid-template-columns: 1fr;
            }
        }
        
        .status-message {
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 16px;
            font-size: 14px;
            display: none;
            animation: slideIn 0.3s ease;
        }
        
        .status-message.show {
            display: block;
        }
        
        .status-message.error {
            background: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .status-message.success {
            background: rgba(52, 211, 153, 0.2);
            color: #86efac;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }
        
        .loading {
            display: inline-block;
            width: 8px;
            height: 8px;
            margin: 0 4px;
            background: #a78bfa;
            border-radius: 50%;
            animation: pulse 1.4s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 1; }
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .examples {
            margin-top: 30px;
        }
        
        .examples h3 {
            font-size: 16px;
            margin-bottom: 15px;
            color: #a78bfa;
        }
        
        .example-btn {
            display: inline-block;
            padding: 10px 16px;
            background: rgba(167, 139, 250, 0.1);
            border: 1px solid rgba(167, 139, 250, 0.3);
            border-radius: 6px;
            color: #cbd5e1;
            cursor: pointer;
            margin-right: 10px;
            margin-bottom: 10px;
            transition: all 0.3s;
            font-size: 13px;
        }
        
        .examples {
            margin-top: 30px;
        }
        
        .examples h3 {
            font-size: 16px;
            margin-bottom: 15px;
            color: #a78bfa;
        }
        
        .example-btn {
            display: inline-block;
            padding: 10px 16px;
            background: rgba(167, 139, 250, 0.1);
            border: 1px solid rgba(167, 139, 250, 0.3);
            border-radius: 6px;
            color: #cbd5e1;
            cursor: pointer;
            margin-right: 10px;
            margin-bottom: 10px;
            transition: all 0.3s;
            font-size: 13px;
        }
        
        .example-btn:hover {
            background: rgba(167, 139, 250, 0.2);
            border-color: rgba(167, 139, 250, 0.5);
        }
        
        .auth-section {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px;
            background: rgba(167, 139, 250, 0.1);
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .user-info {
            color: #cbd5e1;
            font-size: 14px;
        }
        
        .free-uses {
            color: #60a5fa;
            font-weight: 600;
        }
        
        /* Emotion color highlights */
        .emotion-joy { color: #fbbf24; font-weight: 500; }
        .emotion-sadness { color: #60a5fa; font-weight: 500; }
        .emotion-anger { color: #ef4444; font-weight: 500; }
        .emotion-fear { color: #a78bfa; font-weight: 500; }
        .emotion-trust { color: #34d399; font-weight: 500; }
        .emotion-disgust { color: #84cc16; font-weight: 500; }
        .emotion-surprise { color: #f59e0b; font-weight: 500; }
        .emotion-anticipation { color: #ec4899; font-weight: 500; }
        
        /* RTL support for Hebrew */
        .rtl-text {
            direction: rtl;
            text-align: right;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="auth-section">
            <div id="googleSignInButton"></div>
            <div id="userStatus" style="display: none;">
                <div class="user-info">Welcome, <span id="userName"></span></div>
                <div class="free-uses">Free uses remaining: <span id="freeUsesRemaining">1</span></div>
                <button onclick="handleLogout()" style="padding: 8px 16px; background: rgba(239,68,68,0.2); border: 1px solid rgba(239,68,68,0.3); border-radius: 6px; color: #fca5a5; cursor: pointer; font-size: 12px; margin-top: 5px;">Sign Out</button>
            </div>
        </div>
        <div class="header">
            <h1>✨ EmotionFlow Studio</h1>
            <p>Turn a simple prompt into a polished draft with emotional controls</p>
        </div>
        
        <div class="main-grid">
            <!-- Input Controls -->
            <div class="card">
                <h2>🎛️ Controls</h2>
                
                <div class="status-message" id="statusMessage"></div>
                
                <div class="form-group">
                    <label>Prompt or topic</label>
                    <textarea id="prompt" placeholder="Describe the scene, message, or idea...">The hidden library at dawn</textarea>
                </div>
                
                <div class="form-group">
                    <label>Output format</label>
                    <select id="mode">
                        <option value="story">📖 Story</option>
                        <option value="email">📧 Email</option>
                        <option value="pitch">💼 Pitch</option>
                        <option value="social">📱 Social Media</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Language</label>
                    <select id="language">
                        <option value="english">🇬🇧 English</option>
                        <option value="hebrew">🇮🇱 עברית (Hebrew)</option>
                    </select>
                </div>
                
                <div class="controls-row">
                    <div class="form-group">
                        <div class="slider-label">
                            <label>Length</label>
                            <span class="slider-value" id="lengthValue">3</span>
                        </div>
                        <input type="range" id="length" min="2" max="5" value="3" step="1">
                    </div>
                    
                    <div class="form-group">
                        <div class="slider-label">
                            <label>Creativity</label>
                            <span class="slider-value" id="creativityValue">0.7</span>
                        </div>
                        <input type="range" id="creativity" min="0.2" max="1.0" value="0.7" step="0.1">
                    </div>
                </div>
                
                <div class="form-group">
                    <div class="slider-label">
                        <label>Variations</label>
                        <span class="slider-value" id="samplesValue">3</span>
                    </div>
                    <input type="range" id="samples" min="1" max="4" value="3" step="1">
                </div>
                
                <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 20px; margin-top: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <h3 style="color: #a78bfa; font-size: 14px; margin: 0;">Emotional Tone</h3>
                        <button onclick="toggleWheel()" style="padding: 4px 10px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 11px;">🎭 Wheel View</button>
                    </div>
                    
                    <div id="wheelContainer" style="display: none; margin-bottom: 16px;">
                        <canvas id="plutchikWheel" width="400" height="400" style="display: block; margin: 0 auto; cursor: crosshair; border-radius: 50%; background: rgba(0,0,0,0.3);"></canvas>
                        <p style="text-align: center; font-size: 11px; color: #94a3b8; margin-top: 8px;">Click on the wheel to set emotions</p>
                    </div>
                    
                    <div class="emotion-grid" id="emotionSliders"></div>
                </div>
                
                <div class="form-group" style="margin-top: 20px;">
                    <label>OpenAI API Key (optional)</label>
                    <input type="password" id="apiKey" placeholder="sk-... (leave blank for local generation)">
                </div>
                
                <div style="margin-top: 16px;">
                    <label style="margin-bottom: 8px; display: block;">Quick Emotion Presets</label>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        <button onclick="setEmotionPreset('happy')" style="padding: 6px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 12px;">😊 Happy</button>
                        <button onclick="setEmotionPreset('sad')" style="padding: 6px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 12px;">😢 Sad</button>
                        <button onclick="setEmotionPreset('dramatic')" style="padding: 6px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 12px;">🎭 Dramatic</button>
                        <button onclick="setEmotionPreset('calm')" style="padding: 6px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 12px;">🧘 Calm</button>
                        <button onclick="setEmotionPreset('intense')" style="padding: 6px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 12px;">⚡ Intense</button>
                    </div>
                </div>
                
                <button class="btn-generate" id="generateBtn">Generate Draft</button>
            </div>
            
            <!-- Text Analyzer -->
            <div class="card">
                <h2>🔍 Text Emotion Analyzer</h2>
                <p style="font-size: 13px; color: #94a3b8; margin-bottom: 16px;">Analyze any text to detect its emotional content</p>
                
                <div class="form-group">
                    <label>Text to Analyze</label>
                    <textarea id="analyzeText" placeholder="Paste or type text here to analyze its emotions..." style="width: 100%; min-height: 120px; padding: 12px; background: rgba(0, 0, 0, 0.3); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; color: #e2e8f0; font-family: inherit; font-size: 14px; resize: vertical;"></textarea>
                </div>
                
                <div style="display: flex; gap: 8px; margin-bottom: 16px;">
                    <button onclick="analyzeText()" id="analyzeBtn" style="flex: 1; padding: 12px 16px; background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.3s;">Analyze Emotions</button>
                    <button onclick="applyAnalyzedEmotions()" id="applyBtn" style="flex: 1; padding: 12px 16px; background: rgba(52, 211, 153, 0.2); border: 1px solid rgba(52, 211, 153, 0.4); border-radius: 8px; color: #86efac; cursor: pointer; font-size: 14px; font-weight: 600; display: none; transition: all 0.3s;">Apply to Sliders</button>
                </div>
                
                <div id="analyzeResult"></div>
            </div>
        </div>
        
        <!-- Output Section (Full Width) -->
        <div class="card" style="margin-bottom: 40px;">
            <h2>📝 Output</h2>
            <div id="variationsContainer"></div>
            <div class="output-text" id="outputBox">Your polished draft will appear here...</div>
            <div id="emotionInsight" style="margin-top: 16px; padding: 12px; background: rgba(167, 139, 250, 0.1); border-radius: 8px; color: #cbd5e1; font-size: 13px; border: 1px solid rgba(167, 139, 250, 0.2); display: none;"></div>
        </div>
        
        <!-- Charts -->
        <div class="charts-container">
            <div class="chart-wrapper">
                <canvas id="barChart"></canvas>
            </div>
            <div class="chart-wrapper">
                <canvas id="radarChart"></canvas>
            </div>
        </div>
        
        <!-- Examples -->
        <div class="examples card">
            <h3>🧪 Quick Examples</h3>
            <button class="example-btn" onclick="loadExample(['The hidden library at dawn', 'story', 3, 0.7, [0.3, 0.1, 0, 0.1, 0.3, 0, 0.1, 0.2], 3])">Hidden Library</button>
            <button class="example-btn" onclick="loadExample(['A launch email for a calm app', 'email', 3, 0.8, [0.4, 0, 0, 0, 0.5, 0, 0.1, 0], 2])">Calm Email</button>
            <button class="example-btn" onclick="loadExample(['A heartbreaking farewell', 'story', 3, 0.6, [0, 0.9, 0, 0.1, 0, 0, 0, 0], 2])">Sad Farewell</button>
            <button class="example-btn" onclick="loadExample(['City at night', 'social', 3, 0.5, [0.1, 0.2, 0, 0.2, 0.1, 0, 0.3, 0.1], 3])">Dramatic Post</button>
        </div>
    </div>
    
    <script>
        // Wait for Chart.js to load
        function ensureChartLoaded(callback) {
            if (typeof Chart !== 'undefined') {
                callback();
            } else {
                setTimeout(() => ensureChartLoaded(callback), 100);
            }
        }
        
        // Google Sign-In configuration
        function initGoogleSignIn() {
            if (typeof google !== 'undefined' && google.accounts) {
                const clientId = '""" + os.getenv("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID") + """';
                if (clientId && clientId !== 'YOUR_GOOGLE_CLIENT_ID') {
                    google.accounts.id.initialize({
                        client_id: clientId,
                        callback: handleCredentialResponse
                    });
                    
                    google.accounts.id.renderButton(
                        document.getElementById('googleSignInButton'),
                        { theme: 'filled_blue', size: 'large', text: 'signin_with', shape: 'pill' }
                    );
                } else {
                    document.getElementById('googleSignInButton').innerHTML = '<p style="color: #fca5a5; font-size: 12px;">Google Sign-In not configured</p>';
                }
            } else {
                setTimeout(initGoogleSignIn, 100);
            }
        }
        
        window.onload = function() {
            initGoogleSignIn();
            
            // Check if user is already logged in
            const savedUser = localStorage.getItem('emotionflow_user');
            if (savedUser) {
                const user = JSON.parse(savedUser);
                showUserStatus(user);
            }
            
            // Initialize the app
            initializeEmotions();
            updateSliderDisplays();
        };
        
        function handleCredentialResponse(response) {
            // Decode JWT token to get user info
            const base64Url = response.credential.split('.')[1];
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
                return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
            }).join(''));
            
            const user = JSON.parse(jsonPayload);
            
            // Save user to localStorage with free uses
            const userSession = {
                email: user.email,
                name: user.name,
                picture: user.picture,
                freeUses: 1
            };
            localStorage.setItem('emotionflow_user', JSON.stringify(userSession));
            showUserStatus(userSession);
        }
        
        function showUserStatus(user) {
            document.getElementById('googleSignInButton').style.display = 'none';
            document.getElementById('userStatus').style.display = 'block';
            document.getElementById('userName').textContent = user.name;
            document.getElementById('freeUsesRemaining').textContent = user.freeUses;
        }
        
        function handleLogout() {
            localStorage.removeItem('emotionflow_user');
            document.getElementById('userStatus').style.display = 'none';
            document.getElementById('googleSignInButton').style.display = 'block';
        }
        
        const EMOTIONS = ["joy", "sadness", "anger", "fear", "trust", "disgust", "surprise", "anticipation"];
        const EMOTION_LABELS = {
            "joy": "joyful, uplifting, and optimistic",
            "sadness": "melancholic, reflective, and somber",
            "anger": "intense, forceful, and confrontational",
            "fear": "tense, anxious, and foreboding",
            "trust": "reassuring, steady, and confident",
            "disgust": "critical, skeptical, and sharp",
            "surprise": "unexpected, striking, and dramatic",
            "anticipation": "eager, forward-looking, and exciting"
        };
        
        const EMOTION_COLORS = {
            "joy": "#fbbf24",
            "sadness": "#60a5fa",
            "anger": "#ef4444",
            "fear": "#a78bfa",
            "trust": "#34d399",
            "disgust": "#84cc16",
            "surprise": "#f59e0b",
            "anticipation": "#ec4899"
        };
        
        let charts = { bar: null, radar: null };
        let wheelVisible = false;
        
        // Update slider displays - single event listener setup
        function updateSliderDisplays() {
            const updateValue = (id, suffix = '') => {
                const el = document.getElementById(id);
                const display = document.getElementById(id + 'Value');
                if (el && display) {
                    display.textContent = el.value + suffix;
                }
            };
            
            // Set initial values
            updateValue('length');
            updateValue('creativity');
            updateValue('samples');
            
            // Single event listeners (not duplicated)
            document.getElementById('length').addEventListener('input', () => updateValue('length'));
            document.getElementById('creativity').addEventListener('input', () => {
                const val = document.getElementById('creativity').value;
                document.getElementById('creativityValue').textContent = parseFloat(val).toFixed(1);
            });
            document.getElementById('samples').addEventListener('input', () => updateValue('samples'));
        }
        
        // Initialize emotion sliders
        function initializeEmotions() {
            const container = document.getElementById('emotionSliders');
            const emotionValues = {
                joy: 0.3,
                sadness: 0.1,
                anger: 0,
                fear: 0.1,
                trust: 0.3,
                disgust: 0,
                surprise: 0.1,
                anticipation: 0.2
            };
            
            EMOTIONS.forEach(emotion => {
                const div = document.createElement('div');
                div.className = 'form-group';
                div.innerHTML = `
                    <div class="slider-label">
                        <label style="font-size: 13px; font-weight: 500;">${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</label>
                        <span class="slider-value" id="${emotion}Value">${(emotionValues[emotion] * 100).toFixed(0)}%</span>
                    </div>
                    <input type="range" id="${emotion}" min="0" max="1" value="${emotionValues[emotion]}" step="0.05">
                `;
                container.appendChild(div);
                
                // Update display on change
                document.getElementById(emotion).addEventListener('input', (e) => {
                    document.getElementById(`${emotion}Value`).textContent = (parseFloat(e.target.value) * 100).toFixed(0) + '%';
                    // Redraw wheel if visible
                    if (wheelVisible) {
                        drawPlutchikWheel();
                    }
                });
            });
        }
        
        // Get emotion values
        function getEmotions() {
            const emotions = {};
            EMOTIONS.forEach(emotion => {
                emotions[emotion] = parseFloat(document.getElementById(emotion).value);
            });
            return emotions;
        }
        
        // Normalize emotions
        function normalizeEmotions(emotions) {
            const total = Object.values(emotions).reduce((a, b) => a + b, 0);
            if (total <= 0) return Object.fromEntries(EMOTIONS.map(e => [e, 0]));
            const normalized = {};
            EMOTIONS.forEach(e => normalized[e] = emotions[e] / total);
            return normalized;
        }
        
        // Summarize emotions
        function summarizeEmotions(emotions) {
            const ranked = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
            const dominant = ranked[0][0];
            const strength = Math.round(ranked[0][1] * 100);
            const supporting = ranked.slice(1).filter(([_, val]) => val > 0.08).map(([name, _]) => name).slice(0, 2);
            const supportText = supporting.length > 0 ? supporting.join(', ') : 'a calm balance';
            return `The draft leans ${EMOTION_LABELS[dominant]} with ${strength}% intensity. Secondary notes include ${supportText}.`;
        }
        
        // Emotion presets
        function setEmotionPreset(preset) {
            const presets = {
                happy: { joy: 0.8, sadness: 0, anger: 0, fear: 0, trust: 0.5, disgust: 0, surprise: 0.2, anticipation: 0.3 },
                sad: { joy: 0, sadness: 0.9, anger: 0, fear: 0.2, trust: 0.1, disgust: 0, surprise: 0, anticipation: 0 },
                dramatic: { joy: 0.1, sadness: 0.3, anger: 0.2, fear: 0.3, trust: 0, disgust: 0.1, surprise: 0.4, anticipation: 0.2 },
                calm: { joy: 0.3, sadness: 0, anger: 0, fear: 0, trust: 0.7, disgust: 0, surprise: 0, anticipation: 0.1 },
                intense: { joy: 0.1, sadness: 0.1, anger: 0.6, fear: 0.3, trust: 0, disgust: 0.2, surprise: 0.3, anticipation: 0.5 }
            };
            
            const values = presets[preset];
            if (values) {
                EMOTIONS.forEach(emotion => {
                    document.getElementById(emotion).value = values[emotion];
                    document.getElementById(`${emotion}Value`).textContent = (values[emotion] * 100).toFixed(0) + '%';
                });
            }
        }
        
        // Count words
        function countWords(text) {
            return text.trim().split(/\s+/).length;
        }
        
        // Copy to clipboard
        function copyToClipboard(text, btnElement) {
            navigator.clipboard.writeText(text).then(() => {
                const originalText = btnElement.textContent;
                btnElement.textContent = '✓ Copied!';
                btnElement.style.background = 'rgba(52, 211, 153, 0.2)';
                setTimeout(() => {
                    btnElement.textContent = originalText;
                    btnElement.style.background = '';
                }, 2000);
            });
        }
        
        // Store analyzed emotions
        let lastAnalyzedEmotions = null;
        
        // Analyze text
        async function analyzeText() {
            const text = document.getElementById('analyzeText').value.trim();
            let apiKey = document.getElementById('apiKey').value.trim();
            
            if (!text) {
                showStatus('Please enter text to analyze', 'error');
                return;
            }
            
            // Check for API key
            const user = localStorage.getItem('emotionflow_user');
            if (user && !apiKey) {
                const userSession = JSON.parse(user);
                if (userSession.freeUses > 0) {
                    apiKey = 'FREE_USE';
                } else {
                    showStatus('Please enter your OpenAI API key to analyze', 'error');
                    return;
                }
            } else if (!apiKey) {
                showStatus('Please sign in or enter your API key to analyze', 'error');
                return;
            }
            
            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, api_key: apiKey })
                });
                
                const data = await response.json();
                if (data.ok && data.emotions) {
                    lastAnalyzedEmotions = data.emotions;
                    displayAnalysisResult(data.emotions);
                    document.getElementById('applyBtn').style.display = 'inline-block';
                    showStatus('Analysis complete!', 'success');
                } else {
                    showStatus(data.message || 'Analysis failed', 'error');
                }
            } catch (error) {
                showStatus(`Error: ${error.message}`, 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = 'Analyze Emotions';
            }
        }
        
        // Display analysis result
        function displayAnalysisResult(emotions) {
            const resultDiv = document.getElementById('analyzeResult');
            const sorted = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
            
            let html = '<div style="margin-top: 12px;">';
            html += '<h4 style="color: #a78bfa; font-size: 13px; margin-bottom: 8px;">Detected Emotions:</h4>';
            
            sorted.forEach(([emotion, value]) => {
                const percentage = (value * 100).toFixed(0);
                const barWidth = percentage;
                html += `
                    <div style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
                            <span style="color: #cbd5e1;">${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</span>
                            <span style="color: #60a5fa; font-weight: 600;">${percentage}%</span>
                        </div>
                        <div style="width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden;">
                            <div style="width: ${barWidth}%; height: 100%; background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%); border-radius: 3px;"></div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            resultDiv.innerHTML = html;
        }
        
        // Apply analyzed emotions to sliders
        function applyAnalyzedEmotions() {
            if (!lastAnalyzedEmotions) return;
            
            EMOTIONS.forEach(emotion => {
                const value = lastAnalyzedEmotions[emotion] || 0;
                document.getElementById(emotion).value = value;
                document.getElementById(`${emotion}Value`).textContent = (value * 100).toFixed(0) + '%';
            });
            
            showStatus('Emotions applied to sliders!', 'success');
        }
        
        // Toggle Plutchik Wheel
        function toggleWheel() {
            wheelVisible = !wheelVisible;
            document.getElementById('wheelContainer').style.display = wheelVisible ? 'block' : 'none';
            if (wheelVisible) {
                drawPlutchikWheel();
            }
        }
        
        // Draw Plutchik Wheel
        function drawPlutchikWheel() {
            const canvas = document.getElementById('plutchikWheel');
            const ctx = canvas.getContext('2d');
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = 150;  // Increased from 130
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw emotion segments
            const angleStep = (Math.PI * 2) / EMOTIONS.length;
            EMOTIONS.forEach((emotion, index) => {
                const startAngle = angleStep * index - Math.PI / 2;
                const endAngle = startAngle + angleStep;
                const value = parseFloat(document.getElementById(emotion).value);
                
                // Draw segment
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.arc(centerX, centerY, radius * value, startAngle, endAngle);
                ctx.closePath();
                ctx.fillStyle = EMOTION_COLORS[emotion] + '99';
                ctx.fill();
                ctx.strokeStyle = EMOTION_COLORS[emotion];
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Draw label - positioned further out and with better text alignment
                const labelAngle = startAngle + angleStep / 2;
                const labelRadius = radius + 35;  // Increased from 20
                const labelX = centerX + Math.cos(labelAngle) * labelRadius;
                const labelY = centerY + Math.sin(labelAngle) * labelRadius;
                
                // Set text alignment based on position
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                
                ctx.fillStyle = EMOTION_COLORS[emotion];
                ctx.font = 'bold 13px sans-serif';  // Increased font size and made bold
                ctx.fillText(emotion.charAt(0).toUpperCase() + emotion.slice(1), labelX, labelY);
            });
            
            // Draw center circle
            ctx.beginPath();
            ctx.arc(centerX, centerY, 10, 0, Math.PI * 2);
            ctx.fillStyle = '#a78bfa';
            ctx.fill();
        }
        
        // Wheel click interaction
        document.addEventListener('DOMContentLoaded', () => {
            const canvas = document.getElementById('plutchikWheel');
            canvas.addEventListener('click', (e) => {
                if (!wheelVisible) return;
                
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left - canvas.width / 2;
                const y = e.clientY - rect.top - canvas.height / 2;
                const angle = Math.atan2(y, x) + Math.PI / 2;
                const normalizedAngle = (angle + Math.PI * 2) % (Math.PI * 2);
                const distance = Math.sqrt(x * x + y * y);
                const maxRadius = 150;  // Updated to match new radius
                
                // Determine which emotion was clicked
                const angleStep = (Math.PI * 2) / EMOTIONS.length;
                const emotionIndex = Math.floor(normalizedAngle / angleStep);
                const emotion = EMOTIONS[emotionIndex];
                
                // Set value based on distance from center
                const value = Math.min(distance / maxRadius, 1.0);
                document.getElementById(emotion).value = value;
                document.getElementById(`${emotion}Value`).textContent = (value * 100).toFixed(0) + '%';
                
                drawPlutchikWheel();
            });
        });
        
        // Highlight text with emotion colors
        function highlightEmotions(text) {
            // Escape HTML first to prevent XSS but preserve structure
            const escapeMap = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;'
            };
            
            // Don't double-escape if already has emotion spans
            if (text.includes('class="emotion-')) {
                return text;
            }
            
            // Escape HTML special chars
            let safeText = text.replace(/[&<>"']/g, (char) => escapeMap[char] || char);
            
            // Expanded keyword matching for better detection
            const emotionKeywords = {
                joy: ['happy', 'happiness', 'joyful', 'delighted', 'cheerful', 'bright', 'wonderful', 'glad', 'pleased', 'smile', 'smiling', 'laugh', 'laughing', 'excited', 'excitement', 'love', 'loving'],
                sadness: ['sad', 'sadness', 'sorrow', 'sorrowful', 'melancholy', 'tears', 'crying', 'grief', 'lonely', 'loneliness', 'down', 'blue', 'cry', 'weep', 'weeping', 'depressed', 'gloomy'],
                anger: ['angry', 'anger', 'furious', 'fury', 'rage', 'mad', 'irritated', 'annoyed', 'frustrated', 'frustration', 'hate', 'hatred', 'bitter', 'resentful'],
                fear: ['afraid', 'fear', 'scared', 'terrified', 'terror', 'anxious', 'anxiety', 'worried', 'worry', 'nervous', 'frightened', 'panic', 'panicked', 'dread'],
                trust: ['trust', 'trusting', 'believe', 'belief', 'faith', 'confident', 'confidence', 'reliable', 'secure', 'security', 'safe', 'safety', 'depend', 'dependable'],
                disgust: ['disgusted', 'disgust', 'revolting', 'nasty', 'gross', 'repulsive', 'awful', 'vile', 'horrible', 'terrible'],
                surprise: ['surprised', 'surprise', 'shocked', 'shock', 'amazed', 'amazing', 'astonished', 'unexpected', 'sudden', 'suddenly', 'startle', 'startled', 'wow'],
                anticipation: ['anticipate', 'anticipation', 'expect', 'expecting', 'await', 'awaiting', 'hopeful', 'hope', 'eager', 'eagerly', 'excited', 'looking forward', 'soon', 'ready']
            };
            
            // Apply highlighting with proper HTML
            let highlightedText = safeText;
            Object.entries(emotionKeywords).forEach(([emotion, keywords]) => {
                keywords.forEach(keyword => {
                    const regex = new RegExp(`\\b(${keyword})\\b`, 'gi');
                    highlightedText = highlightedText.replace(regex, `<span class="emotion-${emotion}">$1</span>`);
                });
            });
            
            return highlightedText;
        }
        
        // Detect if text contains Hebrew characters
        function isHebrewText(text) {
            const hebrewRegex = /[\u0590-\u05FF]/;
            return hebrewRegex.test(text);
        }
        
        // Apply RTL class if needed
        function wrapWithRTLIfNeeded(html, text) {
            if (isHebrewText(text)) {
                return `<div class="rtl-text">${html}</div>`;
            }
            return html;
        }
        
        // Update charts with error handling
        function updateCharts(emotions) {
            ensureChartLoaded(() => {
                try {
                    // Use raw values for display, not normalized
                    const ordered = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
                    const labels = ordered.map(([name, _]) => name.charAt(0).toUpperCase() + name.slice(1));
                    const values = ordered.map(([_, val]) => val);
                    const colors = ["#7c3aed", "#2563eb", "#f97316", "#dc2626", "#14b8a6", "#84cc16", "#f43f5e", "#eab308"];
                    
                    // Bar chart
                    if (charts.bar) charts.bar.destroy();
                    const barCtx = document.getElementById('barChart').getContext('2d');
                    charts.bar = new Chart(barCtx, {
                        type: 'bar',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: 'Emotion Intensity',
                                data: values,
                                backgroundColor: colors.slice(0, labels.length),
                                borderRadius: 6
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: true,
                            indexAxis: 'y',
                            plugins: { 
                                legend: { display: false },
                                tooltip: { 
                                    enabled: true,
                                    callbacks: {
                                        label: function(context) {
                                            return (context.parsed.x * 100).toFixed(0) + '%';
                                        }
                                    }
                                }
                            },
                            scales: { 
                                x: { 
                                    max: 1,
                                    ticks: { 
                                        color: '#cbd5e1',
                                        callback: function(value) {
                                            return (value * 100).toFixed(0) + '%';
                                        }
                                    },
                                    grid: { color: 'rgba(255,255,255,0.1)' }
                                },
                                y: {
                                    ticks: { color: '#cbd5e1' },
                                    grid: { color: 'rgba(255,255,255,0.1)' }
                                }
                            }
                        }
                    });
                    
                    // Radar chart - use raw values
                    if (charts.radar) charts.radar.destroy();
                    const radarCtx = document.getElementById('radarChart').getContext('2d');
                    const radarValues = EMOTIONS.map(e => emotions[e] || 0);
                    charts.radar = new Chart(radarCtx, {
                        type: 'radar',
                        data: {
                            labels: EMOTIONS.map(e => e.charAt(0).toUpperCase() + e.slice(1)),
                            datasets: [{
                                label: 'Emotion Profile',
                                data: radarValues,
                                borderColor: '#a78bfa',
                                backgroundColor: 'rgba(167, 139, 250, 0.2)',
                                borderWidth: 2,
                                fill: true,
                                pointBackgroundColor: '#a78bfa',
                                pointBorderColor: '#fff',
                                pointHoverBackgroundColor: '#fff',
                                pointHoverBorderColor: '#a78bfa'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: { 
                                legend: { display: false },
                                tooltip: { 
                                    enabled: true,
                                    callbacks: {
                                        label: function(context) {
                                            return (context.parsed.r * 100).toFixed(0) + '%';
                                        }
                                    }
                                }
                            },
                            scales: {
                                r: {
                                    beginAtZero: true,
                                    max: 1,
                                    ticks: { 
                                        color: '#cbd5e1',
                                        backdropColor: 'transparent',
                                        callback: function(value) {
                                            return (value * 100).toFixed(0) + '%';
                                        }
                                    },
                                    grid: { color: 'rgba(255,255,255,0.1)' },
                                    pointLabels: { color: '#cbd5e1' }
                                }
                            }
                        }
                    });
                } catch (error) {
                    console.error('Chart error:', error);
                    showStatus('Chart rendering error. Please refresh.', 'error');
                }
            });
        }
        
        // Show status
        function showStatus(message, type = 'error') {
            const status = document.getElementById('statusMessage');
            status.textContent = message;
            status.className = `status-message show ${type}`;
            setTimeout(() => status.classList.remove('show'), 5000);
        }
        
        // Load example
        function loadExample(params) {
            const [prompt, mode, length, creativity, emotions, samples] = params;
            document.getElementById('prompt').value = prompt;
            document.getElementById('mode').value = mode;
            document.getElementById('length').value = length;
            document.getElementById('creativity').value = creativity;
            document.getElementById('samples').value = samples;
            
            EMOTIONS.forEach((emotion, idx) => {
                const value = emotions[idx] || 0;
                document.getElementById(emotion).value = value;
                document.getElementById(`${emotion}Value`).textContent = (value * 100).toFixed(0) + '%';
            });
            
            updateSliderDisplays();
        }
        
        // Generate with free use handling
        document.getElementById('generateBtn').addEventListener('click', async () => {
            const prompt = document.getElementById('prompt').value.trim();
            let apiKey = document.getElementById('apiKey').value.trim();
            const mode = document.getElementById('mode').value;
            const creativity = parseFloat(document.getElementById('creativity').value);
            const length = parseInt(document.getElementById('length').value);
            const variations = parseInt(document.getElementById('samples').value);
            const emotions = getEmotions();
            const language = document.getElementById('language').value;
            
            if (!prompt) {
                showStatus('Please enter a prompt', 'error');
                return;
            }
            
            if (Object.values(emotions).reduce((a, b) => a + b, 0) === 0) {
                showStatus('Please move at least one emotion slider', 'error');
                return;
            }
            
            // Check if user is logged in and has free uses
            const user = localStorage.getItem('emotionflow_user');
            if (user && !apiKey) {
                const userSession = JSON.parse(user);
                if (userSession.freeUses > 0) {
                    // Use free generation
                    apiKey = 'FREE_USE'; // Signal to backend to use shared key
                    userSession.freeUses--;
                    localStorage.setItem('emotionflow_user', JSON.stringify(userSession));
                    document.getElementById('freeUsesRemaining').textContent = userSession.freeUses;
                } else {
                    showStatus('Free uses exhausted. Please enter your OpenAI API key.', 'error');
                    return;
                }
            } else if (!apiKey) {
                showStatus('Please sign in with Google or enter your OpenAI API key', 'error');
                return;
            }
            
            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            const loadingMsg = variations > 1 ? `Generating ${variations} variations...` : 'Generating...';
            document.getElementById('outputBox').innerHTML = '<div class="loading"></div><div class="loading"></div><div class="loading"></div> ' + loadingMsg;
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, api_key: apiKey, mode, creativity, length, variations, emotions, language })
                });
                
                const data = await response.json();
                if (data.ok && data.results) {
                    const results = data.results;
                    
                    if (results.length === 1) {
                        // Single result - simple display with emotion highlighting
                        const result = results[0];
                        const highlighted = highlightEmotions(result.draft);
                        document.getElementById('outputBox').innerHTML = wrapWithRTLIfNeeded(highlighted, result.draft);
                        document.getElementById('emotionInsight').textContent = '💡 ' + summarizeEmotions(normalizeEmotions(result.emotions));
                        document.getElementById('emotionInsight').style.display = 'block';
                        updateCharts(result.emotions);
                    } else {
                        // Multiple results - create variation cards
                        let html = '';
                        results.forEach((result, idx) => {
                            const wordCount = countWords(result.draft);
                            const highlighted = highlightEmotions(result.draft);
                            const rtlClass = isHebrewText(result.draft) ? ' class="rtl-text"' : '';
                            html += `
                                <div style="margin-bottom: 24px; padding: 16px; background: rgba(0, 0, 0, 0.2); border-radius: 12px; border: 1px solid rgba(167, 139, 250, 0.2);">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                        <h3 style="color: #a78bfa; font-size: 16px; margin: 0;">Variation ${idx + 1}</h3>
                                        <div style="display: flex; gap: 8px; align-items: center;">
                                            <span style="font-size: 12px; color: #94a3b8;">${wordCount} words</span>
                                            <button onclick="copyToClipboard(\`${result.draft.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`, this)" style="padding: 6px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 12px;">📋 Copy</button>
                                        </div>
                                    </div>
                                    <div${rtlClass} style="white-space: pre-wrap; line-height: 1.6; color: #cbd5e1; font-size: 14px; margin-bottom: 12px;">${highlighted}</div>
                                    <div style="padding: 10px; background: rgba(167, 139, 250, 0.1); border-radius: 6px; font-size: 12px; color: #cbd5e1;">
                                        💡 ${summarizeEmotions(normalizeEmotions(result.emotions))}
                                    </div>
                                </div>
                            `;
                        });
                        
                        document.getElementById('outputBox').innerHTML = html;
                        document.getElementById('emotionInsight').style.display = 'none';
                        // Update charts with first variation's emotions
                        updateCharts(results[0].emotions);
                    }
                    
                    showStatus('Generated successfully!', 'success');
                } else {
                    showStatus(data.message || 'Generation failed', 'error');
                    document.getElementById('outputBox').textContent = 'Generation failed. Please try again.';
                    document.getElementById('emotionInsight').style.display = 'none';
                }
            } catch (error) {
                showStatus(`Error: ${error.message}`, 'error');
                document.getElementById('outputBox').textContent = 'Network error. Please check your connection.';
                document.getElementById('emotionInsight').style.display = 'none';
            } finally {
                btn.disabled = false;
            }
        });
        
        // Load example
        window.loadExample = function(example) {
            document.getElementById('prompt').value = example[0];
            document.getElementById('mode').value = example[1];
            document.getElementById('length').value = example[2];
            document.getElementById('length').dispatchEvent(new Event('input'));
            document.getElementById('creativity').value = example[3];
            document.getElementById('creativity').dispatchEvent(new Event('input'));
            
            const emotionValues = example[4];
            EMOTIONS.forEach((emotion, index) => {
                document.getElementById(emotion).value = emotionValues[index];
                document.getElementById(emotion).dispatchEvent(new Event('input'));
            });
            
            document.getElementById('samples').value = example[5];
            document.getElementById('samples').dispatchEvent(new Event('input'));
        };
    </script>
</body>
</html>"""
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/generate", methods=["POST", "GET"])
def generate_api():
    payload = request.get_json(silent=True) or {}
    if request.method == "GET":
        payload = request.args.to_dict()

    prompt = payload.get("prompt", "") or "A fresh idea for tomorrow"
    mode = payload.get("mode", "story")
    creativity = float(payload.get("creativity", 0.7) or 0.7)
    length = int(payload.get("length", 3) or 3)
    variations = int(payload.get("variations", 1) or 1)
    language = payload.get("language", "english")
    api_key = get_api_key(payload.get("api_key"))
    
    # Get emotions from payload
    emotions = payload.get("emotions", {})
    if isinstance(emotions, str):
        emotions = json.loads(emotions)

    if not api_key:
        return jsonify({
            "ok": False,
            "message": "No API key was provided. Set OPENAI_API_KEY in Vercel or paste a key into the form.",
        }), 400

    # Generate multiple variations if requested
    results = []
    for i in range(min(variations, 4)):  # Max 4 variations
        # Add slight variation to emotions for different results
        if i > 0:
            varied_emotions = {}
            for emotion, value in emotions.items():
                # Add ±5% random variation to each emotion
                variation = (hash(f"{prompt}{i}") % 11 - 5) / 100  # Deterministic but varied
                varied_emotions[emotion] = max(0.0, min(1.0, value + variation))
        else:
            varied_emotions = emotions
            
        # Add slight variation to creativity
        variation_creativity = creativity + (i * 0.05) if i > 0 else creativity
        variation_creativity = min(1.0, variation_creativity)
        
        draft = generate_with_openai(prompt, mode, api_key, creativity=variation_creativity, emotions=varied_emotions, length=length, language=language)
        if draft:
            results.append({
                "draft": draft,
                "emotions": varied_emotions,
                "creativity": variation_creativity
            })
        else:
            break  # Stop if generation fails
    
    if not results:
        return jsonify({
            "ok": False,
            "message": "The key could not be used right now. Please check the key or try again.",
        }), 400

    return jsonify({"ok": True, "results": results})


@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    """Analyze text for emotional content."""
    payload = request.get_json(silent=True) or {}
    
    text = payload.get("text", "").strip()
    api_key = get_api_key(payload.get("api_key"))
    
    if not text:
        return jsonify({
            "ok": False,
            "message": "Please provide text to analyze.",
        }), 400
    
    if not api_key:
        return jsonify({
            "ok": False,
            "message": "No API key available for analysis.",
        }), 400
    
    emotions = analyze_text_emotions(text, api_key)
    
    if not emotions:
        return jsonify({
            "ok": False,
            "message": "Failed to analyze text. Please try again.",
        }), 400
    
    return jsonify({
        "ok": True,
        "emotions": emotions,
        "text": text
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
