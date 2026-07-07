import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request

# Set up the app
app = Flask(__name__)


def get_api_key(user_api_key: str | None) -> str | None:
    if user_api_key and user_api_key.strip():
        return user_api_key.strip()
    for env_name in ("OPENAI_API_KEY", "DEFAULT_OPENAI_API_KEY", "FREE_OPENAI_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


def generate_with_openai(prompt: str, mode: str, api_key: str | None, creativity: float = 0.7) -> str | None:
    if not api_key:
        return None

    request_data = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are a creative writing assistant. Write a concise, polished draft matching the requested format and tone.",
            },
            {
                "role": "user",
                "content": f"Create a {mode} style draft for this prompt: {prompt}. Keep it vivid, useful, and polished.",
            },
        ],
        "temperature": min(1.0, 0.6 + creativity * 0.25),
        "max_tokens": 220,
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
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.0.0/dist/chart.min.js"></script>
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
        
        .example-btn:hover {
            background: rgba(167, 139, 250, 0.2);
            border-color: rgba(167, 139, 250, 0.5);
        }
    </style>
</head>
<body>
    <div class="container">
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
                    <h3 style="color: #a78bfa; font-size: 14px; margin-bottom: 12px;">Emotional Tone</h3>
                    <div class="emotion-grid" id="emotionSliders"></div>
                </div>
                
                <div class="form-group" style="margin-top: 20px;">
                    <label>OpenAI API Key (optional)</label>
                    <input type="password" id="apiKey" placeholder="sk-... (leave blank for local generation)">
                </div>
                
                <button class="btn-generate" id="generateBtn">Generate Draft</button>
            </div>
            
            <!-- Output -->
            <div class="card">
                <h2>📝 Output</h2>
                <div class="output-text" id="outputBox">Your polished draft will appear here...</div>
                <div id="emotionInsight" style="margin-top: 16px; padding: 12px; background: rgba(167, 139, 250, 0.1); border-radius: 8px; color: #cbd5e1; font-size: 13px; border: 1px solid rgba(167, 139, 250, 0.2);"></div>
            </div>
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
            <button class="example-btn" onclick="loadExample(['The hidden library at dawn', 'story', 3, 0.7, [0.1, 0, 0, 0.5, 0.4, 0, 0.1, 0.1], 3])">Hidden Library (Story)</button>
            <button class="example-btn" onclick="loadExample(['A launch email for a calm new app', 'email', 3, 0.8, [0.2, 0, 0, 0.4, 0, 0.1, 0.2, 0.1], 2])">Launch Email</button>
            <button class="example-btn" onclick="loadExample(['A product pitch for a smart notebook', 'pitch', 3, 0.7, [0, 0, 0, 0.3, 0, 0.2, 0.2, 0.1], 2])">Product Pitch</button>
            <button class="example-btn" onclick="loadExample(['A dramatic post about a city at night', 'social', 3, 0.5, [0.2, 0.1, 0.3, 0.1, 0, 0.3, 0.1, 0], 3])">Dramatic City Post</button>
        </div>
    </div>
    
    <script>
        const EMOTIONS = ["joy", "sadness", "anger", "fear", "trust", "disgust", "surprise", "anticipation"];
        const EMOTION_LABELS = {
            "joy": "bright and uplifting",
            "sadness": "reflective and tender",
            "anger": "firm and forceful",
            "fear": "tense and shadowed",
            "trust": "steady and reassuring",
            "disgust": "sharp and skeptical",
            "surprise": "electric and unexpected",
            "anticipation": "eager and forward-looking"
        };
        
        let charts = { bar: null, radar: null };
        
        // Initialize emotion sliders
        function initializeEmotions() {
            const container = document.getElementById('emotionSliders');
            const emotionValues = {
                joy: 0.3, sadness: 0, anger: 0, fear: 0,
                trust: 0.3, disgust: 0, surprise: 0.2, anticipation: 0.2
            };
            
            EMOTIONS.forEach(emotion => {
                const div = document.createElement('div');
                div.className = 'form-group';
                div.innerHTML = `
                    <div class="slider-label">
                        <label style="font-size: 12px;">${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</label>
                        <span class="slider-value" id="${emotion}Value">${emotionValues[emotion].toFixed(2)}</span>
                    </div>
                    <input type="range" id="${emotion}" min="0" max="1" value="${emotionValues[emotion]}" step="0.05">
                `;
                container.appendChild(div);
                
                // Update display on change
                document.getElementById(emotion).addEventListener('input', (e) => {
                    document.getElementById(`${emotion}Value`).textContent = parseFloat(e.target.value).toFixed(2);
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
        
        // Update charts
        function updateCharts(emotions) {
            const normalized = normalizeEmotions(emotions);
            const ordered = Object.entries(normalized).sort((a, b) => b[1] - a[1]);
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
                        label: 'Emotion Balance',
                        data: values,
                        backgroundColor: colors.slice(0, labels.length),
                        borderRadius: 6
                    }]
                },
                options: {
                    indexAxis: 'y',
                    plugins: { legend: { display: false } },
                    scales: { x: { max: 1 } }
                }
            });
            
            // Radar chart
            if (charts.radar) charts.radar.destroy();
            const radarCtx = document.getElementById('radarChart').getContext('2d');
            const radarValues = EMOTIONS.map(e => normalized[e] || 0);
            charts.radar = new Chart(radarCtx, {
                type: 'radar',
                data: {
                    labels: EMOTIONS.map(e => e.charAt(0).toUpperCase() + e.slice(1)),
                    datasets: [{
                        label: 'Emotion Radar',
                        data: radarValues,
                        borderColor: '#a78bfa',
                        backgroundColor: 'rgba(167, 139, 250, 0.2)',
                        fill: true,
                        pointBackgroundColor: '#a78bfa'
                    }]
                },
                options: { plugins: { legend: { display: false } } }
            });
        }
        
        // Show status
        function showStatus(message, type = 'error') {
            const status = document.getElementById('statusMessage');
            status.textContent = message;
            status.className = `status-message show ${type}`;
            setTimeout(() => status.classList.remove('show'), 5000);
        }
        
        // Generate
        document.getElementById('generateBtn').addEventListener('click', async () => {
            const prompt = document.getElementById('prompt').value.trim();
            const apiKey = document.getElementById('apiKey').value.trim();
            const mode = document.getElementById('mode').value;
            const creativity = parseFloat(document.getElementById('creativity').value);
            const emotions = getEmotions();
            
            if (!prompt) {
                showStatus('Please enter a prompt', 'error');
                return;
            }
            
            if (Object.values(emotions).reduce((a, b) => a + b, 0) === 0) {
                showStatus('Please move at least one emotion slider', 'error');
                return;
            }
            
            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            document.getElementById('outputBox').innerHTML = '<div class="loading"></div><div class="loading"></div><div class="loading"></div> Generating...';
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, api_key: apiKey, mode, creativity })
                });
                
                const data = await response.json();
                if (data.ok) {
                    document.getElementById('outputBox').textContent = data.draft;
                    document.getElementById('emotionInsight').textContent = '💡 ' + summarizeEmotions(normalizeEmotions(emotions));
                    updateCharts(emotions);
                    showStatus('Generated successfully!', 'success');
                } else {
                    showStatus(data.message, 'error');
                    document.getElementById('outputBox').textContent = '';
                }
            } catch (error) {
                showStatus(`Error: ${error.message}`, 'error');
                document.getElementById('outputBox').textContent = '';
            } finally {
                btn.disabled = false;
            }
        });
        
        // Update slider displays
        document.getElementById('length').addEventListener('input', (e) => {
            document.getElementById('lengthValue').textContent = e.target.value;
        });
        document.getElementById('creativity').addEventListener('input', (e) => {
            document.getElementById('creativityValue').textContent = parseFloat(e.target.value).toFixed(1);
        });
        document.getElementById('samples').addEventListener('input', (e) => {
            document.getElementById('samplesValue').textContent = e.target.value;
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
        
        // Initialize
        initializeEmotions();
        updateCharts(getEmotions());
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
    api_key = get_api_key(payload.get("api_key"))

    if not api_key:
        return jsonify({
            "ok": False,
            "message": "No API key was provided. Set OPENAI_API_KEY in Vercel or paste a key into the form.",
        }), 400

    draft = generate_with_openai(prompt, mode, api_key, creativity=creativity)
    if not draft:
        return jsonify({
            "ok": False,
            "message": "The key could not be used right now. Please check the key or try again.",
        }), 400

    return jsonify({"ok": True, "draft": draft})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
