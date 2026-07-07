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
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 900px;
            width: 100%;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 32px;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 16px;
            opacity: 0.9;
        }
        
        .content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 40px;
        }
        
        @media (max-width: 768px) {
            .content {
                grid-template-columns: 1fr;
            }
        }
        
        .form-section h2 {
            font-size: 20px;
            margin-bottom: 20px;
            color: #333;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
            font-size: 14px;
        }
        
        input[type="text"],
        input[type="password"],
        input[type="number"],
        select,
        textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-family: inherit;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus,
        input[type="password"]:focus,
        input[type="number"]:focus,
        select:focus,
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        textarea {
            resize: vertical;
            min-height: 80px;
        }
        
        .slider-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        input[type="range"] {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
            appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        
        input[type="range"]::-moz-range-thumb {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
            border: none;
        }
        
        .slider-value {
            min-width: 40px;
            text-align: right;
            font-weight: 600;
            color: #667eea;
        }
        
        .button-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 30px;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .btn-generate {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            grid-column: span 2;
        }
        
        .btn-generate:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        
        .btn-generate:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .btn-clear {
            background: #f0f0f0;
            color: #333;
        }
        
        .btn-clear:hover {
            background: #e0e0e0;
        }
        
        .output-section {
            display: flex;
            flex-direction: column;
        }
        
        .output-section h2 {
            font-size: 20px;
            margin-bottom: 20px;
            color: #333;
        }
        
        .output-box {
            flex: 1;
            background: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 20px;
            overflow-y: auto;
            font-size: 14px;
            line-height: 1.6;
            color: #333;
            min-height: 250px;
        }
        
        .output-box.empty {
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
        }
        
        .loading {
            display: inline-block;
            width: 8px;
            height: 8px;
            margin: 0 4px;
            background: #667eea;
            border-radius: 50%;
            animation: pulse 1.4s infinite;
        }
        
        .loading:nth-child(1) {
            animation-delay: 0s;
        }
        
        .loading:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .loading:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes pulse {
            0%, 100% {
                opacity: 0.3;
            }
            50% {
                opacity: 1;
            }
        }
        
        .status-message {
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 14px;
            display: none;
        }
        
        .status-message.show {
            display: block;
        }
        
        .status-message.error {
            background: #fee;
            color: #c33;
            border: 1px solid #fcc;
        }
        
        .status-message.success {
            background: #efe;
            color: #3c3;
            border: 1px solid #cfc;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎭 EmotionFlow Studio</h1>
            <p>Generate creative text with emotional controls</p>
        </div>
        
        <div class="content">
            <div class="form-section">
                <h2>Inputs</h2>
                
                <div class="status-message" id="statusMessage"></div>
                
                <div class="form-group">
                    <label for="prompt">Prompt</label>
                    <textarea id="prompt" placeholder="Enter your prompt here...">A hidden garden at dawn</textarea>
                </div>
                
                <div class="form-group">
                    <label for="apiKey">OpenAI API Key</label>
                    <input type="password" id="apiKey" placeholder="sk-...">
                </div>
                
                <div class="form-group">
                    <label for="mode">Mode</label>
                    <select id="mode">
                        <option value="story">Story</option>
                        <option value="email">Email</option>
                        <option value="pitch">Pitch</option>
                        <option value="social">Social Media</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="creativity">Creativity</label>
                    <div class="slider-group">
                        <input type="range" id="creativity" min="0" max="1" step="0.1" value="0.7">
                        <span class="slider-value" id="creativityValue">0.7</span>
                    </div>
                </div>
                
                <div class="button-group">
                    <button class="btn-generate" id="generateBtn">Generate</button>
                    <button class="btn-clear" id="clearBtn">Clear</button>
                </div>
            </div>
            
            <div class="output-section">
                <h2>Output</h2>
                <div class="output-box empty" id="outputBox">
                    Waiting for generation...
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const promptInput = document.getElementById('prompt');
        const apiKeyInput = document.getElementById('apiKey');
        const modeSelect = document.getElementById('mode');
        const creativitySlider = document.getElementById('creativity');
        const creativityValue = document.getElementById('creativityValue');
        const generateBtn = document.getElementById('generateBtn');
        const clearBtn = document.getElementById('clearBtn');
        const outputBox = document.getElementById('outputBox');
        const statusMessage = document.getElementById('statusMessage');
        
        // Update creativity value display
        creativitySlider.addEventListener('input', (e) => {
            creativityValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
        
        // Show status message
        function showStatus(message, type = 'error') {
            statusMessage.textContent = message;
            statusMessage.className = `status-message show ${type}`;
            setTimeout(() => {
                statusMessage.classList.remove('show');
            }, 5000);
        }
        
        // Generate text
        generateBtn.addEventListener('click', async () => {
            const prompt = promptInput.value.trim();
            const apiKey = apiKeyInput.value.trim();
            const mode = modeSelect.value;
            const creativity = parseFloat(creativitySlider.value);
            
            if (!prompt) {
                showStatus('Please enter a prompt', 'error');
                return;
            }
            
            if (!apiKey) {
                showStatus('Please enter your OpenAI API key', 'error');
                return;
            }
            
            generateBtn.disabled = true;
            outputBox.innerHTML = '<div class="loading"></div><div class="loading"></div><div class="loading"></div>';
            outputBox.classList.remove('empty');
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt,
                        api_key: apiKey,
                        mode,
                        creativity,
                    }),
                });
                
                const data = await response.json();
                
                if (data.ok) {
                    outputBox.innerHTML = data.draft.replace(/\\n/g, '<br>');
                    showStatus('Generated successfully!', 'success');
                } else {
                    outputBox.innerHTML = '';
                    outputBox.classList.add('empty');
                    showStatus(data.message || 'Generation failed', 'error');
                }
            } catch (error) {
                outputBox.innerHTML = '';
                outputBox.classList.add('empty');
                showStatus(`Error: ${error.message}`, 'error');
            } finally {
                generateBtn.disabled = false;
            }
        });
        
        // Clear form
        clearBtn.addEventListener('click', () => {
            promptInput.value = '';
            outputBox.innerHTML = 'Waiting for generation...';
            outputBox.classList.add('empty');
            statusMessage.classList.remove('show');
        });
        
        // Allow Enter key to generate
        promptInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                generateBtn.click();
            }
        });
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
