import json
import os
import urllib.error
import urllib.request

from flask import Flask, jsonify, request

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


@app.route("/api/generate", methods=["POST", "GET"])
def generate_route():
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


if __name__ == "__main__":
    app.run(debug=True)
