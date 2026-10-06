"""Flask entrypoint (Vercel serverless function). Routes only; logic lives in ``emotionflow_studio``."""

import json
import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotionflow_studio import services  # noqa: E402
from emotionflow_studio.emotions import PLUTCHIK_EMOTIONS  # noqa: E402,F401  (re-exported for tests)
from emotionflow_studio.services import (  # noqa: E402,F401
    analyze_text_emotions,
    annotate_text_with_emotions,
    generate_draft,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = Flask(__name__, template_folder=str(ROOT / "emotionflow_studio" / "templates"), static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024


def _number(payload: dict, key: str, default: float, low: float, high: float, cast=float):
    try:
        value = cast(payload.get(key, default))
    except (TypeError, ValueError):
        value = default
    return cast(services.clamp(value, low, high))


def _choice(payload: dict, key: str, options, default: str) -> str:
    value = str(payload.get(key, default) or default).lower().strip()
    return value if value in options else default


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST", "GET"])
def generate_api():
    payload = request.args.to_dict() if request.method == "GET" else (request.get_json(silent=True) or {})

    prompt = str(payload.get("prompt") or "A fresh idea for tomorrow").strip()[: services.MAX_PROMPT_CHARS]
    mode = _choice(payload, "mode", services.MODES, "story")
    language = _choice(payload, "language", services.LANGUAGES, "english")
    creativity = _number(payload, "creativity", 0.7, 0.0, 1.0)
    length = _number(payload, "length", 3, 2, 5, cast=int)
    variations = _number(payload, "variations", 1, 1, services.MAX_VARIATIONS, cast=int)

    emotions = payload.get("emotions", {})
    if isinstance(emotions, str):
        try:
            emotions = json.loads(emotions)
        except ValueError:
            emotions = {}
    emotions = services.normalize_emotions(emotions)

    results = services.generate_variations(prompt, mode, creativity, emotions, length, variations, language)
    if not results:
        return jsonify({"ok": False, "message": "Generation failed. Please try again."}), 500
    return jsonify({"ok": True, "results": results})


@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "message": "Please provide text to analyze."}), 400
    if len(text) > services.MAX_ANALYZE_CHARS:
        return jsonify({"ok": False, "message": f"Text is too long (max {services.MAX_ANALYZE_CHARS} characters)."}), 413
    language = _choice(payload, "language", services.LANGUAGES, "english")

    return jsonify(
        {
            "ok": True,
            "emotions": analyze_text_emotions(text),
            "annotations": annotate_text_with_emotions(text, language=language),
            "text": text,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
