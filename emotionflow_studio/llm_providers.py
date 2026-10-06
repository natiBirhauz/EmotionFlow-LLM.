"""Thin HTTP clients for the free-tier LLM providers (Gemini, Groq, OpenAI).

Every provider returns the raw text of the model reply, or ``None`` when the
provider is not configured or the request failed, so callers can fall through
to the next provider and finally to the local engine.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

GEMINI_MODELS = tuple(
    m.strip() for m in os.getenv("GEMINI_MODELS", "gemini-2.0-flash,gemini-1.5-flash").split(",") if m.strip()
)
GROQ_MODELS = tuple(
    m.strip() for m in os.getenv("GROQ_MODELS", "llama-3.3-70b-versatile,llama-3.1-8b-instant").split(",") if m.strip()
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))


def call_gemini(user_prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False, temperature: float = 0.7, max_tokens: int = 800) -> Optional[str]:
    """Call Google Gemini Free Tier API."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        return None

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": min(1.0, max(0.0, temperature)),
            "maxOutputTokens": max_tokens,
        },
    }
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    if system_prompt:
        body["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    for model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8"))
                candidates = payload.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "").strip()
                        if text:
                            return text
        except Exception as e:
            logger.warning("Gemini (%s) API error: %s", model, e)
            continue
    return None


def call_groq(user_prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False, temperature: float = 0.7, max_tokens: int = 800) -> Optional[str]:
    """Call Groq Free Tier API."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    for model in GROQ_MODELS:
        body = {
            "model": model,
            "messages": messages,
            "temperature": min(1.0, max(0.0, temperature)),
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {groq_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8"))
                msg = payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if msg:
                    return msg
        except Exception as e:
            logger.warning("Groq (%s) API error: %s", model, e)
            continue
    return None


def call_openai(user_prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False, temperature: float = 0.7, max_tokens: int = 800) -> Optional[str]:
    """Call OpenAI API if configured in server environment."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEFAULT_OPENAI_API_KEY")
    if not api_key:
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    body = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": min(1.0, max(0.0, temperature)),
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
            msg = payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if msg:
                return msg
    except Exception as e:
        logger.warning("OpenAI API error: %s", e)
        return None


def call_free_llm(user_prompt: str, system_prompt: Optional[str] = None, json_mode: bool = False, temperature: float = 0.7, max_tokens: int = 800) -> Optional[str]:
    """Dispatcher: tries Gemini, then Groq, then OpenAI; ``None`` if all are unavailable."""
    # 1. Gemini (Free Tier)
    res = call_gemini(user_prompt, system_prompt, json_mode, temperature, max_tokens)
    if res:
        return res
    # 2. Groq (Free Tier)
    res = call_groq(user_prompt, system_prompt, json_mode, temperature, max_tokens)
    if res:
        return res
    # 3. OpenAI (Fallback)
    res = call_openai(user_prompt, system_prompt, json_mode, temperature, max_tokens)
    if res:
        return res
    return None


def parse_llm_json(raw: Optional[str]):
    """Parse a JSON reply, tolerating ```json fences and leading prose. Returns None on failure."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    start = min((i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0), default=-1)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except ValueError:
            logger.warning("LLM returned non-JSON payload")
    return None
