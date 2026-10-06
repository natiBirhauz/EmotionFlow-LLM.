"""Application services: LLM-first generation/analysis with local fallback.

The Flask layer (``api/index.py``) only parses/validates requests and calls
these functions; everything here is framework-free and unit-testable.
"""

import hashlib
import logging
from typing import Dict, List, Optional

from .emotions import EMOTION_LABELS, PLUTCHIK_EMOTIONS
from .llm_providers import call_free_llm, parse_llm_json
from .local_engine import analyze_emotions_local, annotate_text_local, generate_draft_local

logger = logging.getLogger(__name__)

MODES = ("story", "email", "pitch", "social")
LANGUAGES = ("english", "hebrew")
MAX_VARIATIONS = 4
MAX_PROMPT_CHARS = 2000
MAX_ANALYZE_CHARS = 20000

LENGTH_DESCRIPTIONS = {
    2: "very brief and concise (1-2 short paragraphs)",
    3: "moderate length (2-3 paragraphs)",
    4: "detailed and thorough (3-4 paragraphs)",
    5: "comprehensive and elaborate (4-5 paragraphs)",
}
LENGTH_MAX_TOKENS = {2: 250, 3: 450, 4: 650, 5: 900}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_emotions(raw) -> Dict[str, float]:
    """Keep only known Plutchik keys with numeric values clamped to [0, 1]."""
    if not isinstance(raw, dict):
        return {}
    clean = {}
    for name, value in raw.items():
        key = str(name).lower().strip()
        if key not in PLUTCHIK_EMOTIONS:
            continue
        try:
            clean[key] = round(clamp(float(value), 0.0, 1.0), 3)
        except (TypeError, ValueError):
            continue
    return clean


def variation_offset(prompt: str, index: int) -> float:
    """Stable per-(prompt, index) nudge in [-0.05, 0.05]; unlike ``hash()`` it does not change per process."""
    digest = hashlib.sha256(f"{prompt}|{index}".encode("utf-8")).digest()
    return (digest[0] % 11 - 5) / 100


def vary_emotions(emotions: Dict[str, float], prompt: str, index: int) -> Dict[str, float]:
    if index == 0:
        return dict(emotions)
    offset = variation_offset(prompt, index)
    return {name: round(clamp(value + offset, 0.0, 1.0), 3) for name, value in emotions.items()}


def _emotion_instruction(emotions: Optional[Dict[str, float]]) -> str:
    if not emotions:
        return ""
    active = sorted(((n, v) for n, v in emotions.items() if v > 0.1), key=lambda kv: kv[1], reverse=True)
    if not active:
        return ""
    primary_name, primary_val = active[0]
    desc = EMOTION_LABELS.get(primary_name, "vivid")
    text = (
        f" IMPORTANT: The writing MUST strongly convey {primary_name} emotion ({desc}) "
        f"at {int(primary_val * 100)}% intensity."
    )
    if len(active) > 1:
        secondary = [f"{name} ({int(val * 100)}%)" for name, val in active[1:3]]
        text += f" Include subtle undertones of {', '.join(secondary)}."
    return text


def generate_draft(
    prompt: str,
    mode: str,
    creativity: float = 0.7,
    emotions: Optional[Dict[str, float]] = None,
    length: int = 3,
    language: str = "english",
) -> str:
    """Generate a draft with the free LLM chain, falling back to the local template engine."""
    length_instruction = LENGTH_DESCRIPTIONS.get(length, LENGTH_DESCRIPTIONS[3])
    max_tokens = LENGTH_MAX_TOKENS.get(length, 450)
    language_instruction = (
        " Write the entire response in Hebrew (עברית). Use proper Hebrew grammar, natural phrasing, and rich vocabulary."
        if language == "hebrew"
        else ""
    )
    system_prompt = (
        "You are a creative writing assistant specialized in emotional storytelling. "
        f"Write polished drafts that strongly match the requested emotional tone and intensity.{language_instruction}"
    )
    user_prompt = (
        f"Write a {length_instruction} {mode} about: {prompt}.{_emotion_instruction(emotions)} "
        f"Make the emotional tone very clear, evocative, and consistent throughout the entire piece.{language_instruction}"
    )

    llm_result = call_free_llm(
        user_prompt,
        system_prompt=system_prompt,
        json_mode=False,
        temperature=min(1.0, 0.6 + creativity * 0.25),
        max_tokens=max_tokens,
    )
    if llm_result:
        return llm_result
    return generate_draft_local(prompt, mode, emotions=emotions, length=length, language=language)


def analyze_text_emotions(text: str) -> Dict[str, float]:
    """Score Plutchik's 8 emotions in [0, 1] via LLM, with lexicon fallback."""
    system_prompt = (
        "You are an emotion analysis expert. Analyze text and rate the presence of Plutchik's 8 emotions "
        "(joy, sadness, anger, fear, trust, disgust, surprise, anticipation) on a scale of 0.0 to 1.0. "
        "Return ONLY a valid JSON object with emotion names as keys and decimal values between 0.0 and 1.0."
    )
    user_prompt = (
        f"Analyze the emotional content of this text and return emotion scores:\n\n{text}\n\n"
        'Return format: {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "trust": 0.0, '
        '"disgust": 0.0, "surprise": 0.0, "anticipation": 0.0}'
    )
    parsed = parse_llm_json(
        call_free_llm(user_prompt, system_prompt=system_prompt, json_mode=True, temperature=0.2, max_tokens=250)
    )
    if isinstance(parsed, dict):
        scores = normalize_emotions(parsed)
        if scores:
            return {e: round(scores.get(e, 0.0), 2) for e in PLUTCHIK_EMOTIONS}
    return analyze_emotions_local(text)


def annotate_text_with_emotions(text: str, language: str = "english") -> List[Dict[str, str]]:
    """Label each sentence with one emotion via LLM, with lexicon fallback."""
    lang_name = "Hebrew" if language == "hebrew" else "English"
    example = "משפט" if language == "hebrew" else "sentence"
    system_prompt = (
        f"You are an emotion expert. Analyze {lang_name} text and label each sentence with one emotion: "
        "joy, sadness, anger, fear, trust, disgust, surprise, anticipation, or neutral. "
        "Copy each sentence verbatim. Return ONLY a valid JSON array."
    )
    user_prompt = (
        f"Label each sentence in this {lang_name} text with its emotion:\n\n{text}\n\n"
        f'Return JSON: [{{"text": "{example}", "emotion": "joy"}}, ...]'
    )
    parsed = parse_llm_json(
        call_free_llm(user_prompt, system_prompt=system_prompt, json_mode=True, temperature=0.2, max_tokens=1000)
    )
    if isinstance(parsed, dict):
        parsed = next((v for v in parsed.values() if isinstance(v, list)), None)
    if isinstance(parsed, list):
        annotations = []
        for item in parsed:
            if not isinstance(item, dict) or not str(item.get("text", "")).strip():
                continue
            emotion = str(item.get("emotion", "")).lower().strip()
            if emotion not in PLUTCHIK_EMOTIONS:
                emotion = "neutral"
            annotations.append({"text": str(item["text"]), "emotion": emotion})
        if annotations:
            return annotations
    return annotate_text_local(text, language=language)


def generate_variations(
    prompt: str,
    mode: str,
    creativity: float,
    emotions: Dict[str, float],
    length: int,
    variations: int,
    language: str,
) -> List[dict]:
    """Produce up to ``MAX_VARIATIONS`` drafts, each with slightly shifted emotions/creativity."""
    results = []
    for i in range(variations):
        varied_emotions = vary_emotions(emotions, prompt, i)
        variation_creativity = round(min(1.0, creativity + i * 0.05), 3)
        draft = generate_draft(
            prompt, mode, creativity=variation_creativity, emotions=varied_emotions, length=length, language=language
        )
        if not draft:
            break
        results.append(
            {
                "draft": draft,
                "emotions": varied_emotions,
                "creativity": variation_creativity,
                "annotations": annotate_text_with_emotions(draft, language=language),
            }
        )
    logger.info("generated %d variation(s) lang=%s mode=%s", len(results), language, mode)
    return results
