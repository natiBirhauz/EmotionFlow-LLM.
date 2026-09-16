import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request

# Set up the app
app = Flask(__name__)

PLUTCHIK_EMOTIONS = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "trust",
    "disgust",
    "surprise",
    "anticipation",
]

EMOTION_LABELS = {
    "joy": "joyful, uplifting, and optimistic",
    "sadness": "melancholic, reflective, and somber",
    "anger": "intense, forceful, and confrontational",
    "fear": "tense, anxious, and foreboding",
    "trust": "reassuring, steady, and confident",
    "disgust": "critical, repulsive, and disgusted",
    "surprise": "unexpected, striking, and dramatic",
    "anticipation": "eager, forward-looking, and exciting",
}

EMOTION_LABELS_HE = {
    "joy": "שמחה, אופטימית ומרוממת",
    "sadness": "עצובה, מלנכולית ומהורהרת",
    "anger": "זועמת, תקיפה ועוצמתית",
    "fear": "חרדה, מתוחה ומזהירה",
    "trust": "בטוחה, אמינה ומרגיעה",
    "disgust": "ביקורתית, נוקבת ודוחה",
    "surprise": "מפתיעה, דרמטית ובלתי צפויה",
    "anticipation": "דרוכה, סקרנית ומלאת ציפייה",
}

EMOTION_KEYWORDS = {
    "joy": ["happy", "joy", "delight", "cheerful", "glad", "celebrate", "radiant", "smile", "laugh", "uplift", "positive", "warmth", "שמח", "שמחה", "מאושר", "נהדר", "כיף", "נפלא", "אושר", "חיוך", "שמחתי"],
    "sadness": ["sad", "sorrow", "grief", "depressed", "heartbroken", "melancholy", "tears", "mourn", "lonely", "weep", "lost", "עצב", "עצוב", "בכי", "דמעות", "כואב", "אבל", "שברון", "בודד"],
    "anger": ["angry", "rage", "furious", "outraged", "mad", "bitter", "wrath", "hostile", "irritated", "fury", "resent", "כעס", "זעם", "רותח", "עצבני", "מרגיז", "טינה", "קריזה", "זועם"],
    "fear": ["fear", "afraid", "scared", "scary", "terrified", "panic", "dread", "horror", "anxious", "nervous", "alarm", "פחד", "חרדה", "מבוהל", "אימה", "חושש", "בהלה", "חרד", "מפחיד"],
    "trust": ["trust", "faith", "reliable", "confident", "loyal", "honest", "secure", "bond", "dependable", "true", "truth", "אמון", "בטוח", "נאמן", "ביטחון", "אמין", "שותפות", "אמונה", "אמיתי"],
    "disgust": ["disgust", "repulsive", "gross", "revolting", "vile", "sickening", "nasty", "foul", "loathe", "distaste", "גועל", "מגעיל", "דוחה", "מתעב", "מאוס", "סלידה", "בחילה", "מאוסה"],
    "surprise": ["surprise", "shock", "astonished", "stunned", "unexpected", "amazed", "wonder", "startled", "abrupt", "הפתעה", "נדהם", "בהלם", "בלתי צפוי", "פתאומי", "פליאה", "תדהמה", "מופתע"],
    "anticipation": ["anticipate", "eager", "expect", "hope", "forward", "await", "curious", "ready", "longing", "yearn", "ציפייה", "תקווה", "מחכה", "מצפה", "להוט", "דריכות", "סקרנות", "מייחל"],
}


def call_gemini(user_prompt: str, system_prompt: str = None, json_mode: bool = False, temperature: float = 0.7, max_tokens: int = 800) -> str | None:
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

    for model in ("gemini-2.0-flash", "gemini-1.5-flash"):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
                candidates = payload.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "").strip()
                        if text:
                            return text
        except Exception as e:
            print(f"Gemini ({model}) API error: {e}")
            continue
    return None


def call_groq(user_prompt: str, system_prompt: str = None, json_mode: bool = False, temperature: float = 0.7, max_tokens: int = 800) -> str | None:
    """Call Groq Free Tier API."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    for model in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
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
            with urllib.request.urlopen(req, timeout=25) as response:
                payload = json.loads(response.read().decode("utf-8"))
                msg = payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if msg:
                    return msg
        except Exception as e:
            print(f"Groq ({model}) API error: {e}")
            continue
    return None


def call_openai(user_prompt: str, system_prompt: str = None, json_mode: bool = False, temperature: float = 0.7, max_tokens: int = 800) -> str | None:
    """Call OpenAI API if configured in server environment."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEFAULT_OPENAI_API_KEY")
    if not api_key:
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    body = {
        "model": "gpt-4o-mini",
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
        with urllib.request.urlopen(req, timeout=25) as response:
            payload = json.loads(response.read().decode("utf-8"))
            msg = payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if msg:
                return msg
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return None


def call_free_llm(user_prompt: str, system_prompt: str = None, json_mode: bool = False, temperature: float = 0.7, max_tokens: int = 800) -> str | None:
    """Dispatcher: Tries Google Gemini free tier, then Groq free tier, then OpenAI."""
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


def generate_draft_local(prompt: str, mode: str, emotions: dict = None, length: int = 3, language: str = "english") -> str:
    """Built-in local emotion generator that requires zero external APIs."""
    emotions = emotions or {e: 0.125 for e in PLUTCHIK_EMOTIONS}
    active = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    dominant = active[0][0] if active else "joy"
    secondary = [name for name, val in active[1:3] if val > 0.08] or ["clarity"]
    topic = (prompt or "a fresh idea").strip()
    is_hebrew = language == "hebrew" or bool(re.search(r"[\u0590-\u05FF]", topic))

    if is_hebrew:
        tone_he = EMOTION_LABELS_HE.get(dominant, "מרגשת")
        sec_he = ", ".join([EMOTION_LABELS_HE.get(s, s) for s in secondary])
        if mode == "story":
            sentences = [
                f"סביב {topic}, אנרגיה {tone_he} החלה לפעום ולמלא את החלל בעוצמה ייחודית.",
                f"כל פרט ברגע נראה כאילו נע מתוך כוונה פנימית, והפך את החוויה לבלתי נשכחת.",
                f"ככל שהזמן חלף, האווירה השאירה תחושה עמוקה של {sec_he} שהדהדה לאורך זמן.",
            ]
            if length > 3:
                sentences.append("זה היה רגע שנחרט בזיכרון בבהירות שקטה ומשמעותית.")
            return " ".join(sentences[: max(2, length)])
        elif mode == "email":
            subject = f"נושא: עדכון מעורר השראה לגבי {topic}"
            body = [
                f"שלום רב,\n\nרציתי לשתף עדכון קצר וממוקד בנוגע ל-{topic}, שכן התחושה המובילה כרגע היא {tone_he} וחשוב לרתום אותה.",
                f"הרעיון המרכזי הוא לשמור על בהירות, ביטחון וקשר ישיר, תוך הדגשת תחושת ה-{sec_he} שנוצרה.",
                f"נשמח לקבל את המשוב שלך ולבחון יחד את הצעדים הבאים לקידום הפרויקט.",
            ]
            return f"{subject}\n\n" + "\n\n".join(body[: max(2, length)])
        elif mode == "pitch":
            bullets = [
                f"• {topic} מציג גישה חדשנית המשלבת נוכחות {tone_he} עם פתרון מעשי מוכח.",
                f"• המסר מעוצב כך שיבלוט בייחודיותו ויעביר אמינות גבוהה ומיקוד סביב {sec_he}.",
                f"• המיזם מוכן ליישום מיידי ומייצר מעורבות רגשית עמוקה בקרב קהל היעד.",
            ]
            if length > 3:
                bullets.append("• זהו בדיוק התזמון הנכון להפוך את החזון להצלחה ממשית.")
            return "\n".join(bullets[: max(2, length)])
        else:  # social
            post = [
                f"{topic} מתפתח בכיוון שמרגיש מלא אנרגיה {tone_he} וחיבור אנושי אמיתי.",
                f"המיקוד מדויק, ומביא לידי ביטוי תחושה בלתי מתפשרת של {sec_he}.",
                "צעד קדימה, והדברים מתחילים להתחבר בצורה מדויקת ומעוררת השראה.",
            ]
            if length > 3:
                post.append("#חדשנות #רגש #יצירתיות")
            return "\n\n".join(post[: max(2, length)])
    else:
        tone_en = EMOTION_LABELS.get(dominant, "vivid")
        sec_en = ", ".join(secondary)
        if mode == "story":
            sentences = [
                f"In the atmosphere surrounding {topic}, {tone_en} energy took root and gave the scene a distinctive pulse.",
                f"Every detail seemed composed with intention, turning the moment into something indelible and sharply felt.",
                f"By the conclusion, a lingering current of {sec_en} remained in the air, quiet yet unmistakable.",
            ]
            if length > 3:
                sentences.append("It settled into memory with the quiet authority of something truly felt.")
            return " ".join(sentences[: max(2, length)])
        elif mode == "email":
            subject = f"Subject: Thoughtful update on {topic}"
            body = [
                f"Hi there,\n\nI wanted to share a quick perspective on {topic}. The direction feels distinctly {tone_en} and presents an ideal moment to move forward.",
                f"Our core aim is to keep the communication grounded, impactful, and clearly guided by {sec_en}.",
                f"Looking forward to hearing your thoughts as we shape the next iteration.",
            ]
            return f"{subject}\n\n" + "\n\n".join(body[: max(2, length)])
        elif mode == "pitch":
            bullets = [
                f"• {topic.capitalize()} presents a compelling direction marked by a {tone_en} presence.",
                f"• The concept cuts through conventional noise, pairing clear utility with undertones of {sec_en}.",
                f"• Designed for immediate traction, turning emotional resonance into tangible advantage.",
            ]
            if length > 3:
                bullets.append("• This is the ideal moment to capture attention and build momentum.")
            return "\n".join(bullets[: max(2, length)])
        else:  # social
            post = [
                f"{topic.capitalize()} is developing with an energy that feels authentically {tone_en}.",
                f"Focused, clear, and designed to connect with anyone seeking genuine {sec_en}.",
                "A thoughtful shift in perspective makes all the difference.",
            ]
            if length > 3:
                post.append("#Creative #Storytelling #Innovation")
            return "\n\n".join(post[: max(2, length)])


def analyze_emotions_local(text: str) -> dict:
    """Built-in text emotion analyzer based on Plutchik's lexicon."""
    if not text:
        return {e: 0.125 for e in PLUTCHIK_EMOTIONS}
    text_lower = text.lower()
    scores = {e: 0.05 for e in PLUTCHIK_EMOTIONS}
    total = 0.0
    for emotion, keywords in EMOTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        scores[emotion] += count * 0.35
        total += scores[emotion]
    if total > 0:
        return {e: round(scores[e] / total, 2) for e in PLUTCHIK_EMOTIONS}
    return {e: 0.125 for e in PLUTCHIK_EMOTIONS}


def annotate_text_local(text: str, language: str = "english") -> list:
    """Built-in sentence annotator."""
    if not text:
        return []
    parts = re.split(r'([.!?;\n]+)', text)
    sentences = []
    for i in range(0, len(parts), 2):
        s = parts[i].strip()
        punct = parts[i+1] if i + 1 < len(parts) else ""
        full_s = f"{s}{punct}".strip()
        if full_s:
            sentences.append(full_s)
    if not sentences:
        sentences = [text]

    annotations = []
    for s in sentences:
        emo_scores = analyze_emotions_local(s)
        best_emotion = max(emo_scores.items(), key=lambda x: x[1])[0]
        annotations.append({"text": s, "emotion": best_emotion})
    return annotations


def generate_draft(prompt: str, mode: str, creativity: float = 0.7, emotions: dict = None, length: int = 3, language: str = "english") -> str:
    """Generate draft using free LLM with local fallback."""
    emotion_instruction = ""
    if emotions:
        active_emotions = [(name, val) for name, val in emotions.items() if val > 0.1]
        active_emotions.sort(key=lambda x: x[1], reverse=True)
        if active_emotions:
            primary = active_emotions[0]
            desc = EMOTION_LABELS.get(primary[0], "vivid")
            emotion_instruction = f" IMPORTANT: The writing MUST strongly convey {primary[0]} emotion ({desc}) at {int(primary[1]*100)}% intensity."
            if len(active_emotions) > 1:
                secondary = [f"{name} ({int(val*100)}%)" for name, val in active_emotions[1:3]]
                emotion_instruction += f" Include subtle undertones of {', '.join(secondary)}."

    length_map = {
        2: "very brief and concise (1-2 short paragraphs)",
        3: "moderate length (2-3 paragraphs)",
        4: "detailed and thorough (3-4 paragraphs)",
        5: "comprehensive and elaborate (4-5 paragraphs)",
    }
    length_instruction = length_map.get(length, "moderate length (2-3 paragraphs)")
    max_tokens_map = {2: 250, 3: 450, 4: 650, 5: 900}
    max_tokens = max_tokens_map.get(length, 450)

    language_instruction = ""
    if language == "hebrew":
        language_instruction = " Write the entire response in Hebrew (עברית). Use proper Hebrew grammar, natural phrasing, and rich vocabulary."

    system_prompt = (
        "You are a creative writing assistant specialized in emotional storytelling. "
        f"Write polished drafts that strongly match the requested emotional tone and intensity.{language_instruction}"
    )
    user_prompt = (
        f"Write a {length_instruction} {mode} about: {prompt}.{emotion_instruction} "
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

    # Fallback to local emotion engine
    return generate_draft_local(prompt, mode, emotions=emotions, length=length, language=language)


def analyze_text_emotions(text: str) -> dict:
    """Analyze text for Plutchik's 8 emotions using free LLM with local fallback."""
    system_prompt = (
        "You are an emotion analysis expert. Analyze text and rate the presence of Plutchik's 8 emotions "
        "(joy, sadness, anger, fear, trust, disgust, surprise, anticipation) on a scale of 0.0 to 1.0. "
        "Return ONLY a valid JSON object with emotion names as keys and decimal values between 0.0 and 1.0."
    )
    user_prompt = (
        f"Analyze the emotional content of this text and return emotion scores:\n\n{text}\n\n"
        'Return format: {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "trust": 0.0, "disgust": 0.0, "surprise": 0.0, "anticipation": 0.0}'
    )

    raw = call_free_llm(user_prompt, system_prompt=system_prompt, json_mode=True, temperature=0.2, max_tokens=250)
    if raw:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return {e: round(float(parsed.get(e, 0.0)), 2) for e in PLUTCHIK_EMOTIONS}
        except Exception as e:
            print(f"Emotion JSON parse error: {e}")

    return analyze_emotions_local(text)


def annotate_text_with_emotions(text: str, language: str = "english") -> list:
    """Annotate text with emotions using free LLM with local fallback."""
    if language == "hebrew":
        system_prompt = (
            "You are an emotion expert. Analyze Hebrew text and label each sentence with one emotion: "
            "joy, sadness, anger, fear, trust, disgust, surprise, or anticipation. Return ONLY a valid JSON array."
        )
        user_prompt = f'Label each sentence in this Hebrew text with its emotion:\n\n{text}\n\nReturn JSON: [{{"text": "משפט", "emotion": "joy"}}, ...]'
    else:
        system_prompt = (
            "You are an emotion expert. Analyze English text and label each sentence with one emotion: "
            "joy, sadness, anger, fear, trust, disgust, surprise, or anticipation. Return ONLY a valid JSON array."
        )
        user_prompt = f'Label each sentence with its emotion:\n\n{text}\n\nReturn JSON: [{{"text": "sentence", "emotion": "joy"}}, ...]'

    raw = call_free_llm(user_prompt, system_prompt=system_prompt, json_mode=True, temperature=0.2, max_tokens=1000)
    if raw:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and len(parsed) > 0:
                valid_annotations = []
                for item in parsed:
                    if isinstance(item, dict) and "text" in item and "emotion" in item:
                        emo = item["emotion"].lower()
                        if emo not in PLUTCHIK_EMOTIONS:
                            emo = "joy"
                        valid_annotations.append({"text": item["text"], "emotion": emo})
                if valid_annotations:
                    return valid_annotations
        except Exception as e:
            print(f"Annotation JSON parse error: {e}")

    return annotate_text_local(text, language=language)



@app.route("/", methods=["GET"])
def index():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EmotionFlow Studio</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script>
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
        

        
        /* Emotion color highlights - bright and visible */
        .emotion-joy { color: #fbbf24 !important; font-weight: 600 !important; }
        .emotion-sadness { color: #60a5fa !important; font-weight: 600 !important; }
        .emotion-anger { color: #ef4444 !important; font-weight: 600 !important; }
        .emotion-fear { color: #a78bfa !important; font-weight: 600 !important; }
        .emotion-trust { color: #34d399 !important; font-weight: 600 !important; }
        .emotion-disgust { color: #84cc16 !important; font-weight: 600 !important; }
        .emotion-surprise { color: #f59e0b !important; font-weight: 600 !important; }
        .emotion-anticipation { color: #ec4899 !important; font-weight: 600 !important; }
        
        /* RTL support for Hebrew */
        .rtl-text {
            direction: rtl;
            text-align: right;
        }
    </style>
</head>
<body>
    <div class="container">

        <div class="header">
            <h1>✨ EmotionFlow Studio</h1>
            <p>Turn a simple prompt into a polished draft with emotional controls</p>
        </div>
        
        <!-- Quick Examples at top -->
        <div class="examples card" style="margin-bottom: 30px;">
            <h3>🧪 Quick Examples</h3>
            <button class="example-btn" id="example1">Hidden Library</button>
            <button class="example-btn" id="example2">Calm Email</button>
            <button class="example-btn" id="example3">Sad Farewell</button>
            <button class="example-btn" id="example4">Dramatic Post</button>
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
                        <button id="toggleWheelBtn" style="padding: 4px 10px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 11px;">🎭 Wheel View</button>
                    </div>
                    
                    <div id="wheelContainer" style="display: none; margin-bottom: 16px;">
                        <canvas id="plutchikWheel" width="600" height="600" style="display: block; margin: 0 auto; cursor: crosshair; border-radius: 50%; background: rgba(0,0,0,0.3);"></canvas>
                        <p style="text-align: center; font-size: 11px; color: #94a3b8; margin-top: 8px;">Click on the wheel to set emotions</p>
                    </div>
                    
                    <div style="margin-bottom: 16px; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                        <h4 style="font-size: 12px; color: #a78bfa; margin-bottom: 8px;">Emotion Colors Guide:</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 11px;">
                            <div><span style="color: #fbbf24; font-weight: 600;">●</span> Joy</div>
                            <div><span style="color: #60a5fa; font-weight: 600;">●</span> Sadness</div>
                            <div><span style="color: #ef4444; font-weight: 600;">●</span> Anger</div>
                            <div><span style="color: #a78bfa; font-weight: 600;">●</span> Fear</div>
                            <div><span style="color: #34d399; font-weight: 600;">●</span> Trust</div>
                            <div><span style="color: #84cc16; font-weight: 600;">●</span> Disgust</div>
                            <div><span style="color: #f59e0b; font-weight: 600;">●</span> Surprise</div>
                            <div><span style="color: #ec4899; font-weight: 600;">●</span> Anticipation</div>
                        </div>
                    </div>
                    
                    <div class="emotion-grid" id="emotionSliders"></div>
                </div>
                

                
                <div style="margin-top: 16px;">
                    <label style="margin-bottom: 8px; display: block;">Quick Emotion Presets</label>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        <button id="presetHappy" style="padding: 6px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 12px;">😊 Happy</button>
                        <button id="presetSad" style="padding: 6px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 12px;">😢 Sad</button>
                        <button id="presetDramatic" style="padding: 6px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 12px;">🎭 Dramatic</button>
                        <button id="presetCalm" style="padding: 6px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 12px;">🧘 Calm</button>
                        <button id="presetIntense" style="padding: 6px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 12px;">⚡ Intense</button>
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
                
                <div style="margin-bottom: 16px;">
                    <button id="analyzeBtn" style="width: 100%; padding: 12px 16px; background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.3s;">Analyze Emotions</button>
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
        
        window.onload = function() {
            // Initialize the app
            initializeEmotions();
            updateSliderDisplays();
        };
        
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
                        <label style="font-size: 13px; font-weight: 600; color: ${EMOTION_COLORS[emotion]};">${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</label>
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
        
        // Summarize emotions in Hebrew or English
        function summarizeEmotions(emotions, isHebrew = false) {
            const ranked = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
            const dominant = ranked[0][0];
            const strength = Math.round(ranked[0][1] * 100);
            const supporting = ranked.slice(1).filter(([_, val]) => val > 0.08).map(([name, _]) => name).slice(0, 2);
            
            if (isHebrew) {
                const hebrewEmotions = {
                    joy: 'שמח',
                    sadness: 'עצוב',
                    anger: 'כועס',
                    fear: 'מפוחד',
                    trust: 'אמון',
                    disgust: 'גועל',
                    surprise: 'הפתעה',
                    anticipation: 'ציפייה'
                };
                const hebrewLabels = {
                    joy: 'משמח, מרומם ואופטימי',
                    sadness: 'מלנכולי, רפלקטיבי ועצוב',
                    anger: 'אינטנסיבי, כוחני וקונפרונטטיבי',
                    fear: 'מתוח, חרד ומאיים',
                    trust: 'מרגיע, יציב ובטוח',
                    disgust: 'ביקורתי, ספקן וחריף',
                    surprise: 'בלתי צפוי, מפתיע ודרמטי',
                    anticipation: 'נלהב, צופה פני עתיד ומרגש'
                };
                const dominantHebrew = hebrewEmotions[dominant] || dominant;
                const supportText = supporting.length > 0 
                    ? supporting.map(e => hebrewEmotions[e] || e).join(', ')
                    : 'איזון רגוע';
                return `הטקסט נוטה להיות ${hebrewLabels[dominant]} בעוצמה של ${strength}%. רגשות משניים כוללים ${supportText}.`;
            } else {
                const supportText = supporting.length > 0 ? supporting.join(', ') : 'a calm balance';
                return `The draft leans ${EMOTION_LABELS[dominant]} with ${strength}% intensity. Secondary notes include ${supportText}.`;
            }
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
            return text.trim().split(/\\s+/).length;
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
        
        // Store analyzed emotions (for future use if needed)
        let lastAnalyzedEmotions = null;
        
        // Analyze text
        async function analyzeText() {
            const text = document.getElementById('analyzeText').value.trim();
            const language = document.getElementById('language').value;
            
            if (!text) {
                showStatus('Please enter text to analyze', 'error');
                return;
            }
            
            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, language })
                });
                
                const data = await response.json();
                if (data.ok && data.emotions) {
                    lastAnalyzedEmotions = data.emotions;
                    displayAnalysisResult(data.emotions, data.annotations, text);
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
        function displayAnalysisResult(emotions, annotations, originalText) {
            const resultDiv = document.getElementById('analyzeResult');
            const sorted = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
            
            let html = '<div style="margin-top: 12px;">';
            
            // Show colored text if annotations available
            if (annotations && annotations.length > 0) {
                const isRtl = isHebrewText(originalText);
                const rtlStyle = isRtl ? 'direction: rtl; text-align: right;' : '';
                html += '<h4 style="color: #a78bfa; font-size: 13px; margin-bottom: 8px;">Emotion-Colored Text:</h4>';
                html += '<div style="padding: 12px; background: rgba(0,0,0,0.3); border-radius: 8px; margin-bottom: 16px; line-height: 1.8; ' + rtlStyle + '">';
                html += applyEmotionColors(annotations, originalText);
                html += '</div>';
            } else {
                console.log('No annotations received for analyzer');
            }
            
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
            const radius = 220;  // Increased for 600x600 canvas
            
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
                
                // Draw label - positioned closer to avoid cutoff
                const labelAngle = startAngle + angleStep / 2;
                const labelRadius = radius + 45;  // Reduced from 65 to 45
                const labelX = centerX + Math.cos(labelAngle) * labelRadius;
                const labelY = centerY + Math.sin(labelAngle) * labelRadius;
                
                // Set text alignment based on position
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                
                ctx.fillStyle = EMOTION_COLORS[emotion];
                ctx.font = 'bold 13px sans-serif';  // Smaller font to fit better
                ctx.fillText(emotion.charAt(0).toUpperCase() + emotion.slice(1), labelX, labelY);
            });
            
            // Draw center circle
            ctx.beginPath();
            ctx.arc(centerX, centerY, 15, 0, Math.PI * 2);
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
                const maxRadius = 220;  // Updated to match new radius
                
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
        
        // Apply emotion annotations from API with fallback
        function applyEmotionColors(annotations, originalText) {
            if (!annotations || annotations.length === 0) {
                // Fallback: return original text without coloring
                console.log('No annotations, using original text');
                return originalText || '';
            }
            
            const emotionColors = {
                joy: '#fbbf24',
                sadness: '#60a5fa',
                anger: '#ef4444',
                fear: '#a78bfa',
                trust: '#34d399',
                disgust: '#84cc16',
                surprise: '#f59e0b',
                anticipation: '#ec4899'
            };
            
            console.log('Applying colors to annotations:', annotations.length, 'items');
            
            // Join with space and preserve line breaks
            return annotations.map(item => {
                const color = emotionColors[item.emotion] || '#cbd5e1';
                const text = item.text || '';
                return '<span style="color: ' + color + '; font-weight: 600;">' + text + '</span>';
            }).join(' ');
        }
        
        // Calculate detected emotions from annotations
        function calculateDetectedEmotions(annotations) {
            const emotionCounts = {
                joy: 0, sadness: 0, anger: 0, fear: 0,
                trust: 0, disgust: 0, surprise: 0, anticipation: 0
            };
            
            annotations.forEach(item => {
                if (item.emotion && emotionCounts.hasOwnProperty(item.emotion)) {
                    emotionCounts[item.emotion]++;
                }
            });
            
            // Convert counts to percentages
            const total = annotations.length;
            const emotions = {};
            Object.keys(emotionCounts).forEach(emotion => {
                emotions[emotion] = total > 0 ? emotionCounts[emotion] / total : 0;
            });
            
            return emotions;
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
        
        // Generate draft
        document.getElementById('generateBtn').addEventListener('click', async () => {
            const prompt = document.getElementById('prompt').value.trim();
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
            
            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            const loadingMsg = variations > 1 ? `Generating ${variations} variations...` : 'Generating...';
            document.getElementById('outputBox').innerHTML = '<div class="loading"></div><div class="loading"></div><div class="loading"></div> ' + loadingMsg;
            
            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, mode, creativity, length, variations, emotions, language })
                });
                
                const data = await response.json();
                if (data.ok && data.results) {
                    const results = data.results;
                    
                    console.log('Results received:', results); // Debug log
                    
                    if (results.length === 1) {
                        // Single result - simple display with emotion highlighting
                        const result = results[0];
                        console.log('Single result annotations:', result.annotations);
                        const coloredText = result.annotations && result.annotations.length > 0 
                            ? applyEmotionColors(result.annotations, result.draft)
                            : result.draft;
                        const isHebrew = isHebrewText(result.draft);
                        document.getElementById('outputBox').innerHTML = wrapWithRTLIfNeeded(coloredText, result.draft);
                        
                        // Show DETECTED emotions if we have annotations
                        if (result.annotations && result.annotations.length > 0) {
                            const detectedEmotions = calculateDetectedEmotions(result.annotations);
                            document.getElementById('emotionInsight').textContent = '💡 ' + summarizeEmotions(normalizeEmotions(detectedEmotions), isHebrew);
                            updateCharts(detectedEmotions);
                        } else {
                            document.getElementById('emotionInsight').textContent = '💡 ' + summarizeEmotions(normalizeEmotions(result.emotions), isHebrew);
                            updateCharts(result.emotions);
                        }
                        document.getElementById('emotionInsight').style.display = 'block';
                    } else {
                        // Multiple results - create variation cards
                        let html = '';
                        results.forEach((result, idx) => {
                            const wordCount = countWords(result.draft);
                            console.log('Variation', idx, 'annotations:', result.annotations);
                            const coloredText = result.annotations && result.annotations.length > 0 
                                ? applyEmotionColors(result.annotations, result.draft)
                                : result.draft;
                            const isHebrew = isHebrewText(result.draft);
                            const rtlClass = isHebrew ? ' class="rtl-text"' : '';
                            
                            // Calculate detected emotions from annotations
                            let insightText;
                            if (result.annotations && result.annotations.length > 0) {
                                const detectedEmotions = calculateDetectedEmotions(result.annotations);
                                insightText = summarizeEmotions(normalizeEmotions(detectedEmotions), isHebrew);
                            } else {
                                insightText = summarizeEmotions(normalizeEmotions(result.emotions), isHebrew);
                            }
                            
                            html += `
                                <div style="margin-bottom: 24px; padding: 16px; background: rgba(0, 0, 0, 0.2); border-radius: 12px; border: 1px solid rgba(167, 139, 250, 0.2);">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                        <h3 style="color: #a78bfa; font-size: 16px; margin: 0;">Variation ${idx + 1}</h3>
                                        <div style="display: flex; gap: 8px; align-items: center;">
                                            <span style="font-size: 12px; color: #94a3b8;">${wordCount} words</span>
                                            <button class="copy-btn-${idx}" style="padding: 6px 12px; background: rgba(167, 139, 250, 0.2); border: 1px solid rgba(167, 139, 250, 0.4); border-radius: 6px; color: #cbd5e1; cursor: pointer; font-size: 12px;">📋 Copy</button>
                                        </div>
                                    </div>
                                    <div${rtlClass} style="white-space: pre-wrap; line-height: 1.6; font-size: 14px; margin-bottom: 12px;">${coloredText}</div>
                                    <div style="padding: 10px; background: rgba(167, 139, 250, 0.1); border-radius: 6px; font-size: 12px; color: #cbd5e1;">
                                        💡 ${insightText}
                                    </div>
                                </div>
                            `;
                        });
                        
                        document.getElementById('outputBox').innerHTML = html;
                        document.getElementById('emotionInsight').style.display = 'none';
                        
                        // Wire up copy buttons
                        results.forEach((result, idx) => {
                            const btn = document.querySelector('.copy-btn-' + idx);
                            if (btn) {
                                btn.addEventListener('click', function() {
                                    copyToClipboard(result.draft, this);
                                });
                            }
                        });
                        
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
        
        // Wire up example buttons
        document.getElementById('example1').addEventListener('click', () => loadExample(['The hidden library at dawn', 'story', 3, 0.7, [0.3, 0.1, 0, 0.1, 0.3, 0, 0.1, 0.2], 3]));
        document.getElementById('example2').addEventListener('click', () => loadExample(['A launch email for a calm app', 'email', 3, 0.8, [0.4, 0, 0, 0, 0.5, 0, 0.1, 0], 2]));
        document.getElementById('example3').addEventListener('click', () => loadExample(['A heartbreaking farewell', 'story', 3, 0.6, [0, 0.9, 0, 0.1, 0, 0, 0, 0], 2]));
        document.getElementById('example4').addEventListener('click', () => loadExample(['City at night', 'social', 3, 0.5, [0.1, 0.2, 0, 0.2, 0.1, 0, 0.3, 0.1], 3]));
        
        // Wire up other buttons
        document.getElementById('toggleWheelBtn').addEventListener('click', toggleWheel);
        document.getElementById('presetHappy').addEventListener('click', () => setEmotionPreset('happy'));
        document.getElementById('presetSad').addEventListener('click', () => setEmotionPreset('sad'));
        document.getElementById('presetDramatic').addEventListener('click', () => setEmotionPreset('dramatic'));
        document.getElementById('presetCalm').addEventListener('click', () => setEmotionPreset('calm'));
        document.getElementById('presetIntense').addEventListener('click', () => setEmotionPreset('intense'));
        document.getElementById('analyzeBtn').addEventListener('click', analyzeText);
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
    
    # Get emotions from payload
    emotions = payload.get("emotions", {})
    if isinstance(emotions, str):
        try:
            emotions = json.loads(emotions)
        except Exception:
            emotions = {}

    # Generate multiple variations if requested
    results = []
    for i in range(min(variations, 4)):  # Max 4 variations
        if i > 0:
            varied_emotions = {}
            for emotion, value in emotions.items():
                variation = (hash(f"{prompt}{i}") % 11 - 5) / 100
                varied_emotions[emotion] = max(0.0, min(1.0, value + variation))
        else:
            varied_emotions = emotions
            
        variation_creativity = creativity + (i * 0.05) if i > 0 else creativity
        variation_creativity = min(1.0, variation_creativity)
        
        draft = generate_draft(prompt, mode, creativity=variation_creativity, emotions=varied_emotions, length=length, language=language)
        if draft:
            annotations = annotate_text_with_emotions(draft, language=language)
            print(f"Generation {i} - Language: {language}, Draft length: {len(draft)}, Annotations: {annotations is not None}")
            
            results.append({
                "draft": draft,
                "emotions": varied_emotions,
                "creativity": variation_creativity,
                "annotations": annotations if annotations else []
            })
        else:
            break
    
    if not results:
        return jsonify({
            "ok": False,
            "message": "Generation failed. Please try again.",
        }), 500

    return jsonify({"ok": True, "results": results})


@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    """Analyze text for emotional content and provide annotations."""
    payload = request.get_json(silent=True) or {}
    
    text = payload.get("text", "").strip()
    language = payload.get("language", "english")
    
    if not text:
        return jsonify({
            "ok": False,
            "message": "Please provide text to analyze.",
        }), 400
    
    emotions = analyze_text_emotions(text)
    annotations = annotate_text_with_emotions(text, language=language)
    
    print(f"Analyze API - Language: {language}, Annotations: {annotations is not None}")
    
    return jsonify({
        "ok": True,
        "emotions": emotions,
        "annotations": annotations if annotations else [],
        "text": text
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
