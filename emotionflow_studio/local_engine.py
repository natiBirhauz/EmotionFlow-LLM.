"""Zero-dependency fallback engine: template drafts + lexicon-based emotion analysis.

Used whenever no LLM provider is configured or every provider fails, so the
app always returns *something* (and the test-suite runs offline).
"""

import re
from typing import Dict, List

from .emotions import EMOTION_KEYWORDS, EMOTION_LABELS, EMOTION_LABELS_HE, NEUTRAL, PLUTCHIK_EMOTIONS

HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
SENTENCE_SPLIT_RE = re.compile(r"([.!?;\n]+)")
WORD_RE = re.compile(r"[a-z']+")


def is_hebrew(text: str) -> bool:
    return bool(HEBREW_RE.search(text or ""))


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
                "כל פרט ברגע נראה כאילו נע מתוך כוונה פנימית, והפך את החוויה לבלתי נשכחת.",
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
                "נשמח לקבל את המשוב שלך ולבחון יחד את הצעדים הבאים לקידום הפרויקט.",
            ]
            return f"{subject}\n\n" + "\n\n".join(body[: max(2, length)])
        elif mode == "pitch":
            bullets = [
                f"• {topic} מציג גישה חדשנית המשלבת נוכחות {tone_he} עם פתרון מעשי מוכח.",
                f"• המסר מעוצב כך שיבלוט בייחודיותו ויעביר אמינות גבוהה ומיקוד סביב {sec_he}.",
                "• המיזם מוכן ליישום מיידי ומייצר מעורבות רגשית עמוקה בקרב קהל היעד.",
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
                "Every detail seemed composed with intention, turning the moment into something indelible and sharply felt.",
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
                "Looking forward to hearing your thoughts as we shape the next iteration.",
            ]
            return f"{subject}\n\n" + "\n\n".join(body[: max(2, length)])
        elif mode == "pitch":
            bullets = [
                f"• {topic.capitalize()} presents a compelling direction marked by a {tone_en} presence.",
                f"• The concept cuts through conventional noise, pairing clear utility with undertones of {sec_en}.",
                "• Designed for immediate traction, turning emotional resonance into tangible advantage.",
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


def _keyword_pattern(keywords: List[str]) -> "re.Pattern[str]":
    """Whole-word matcher; Hebrew keywords may carry up to two clitic prefixes (ו, ה, ב, ל, מ, ש, כ)."""
    alternatives = []
    for kw in keywords:
        escaped = re.escape(kw.lower())
        if is_hebrew(kw):
            alternatives.append(rf"(?<![֐-׿])[והבלמשכ]{{0,2}}{escaped}(?![֐-׿])")
        else:
            alternatives.append(rf"(?<![a-z'])(?:{escaped}(?:s|es|ed|ing|ly)?)(?![a-z'])")
    return re.compile("|".join(alternatives), re.IGNORECASE)


EMOTION_PATTERNS = {emotion: _keyword_pattern(words) for emotion, words in EMOTION_KEYWORDS.items()}


def count_emotion_hits(text: str) -> Dict[str, int]:
    text_lower = (text or "").lower()
    return {emotion: len(pattern.findall(text_lower)) for emotion, pattern in EMOTION_PATTERNS.items()}


def analyze_emotions_local(text: str) -> Dict[str, float]:
    """Lexicon analyzer. Scores sum to 1.0; text with no emotional cue scores 0.0 everywhere."""
    hits = count_emotion_hits(text)
    total = sum(hits.values())
    if total == 0:
        return {e: 0.0 for e in PLUTCHIK_EMOTIONS}
    return {e: round(hits[e] / total, 2) for e in PLUTCHIK_EMOTIONS}


def split_sentences(text: str) -> List[str]:
    """Split on sentence punctuation / newlines, keeping the punctuation attached."""
    parts = SENTENCE_SPLIT_RE.split(text or "")
    sentences = []
    for i in range(0, len(parts), 2):
        body = parts[i].strip()
        punct = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            sentences.append(f"{body}{punct}")
    return sentences or ([text.strip()] if text and text.strip() else [])


def annotate_text_local(text: str, language: str = "english") -> List[Dict[str, str]]:
    """Label each sentence with its dominant emotion, or ``neutral`` when no cue is found."""
    annotations = []
    for sentence in split_sentences(text):
        hits = count_emotion_hits(sentence)
        best_emotion, best_count = max(hits.items(), key=lambda kv: kv[1])
        annotations.append({"text": sentence, "emotion": best_emotion if best_count > 0 else NEUTRAL})
    return annotations
