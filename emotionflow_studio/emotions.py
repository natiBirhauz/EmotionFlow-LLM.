"""Plutchik emotion vocabulary shared by the web app (labels, prompts, lexicon)."""

NEUTRAL = "neutral"

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

