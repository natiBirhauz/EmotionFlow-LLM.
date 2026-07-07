"""EmotionFlow-LLM UI redesign.

This version replaces the placeholder demo with a more useful content
studio that generates readable drafts for stories, emails, pitches, and
short social posts from emotional controls.
"""

import json
import os
import random
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

EMOTIONS = ["joy", "sadness", "anger", "fear", "trust", "disgust", "surprise", "anticipation"]
EMOTION_LABELS = {
    "joy": "bright and uplifting",
    "sadness": "reflective and tender",
    "anger": "firm and forceful",
    "fear": "tense and shadowed",
    "trust": "steady and reassuring",
    "disgust": "sharp and skeptical",
    "surprise": "electric and unexpected",
    "anticipation": "eager and forward-looking",
}


def normalize_emotions(emotions: Dict[str, float]) -> Dict[str, float]:
    """Normalize emotion values so they sum to 1.0."""
    total = sum(emotions.values())
    if total <= 0:
        return {emotion: 0.0 for emotion in EMOTIONS}
    return {emotion: emotions.get(emotion, 0.0) / total for emotion in EMOTIONS}


def summarize_emotions(emotions: Dict[str, float]) -> str:
    """Return a human-friendly description of the current emotional profile."""
    ranked = sorted(emotions.items(), key=lambda item: item[1], reverse=True)
    dominant = ranked[0][0]
    strength = round(ranked[0][1] * 100)
    supporting = [name for name, value in ranked[1:] if value > 0.08]
    support_text = ", ".join(supporting[:2]) if supporting else "a calm balance"
    return (
        f"The draft leans {EMOTION_LABELS[dominant]} with {strength}% intensity. "
        f"Secondary notes include {support_text}."
    )


def get_active_api_key(user_api_key: str | None) -> str | None:
    """Return the first usable API key from the user input or shared environment fallback."""
    if user_api_key and user_api_key.strip():
        return user_api_key.strip()
    for env_name in ("OPENAI_API_KEY", "DEFAULT_OPENAI_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value
    return None


def generate_with_openai(prompt: str, mode: str, api_key: str | None, creativity: float = 0.7) -> str | None:
    """Use the OpenAI chat API for a higher-quality draft when a key is available."""
    if not api_key:
        return None

    system_prompt = (
        "You are a creative writing assistant. Write a concise, polished draft that matches the user's prompt "
        "and the requested tone. Keep the response useful and readable."
    )
    user_prompt = (
        f"Create a {mode} style draft for this prompt: {prompt}. "
        f"Make the voice feel {'more vivid' if creativity > 0.7 else 'balanced'} and polished."
    )

    request_data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
            message = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
            return message.strip() or None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
        return None


def _pick_word(prompt: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", prompt.lower()).strip()
    words = [word for word in cleaned.split() if len(word) > 2]
    return words[0] if words else fallback


def generate_story_text(prompt: str, emotions: Dict[str, float], length: int = 3, mode: str = "story") -> str:
    """Generate a polished draft from a prompt and emotional profile."""
    normalized = normalize_emotions(emotions)
    ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    dominant = ranked[0][0]
    secondary = [name for name, value in ranked[1:] if value > 0.08][:2]
    topic = (prompt or "a new idea").strip() or "a new idea"
    anchor = _pick_word(topic, "moment")
    tone = EMOTION_LABELS[dominant]
    detail = "glow" if dominant in {"joy", "anticipation", "surprise"} else "shadow"
    if mode == "story":
        sentences = [
            f"In the hush around {topic}, {tone} energy gathered and gave the scene a striking pulse.",
            f"Every detail seemed to move with intention, turning {anchor} into something vivid and memorable.",
            f"By the end, the atmosphere felt {detail}-filled and alive, leaving behind a sense of {', '.join(secondary or ['clarity'])}.",
        ]
        if length > 3:
            sentences.append("The moment settled into memory with a quiet, lasting force.")
        return " ".join(sentences[: max(2, length)])

    if mode == "email":
        subject = f"Subject: A thoughtful update about {topic}"
        body = [
            f"Hi there, I wanted to share a quick note about {topic} because the current mood feels {tone} and worth acting on.",
            f"The main idea is simple: make the message feel clear, grounded, and useful while keeping the energy {detail} and inviting.",
            f"If this direction feels right, I can help shape it into a polished note or a fuller draft.",
        ]
        if length > 3:
            body.append("The goal is to keep the tone warm, actionable, and easy to respond to.")
        return "\n\n".join([subject, *body[: max(2, length)]])

    if mode == "pitch":
        bullets = [
            f"{topic.capitalize()} is presented as a bold idea that turns {anchor} into a practical advantage.",
            f"The experience feels {tone}, which helps the concept stand out and stay memorable.",
            f"This direction makes the message feel fresh, credible, and ready for real momentum.",
        ]
        if length > 3:
            bullets.append("It gives the audience a clear reason to pay attention and act.")
        return "\n".join([f"• {line}" for line in bullets[: max(2, length)]])

    # social mode
    post = [
        f"{topic.capitalize()} is shaping up in a way that feels {tone} and surprisingly human.",
        f"The energy is focused, vivid, and ready to connect with people who care about {anchor}.",
        f"A little more clarity, a little more courage, and the message starts to land.",
    ]
    if length > 3:
        post.append("#creative #writing #ideas")
    return " ".join(post[: max(2, length)])


def generate_multiple_samples(prompt: str, emotions: Dict[str, float], num_samples: int, mode: str, length: int) -> List[str]:
    """Generate a small set of polished variations."""
    samples = []
    for index in range(num_samples):
        sample_emotions = dict(emotions)
        if index > 0:
            for emotion in sample_emotions:
                noise = random.uniform(-0.03, 0.03)
                sample_emotions[emotion] = max(0.0, min(1.0, sample_emotions[emotion] + noise))
            sample_emotions = normalize_emotions(sample_emotions)
        samples.append(generate_story_text(prompt, sample_emotions, length=length, mode=mode))
    return samples


def generate_with_ui(
    prompt_text: str,
    mode: str,
    length: int,
    creativity: float,
    joy: float,
    sadness: float,
    anger: float,
    fear: float,
    trust: float,
    disgust: float,
    surprise: float,
    anticipation: float,
    num_samples: int,
    api_key: str,
):
    """Generate a polished response and supporting visuals."""
    if not prompt_text or not prompt_text.strip():
        prompt_text = "A fresh idea for tomorrow"

    emotions = {
        "joy": joy,
        "sadness": sadness,
        "anger": anger,
        "fear": fear,
        "trust": trust,
        "disgust": disgust,
        "surprise": surprise,
        "anticipation": anticipation,
    }
    if sum(emotions.values()) <= 0:
        return "⚠️ Move at least one slider to create a draft.", None, None, None

    normalized = normalize_emotions(emotions)
    active_key = get_active_api_key(api_key)
    ai_draft = None
    source_label = "local emotional draft"
    if active_key:
        ai_draft = generate_with_openai(prompt_text, mode, active_key, creativity=creativity)
        if ai_draft:
            source_label = "AI draft"

    if ai_draft:
        samples = [ai_draft]
        if num_samples > 1:
            samples.extend(generate_multiple_samples(prompt_text, normalized, max(1, num_samples - 1), mode, length))
    else:
        samples = generate_multiple_samples(prompt_text, normalized, num_samples, mode, length)

    intro = f"## ✨ {mode.title()} draft\n\n**Prompt:** {prompt_text}\n\n**Emotion profile:** {summarize_emotions(normalized)}\n\n**Source:** {source_label}"
    body = []
    for index, sample in enumerate(samples, 1):
        body.append(f"### Option {index}\n{sample}\n")

    if creativity > 0.7:
        body.append("\n💡 Tip: Increase the intensity of surprise or anticipation for a more vivid, cinematic result.")

    text_output = intro + "\n" + "\n".join(body)

    fig1, ax1 = plt.subplots(figsize=(8, 3.8))
    ordered = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    labels = [name.capitalize() for name, _ in ordered]
    values = [value for _, value in ordered]
    colors = ["#7c3aed", "#2563eb", "#f97316", "#dc2626", "#14b8a6", "#84cc16", "#f43f5e", "#eab308"]
    ax1.bar(labels, values, color=colors[: len(labels)])
    ax1.set_title("Emotion balance")
    ax1.set_ylim(0, 1.0)
    for bar, value in zip(ax1.patches, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.0%}", ha="center", fontsize=9)
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"projection": "polar"})
    angles = np.linspace(0, 2 * np.pi, len(EMOTIONS), endpoint=False).tolist()
    values_radar = [normalized[emotion] for emotion in EMOTIONS]
    values_radar += values_radar[:1]
    angles += angles[:1]
    ax2.plot(angles, values_radar, color="#7c3aed", linewidth=2)
    ax2.fill(angles, values_radar, color="#8b5cf6", alpha=0.22)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([emotion.capitalize() for emotion in EMOTIONS], fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Emotion radar")
    plt.tight_layout()

    return text_output, f"**Why this works:** {summarize_emotions(normalized)}", fig1, fig2


def build_interface():
    """Create the redesigned Gradio experience."""
    custom_css = """
    .gradio-container { max-width: 1400px !important; font-family: 'Segoe UI', sans-serif; }
    .main-panel { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 16px; border-radius: 18px; }
    .panel-card { border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 14px; }
    """

    with gr.Blocks(title="EmotionFlow Studio", css=custom_css, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")) as demo:
        gr.Markdown(
            """
            # ✨ EmotionFlow Studio
            Turn a simple prompt into a polished draft that feels intentional, vivid, and useful.
            Choose a format, steer the emotions, and generate something that feels closer to a real writing assistant.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(elem_classes=["panel-card"]):
                    prompt = gr.Textbox(
                        label="Prompt or topic",
                        value="The hidden library at dawn",
                        placeholder="Describe the scene, message, or idea you want to shape",
                        lines=2,
                    )
                    mode = gr.Dropdown(
                        choices=["story", "email", "pitch", "social"],
                        value="story",
                        label="Output format",
                    )
                    length = gr.Slider(minimum=2, maximum=5, value=3, step=1, label="Length")
                    creativity = gr.Slider(minimum=0.2, maximum=1.0, value=0.7, step=0.1, label="Creativity")

                    gr.Markdown("### 🎛️ Emotional tone")
                    with gr.Row():
                        with gr.Column():
                            joy = gr.Slider(0, 1, value=0.3, label="Joy", step=0.05)
                            sadness = gr.Slider(0, 1, value=0.0, label="Sadness", step=0.05)
                            anger = gr.Slider(0, 1, value=0.0, label="Anger", step=0.05)
                            fear = gr.Slider(0, 1, value=0.0, label="Fear", step=0.05)
                        with gr.Column():
                            trust = gr.Slider(0, 1, value=0.3, label="Trust", step=0.05)
                            disgust = gr.Slider(0, 1, value=0.0, label="Disgust", step=0.05)
                            surprise = gr.Slider(0, 1, value=0.2, label="Surprise", step=0.05)
                            anticipation = gr.Slider(0, 1, value=0.2, label="Anticipation", step=0.05)

                    num_samples = gr.Slider(minimum=1, maximum=4, value=3, step=1, label="Variations")
                    api_key = gr.Textbox(
                        label="OpenAI API key (optional)",
                        type="password",
                        placeholder="Paste your key here, or leave blank to use the shared deployment key",
                    )
                    generate_btn = gr.Button("Generate draft", variant="primary", size="lg")

                gr.Markdown(
                    """
                    💡 Leave the API key blank to use the shared deployment key if one is configured in Vercel. You can also paste your own key for private use.
                    """
                )

            with gr.Column(scale=1):
                output_text = gr.Markdown(value="*Your polished draft will appear here.*")
                insight = gr.Markdown(value="*The emotional rationale will appear here.*")
                with gr.Row():
                    plot_one = gr.Plot(label="Emotion balance")
                    plot_two = gr.Plot(label="Emotion radar")

        gr.Markdown("### 🧪 Quick starters")
        gr.Examples(
            examples=[
                ["The hidden library at dawn", "story", 3, 0.7, 0.1, 0.0, 0.0, 0.5, 0.4, 0.0, 0.1, 0.1, 3],
                ["A launch email for a calm new app", "email", 3, 0.8, 0.2, 0.0, 0.0, 0.4, 0.0, 0.1, 0.2, 0.1, 2],
                ["A product pitch for a smart notebook", "pitch", 3, 0.7, 0.0, 0.0, 0.0, 0.3, 0.0, 0.2, 0.2, 0.1, 2],
                ["A dramatic post about a city at night", "social", 3, 0.5, 0.2, 0.1, 0.3, 0.1, 0.0, 0.3, 0.1, 0.0, 3],
            ],
            inputs=[prompt, mode, length, creativity, joy, sadness, anger, fear, trust, disgust, surprise, anticipation, num_samples],
            label="Click an example to load it",
        )

        generate_btn.click(
            fn=generate_with_ui,
            inputs=[prompt, mode, length, creativity, joy, sadness, anger, fear, trust, disgust, surprise, anticipation, num_samples, api_key],
            outputs=[output_text, insight, plot_one, plot_two],
        )

    return demo


def main():
    """Launch the redesigned web UI."""
    print("EmotionFlow Studio")
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)


if __name__ == "__main__":
    main()

