from app import generate_story_text


def test_generate_story_text_uses_prompt_context_and_emotion_tone():
    story = generate_story_text(
        "The hidden library at dawn",
        {"joy": 0.4, "fear": 0.6},
        length=2,
        mode="scene",
    )

    assert isinstance(story, str)
    assert len(story) > 80
    assert "hidden" in story.lower() or "library" in story.lower()
    assert "dawn" in story.lower() or "shadow" in story.lower() or "glow" in story.lower()
