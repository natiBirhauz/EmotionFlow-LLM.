"""Offline tests for the emotionflow_studio package (no LLM keys needed)."""

import pytest

from api.index import app
from emotionflow_studio import services
from emotionflow_studio.llm_providers import parse_llm_json
from emotionflow_studio.local_engine import analyze_emotions_local, annotate_text_local, split_sentences


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_variation_offset_is_deterministic_and_bounded():
    for i in range(1, 5):
        a = services.variation_offset("a quiet morning", i)
        assert a == services.variation_offset("a quiet morning", i)
        assert -0.05 <= a <= 0.05
    assert services.vary_emotions({"joy": 0.5}, "p", 0) == {"joy": 0.5}
    assert 0.0 <= services.vary_emotions({"joy": 1.0}, "p", 2)["joy"] <= 1.0


def test_normalize_emotions_drops_unknown_and_clamps():
    out = services.normalize_emotions({"joy": "1.7", "anger": -3, "bogus": 1, "Fear": 0.25})
    assert out == {"joy": 1.0, "anger": 0.0, "fear": 0.25}
    assert services.normalize_emotions("not a dict") == {}


def test_parse_llm_json_handles_fences_and_prose():
    assert parse_llm_json('```json\n{"joy": 0.5}\n```') == {"joy": 0.5}
    assert parse_llm_json('Sure! Here it is: [{"text": "hi", "emotion": "joy"}]') == [{"text": "hi", "emotion": "joy"}]
    assert parse_llm_json("nope") is None
    assert parse_llm_json(None) is None


def test_local_analyzer_is_neutral_without_cues_and_matches_whole_words():
    assert set(analyze_emotions_local("The meeting is at noon.").values()) == {0.0}
    # "sad" must not fire inside "Saddle"; "mad" must not fire inside "made"
    assert analyze_emotions_local("Saddle up, we made it.")["sadness"] == 0.0
    scores = analyze_emotions_local("I am so happy and glad today!")
    assert scores["joy"] == 1.0
    assert abs(sum(scores.values()) - 1.0) < 0.05


def test_local_analyzer_hebrew_with_prefix():
    assert analyze_emotions_local("הייתי בשמחה גדולה")["joy"] > 0.5


def test_annotate_local_marks_neutral_sentences():
    ann = annotate_text_local("The train left at nine. I was so happy to see you!")
    assert [a["emotion"] for a in ann] == ["neutral", "joy"]
    assert ann[0]["text"] == "The train left at nine."
    assert split_sentences("") == []


def test_generate_api_clamps_and_validates_input(client):
    res = client.post(
        "/api/generate",
        json={
            "prompt": "x",
            "mode": "<script>",
            "creativity": "abc",
            "length": 99,
            "variations": 10,
            "language": "klingon",
            "emotions": {"joy": 5, "nonsense": 1},
        },
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert len(data["results"]) == services.MAX_VARIATIONS
    assert data["results"][0]["emotions"] == {"joy": 1.0}
    assert data["results"][0]["creativity"] == 0.7
    assert data["results"][1]["creativity"] == 0.75


def test_generate_api_get_with_query_string(client):
    res = client.get("/api/generate?prompt=hello&emotions=%7B%22fear%22%3A0.9%7D&variations=1")
    assert res.status_code == 200
    assert res.get_json()["results"][0]["emotions"] == {"fear": 0.9}


def test_analyze_api_rejects_empty_and_huge_text(client):
    assert client.post("/api/analyze", json={"text": "   "}).status_code == 400
    assert client.post("/api/analyze", json={"text": "a" * (services.MAX_ANALYZE_CHARS + 1)}).status_code == 413


def test_index_escapes_model_output(client):
    html = client.get("/").get_data(as_text=True)
    assert "function escapeHtml" in html
    assert "escapeHtml(result.draft)" in html
