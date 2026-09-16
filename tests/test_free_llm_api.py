import json
import pytest
from api.index import (
    app,
    analyze_text_emotions,
    annotate_text_with_emotions,
    generate_draft,
    PLUTCHIK_EMOTIONS,
)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok"}


def test_index_has_no_google_oauth_or_api_key_field(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "googleSignInButton" not in html
    assert "freeUsesRemaining" not in html
    assert "OpenAI API Key" not in html
    assert "apiKey" not in html


def test_generate_api_without_api_key(client):
    res = client.post(
        "/api/generate",
        json={
            "prompt": "Morning sunlight breaking through clouds",
            "mode": "story",
            "creativity": 0.7,
            "emotions": {"joy": 0.8, "anticipation": 0.4},
            "length": 2,
            "variations": 1,
            "language": "english",
        },
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("ok") is True
    assert len(data.get("results", [])) == 1
    assert "draft" in data["results"][0]
    assert len(data["results"][0]["draft"]) > 10
    assert "annotations" in data["results"][0]


def test_analyze_api_without_api_key(client):
    res = client.post(
        "/api/analyze",
        json={
            "text": "I feel absolutely ecstatic and full of joy!",
            "language": "english",
        },
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("ok") is True
    emotions = data.get("emotions", {})
    for emotion in PLUTCHIK_EMOTIONS:
        assert emotion in emotions
    assert emotions["joy"] > 0.2


def test_hebrew_generate_and_analyze(client):
    res_gen = client.post(
        "/api/generate",
        json={
            "prompt": "רגע מיוחד של שקט",
            "mode": "story",
            "emotions": {"joy": 0.6, "trust": 0.5},
            "length": 2,
            "variations": 1,
            "language": "hebrew",
        },
    )
    assert res_gen.status_code == 200
    data_gen = res_gen.get_json()
    assert data_gen.get("ok") is True

    res_an = client.post(
        "/api/analyze",
        json={
            "text": "אני מרגיש שמחה עצומה והתרגשות גדולה",
            "language": "hebrew",
        },
    )
    assert res_an.status_code == 200
    data_an = res_an.get_json()
    assert data_an.get("ok") is True
    assert data_an["emotions"]["joy"] > 0.2
