"""API tests for CineMatch (FastAPI TestClient — no server needed)."""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_root_serves_ui(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "CineMatch" in res.text


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["movies"] >= 300


def test_recommend_known_title(client):
    res = client.get("/recommend", params={"title": "Inception", "n": 5})
    assert res.status_code == 200
    data = res.json()
    assert data["found"] is True
    assert len(data["recommendations"]) == 5


def test_recommend_typo_returns_suggestions_not_error(client):
    # Cold-start handling: 200 + "did you mean", never a 500.
    res = client.get("/recommend", params={"title": "Inceptoin"})
    assert res.status_code == 200
    data = res.json()
    assert data["found"] is False
    assert "Inception" in data["suggestions"]


def test_recommend_gibberish_returns_empty_suggestions(client):
    res = client.get("/recommend", params={"title": "zzzzqqqqxxxx"})
    assert res.status_code == 200
    data = res.json()
    assert data["found"] is False
    assert data["suggestions"] == []


def test_search(client):
    res = client.get("/search", params={"q": "dangal"})
    assert res.status_code == 200
    titles = [m["title"] for m in res.json()["results"]]
    assert "Dangal" in titles


def test_genres(client):
    res = client.get("/genres")
    assert res.status_code == 200
    names = [g["name"] for g in res.json()["genres"]]
    assert "Sci-Fi" in names and "Comedy" in names


def test_mood_valid(client):
    res = client.get("/mood/feel-good", params={"n": 5})
    assert res.status_code == 200
    assert len(res.json()["recommendations"]) == 5


def test_mood_invalid_is_404_with_options(client):
    res = client.get("/mood/sleepy")
    assert res.status_code == 404
    assert "available_moods" in res.json()["detail"]


def test_surprise(client):
    res = client.get("/surprise")
    assert res.status_code == 200
    assert res.json()["movie"]["rating"] >= 7.5
