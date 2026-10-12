"""Tests for the CineMatch recommendation engine."""

import math

import pytest

from recommender import MOOD_MAP, MovieRecommender


@pytest.fixture(scope="module")
def rec():
    # Built once for the whole module — the similarity matrix is precomputed.
    return MovieRecommender()


# ---------------- dataset sanity ----------------

def test_dataset_loads_enough_movies(rec):
    assert len(rec.df) >= 300


def test_dataset_columns(rec):
    assert {"title", "year", "genres", "overview", "rating"} <= set(rec.df.columns)


def test_no_duplicate_titles(rec):
    assert rec.df["title"].str.lower().nunique() == len(rec.df)


def test_similarity_matrix_shape_and_diagonal(rec):
    n = len(rec.df)
    assert rec.similarity.shape == (n, n)
    # A movie is perfectly similar to itself.
    for i in (0, n // 2, n - 1):
        assert math.isclose(rec.similarity[i][i], 1.0, rel_tol=1e-6)


# ---------------- recommend_by_title ----------------

def test_recommend_by_title_returns_n(rec):
    out = rec.recommend_by_title("Inception", n=5)
    assert out["found"] is True
    assert len(out["recommendations"]) == 5
    # Never recommends the queried movie itself.
    assert all(m["title"] != "Inception" for m in out["recommendations"])
    # Scores are present and sorted best-first.
    scores = [m["similarity"] for m in out["recommendations"]]
    assert all(0 < s <= 1 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_recommendations_are_genre_sane(rec):
    # Inception is Sci-Fi/Action/Thriller — most neighbours should share a genre.
    out = rec.recommend_by_title("Inception", n=5)
    inception_genres = set(out["movie"]["genres"])
    shared = sum(
        1 for m in out["recommendations"]
        if inception_genres & set(m["genres"])
    )
    assert shared >= 3


def test_title_lookup_is_case_insensitive(rec):
    lower = rec.recommend_by_title("inception", n=3)
    upper = rec.recommend_by_title("INCEPTION", n=3)
    assert lower["found"] and upper["found"]
    assert [m["title"] for m in lower["recommendations"]] == [
        m["title"] for m in upper["recommendations"]
    ]


def test_unknown_title_returns_suggestions(rec):
    out = rec.recommend_by_title("Inceptoin", n=5)  # typo
    assert out["found"] is False
    assert "Inception" in out["suggestions"]


def test_gibberish_title_returns_no_suggestions(rec):
    out = rec.recommend_by_title("zzzzqqqqxxxx", n=5)
    assert out["found"] is False
    assert out["suggestions"] == []


# ---------------- search ----------------

def test_search_finds_partial_titles(rec):
    results = rec.search("godfather")
    titles = [m["title"] for m in results]
    assert "The Godfather" in titles
    assert "The Godfather Part II" in titles


def test_search_empty_query(rec):
    assert rec.search("   ") == []


# ---------------- recommend_by_genre ----------------

def test_genre_returns_top_rated_in_genre(rec):
    out = rec.recommend_by_genre("Sci-Fi", n=8)
    assert out["found"] is True
    assert len(out["recommendations"]) == 8
    assert all("Sci-Fi" in m["genres"] for m in out["recommendations"])
    ratings = [m["rating"] for m in out["recommendations"]]
    assert ratings == sorted(ratings, reverse=True)


def test_unknown_genre_lists_available(rec):
    out = rec.recommend_by_genre("Noir-ish", n=5)
    assert out["found"] is False
    assert "Sci-Fi" in out["available_genres"]


# ---------------- recommend_by_mood ----------------

def test_mood_maps_to_genres(rec):
    out = rec.recommend_by_mood("feel-good", n=8)
    assert out["found"] is True
    mood_genres = set(MOOD_MAP["feel-good"])
    assert all(mood_genres & set(m["genres"]) for m in out["recommendations"])


def test_every_genre_is_reachable_from_some_mood(rec):
    covered = {g for genres in MOOD_MAP.values() for g in genres}
    assert set(rec.all_genres) <= covered


def test_unknown_mood_lists_available(rec):
    out = rec.recommend_by_mood("sleepy", n=5)
    assert out["found"] is False
    assert "feel-good" in out["available_moods"]


# ---------------- surprise_me ----------------

def test_surprise_returns_highly_rated_movie(rec):
    out = rec.surprise_me(seed=42)
    assert out["movie"]["rating"] >= 7.5
    assert out["movie"]["title"]


def test_surprise_is_reproducible_with_seed(rec):
    a = rec.surprise_me(seed=7)["movie"]["title"]
    b = rec.surprise_me(seed=7)["movie"]["title"]
    assert a == b
