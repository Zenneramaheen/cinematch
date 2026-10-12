"""CineMatch API — FastAPI wrapper around the recommendation engine.

Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from recommender import MovieRecommender

BASE_DIR = Path(__file__).parent

app = FastAPI(
    title="CineMatch",
    description="Content-based movie recommendations (TF-IDF + cosine similarity).",
    version="1.0.0",
)

# Built once at import time so every request (and the test client) reuses it.
recommender = MovieRecommender()

# Frontend assets.
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the web UI."""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "movies": len(recommender.df)}


@app.get("/recommend")
def recommend(
    title: str = Query(..., description="Movie title, e.g. 'Inception'"),
    n: int = Query(5, ge=1, le=20),
) -> dict:
    """Recommend movies similar to a title.

    Unknown titles don't error — they return close "did you mean" matches.
    """
    return recommender.recommend_by_title(title, n)


@app.get("/search")
def search(
    q: str = Query(..., description="Fuzzy title search, e.g. 'godfather'"),
    n: int = Query(8, ge=1, le=20),
) -> dict:
    return {"query": q, "results": recommender.search(q, n)}


@app.get("/genres")
def genres() -> dict:
    counts = (
        recommender.df["genres"]
        .str.split("|")
        .explode()
        .value_counts()
        .rename_axis("genre")
        .reset_index(name="count")
    )
    return {
        "genres": [
            {"name": row["genre"], "count": int(row["count"])}
            for _, row in counts.iterrows()
        ]
    }


@app.get("/mood/{mood}")
def mood(mood: str, n: int = Query(8, ge=1, le=20)) -> dict:
    result = recommender.recommend_by_mood(mood, n)
    if not result["found"]:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Unknown mood '{mood}'.",
                "available_moods": result["available_moods"],
            },
        )
    return result


@app.get("/surprise")
def surprise(seed: int | None = Query(None, description="Optional seed")) -> dict:
    return recommender.surprise_me(seed)
