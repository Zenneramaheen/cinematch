"""CineMatch recommendation engine.

Content-based filtering: every movie is turned into a TF-IDF vector built
from its genres + plot overview, and "similar movies" are the ones with the
highest cosine similarity to the movie you picked.

Why this approach?
  * It needs no user history (no cold-start problem for new users).
  * It is fully explainable: two movies are similar because they share
    descriptive words — you can point at the overlapping terms.
"""

from __future__ import annotations

import difflib
import random
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "movies.csv"

# Human moods mapped onto the genre vocabulary used in movies.csv.
# A movie matches a mood if it has ANY of the mapped genres.
MOOD_MAP: dict[str, list[str]] = {
    "feel-good": ["Comedy", "Family", "Musical", "Romance", "Animation"],
    "funny": ["Comedy", "Animation", "Family"],
    "romantic": ["Romance", "Drama"],
    "thriller": ["Thriller", "Crime", "Mystery"],
    "mind-bending": ["Sci-Fi", "Mystery", "Thriller"],
    "dark": ["Crime", "Horror", "Thriller"],
    "scary": ["Horror", "Thriller"],
    "adventurous": ["Adventure", "Action", "Fantasy"],
    "emotional": ["Drama", "Romance"],
    "epic": ["Adventure", "War", "History", "Fantasy", "Action", "Western"],
    "sporty": ["Sport", "Drama"],
}

# Genres count double: a shared genre is a stronger signal than a shared word.
GENRE_WEIGHT = 2


class MovieRecommender:
    """Loads movies.csv once, precomputes the similarity matrix, serves queries."""

    def __init__(self, csv_path: Path = DATA_PATH) -> None:
        self.df = pd.read_csv(csv_path)
        # One text blob per movie: genres (weighted) + plot overview.
        self.df["soup"] = self.df.apply(
            lambda row: " ".join(
                [row["genres"].replace("|", " ")] * GENRE_WEIGHT + [row["overview"]]
            ),
            axis=1,
        )
        # TF-IDF: rare, distinctive words ("heist", "samurai") weigh more
        # than common ones ("man", "life"). Bigrams catch phrases like
        # "time travel".
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=2
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["soup"])
        # Cosine similarity between every pair of movies (computed once).
        self.similarity = cosine_similarity(self.tfidf_matrix)

        # Case-insensitive title -> row index lookup.
        self.title_index = {
            str(title).strip().lower(): i for i, title in enumerate(self.df["title"])
        }
        self.all_genres = sorted(
            {g.strip() for genres in self.df["genres"] for g in str(genres).split("|")}
        )

    # ---------------- helpers ----------------

    @staticmethod
    def _movie_to_dict(row: pd.Series, score: float | None = None) -> dict:
        movie = {
            "title": row["title"],
            "year": int(row["year"]),
            "genres": str(row["genres"]).split("|"),
            "overview": row["overview"],
            "rating": float(row["rating"]),
        }
        if score is not None:
            movie["similarity"] = round(float(score), 3)
        return movie

    def _resolve_title(self, title: str) -> tuple[int | None, list[str]]:
        """Return (row index, []) on exact match, else (None, close matches)."""
        key = title.strip().lower()
        if key in self.title_index:
            return self.title_index[key], []
        suggestions = difflib.get_close_matches(
            title.strip(), list(self.df["title"]), n=5, cutoff=0.6
        )
        return None, suggestions

    # ---------------- public API ----------------

    def search(self, query: str, n: int = 8) -> list[dict]:
        """Fuzzy title search: substring hits first, then close matches."""
        q = query.strip().lower()
        if not q:
            return []
        substring_hits = [
            title for title in self.df["title"] if q in str(title).lower()
        ]
        fuzzy_hits = difflib.get_close_matches(
            query.strip(), list(self.df["title"]), n=n, cutoff=0.6
        )
        ordered = list(dict.fromkeys(substring_hits + fuzzy_hits))[:n]
        return [
            self._movie_to_dict(self.df.iloc[self.title_index[t.lower()]])
            for t in ordered
        ]

    def recommend_by_title(self, title: str, n: int = 5) -> dict:
        """Movies most similar to the given title (excluding itself).

        Cold-start safe: unknown titles return {"found": False, "suggestions"}
        instead of raising.
        """
        idx, suggestions = self._resolve_title(title)
        if idx is None:
            return {"found": False, "suggestions": suggestions}
        # Similarity scores for this movie against all others, best first.
        ranked = sorted(
            enumerate(self.similarity[idx]), key=lambda pair: pair[1], reverse=True
        )
        # Skip index 0 of the ranking — that's the movie itself (score 1.0).
        top = [i for i, _ in ranked if i != idx][:n]
        return {
            "found": True,
            "movie": self._movie_to_dict(self.df.iloc[idx]),
            "recommendations": [
                self._movie_to_dict(self.df.iloc[i], self.similarity[idx][i])
                for i in top
            ],
        }

    def recommend_by_genre(self, genre: str, n: int = 8) -> dict:
        """Top-rated movies in a genre. Unknown genres list what's available."""
        wanted = genre.strip().lower()
        match = next((g for g in self.all_genres if g.lower() == wanted), None)
        if match is None:
            return {"found": False, "available_genres": self.all_genres}
        mask = self.df["genres"].str.split("|").apply(lambda gs: match in gs)
        top = self.df[mask].sort_values("rating", ascending=False).head(n)
        return {
            "found": True,
            "genre": match,
            "recommendations": [self._movie_to_dict(row) for _, row in top.iterrows()],
        }

    def recommend_by_mood(self, mood: str, n: int = 8) -> dict:
        """Map a human mood onto genres, rank by genre overlap then rating."""
        key = mood.strip().lower().replace(" ", "-")
        if key not in MOOD_MAP:
            return {"found": False, "available_moods": sorted(MOOD_MAP)}
        wanted = set(MOOD_MAP[key])

        def overlap(genres: str) -> int:
            return len(wanted & set(str(genres).split("|")))

        scored = self.df.copy()
        scored["overlap"] = scored["genres"].apply(overlap)
        top = scored[scored["overlap"] > 0].sort_values(
            ["overlap", "rating"], ascending=False
        ).head(n)
        return {
            "found": True,
            "mood": key,
            "genres": sorted(wanted),
            "recommendations": [
                self._movie_to_dict(row) for _, row in top.drop(columns="overlap").iterrows()
            ],
        }

    def surprise_me(self, seed: int | None = None) -> dict:
        """A random highly-rated pick (rating >= 7.5). Seed for reproducibility."""
        rng = random.Random(seed)
        pool = self.df[self.df["rating"] >= 7.5]
        row = pool.iloc[rng.randrange(len(pool))]
        return {"movie": self._movie_to_dict(row)}
