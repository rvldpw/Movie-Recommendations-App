import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.feature_engineering import (
    add_recency_weight,
    build_movie_feature_matrix,
    build_user_profile,
    get_available_genre_cols,
)
from src.collaborative import CollaborativeEngine


class RecommenderSystem:
    """Hybrid movie recommender (content + collaborative + popularity)."""

    # Blending weights — must sum to 1.0
    W_CF = 0.40
    W_CONTENT = 0.35
    W_POPULAR = 0.25

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.movies: pd.DataFrame | None = None
        self.cf: CollaborativeEngine | None = None
        self._genre_cols: list[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self) -> None:
        self.df = add_recency_weight(self.df)
        self.movies = build_movie_feature_matrix(self.df)
        self._genre_cols = get_available_genre_cols(self.df)
        self.cf = CollaborativeEngine()
        self.cf.fit(self.df)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def user_exists(self, user_id: int) -> bool:
        return user_id in self.df["userId"].values

    def all_user_ids(self) -> list[int]:
        return sorted(self.df["userId"].unique().tolist())

    def get_recent_activity(
        self, user_id: int, top_n: int = 5
    ) -> pd.DataFrame:
        """Return the user's top-scored recent movies."""
        user = (
            self.df[self.df["userId"] == user_id]
            .sort_values("interaction_score", ascending=False)
            .drop_duplicates("movieId")
            .head(top_n)
        )
        return user[["title", "rating", "datetime"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def content_scores(self, user_id: int) -> pd.DataFrame:
        """Cosine similarity between the user's genre profile and all movies."""
        user = self.df[self.df["userId"] == user_id]
        profile = build_user_profile(user, self._genre_cols)

        if profile.empty or profile.sum() == 0:
            # Return zero scores for all movies
            return pd.DataFrame(
                {"movieId": self.movies.index, "content_score": 0.0}
            )

        movie_vectors = self.movies[self._genre_cols]
        scores = cosine_similarity([profile.values], movie_vectors.values)[0]
        return pd.DataFrame({"movieId": movie_vectors.index, "content_score": scores})

    def collaborative_scores(self, user_id: int) -> pd.DataFrame:
        """Average rating of similar users, per movie."""
        neighbors = self.cf.similar_users(user_id)
        if not neighbors:
            return pd.DataFrame(columns=["movieId", "cf_score"])

        cf = (
            self.df[self.df["userId"].isin(neighbors)]
            .groupby("movieId")["rating"]
            .mean()
            .reset_index()
            .rename(columns={"rating": "cf_score"})
        )
        return cf

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------

    def recommend(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Return top-N hybrid recommendations (unseen movies only)."""
        watched = set(self.df[self.df["userId"] == user_id]["movieId"])

        content = self.content_scores(user_id)
        collab = self.collaborative_scores(user_id)

        rank = content.merge(collab, on="movieId", how="left").fillna(0)

        popularity = (
            self.df.groupby("movieId")["rating"]
            .mean()
            .reset_index()
            .rename(columns={"rating": "popularity"})
        )

        rank = rank.merge(popularity, on="movieId")

        # Normalise cf_score to [0, 1] so it's on the same scale as content_score
        cf_max = rank["cf_score"].max()
        if cf_max > 0:
            rank["cf_score"] = rank["cf_score"] / cf_max

        pop_max = rank["popularity"].max()
        if pop_max > 0:
            rank["popularity"] = rank["popularity"] / pop_max

        rank["score"] = (
            self.W_CF * rank["cf_score"]
            + self.W_CONTENT * rank["content_score"]
            + self.W_POPULAR * rank["popularity"]
        )

        rank = (
            rank[~rank["movieId"].isin(watched)]
            .merge(
                self.movies[["title"]],
                left_on="movieId",
                right_index=True,
            )
            .sort_values("score", ascending=False)
            .head(top_n)
        )

        return rank[["title", "score"]].reset_index(drop=True)

    def get_user_profile(self, user_id: int) -> list[tuple[str, float]]:
        """Return the top-8 genre weights for a user (for visualisation)."""
        user = self.df[self.df["userId"] == user_id]
        p = (
            build_user_profile(user, self._genre_cols)
            .sort_values(ascending=False)
            .head(8)
        )
        return list(zip(p.index, p.values))
