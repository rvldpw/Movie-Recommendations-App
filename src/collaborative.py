import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Keep only top N users and movies to avoid memory explosion
# even with 500k rows, userId x movieId can be billions of cells
TOP_USERS  = 5_000
TOP_MOVIES = 2_000


class CollaborativeEngine:
    """SVD-based user-user collaborative filtering engine."""

    def __init__(self, n_components: int = 50):
        self.n_components  = n_components
        self.user_matrix   = None
        self.similarity    = None

    def fit(self, df: pd.DataFrame) -> None:
        # Keep only the most active users and most rated movies
        top_users  = df["userId"].value_counts().head(TOP_USERS).index
        top_movies = df["movieId"].value_counts().head(TOP_MOVIES).index

        df_sub = df[
            df["userId"].isin(top_users) &
            df["movieId"].isin(top_movies)
        ]

        print(f"[collaborative] Building matrix: {df_sub['userId'].nunique()} users x "
              f"{df_sub['movieId'].nunique()} movies …")

        matrix = df_sub.pivot_table(
            index="userId",
            columns="movieId",
            values="interaction_score",
            fill_value=0,
        )

        self.user_matrix = matrix

        n_components = min(self.n_components, min(matrix.shape) - 1)
        svd    = TruncatedSVD(n_components=n_components, random_state=42)
        latent = svd.fit_transform(matrix)

        self.similarity = cosine_similarity(latent)
        print(f"[collaborative] SVD done — similarity matrix: {self.similarity.shape}")

    def similar_users(self, user_id: int, n: int = 20) -> list[int]:
        if self.user_matrix is None or user_id not in self.user_matrix.index:
            return []
        idx = self.user_matrix.index.get_loc(user_id)
        sims = self.similarity[idx]
        similar_ids = (
            pd.Series(sims, index=self.user_matrix.index)
            .drop(user_id, errors="ignore")
            .sort_values(ascending=False)
            .head(n)
            .index
            .tolist()
        )
        return similar_ids
