import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeEngine:
    """SVD-based user–user collaborative filtering engine."""

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.user_matrix: pd.DataFrame | None = None
        self.similarity: "np.ndarray | None" = None  # noqa: F821

    def fit(self, df: pd.DataFrame) -> None:
        """Build the user–item matrix and compute cosine similarities."""
        matrix = df.pivot_table(
            index="userId",
            columns="movieId",
            values="interaction_score",
            fill_value=0,
        )
        self.user_matrix = matrix

        # Cap n_components to matrix rank
        n_components = min(self.n_components, min(matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        latent = svd.fit_transform(matrix)
        self.similarity = cosine_similarity(latent)

    def similar_users(self, user_id: int, n: int = 20) -> list[int]:
        """Return the n most similar user IDs for a given user."""
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
