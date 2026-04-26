import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeEngine:

    def __init__(self):
        self.user_matrix=None
        self.similarity=None


    def fit(self,df):

        matrix=(
            df.pivot_table(
                index='userId',
                columns='movieId',
                values='interaction_score',
                fill_value=0
            )
        )

        self.user_matrix=matrix

        svd=TruncatedSVD(
            n_components=50,
            random_state=42
        )

        latent=svd.fit_transform(matrix)

        self.similarity=cosine_similarity(latent)


    def similar_users(
        self,
        user_id,
        n=20
    ):

        if user_id not in self.user_matrix.index:
            return []

        idx=(
            self.user_matrix
            .index
            .get_loc(user_id)
        )

        sims=self.similarity[idx]

        similar_ids=(
            pd.Series(
                sims,
                index=self.user_matrix.index
            )
            .drop(user_id)
            .sort_values(
                ascending=False
            )
            .head(n)
            .index
            .tolist()
        )

        return similar_ids
