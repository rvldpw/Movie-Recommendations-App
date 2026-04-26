import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from src.feature_engineering import (
    add_recency_weight,
    build_movie_feature_matrix,
    build_user_profile,
    GENRE_COLS
)

from src.collaborative import CollaborativeEngine

class RecommenderSystem:

    def __init__(self,df):
        self.df=df.copy()
        self.movies=None
        self.cf=None


    def fit(self):

        self.df=add_recency_weight(
            self.df
        )

        self.movies=(
            build_movie_feature_matrix(
                self.df
            )
        )

        self.cf=CollaborativeEngine()
        self.cf.fit(self.df)

def get_recent_activity(
                self.df.userId==user_id
            ]
            .sort_values(
                'interaction_score',
                ascending=False
            )
            .drop_duplicates(
                'movieId'
            )
            .head(top_n)
        )

        return user[
            ['title','rating','datetime']
        ]


    def content_scores(
        self,
        user_id
    ):

        user=(
            self.df[
                self.df.userId==user_id
            ]
        )

        profile=(
            build_user_profile(user)
        )

        movie_vectors=(
            self.movies[GENRE_COLS]
        )

        scores=cosine_similarity(
            [profile.values],
            movie_vectors.values
        )[0]

        score_df=pd.DataFrame({
            'movieId':movie_vectors.index,
            'content_score':scores
        })

        return score_df


    def collaborative_scores(
        self,
        user_id
    ):

        neighbors=(
            self.cf.similar_users(
                user_id
            )
        )

        if len(neighbors)==0:
            return pd.DataFrame(
                columns=[
                    'movieId',
                    'cf_score'
                ]
            )

        cf=(
            self.df[
                self.df.userId.isin(
                    neighbors
                )
            ]
            .groupby('movieId')
            ['rating']
            .mean()
            .reset_index()
        )

        cf.columns=[
            'movieId',
            'cf_score'
        ]

        return cf


    def recommend(
        self,
        user_id,
        top_n=10
    ):

        watched=set(
            ]['movieId']
        )

        content=(
            self.content_scores(
                user_id
            )
        )

        collab=(
            self.collaborative_scores(
                user_id
            )
        )

        rank=(
            content.merge(
                collab,
                on='movieId',
                how='left'
            )
            .fillna(0)
        )

        popularity=(
            self.df.groupby(
                'movieId'
            )['rating']
            .mean()
            .reset_index()
        )

        popularity.columns=[
            'movieId',
            'popularity'
        ]

        rank=rank.merge(
            popularity,
            on='movieId'
        )

        rank['score']=(
            .4*rank['cf_score']+
            .35*rank['content_score']+
            .25*rank['popularity']
        )

        rank=(
            rank[
                ~rank.movieId.isin(
                    watched
                )
            ]
            .merge(
                self.movies[
                    ['title']
                ],
                left_on='movieId',
                right_index=True
            )
            .sort_values(
                'score',
                ascending=False
            )
            .head(top_n)
        )

        return rank[
            ['title','score']
        ]


    def get_user_profile(
        self,
        user_id
    ):

        user=(
            self.df[
                self.df.userId==user_id
            ]
        )

        p=(
            build_user_profile(
                user
            )
            .sort_values(
                ascending=False
            )
            .head(8)
        )

        return list(
            zip(
                p.index,
                p.values
            )
        )
