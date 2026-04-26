import numpy as np
import pandas as pd


GENRE_COLS = [
''Action','Adventure','Animation','Children','Comedy',
'Crime','Documentary','Drama','Fantasy','Film-Noir',
'Horror','IMAX','Musical','Mystery','Romance',
'Sci-Fi','Thriller','War','Western'
]


def add_recency_weight(df, decay=.002):

    latest_date = df['datetime'].max()

    df['recency_days'] = (
        latest_date - df['datetime']
    ).dt.days

    df['recency_weight'] = np.exp(
        -decay * df['recency_days']
    )

    df['interaction_score'] = (
        df['rating'] * df['recency_weight']
    )

    return df


def build_movie_feature_matrix(df):
    movie_features = (
        df[['movieId','title'] + GENRE_COLS]
        .drop_duplicates('movieId')
        .set_index('movieId')
    )

    return movie_features


def build_user_profile(
    user_df,
    genre_cols=GENRE_COLS
):

    weighted = (
        user_df[genre_cols]
        .multiply(
            user_df['interaction_score'],
            axis=0
        )
    )

    profile = weighted.sum()

    if profile.sum() > 0:
        profile = profile / profile.sum()

    return profile
