import numpy as np
import pandas as pd

GENRE_COLS = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "IMAX", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western",
]


def get_available_genre_cols(df: pd.DataFrame) -> list[str]:
    """Return only the genre columns that actually exist in the dataframe."""
    return [c for c in GENRE_COLS if c in df.columns]


def add_recency_weight(df: pd.DataFrame, decay: float = 0.002) -> pd.DataFrame:
    """Add exponential recency weight and a weighted interaction score."""
    latest_date = df["datetime"].max()
    df = df.copy()
    df["recency_days"] = (latest_date - df["datetime"]).dt.days
    df["recency_weight"] = np.exp(-decay * df["recency_days"])
    df["interaction_score"] = df["rating"] * df["recency_weight"]
    return df


def build_movie_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by movieId with title + genre columns."""
    genre_cols = get_available_genre_cols(df)
    keep = ["movieId", "title"] + genre_cols
    movie_features = (
        df[keep]
        .drop_duplicates("movieId")
        .set_index("movieId")
    )
    return movie_features


def build_user_profile(
    user_df: pd.DataFrame,
    genre_cols: list[str] | None = None,
) -> pd.Series:
    """Build a normalised genre-preference vector for one user."""
    if genre_cols is None:
        genre_cols = get_available_genre_cols(user_df)

    # Keep only genre cols that are present in user_df
    genre_cols = [c for c in genre_cols if c in user_df.columns]

    if not genre_cols or user_df.empty:
        return pd.Series(dtype=float)

    weighted = user_df[genre_cols].multiply(user_df["interaction_score"], axis=0)
    profile = weighted.sum()

    if profile.sum() > 0:
        profile = profile / profile.sum()

    return profile
