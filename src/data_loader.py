import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess the ratings CSV.

    Expects columns: userId, movieId, title, rating, timestamp,
    plus one-hot genre columns.
    """
    df = pd.read_csv(path)

    # Drop unnamed index column if present
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Parse timestamp → datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    return df
