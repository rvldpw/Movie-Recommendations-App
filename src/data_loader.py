import os
import gdown
import pandas as pd


FILE_ID = "1cQUO_1eKa3VxHzeGapHmGEmDwrvu0ZYz"

LOCAL_FILE = "data/big.parquet"


def load_data() -> pd.DataFrame:
    """
    Download parquet dataset from Google Drive
    (first run only), cache locally,
    then load and preprocess.
    """

    os.makedirs(
        "data",
        exist_ok=True
    )

    # Download only if not cached
    if not os.path.exists(
        LOCAL_FILE
    ):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            LOCAL_FILE,
            quiet=False
        )

    # Load parquet
    df = pd.read_parquet(
        LOCAL_FILE
    )

    # Drop accidental unnamed cols
    unnamed_cols = [
        c for c in df.columns
        if c.startswith("Unnamed")
    ]

    if unnamed_cols:
        df = df.drop(
            columns=unnamed_cols
        )

    # Unix timestamp → datetime
    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s"
    )

    return df
