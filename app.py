import os
import gdown
import pandas as pd

FILE_ID   = "1V7ZJu-kkslnaJ1KVI8yQ7qgMi0zsIQed"
LOCAL_FILE = "data/big.parquet"

# Minimum expected file size in bytes (3 GB file — reject anything smaller than 100 MB)
MIN_BYTES = 100 * 1024 * 1024


def _is_valid_parquet(path: str) -> bool:
    """Return True only if the file exists and looks like a real parquet (not an HTML error page)."""
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size < MIN_BYTES:
        return False
    # Parquet files start with the magic bytes PAR1
    with open(path, "rb") as f:
        magic = f.read(4)
    return magic == b"PAR1"


def _download(file_id: str, dest: str) -> None:
    """
    Download from Google Drive using gdown's fuzzy mode.
    Falls back to the direct export URL if the first attempt produces a bad file.
    """
    url = f"https://drive.google.com/uc?id={file_id}"

    # Remove any partial / corrupt file before downloading
    if os.path.exists(dest):
        os.remove(dest)

    print(f"Downloading dataset from Google Drive → {dest}")
    gdown.download(url, dest, quiet=False, fuzzy=True)

    if not _is_valid_parquet(dest):
        # gdown sometimes fails silently on large files; try the direct URL variant
        if os.path.exists(dest):
            os.remove(dest)
        fallback_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
        print("First attempt failed — retrying with confirm=t URL …")
        gdown.download(fallback_url, dest, quiet=False, fuzzy=True)

    if not _is_valid_parquet(dest):
        # Remove the bad file so the next run tries again
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(
            "Downloaded file is not a valid Parquet file. "
            "Possible causes:\n"
            "  1. The Google Drive file is not shared as 'Anyone with the link can view'.\n"
            "  2. Google Drive daily download quota was exceeded — try again in 24 hours.\n"
            "  3. The FILE_ID in data_loader.py is wrong.\n"
            f"  FILE_ID used: {file_id}"
        )


def load_data() -> pd.DataFrame:
    """
    Download parquet dataset from Google Drive (first run only),
    cache locally, then load and preprocess.
    """
    os.makedirs("data", exist_ok=True)

    # Download only if a valid cached file does not already exist
    if not _is_valid_parquet(LOCAL_FILE):
        _download(FILE_ID, LOCAL_FILE)

    df = pd.read_parquet(LOCAL_FILE)

    # Drop accidental unnamed columns
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Unix timestamp → datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    return df
