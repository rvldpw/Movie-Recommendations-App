import os
import gdown
import pandas as pd

FILE_ID    = "1V7ZJu-kkslnaJ1KVI8yQ7qgMi0zsIQed"   # ← your new file
LOCAL_FILE = "data/big.parquet"

# Reject anything smaller than 100 MB (HTML error pages are ~10 KB)
MIN_BYTES = 100 * 1024 * 1024


def _is_valid_parquet(path: str) -> bool:
    """True only if the file exists, is large enough, and starts with PAR1 magic bytes."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < MIN_BYTES:
        return False
    with open(path, "rb") as f:
        magic = f.read(4)
    return magic == b"PAR1"


def _download(file_id: str, dest: str) -> None:
    """Download from Google Drive with two attempts and file validation."""

    # Wipe any corrupt/partial file first
    if os.path.exists(dest):
        os.remove(dest)

    # Attempt 1 — fuzzy mode handles most large files
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[data_loader] Downloading dataset → {dest}")
    gdown.download(url, dest, quiet=False, fuzzy=True)

    if _is_valid_parquet(dest):
        return

    # Attempt 2 — force-confirm to bypass Google's virus-scan warning page
    if os.path.exists(dest):
        os.remove(dest)
    fallback_url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
    print("[data_loader] Retrying with confirm=t …")
    gdown.download(fallback_url, dest, quiet=False, fuzzy=True)

    if _is_valid_parquet(dest):
        return

    # Both failed — clean up and raise a readable error
    if os.path.exists(dest):
        os.remove(dest)
    raise RuntimeError(
        "\n\n[data_loader] Download failed — file is not a valid Parquet.\n"
        "Check the following:\n"
        "  1. Google Drive sharing is set to 'Anyone with the link can VIEW'.\n"
        "  2. The daily download quota has not been exceeded (try again in 24 h).\n"
        f"  3. FILE_ID is correct: {file_id}\n"
    )


def load_data() -> pd.DataFrame:
    """
    Download the parquet dataset from Google Drive on first run,
    cache it locally, then load and preprocess.
    """
    os.makedirs("data", exist_ok=True)

    if not _is_valid_parquet(LOCAL_FILE):
        _download(FILE_ID, LOCAL_FILE)

    df = pd.read_parquet(LOCAL_FILE)

    # Drop accidental unnamed columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Unix timestamp → datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    return df
