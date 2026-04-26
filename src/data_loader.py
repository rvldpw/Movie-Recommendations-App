import os
import requests
import pandas as pd

FILE_ID = "1V7ZJu-kkslnaJ1KVI8yQ7qgMi0zsIQed"   # ← your new file
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


def _download_from_drive(file_id: str, dest: str) -> None:
    """
    Stream a large Google Drive file to disk using requests only.
    Handles the virus-scan confirmation page automatically.
    No gdown dependency — works on all Python versions.
    """
    if os.path.exists(dest):
        os.remove(dest)

    session = requests.Session()

    # Step 1 — initial request (may get a confirm page for large files)
    url = "https://drive.google.com/uc"
    params = {"id": file_id, "export": "download"}
    print(f"[data_loader] Connecting to Google Drive …")
    response = session.get(url, params=params, stream=True, timeout=30)
    response.raise_for_status()

    # Step 2 — check if Google returned a confirmation page
    # (happens for files > ~40 MB)
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    # Also check the response URL for a confirm param (newer Drive behaviour)
    if confirm_token is None and "confirm=" in response.url:
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(response.url).query)
        confirm_token = qs.get("confirm", [None])[0]

    if confirm_token:
        print("[data_loader] Got confirmation token — re-requesting …")
        params["confirm"] = confirm_token
        response = session.get(url, params=params, stream=True, timeout=30)
        response.raise_for_status()

    # Step 3 — stream to disk in 8 MB chunks
    print(f"[data_loader] Downloading → {dest}")
    chunk_size = 8 * 1024 * 1024  # 8 MB
    bytes_written = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
                mb = bytes_written / (1024 * 1024)
                print(f"[data_loader]   {mb:.0f} MB downloaded …", flush=True)

    print(f"[data_loader] Download complete — {bytes_written / (1024**2):.1f} MB")

    if not _is_valid_parquet(dest):
        size_kb = os.path.getsize(dest) / 1024 if os.path.exists(dest) else 0
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(
            f"\n\n[data_loader] Downloaded file is not a valid Parquet (got {size_kb:.0f} KB).\n"
            "Possible causes:\n"
            "  1. Google Drive sharing is NOT set to 'Anyone with the link — Viewer'.\n"
            "  2. Daily download quota exceeded — try again in 24 hours.\n"
            f"  3. FILE_ID is wrong: {file_id}\n"
        )


def load_data() -> pd.DataFrame:
    """
    Download the parquet dataset from Google Drive on first run,
    cache it locally, then load and preprocess.
    """
    os.makedirs("data", exist_ok=True)

    if not _is_valid_parquet(LOCAL_FILE):
        _download_from_drive(FILE_ID, LOCAL_FILE)

    df = pd.read_parquet(LOCAL_FILE)

    # Drop accidental unnamed columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Unix timestamp → datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    return df
    return df
