import os
import requests
import pandas as pd

FILE_ID    = "1h10NTfIaxGbCbOmLvYRv07mjasXfyJ8a"
LOCAL_FILE = "/tmp/big.parquet"   # /tmp is always writable on Streamlit Cloud

MIN_BYTES  = 100 * 1024 * 1024   # 100 MB — rejects HTML error pages (~10 KB)


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
    Works on all Python versions (no gdown dependency).
    """
    if os.path.exists(dest):
        os.remove(dest)

    session  = requests.Session()
    url      = "https://drive.google.com/uc"
    params   = {"id": file_id, "export": "download"}

    print("[data_loader] Connecting to Google Drive …")
    response = session.get(url, params=params, stream=True, timeout=60)
    response.raise_for_status()

    # Google shows a virus-scan warning page for large files.
    # Detect the confirmation token from cookies or redirect URL.
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    if confirm_token is None and "confirm=" in response.url:
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(response.url).query)
        confirm_token = qs.get("confirm", [None])[0]

    if confirm_token:
        print("[data_loader] Confirmation token found — re-requesting …")
        params["confirm"] = confirm_token
        response = session.get(url, params=params, stream=True, timeout=60)
        response.raise_for_status()

    # Stream to disk in 8 MB chunks so 3 GB never loads into RAM at once
    print(f"[data_loader] Streaming to {dest} …")
    chunk_size    = 8 * 1024 * 1024
    bytes_written = 0

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
                print(f"[data_loader]   {bytes_written / 1024**2:.0f} MB …", flush=True)

    print(f"[data_loader] Done — {bytes_written / 1024**2:.1f} MB written.")

    if not _is_valid_parquet(dest):
        size_kb = os.path.getsize(dest) / 1024 if os.path.exists(dest) else 0
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(
            f"[data_loader] File is not valid Parquet (got {size_kb:.0f} KB).\n"
            "Check:\n"
            "  1. Google Drive sharing = 'Anyone with the link — Viewer'\n"
            "  2. Daily quota not exceeded (wait 24 h if so)\n"
            f"  3. FILE_ID correct: {file_id}"
        )


def load_data() -> pd.DataFrame:
    """
    Download the parquet dataset from Google Drive on first run,
    cache it in /tmp, then load and preprocess.
    """
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
