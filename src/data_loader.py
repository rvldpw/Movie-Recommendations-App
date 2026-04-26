import os
import re
import requests
import pandas as pd

FILE_ID    = "1h10NTfIaxGbCbOmLvYRv07mjasXfyJ8a"
LOCAL_FILE = "/tmp/big.parquet"

MIN_BYTES  = 100 * 1024 * 1024   # 100 MB — rejects HTML error pages


def _is_valid_parquet(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < MIN_BYTES:
        return False
    with open(path, "rb") as f:
        magic = f.read(4)
    return magic == b"PAR1"


def _stream_to_disk(response: requests.Response, dest: str) -> int:
    """Write a streaming response to disk, return bytes written."""
    chunk_size    = 8 * 1024 * 1024
    bytes_written = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)
                print(f"[data_loader]   {bytes_written / 1024**2:.0f} MB …", flush=True)
    return bytes_written


def _download_from_drive(file_id: str, dest: str) -> None:
    if os.path.exists(dest):
        os.remove(dest)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36"
    })

    # ── Attempt 1: drive/v3 API-style direct download ──────────────────────────
    print("[data_loader] Attempt 1 — drive/v3 direct …")
    url1 = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
    r = session.get(url1, stream=True, timeout=60)
    r.raise_for_status()

    # If Google returns HTML instead of binary, extract the confirm form action
    content_type = r.headers.get("Content-Type", "")
    if "text/html" in content_type or "application/json" in content_type:
        html = r.text

        # Try to find the download form action URL
        match = re.search(r'action="([^"]+)"', html)
        if match:
            form_url = match.group(1).replace("&amp;", "&")
            print(f"[data_loader] Found form URL, following …")
            r = session.get(form_url, stream=True, timeout=60)
            r.raise_for_status()
        else:
            # Try to extract uuid-based download link (newer Drive UI)
            uuid_match = re.search(r'"(https://drive\.usercontent\.google\.com/download[^"]+)"', html)
            if uuid_match:
                dl_url = uuid_match.group(1).replace("\\u003d", "=").replace("\\u0026", "&")
                print(f"[data_loader] Found usercontent URL, following …")
                r = session.get(dl_url, stream=True, timeout=60)
                r.raise_for_status()

    n = _stream_to_disk(r, dest)
    print(f"[data_loader] Attempt 1 wrote {n / 1024**2:.1f} MB")

    if _is_valid_parquet(dest):
        return

    # ── Attempt 2: drive.usercontent.google.com (newer endpoint) ──────────────
    if os.path.exists(dest):
        os.remove(dest)
    print("[data_loader] Attempt 2 — usercontent endpoint …")
    url2 = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    r = session.get(url2, stream=True, timeout=60)
    r.raise_for_status()
    n = _stream_to_disk(r, dest)
    print(f"[data_loader] Attempt 2 wrote {n / 1024**2:.1f} MB")

    if _is_valid_parquet(dest):
        return

    # ── Both failed ────────────────────────────────────────────────────────────
    size_kb = os.path.getsize(dest) / 1024 if os.path.exists(dest) else 0
    if os.path.exists(dest):
        os.remove(dest)
    raise RuntimeError(
        f"[data_loader] Both download attempts failed (got {size_kb:.0f} KB, expected >100 MB).\n"
        "This usually means Google Drive quota is exceeded for this file.\n"
        "Options:\n"
        "  1. Wait 24 hours for quota reset.\n"
        "  2. Copy the file to a new Drive file (new FILE_ID resets quota).\n"
        "  3. Use a different hosting service (Hugging Face Datasets is free & better for large files).\n"
        f"  Current FILE_ID: {file_id}"
    )


def load_data() -> pd.DataFrame:
    if not _is_valid_parquet(LOCAL_FILE):
        _download_from_drive(FILE_ID, LOCAL_FILE)

    df = pd.read_parquet(LOCAL_FILE)

    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    return df
