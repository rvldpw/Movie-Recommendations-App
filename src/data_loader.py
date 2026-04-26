import os
import requests
import pandas as pd

HF_PARQUET_URL = "https://huggingface.co/api/datasets/rvlpw/movie-recommendations/parquet/default/train"
LOCAL_FILE     = "/tmp/big.parquet"
MIN_BYTES      = 50 * 1024 * 1024

# ── How many rows to keep ─────────────────────────────────────────────────────
# 25M rows is too large for Streamlit Cloud free tier (~1 GB RAM).
# 2M rows covers ~99% of users and loads in seconds.
MAX_ROWS = 2_000_000


def _is_valid_parquet(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < MIN_BYTES:
        return False
    with open(path, "rb") as f:
        return f.read(4) == b"PAR1"


def _get_parquet_shard_url() -> str:
    print("[data_loader] Fetching shard URL from HF API …")
    r = requests.get(HF_PARQUET_URL, timeout=30)
    r.raise_for_status()
    urls = r.json()
    if not urls:
        raise RuntimeError("HF API returned no parquet URLs. Is the dataset public?")
    print(f"[data_loader] Found {len(urls)} shard(s).")
    return urls[0]


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        os.remove(dest)
    print("[data_loader] Downloading …")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        bytes_written = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
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
            f"[data_loader] Not valid Parquet (got {size_kb:.0f} KB). "
            "Check dataset is Public on Hugging Face."
        )


def load_data() -> pd.DataFrame:
    if not _is_valid_parquet(LOCAL_FILE):
        shard_url = _get_parquet_shard_url()
        _download(shard_url, LOCAL_FILE)

    print(f"[data_loader] Reading parquet (sampling up to {MAX_ROWS:,} rows) …")
    df = pd.read_parquet(LOCAL_FILE)

    # Sample down to MAX_ROWS to stay within Streamlit Cloud memory limits
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        print(f"[data_loader] Sampled to {MAX_ROWS:,} rows.")

    # Drop accidental unnamed columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Unix timestamp → datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    print(f"[data_loader] Ready — {len(df):,} rows, {df['userId'].nunique():,} users.")
    return df
