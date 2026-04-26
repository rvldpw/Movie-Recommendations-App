import os
import requests
import pandas as pd

# Hugging Face auto-converts uploaded files to parquet shards.
# This URL points to the converted parquet shard directly.
HF_PARQUET_URL = "https://huggingface.co/api/datasets/rvlpw/movie-recommendations/parquet/default/train"
LOCAL_FILE     = "/tmp/big.parquet"
MIN_BYTES      = 50 * 1024 * 1024   # 50 MB minimum


def _is_valid_parquet(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < MIN_BYTES:
        return False
    with open(path, "rb") as f:
        return f.read(4) == b"PAR1"


def _get_parquet_shard_url() -> str:
    """
    Call the HF API to get the actual download URL of the parquet shard(s).
    Returns the first (and usually only) shard URL.
    """
    print("[data_loader] Fetching parquet shard URL from HF API …")
    r = requests.get(HF_PARQUET_URL, timeout=30)
    r.raise_for_status()
    urls = r.json()   # returns a list of shard URLs
    if not urls:
        raise RuntimeError("HF API returned no parquet URLs. Is the dataset public?")
    print(f"[data_loader] Found {len(urls)} shard(s). Using first shard.")
    return urls[0]


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        os.remove(dest)

    print(f"[data_loader] Downloading parquet shard …")
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
            f"[data_loader] Downloaded file is not valid Parquet (got {size_kb:.0f} KB).\n"
            "Make sure the dataset is set to Public on Hugging Face."
        )


def load_data() -> pd.DataFrame:
    if not _is_valid_parquet(LOCAL_FILE):
        shard_url = _get_parquet_shard_url()
        _download(shard_url, LOCAL_FILE)

    df = pd.read_parquet(LOCAL_FILE)

    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    return df
