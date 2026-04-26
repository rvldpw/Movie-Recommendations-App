import os
import requests
import pandas as pd

# Point this to your small.parquet after uploading to HF
HF_PARQUET_URL = "https://huggingface.co/datasets/rvlpw/movie-recommendations"
LOCAL_FILE     = "/tmp/small.parquet"
MIN_BYTES      = 1 * 1024 * 1024   # 1 MB minimum (small file now)


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
    return urls[0]


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        os.remove(dest)
    print("[data_loader] Downloading …")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        bytes_written = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bytes_written += len(chunk)
                    print(f"[data_loader]   {bytes_written / 1024**2:.0f} MB …", flush=True)
    print(f"[data_loader] Done — {bytes_written / 1024**2:.1f} MB written.")


def load_data() -> pd.DataFrame:
    if not _is_valid_parquet(LOCAL_FILE):
        shard_url = _get_parquet_shard_url()
        _download(shard_url, LOCAL_FILE)

    print("[data_loader] Reading parquet …")
    df = pd.read_parquet(LOCAL_FILE)

    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    print(f"[data_loader] Ready — {len(df):,} rows, {df['userId'].nunique():,} users.")
    return df
