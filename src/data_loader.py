import os
import requests
import pandas as pd

HF_URL     = "https://huggingface.co/datasets/rvlpw/movie-recommendations/resolve/main/small.parquet"
LOCAL_FILE = "/tmp/small.parquet"
MIN_BYTES  = 1 * 1024 * 1024   # 1 MB


def _is_valid_parquet(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < MIN_BYTES:
        return False
    with open(path, "rb") as f:
        return f.read(4) == b"PAR1"


def load_data() -> pd.DataFrame:
    if not _is_valid_parquet(LOCAL_FILE):
        print("[data_loader] Downloading small.parquet from Hugging Face …")
        with requests.get(HF_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            bytes_written = 0
            with open(LOCAL_FILE, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)
                        print(f"[data_loader]   {bytes_written / 1024**2:.0f} MB …", flush=True)
        print(f"[data_loader] Done — {bytes_written / 1024**2:.1f} MB written.")

    print("[data_loader] Reading parquet …")
    df = pd.read_parquet(LOCAL_FILE)

    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    print(f"[data_loader] Ready — {len(df):,} rows, {df['userId'].nunique():,} users.")
    return df
