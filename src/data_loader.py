import os
import pandas as pd
from huggingface_hub import hf_hub_download

# ── Change these two values to match your Hugging Face dataset ────────────────
HF_REPO_ID = "rvldpw/movie-recommendations"
HF_FILENAME = "big.parquet"
# ─────────────────────────────────────────────────────────────────────────────

LOCAL_FILE = "/tmp/big.parquet"


def load_data() -> pd.DataFrame:
    """
    Download parquet from Hugging Face Datasets on first run,
    cache in /tmp, then load and preprocess.
    """
    if not os.path.exists(LOCAL_FILE):
        print(f"[data_loader] Downloading {HF_FILENAME} from Hugging Face …")
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            local_dir="/tmp",
        )
        print(f"[data_loader] Downloaded to {path}")

    df = pd.read_parquet(LOCAL_FILE)

    # Drop accidental unnamed columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Unix timestamp → datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    return df
