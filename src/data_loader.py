import os
import pandas as pd

# Force HF cache to a writable temp directory before importing huggingface_hub
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_home"
os.makedirs("/tmp/hf_home", exist_ok=True)

from huggingface_hub import hf_hub_download

HF_REPO_ID  = "rvlpw/movie-recommendations"
HF_FILENAME = "big.parquet"
LOCAL_FILE  = "/tmp/big.parquet"


def load_data() -> pd.DataFrame:
    """
    Download parquet from Hugging Face Datasets on first run,
    cache in /tmp, then load and preprocess.
    """
    if not os.path.exists(LOCAL_FILE):
        print(f"[data_loader] Downloading {HF_FILENAME} from Hugging Face …")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            local_dir="/tmp",
            local_dir_use_symlinks=False,   # write the actual file, not a symlink
        )
        print("[data_loader] Download complete.")

    df = pd.read_parquet(LOCAL_FILE)

    # Drop accidental unnamed columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Unix timestamp → datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    return df
