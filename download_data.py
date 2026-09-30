"""Download the Biohub Cell Tracking competition data using a Kaggle API token.

Supports two credential sources (in priority order):
  1. Environment variables KAGGLE_USERNAME / KAGGLE_KEY
  2. A kaggle.json file placed in ./data/  (or ~/.kaggle/kaggle.json)

Usage:
    python download_data.py
"""

import os
import shutil
import subprocess
import sys

COMPETITION = "biohub-cell-tracking-during-development"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
KAGGLE_JSON_CANDIDATES = [
    os.path.join(DATA_DIR, "kaggle.json"),
    os.path.expanduser("~/.kaggle/kaggle.json"),
]


def setup_credentials() -> None:
    """Make sure kaggle credentials are visible to the kaggle python client."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("[ok] Using KAGGLE_USERNAME / KAGGLE_KEY environment variables.")
        return

    for cand in KAGGLE_JSON_CANDIDATES:
        if os.path.isfile(cand):
            os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
            dest = os.path.expanduser("~/.kaggle/kaggle.json")
            if os.path.abspath(cand) != os.path.abspath(dest):
                shutil.copy(cand, dest)
            os.chmod(dest, 0o600)  # kaggle requires strict permissions
            print(f"[ok] Found kaggle.json at {cand} -> copied to ~/.kaggle/kaggle.json")
            return

    sys.exit(
        "[error] No Kaggle credentials found.\n"
        "  Option A: put kaggle.json in the data/ folder\n"
        "  Option B: export KAGGLE_USERNAME and KAGGLE_KEY\n"
        "  (create a token at https://www.kaggle.com/settings -> API)"
    )


def download() -> None:
    target = os.path.join(DATA_DIR, f"{COMPETITION}.zip")
    if os.path.isdir(os.path.join(DATA_DIR, "test")) and os.path.isdir(os.path.join(DATA_DIR, "train")):
        print("[ok] Data already present in data/, skipping download.")
        return

    print(f"[..] Downloading competition data for '{COMPETITION}' ...")
    subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "download",
         "-c", COMPETITION, "-p", DATA_DIR],
        check=True,
    )

    print("[..] Extracting archive ...")
    import zipfile
    with zipfile.ZipFile(target, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(target)
    print(f"[done] Data ready in {DATA_DIR}/")


if __name__ == "__main__":
    setup_credentials()
    download()
