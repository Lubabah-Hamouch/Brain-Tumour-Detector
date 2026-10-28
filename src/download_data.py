"""Download raw datasets from Kaggle into data/raw/.

Usage:
    python -m src.download_data classification   # 4-class, primary
    python -m src.download_data brats
    python -m src.download_data figshare3        # original 3-class release
    python -m src.download_data all
"""
import argparse
import tarfile
import zipfile
from pathlib import Path

from src.config import BRATS_DIR, CLASSIFICATION_DIR, FIGSHARE_3CLASS_DIR, require_env

BRATS_DATASET = "dschettler8845/brats-2021-task1"
BRATS_ARCHIVE = "BraTS2021_Training_Data.tar"
CLASSIFICATION_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
FIGSHARE_3CLASS_DATASET = "ashkhagan/figshare-brain-tumor-dataset"


def kaggle_api():
    # The kaggle package reads credentials at import time, so the
    # environment must already be loaded (src.config does that).
    require_env("KAGGLE_USERNAME")
    require_env("KAGGLE_KEY")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def unzip_all(directory):
    """Extract and remove any .zip archives in directory (Kaggle wraps some files)."""
    for archive in Path(directory).glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(directory)
        archive.unlink()


def download_brats(api):
    BRATS_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = BRATS_DIR / BRATS_ARCHIVE
    if not tar_path.exists():
        print(f"Downloading {BRATS_DATASET}/{BRATS_ARCHIVE} -> {BRATS_DIR}")
        api.dataset_download_file(BRATS_DATASET, BRATS_ARCHIVE, path=str(BRATS_DIR), quiet=False)
        unzip_all(BRATS_DIR)

    print(f"Extracting {tar_path}")
    with tarfile.open(tar_path) as tf:
        tf.extractall(BRATS_DIR, filter="data")
    # The tar is ~13 GB and fully re-downloadable; drop it once extracted.
    tar_path.unlink()


def download_classification(api):
    CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {CLASSIFICATION_DATASET} -> {CLASSIFICATION_DIR}")
    api.dataset_download_files(CLASSIFICATION_DATASET, path=str(CLASSIFICATION_DIR), unzip=True, quiet=False)


def download_figshare_3class(api):
    FIGSHARE_3CLASS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {FIGSHARE_3CLASS_DATASET} -> {FIGSHARE_3CLASS_DIR}")
    api.dataset_download_files(FIGSHARE_3CLASS_DATASET, path=str(FIGSHARE_3CLASS_DIR), unzip=True, quiet=False)
    # The original Figshare release nests per-chunk zip files inside the archive.
    for nested in FIGSHARE_3CLASS_DIR.rglob("*.zip"):
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(nested.parent)
        nested.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["classification", "brats", "figshare3", "all"])
    args = parser.parse_args()

    api = kaggle_api()
    if args.dataset in ("classification", "all"):
        download_classification(api)
    if args.dataset in ("figshare3", "all"):
        download_figshare_3class(api)
    if args.dataset in ("brats", "all"):
        download_brats(api)


if __name__ == "__main__":
    main()
