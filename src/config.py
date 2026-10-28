"""Project configuration loaded from the project-local .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing environment variable {name}; see .env.example")
    return value


# Datasets live under data/ in the repo but are gitignored.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

BRATS_DIR = RAW_DIR / "brats2021"
FIGSHARE_DIR = RAW_DIR / "figshare_classification"
