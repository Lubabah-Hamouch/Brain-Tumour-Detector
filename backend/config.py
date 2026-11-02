"""API settings, read from environment variables (all optional).

    BT_DEVICE                 auto (default: CUDA if available, else CPU) | cuda | cpu
    BT_CLASSIFIER_CHECKPOINT  default models/classification_crop_resdegrade_long/best.pt
    BT_SEGMENTATION_DIR       default models/segmentation (expects fold<k>/best.pt)
    BT_SEGMENTATION_FOLDS     default 1,2,3,4,5
    BT_ET_MIN_VOXELS          ET post-processing threshold; default: the value tuned on
                              out-of-fold validation data (reports/results/segmentation_test.json)
    BT_CORS_ORIGINS           comma-separated; default common local dev origins
    ALLOWED_ORIGIN            comma-separated origins allowed in addition to BT_CORS_ORIGINS, e.g. the
                              deployed web client (https://your-app.vercel.app); default none
    BT_MAX_IMAGE_MB           classification upload limit, default 20
    BT_MAX_VOLUME_MB          per-modality NIfTI upload limit, default 200
    BT_RESULT_CACHE_SIZE      recent results kept for download (segmentation masks, classification
                              reports, finished jobs), default 16
    BT_SAMPLE_RESULTS         file for the pre-computed sample results (backend/services/samples.py),
                              default models/sample_results.npz; "off" disables pre-computing samples
    BT_SAMPLE_RESULTS_REQUIRED  1: refuse to start unless BT_SAMPLE_RESULTS holds results for the served
                              models (the Hugging Face Space, which cannot recompute them); default 0
    BT_SAMPLE_STUDY_FILES     0: do not serve the sample study's NIfTI files (GET /api/sample-study and
                              /api/sample-study/{name} do not exist); default 1. The Space sets 0: the
                              BraTS terms do not allow redistributing the raw volumes.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from src.config import PROJECT_ROOT

DEFAULT_ET_MIN_VOXELS = 50
DEV_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173")


def _path(name, default):
    value = os.environ.get(name)
    path = Path(value) if value else PROJECT_ROOT / default
    return path if path.is_absolute() else PROJECT_ROOT / path


def tuned_et_threshold():
    """ET threshold chosen on out-of-fold validation predictions during evaluation."""
    path = PROJECT_ROOT / "reports" / "results" / "segmentation_test.json"
    try:
        return int(json.loads(path.read_text())["et_threshold_voxels"])
    except (OSError, KeyError, ValueError):
        return DEFAULT_ET_MIN_VOXELS


def _origins(value):
    # Browsers send the origin without a trailing slash, so "https://x.vercel.app/" would never match.
    return tuple(o.strip().rstrip("/") for o in value.split(",") if o.strip())


def cors_origins(env):
    """BT_CORS_ORIGINS (default: local dev origins) plus ALLOWED_ORIGIN, the deployed web client."""
    base = _origins(env["BT_CORS_ORIGINS"]) if "BT_CORS_ORIGINS" in env else DEV_ORIGINS
    return tuple(dict.fromkeys(base + _origins(env.get("ALLOWED_ORIGIN", ""))))


@dataclass
class Settings:
    device: str = "auto"
    classifier_checkpoint: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "classification_crop_resdegrade_long" / "best.pt")
    segmentation_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "segmentation")
    segmentation_folds: tuple = (1, 2, 3, 4, 5)
    et_min_voxels: int = DEFAULT_ET_MIN_VOXELS
    cors_origins: tuple = DEV_ORIGINS
    max_image_mb: float = 20
    max_volume_mb: float = 200
    result_cache_size: int = 16
    sample_results: Path | None = field(default_factory=lambda: PROJECT_ROOT / "models" / "sample_results.npz")
    sample_results_required: bool = False
    sample_study_files: bool = True

    @classmethod
    def from_env(cls):
        env = os.environ
        return cls(
            device=env.get("BT_DEVICE", "auto").lower(),
            classifier_checkpoint=_path("BT_CLASSIFIER_CHECKPOINT", "models/classification_crop_resdegrade_long/best.pt"),
            segmentation_dir=_path("BT_SEGMENTATION_DIR", "models/segmentation"),
            segmentation_folds=tuple(int(k) for k in env.get("BT_SEGMENTATION_FOLDS", "1,2,3,4,5").split(",")),
            et_min_voxels=int(env["BT_ET_MIN_VOXELS"]) if "BT_ET_MIN_VOXELS" in env else tuned_et_threshold(),
            cors_origins=cors_origins(env),
            max_image_mb=float(env.get("BT_MAX_IMAGE_MB", 20)),
            max_volume_mb=float(env.get("BT_MAX_VOLUME_MB", 200)),
            result_cache_size=int(env.get("BT_RESULT_CACHE_SIZE", 16)),
            sample_results=None if env.get("BT_SAMPLE_RESULTS", "").lower() == "off"
            else _path("BT_SAMPLE_RESULTS", "models/sample_results.npz"),
            sample_results_required=env.get("BT_SAMPLE_RESULTS_REQUIRED", "0") == "1",
            sample_study_files=env.get("BT_SAMPLE_STUDY_FILES", "1") != "0",
        )
