"""API settings, read from environment variables (all optional).

    BT_DEVICE                 auto (default: CUDA if available, else CPU) | cuda | cpu
    BT_CLASSIFIER_CHECKPOINT  default models/classification_crop_resdegrade_long/best.pt
    BT_SEGMENTATION_DIR       default models/segmentation (expects fold<k>/best.pt)
    BT_SEGMENTATION_FOLDS     default 1,2,3,4,5
    BT_ET_MIN_VOXELS          ET post-processing threshold; default: the value tuned on
                              out-of-fold validation data (reports/results/segmentation_test.json)
    BT_CORS_ORIGINS           comma-separated; default common local dev origins
    BT_MAX_IMAGE_MB           classification upload limit, default 20
    BT_MAX_VOLUME_MB          per-modality NIfTI upload limit, default 200
    BT_RESULT_CACHE_SIZE      segmentation masks kept for download, default 16
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

    @classmethod
    def from_env(cls):
        env = os.environ
        return cls(
            device=env.get("BT_DEVICE", "auto").lower(),
            classifier_checkpoint=_path("BT_CLASSIFIER_CHECKPOINT", "models/classification_crop_resdegrade_long/best.pt"),
            segmentation_dir=_path("BT_SEGMENTATION_DIR", "models/segmentation"),
            segmentation_folds=tuple(int(k) for k in env.get("BT_SEGMENTATION_FOLDS", "1,2,3,4,5").split(",")),
            et_min_voxels=int(env["BT_ET_MIN_VOXELS"]) if "BT_ET_MIN_VOXELS" in env else tuned_et_threshold(),
            cors_origins=tuple(o.strip() for o in env["BT_CORS_ORIGINS"].split(",")) if "BT_CORS_ORIGINS" in env else DEV_ORIGINS,
            max_image_mb=float(env.get("BT_MAX_IMAGE_MB", 20)),
            max_volume_mb=float(env.get("BT_MAX_VOLUME_MB", 200)),
            result_cache_size=int(env.get("BT_RESULT_CACHE_SIZE", 16)),
        )
