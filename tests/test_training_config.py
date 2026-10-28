import pytest

from src.config import PROJECT_ROOT
from src.training.config import ClassificationTrainConfig, SegmentationTrainConfig

CONFIGS = PROJECT_ROOT / "configs"


def test_segmentation_defaults_use_accumulation_for_effective_batch_two():
    cfg = SegmentationTrainConfig.from_json(CONFIGS / "train_segmentation.json")
    assert (cfg.batch_size, cfg.grad_accum_steps, cfg.effective_batch_size) == (1, 2, 2)
    assert cfg.mixed_precision
    assert cfg.widths == (32, 64, 128, 256) and cfg.bottleneck == 320
    assert cfg == SegmentationTrainConfig()  # JSON file matches the code defaults


def test_classification_defaults():
    cfg = ClassificationTrainConfig.from_json(CONFIGS / "train_classification.json")
    assert (cfg.batch_size, cfg.effective_batch_size, cfg.mixed_precision) == (16, 16, True)
    assert cfg == ClassificationTrainConfig()


def test_overrides_and_validation():
    cfg = SegmentationTrainConfig.from_json(CONFIGS / "train_segmentation.json", batch_size=2, grad_accum_steps=1)
    assert cfg.effective_batch_size == 2
    with pytest.raises(ValueError):
        SegmentationTrainConfig.from_json(CONFIGS / "train_segmentation.json", batch_sise=2)
    with pytest.raises(ValueError):
        SegmentationTrainConfig(grad_accum_steps=0)
