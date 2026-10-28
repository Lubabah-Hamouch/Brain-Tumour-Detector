"""Typed training configurations, loaded from JSON files in configs/.

Batch settings are split into the per-step batch that must fit in GPU memory
and the number of gradient-accumulation steps, so the effective batch size
can be kept constant when memory is tight.

Memory measured on the RTX 4060 Laptop GPU (8 GB, ~6.9 GiB free), see
src.models.profile_models:
    UNet3D, 4 x 128^3, mixed precision: bs=1 2.36 GiB, bs=2 4.62 GiB
    EfficientNetV2-S, 3 x 384^2, mixed precision: bs=16 3.57 GiB
"""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class TrainConfig:
    batch_size: int = 1
    grad_accum_steps: int = 1
    mixed_precision: bool = True
    seed: int = 42

    @property
    def effective_batch_size(self):
        return self.batch_size * self.grad_accum_steps

    def __post_init__(self):
        if self.batch_size < 1 or self.grad_accum_steps < 1:
            raise ValueError("batch_size and grad_accum_steps must be >= 1")

    @classmethod
    def from_json(cls, path, **overrides):
        values = {**json.loads(Path(path).read_text()), **overrides}
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
        for key in ("widths",):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self):
        return {**asdict(self), "effective_batch_size": self.effective_batch_size}


@dataclass
class SegmentationTrainConfig(TrainConfig):
    # Batch size 1 with 2 accumulation steps (effective 2): about half the
    # memory of a true batch of 2, leaving headroom for the loss and optimizer.
    # Instance norm makes the per-step batch size irrelevant to normalisation.
    batch_size: int = 1
    grad_accum_steps: int = 2
    patch_size: int = 128
    tumor_ratio: float = 0.75
    widths: tuple = (32, 64, 128, 256)
    bottleneck: int = 320
    deep_supervision: int = 2


@dataclass
class ClassificationTrainConfig(TrainConfig):
    batch_size: int = 16
    grad_accum_steps: int = 1
    image_size: int = 384
    pretrained: bool = True
    freeze_backbone: bool = False
