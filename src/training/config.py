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
    epochs: int = 1
    weight_decay: float = 0.0
    num_workers: int = 2
    # A compute step far slower than profiled usually means the Windows driver
    # has spilled GPU memory into system RAM; training stops instead of crawling.
    # A real spill slows every step, so several consecutive slow steps are
    # required; brief contention from other programs using the GPU produces
    # only isolated slow steps and should not end a multi-hour run.
    max_step_seconds: float = 3.0
    slow_step_patience: int = 5
    wandb_project: str = ""

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
        for key in ("widths", "level_weights", "crop_scale", "crop_ratio"):
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
    # Deep-supervision loss weights, coarsest output first (normalised in the loss).
    level_weights: tuple = (0.25, 0.5, 1.0)
    # AdamW at 3e-4 is a standard, stable rate for 3D U-Nets trained from
    # scratch; weight decay 3e-5 follows nnU-Net. Clipping at 12 (also nnU-Net)
    # guards against rare gradient spikes from Dice on near-empty regions.
    lr: float = 3e-4
    weight_decay: float = 3e-5
    grad_clip: float = 12.0
    # Linear warmup then cosine decay to 1% of lr, stepped per optimizer step.
    warmup_epochs: int = 2
    min_lr_ratio: float = 0.01
    # One epoch = samples_per_case random patches from every training case.
    epochs: int = 40
    samples_per_case: int = 1
    flip: bool = True
    num_workers: int = 4
    # Cross-validation: train on every fold except `fold` (1-based), validate on it.
    folds_manifest: str = "data/manifests/brats_5fold_splits.json"
    fold: int = 1
    # Whole-volume sliding-window validation (128^3 windows, Gaussian blending).
    val_every: int = 4
    sw_overlap: float = 0.5
    sw_batch_size: int = 2
    # Fixed validation cases drawn as slice overlays at every validation.
    vis_cases: int = 4
    checkpoint_dir: str = "models/segmentation"
    wandb_project: str = "brain-tumor-segmentation"


@dataclass
class ClassificationTrainConfig(TrainConfig):
    batch_size: int = 16
    grad_accum_steps: int = 1
    image_size: int = 384
    pretrained: bool = True
    freeze_backbone: bool = False
    # Discriminative learning rates: the pretrained backbone already holds useful
    # features and needs only small adjustments (a large rate would overwrite
    # them), while the randomly initialised head must learn from scratch, so it
    # gets a 10x higher rate. 1e-4 is a common AdamW fine-tuning rate for
    # ImageNet-pretrained CNNs; weight decay 1e-2 is the AdamW default.
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-2
    # Linear warmup (protects the pretrained backbone from the untrained head's
    # large early gradients) then cosine decay over `epochs`.
    warmup_epochs: int = 1
    min_lr_ratio: float = 0.01
    epochs: int = 30
    # Stop when validation loss has not improved for `patience` epochs. The
    # checkpoint kept is the epoch with the best validation macro F1.
    patience: int = 5
    augment: bool = True
    # Random crop of the original image (fraction of its area, width/height
    # ratio range) before letterboxing. Off by default; enabled in
    # train_classification_crop.json to break the link between source image
    # format (size, aspect ratio) and class found in the error analysis.
    random_crop: bool = False
    crop_scale: tuple = (0.7, 1.0)
    crop_ratio: tuple = (0.75, 1.3333)
    num_workers: int = 6
    misclassified_every: int = 2
    misclassified_max: int = 24
    checkpoint_dir: str = "models/classification"
    wandb_project: str = "brain-tumor-classification"
