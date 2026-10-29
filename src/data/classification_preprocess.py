"""On-the-fly preprocessing for the cleaned 4-class MRI classification dataset.

Images are standardised at load time (nothing is written to disk):

1. Decode to single-channel grayscale. MRI is intrinsically single-channel; the
   source mixes L, RGB, RGBA and palette files, and the only images with real
   colour are 34 notumor images whose colour comes from website watermarks.
   Keeping colour would hand the model a label shortcut.
2. Letterbox to a square (black padding, aspect ratio preserved) and resize to
   384x384, the evaluation resolution of torchvision's EfficientNetV2-S weights.
   Black padding matches the MRI background; stretching would distort anatomy.
3. Replicate to 3 channels and normalise with ImageNet statistics, matching
   the input the pretrained backbone was trained on.

Usage (writes a preview grid to reports/):
    python -m src.data.classification_preprocess --preview
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import PROCESSED_DIR, PROJECT_ROOT

CLEAN_DIR = PROCESSED_DIR / "figshare_classification_4class_clean"
CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
IMAGE_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def to_grayscale(img):
    """Convert any PIL mode to 8-bit grayscale; transparency is composited onto black."""
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(background, img.convert("RGBA"))
    return img.convert("L")


def letterbox(img, size):
    """Resize the longer side to `size` and pad the shorter side with black, centred."""
    w, h = img.size
    scale = size / max(w, h)
    resized = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def to_tensor(img):
    """Grayscale PIL image -> normalised float tensor of shape (3, H, W)."""
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    x = x.unsqueeze(0).expand(3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


def denormalize(x):
    """Inverse of the ImageNet normalisation, for visualisation."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


class BrainMRIClassificationDataset(Dataset):
    """Images from <root>/<split>/<class>/, returned as (tensor (3, S, S), class index).

    `augment` is an optional callable applied to the letterboxed grayscale PIL
    image before tensor conversion (training-time augmentation only).
    `crop` is an optional callable applied to the original grayscale image
    before letterboxing (training only), e.g. RandomAreaCrop: it changes the
    size and aspect ratio the letterbox sees, which augmentation after
    letterboxing cannot do.
    """

    def __init__(self, split, root=CLEAN_DIR, image_size=IMAGE_SIZE, augment=None, crop=None):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.crop = crop
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}
        self.samples = [
            (p, self.class_to_idx[c])
            for c in CLASSES
            for p in sorted((self.root / split / c).iterdir())
            if p.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not self.samples:
            raise FileNotFoundError(f"No images found under {self.root / split}")

    def __len__(self):
        return len(self.samples)

    def load_image(self, index):
        path, _ = self.samples[index]
        with Image.open(path) as img:
            return letterbox(to_grayscale(img), self.image_size)

    def __getitem__(self, index):
        if self.crop is None:
            img = self.load_image(index)
        else:
            path, _ = self.samples[index]
            with Image.open(path) as raw:
                img = letterbox(self.crop(to_grayscale(raw)), self.image_size)
        if self.augment is not None:
            img = self.augment(img)
        return to_tensor(img), self.samples[index][1]

    def class_counts(self):
        counts = np.bincount([label for _, label in self.samples], minlength=len(CLASSES))
        return dict(zip(CLASSES, counts.tolist()))


def save_preview(out_path, per_class=4, seed=0):
    """Original vs preprocessed grid, deliberately including non-square and non-L images."""
    import matplotlib.pyplot as plt

    ds = BrainMRIClassificationDataset("train")
    rng = np.random.default_rng(seed)
    picks = []
    for label in range(len(CLASSES)):
        indices = [i for i, (_, y) in enumerate(ds.samples) if y == label]
        odd = [i for i in indices if _is_unusual(ds.samples[i][0])]
        chosen = odd[: per_class // 2]
        rest = [i for i in indices if i not in chosen]
        chosen += rng.choice(rest, per_class - len(chosen), replace=False).tolist()
        picks += chosen

    fig, axes = plt.subplots(len(picks) // 4, 8, figsize=(20, 2.6 * len(picks) // 4))
    for n, idx in enumerate(picks):
        path, label = ds.samples[idx]
        x, y = ds[idx]
        with Image.open(path) as raw:
            info = f"{raw.mode} {raw.width}x{raw.height}"
            raw_rgb = raw.convert("RGB")
        ax_raw, ax_out = axes.flat[2 * n], axes.flat[2 * n + 1]
        ax_raw.imshow(raw_rgb)
        ax_raw.set_title(f"{CLASSES[label]}\nraw {info}", fontsize=8)
        ax_out.imshow(denormalize(x).permute(1, 2, 0).numpy())
        ax_out.set_title(f"{path.name}\n{tuple(x.shape)} mean={x.mean():.2f}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def _is_unusual(path):
    with Image.open(path) as img:
        return img.mode != "L" or img.width != img.height


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", action="store_true", help="write reports/classification_preview.png")
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        ds = BrainMRIClassificationDataset(split)
        x, y = ds[0]
        print(f"{split:<5} n={len(ds):5d} sample={tuple(x.shape)} {x.dtype} label={CLASSES[y]} counts={ds.class_counts()}")
    if args.preview:
        out = PROJECT_ROOT / "reports" / "classification_preview.png"
        save_preview(out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
