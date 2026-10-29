"""Counterfactual image-format test for the classification shortcut.

The error analysis found that no-tumour images are mostly small, non-square
files while tumour images are mostly 512 x 512. This test changes only the
format of each test image, not its content: the original is resized to a
typical no-tumour file size before the usual letterbox preprocessing, and the
predictions are compared with those on the unmodified image. A model that
decides from tumour appearance should barely change; a model that uses format
as a cue will move tumour images towards "no tumour".

Conditions (applied to every test image, all classes):
    original        unchanged
    small_square    resized to 225 x 225, the most common no-tumour training size
    small_portrait  resized to 210 x 234, a common non-square no-tumour size
    tiny_square     resized to 128 x 128, smaller than any size seen by the
                    resolution-degradation augmentation (shorter side 150-320),
                    so it also tests robustness outside the trained range

Writes reports/results/format_stress_test.json.

Usage:
    python -m src.training.format_stress_test models/classification/best.pt models/classification_crop/best.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT
from src.data.classification_preprocess import CLASSES, BrainMRIClassificationDataset
from src.models.classifier import EfficientNetV2Classifier
from src.training.evaluate_segmentation import RESULTS_DIR
from src.training.utils import load_checkpoint

CONDITIONS = {"original": None, "small_square": (225, 225), "small_portrait": (210, 234), "tiny_square": (128, 128)}
NO_TUMOUR = CLASSES.index("notumor")


class ResizeTo:
    """Resize the original image to a fixed file size (before letterboxing)."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize(self.size, Image.BICUBIC)


@torch.no_grad()
def predict(model, dataset, device, batch_size=32, workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    preds = []
    for image, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            preds.append(model(image.to(device, non_blocking=True)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = np.array([label for _, label in BrainMRIClassificationDataset(args.split).samples])
    tumour = labels != NO_TUMOUR
    results = {"split": args.split, "images": int(len(labels)), "tumour_images": int(tumour.sum()),
               "conditions": {k: v for k, v in CONDITIONS.items()}, "models": {}}

    for ckpt in args.checkpoints:
        model = EfficientNetV2Classifier(pretrained=False).to(device).eval()
        model.load_state_dict(load_checkpoint(ckpt, device)["model"])
        name = ckpt.resolve().relative_to(PROJECT_ROOT).as_posix()
        per_condition, base_preds = {}, None
        for condition, size in CONDITIONS.items():
            ds = BrainMRIClassificationDataset(args.split, crop=ResizeTo(size) if size else None)
            preds = predict(model, ds, device)
            base_preds = preds if base_preds is None else base_preds
            entry = {
                "accuracy": float((preds == labels).mean()),
                "tumour_images_called_no_tumour": int(((preds == NO_TUMOUR) & tumour).sum()),
                "tumour_recall_any_tumour_class": float((preds[tumour] != NO_TUMOUR).mean()),
                "predictions_changed_vs_original": int((preds != base_preds).sum()),
                "accuracy_by_class": {c: float((preds[labels == i] == i).mean()) for i, c in enumerate(CLASSES)},
            }
            per_condition[condition] = entry
            print(f"{name:<40} {condition:<15} acc {entry['accuracy']:.4f}  tumours called no tumour "
                  f"{entry['tumour_images_called_no_tumour']:4d}/{int(tumour.sum())}  changed "
                  f"{entry['predictions_changed_vs_original']:4d}  per class "
                  + " ".join(f"{c} {a:.3f}" for c, a in entry["accuracy_by_class"].items()), flush=True)
        results["models"][name] = per_condition

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "format_stress_test.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
