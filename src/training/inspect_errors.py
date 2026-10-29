"""Visual inspection of classifier errors: original image, model input, class probabilities, Grad-CAM.

For each image the figure shows the original file (with its size and colour
mode), the letterboxed 384 x 384 input the model sees, the softmax
probabilities, and a Grad-CAM heatmap from the last convolutional stage for
the class given by --cam-class (default: the predicted class).

Grad-CAM is a coarse (12 x 12) attribution and is only indicative; it is used
here to check whether a prediction is driven by the lesion or by something
else in the frame, not as a precise localisation.

Figures contain dataset images, so they are written under reports/ (untracked).

Usage:
    python -m src.training.inspect_errors --split test --files Te-gl_350.jpg Te-gl_97.jpg Te-gl_318.jpg \
        --out reports/figures/glioma_missed_as_notumor.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import PROJECT_ROOT
from src.data.classification_preprocess import CLASSES, CLEAN_DIR, IMAGE_SIZE, letterbox, to_grayscale, to_tensor
from src.models.classifier import EfficientNetV2Classifier
from src.training.utils import load_checkpoint


def find(split, name):
    matches = list((CLEAN_DIR / split).glob(f"*/{name}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{name}: {len(matches)} matches under {CLEAN_DIR / split}")
    return matches[0]


def grad_cam(model, x, class_index):
    """Grad-CAM over the last feature stage, upsampled to the input size and scaled to [0, 1]."""
    store = {}
    layer = model.features[-1]
    hooks = [layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o)),
             layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))]
    try:
        logits = model(x)
        model.zero_grad()
        logits[0, class_index].backward()
    finally:
        for h in hooks:
            h.remove()
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * store["act"]).sum(1))[0].detach().cpu().numpy()
    cam = Image.fromarray(cam / (cam.max() + 1e-8)).resize(x.shape[-2:][::-1], Image.BILINEAR)
    return logits.softmax(1)[0].detach().cpu().numpy(), np.asarray(cam)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--files", nargs="+", required=True, help="image file names, e.g. Te-gl_350.jpg")
    parser.add_argument("--cam-class", choices=CLASSES, help="class for Grad-CAM (default: predicted class)")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "classification" / "best.pt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetV2Classifier(pretrained=False).to(device).eval()
    model.load_state_dict(load_checkpoint(args.checkpoint, device)["model"])

    fig, axes = plt.subplots(len(args.files), 3, figsize=(12, 4 * len(args.files)), squeeze=False)
    for row, name in zip(axes, args.files):
        path = find(args.split, name)
        with Image.open(path) as raw:
            size, mode, gray = raw.size, raw.mode, to_grayscale(raw)
        img = letterbox(gray, IMAGE_SIZE)
        x = to_tensor(img)[None].to(device).requires_grad_(True)
        probs, _ = grad_cam(model, x, 0)
        pred = int(probs.argmax())
        cam_class = CLASSES.index(args.cam_class) if args.cam_class else pred
        _, cam = grad_cam(model, x, cam_class)
        true = path.parent.name
        print(f"{name}: true {true}, predicted {CLASSES[pred]}, size {size[0]}x{size[1]} {mode}, "
              + ", ".join(f"{c} {p:.4f}" for c, p in zip(CLASSES, probs)))

        row[0].imshow(np.asarray(gray), cmap="gray")
        row[0].set_title(f"{name} (true {true})\noriginal {size[0]}x{size[1]} {mode}", fontsize=9)
        row[1].imshow(np.asarray(img), cmap="gray")
        row[1].set_title("model input\n" + ", ".join(f"{c} {p:.3f}" for c, p in zip(CLASSES, probs)), fontsize=8)
        row[2].imshow(np.asarray(img), cmap="gray")
        row[2].imshow(cam, cmap="jet", alpha=0.4)
        row[2].set_title(f"Grad-CAM for '{CLASSES[cam_class]}' (predicted {CLASSES[pred]})", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
