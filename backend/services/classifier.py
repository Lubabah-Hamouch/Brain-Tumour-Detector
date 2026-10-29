"""Tumour-type classification of a single 2D MRI image.

Preprocessing, the model and Grad-CAM are imported from the training code, not
reimplemented: to_grayscale -> letterbox -> to_tensor from
src.data.classification_preprocess (exactly what validation and test images
get), and grad_cam from src.training.inspect_errors. Probabilities use the same
mixed-precision forward pass as src.training.evaluate_classification.
"""
import io
import threading

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from backend.services.errors import ApiError, relative_name
from backend.services.imaging import heatmap_overlay, png_base64
from src.data.classification_preprocess import CLASSES, letterbox, to_grayscale, to_tensor
from src.models.classifier import EfficientNetV2Classifier
from src.training.inspect_errors import grad_cam
from src.training.utils import load_checkpoint

MIN_SIDE = 32


class ClassifierService:
    def __init__(self, checkpoint, device):
        state = load_checkpoint(checkpoint, device)
        self.device = device
        self.checkpoint = checkpoint
        self.checkpoint_name = relative_name(checkpoint)
        self.image_size = int(state["config"]["image_size"])
        self.mixed_precision = bool(state["config"].get("mixed_precision", True)) and device.type == "cuda"
        self.classes = tuple(state.get("classes", CLASSES))
        self.info = {"checkpoint_epoch": state["epoch"] + 1,
                     "validation": {k.removeprefix("val/"): round(float(v), 4) for k, v in state["metrics"].items()
                                    if k in ("val/accuracy", "val/macro_f1", "val/loss")}}
        self.model = EfficientNetV2Classifier(pretrained=False).to(device).eval()
        self.model.load_state_dict(state["model"])
        self._lock = threading.Lock()  # Grad-CAM registers hooks and backpropagates: one request at a time

    @staticmethod
    def open_image(data):
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                return img.copy()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ApiError(422, "unreadable_image", f"The file could not be decoded as an image ({exc}).") from exc

    def predict(self, img):
        warnings = []
        if min(img.size) < MIN_SIDE:
            raise ApiError(422, "image_too_small", f"Image is {img.width}x{img.height}; at least {MIN_SIDE} px per side is required.")
        if img.mode not in ("L", "RGB", "RGBA", "P", "LA", "I;16", "I"):
            warnings.append(f"Unusual image mode {img.mode}; converted to grayscale.")
        if img.mode in ("I;16", "I"):  # 16-bit grayscale: scale to 8 bit as PIL's 'L' conversion would clip
            arr = np.asarray(img, dtype=np.float32)
            img = Image.fromarray((255 * (arr - arr.min()) / max(float(arr.max() - arr.min()), 1.0)).astype(np.uint8))
            warnings.append("16-bit image rescaled to 8 bit; training images were 8 bit.")

        gray = to_grayscale(img)
        boxed = letterbox(gray, self.image_size)
        x = to_tensor(boxed)[None].to(self.device)

        with self._lock:
            with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
                logits = self.model(x)
            probs = logits.float().softmax(1)[0].cpu().numpy()
            pred = int(probs.argmax())
            _, cam = grad_cam(self.model, x.clone(), pred)
            self.model.zero_grad(set_to_none=True)

        scale = self.image_size / max(gray.size)
        resized = (max(1, round(gray.width * scale)), max(1, round(gray.height * scale)))
        boxed_arr = np.asarray(boxed)
        return {
            "predicted_class": self.classes[pred],
            "confidence": float(probs[pred]),
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "image": {"width": img.width, "height": img.height, "mode": img.mode},
            "letterbox": {"size": self.image_size, "scale": scale,
                          "offset_x": (self.image_size - resized[0]) // 2, "offset_y": (self.image_size - resized[1]) // 2,
                          "resized_width": resized[0], "resized_height": resized[1]},
            "gradcam": {"target_class": self.classes[pred],
                        "heatmap_png": png_base64((255 * np.clip(cam, 0, 1)).astype(np.uint8)),
                        "overlay_png": png_base64(heatmap_overlay(boxed_arr, cam)),
                        "resolution_note": "Upsampled from the 12 x 12 last feature map; indicative, not a precise localisation."},
            "model": {"checkpoint": self.checkpoint_name, **self.info},
            "warnings": warnings,
        }
