"""Small image helpers for API responses: PNG encoding and colour overlays."""
import base64
import io

import numpy as np
from matplotlib import colormaps
from PIL import Image


def png_base64(array):
    """uint8 (H, W), (H, W, 3) or (H, W, 4) array -> base64 PNG string."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# Finding colours of the web client's design system (frontend/src/design/tokens.ts):
# edema, necrotic core, enhancing. Grad-CAM is drawn on the same ramp, dark red to pale pink.
HEAT_STOPS = np.array([[122, 17, 27], [215, 38, 61], [255, 194, 202]], dtype=np.float32)


def heat_rgba(heat):
    """[0, 1] heatmap -> uint8 RGBA on the red ramp, transparent where attribution is weak (below 0.15)."""
    h = np.clip(heat, 0, 1).astype(np.float32)
    t = h * (len(HEAT_STOPS) - 1)
    lo = np.clip(np.floor(t).astype(int), 0, len(HEAT_STOPS) - 2)
    frac = (t - lo)[..., None]
    rgb = HEAT_STOPS[lo] * (1 - frac) + HEAT_STOPS[lo + 1] * frac
    alpha = np.clip((h - 0.15) / 0.85, 0, 1) ** 0.9 * 0.92
    return np.dstack([rgb, 255 * alpha]).round().astype(np.uint8)


def heatmap_overlay(gray, heat, alpha=0.4):
    """Blend a [0, 1] heatmap (jet colormap) over a grayscale uint8 image, as in inspect_errors."""
    base = np.repeat(gray[..., None].astype(np.float32) / 255.0, 3, axis=2)
    colour = colormaps["jet"](np.clip(heat, 0, 1))[..., :3]
    return (255 * ((1 - alpha) * base + alpha * colour)).clip(0, 255).astype(np.uint8)


# BraTS labels: 1 necrotic tumour core, 2 peritumoral edema, 4 enhancing tumour.
LABEL_COLOURS = {1: (214, 39, 40), 2: (44, 160, 44), 4: (255, 215, 0)}


def label_overlay(slice_2d, labels_2d, alpha=0.5):
    """FLAIR slice (any float range) with BraTS label colours; returned upright for display."""
    lo, hi = np.percentile(slice_2d, [1, 99])
    gray = np.clip((slice_2d - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    for label, colour in LABEL_COLOURS.items():
        mask = labels_2d == label
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.array(colour) / 255.0
    # Arrays are indexed (x, y); transposing and flipping vertically gives the same orientation as
    # the training visualisations (imshow of the transpose with origin="lower").
    return np.flipud((255 * rgb).astype(np.uint8).transpose(1, 0, 2))
