"""Shareable PNG of a classification result: the slice, the Grad-CAM overlay, the prediction and
all four probabilities, laid out like the web client's result screen.

Rendered on the server with Pillow, using the web client's fonts (Geist, JetBrains Mono, bundled
in backend/assets/fonts under the SIL Open Font License) and its colour tokens, so the export
looks the same whichever browser asked for it. Layout units are CSS pixels of a 1200 x 1500 page,
drawn at SCALE x for high-density screens.
"""
import io
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.services.imaging import heat_rgba

FONTS = Path(__file__).resolve().parent.parent / "assets" / "fonts"
SCALE = 2
WIDTH, HEIGHT, PAD = 1200, 1500, 64

BG = "#0B0C0D"
PANEL = "#000000"
TEXT = "#ECEDEF"
SECONDARY = "#9A9EA4"
MUTED = "#858A90"
HAIRLINE = "#24262A"
TRACK = "#2A2C30"
BAR = "#6A6E74"
FINDING = "#D7263D"

CLASS_NAMES = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumour", "pituitary": "Pituitary"}


@lru_cache(maxsize=None)
def font(family, size, weight):
    f = ImageFont.truetype(str(FONTS / f"{family}-Variable.ttf"), round(size * SCALE))
    f.set_variation_by_axes([weight])
    return f


def sans(size, weight=400):
    return font("Geist", size, weight)


def mono(size, weight=400):
    return font("JetBrainsMono", size, weight)


class Canvas:
    """Pillow drawing in CSS pixels (scaled by SCALE), with letter-spacing for small-caps labels."""

    def __init__(self):
        self.image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BG)
        self.draw = ImageDraw.Draw(self.image)

    def width(self, text, f, tracking=0.0):
        if not tracking:
            return f.getlength(text) / SCALE
        return sum(f.getlength(ch) for ch in text) / SCALE + tracking * f.size / SCALE * (len(text) - 1)

    def text(self, x, y, text, f, fill, tracking=0.0, anchor="ls"):
        """Draw with the baseline at y. anchor 'rs' right-aligns at x."""
        if anchor == "rs":
            x -= self.width(text, f, tracking)
        if not tracking:
            self.draw.text((x * SCALE, y * SCALE), text, font=f, fill=fill, anchor="ls")
            return
        cx = x * SCALE
        for ch in text:
            self.draw.text((cx, y * SCALE), ch, font=f, fill=fill, anchor="ls")
            cx += f.getlength(ch) + tracking * f.size

    def rect(self, x0, y0, x1, y1, fill):
        self.draw.rectangle((round(x0 * SCALE), round(y0 * SCALE), round(x1 * SCALE) - 1, round(y1 * SCALE) - 1),
                            fill=fill)

    def hline(self, x0, x1, y, fill=HAIRLINE):
        self.rect(x0, y, x1, y + 1, fill)

    def fit(self, text, f, max_width):
        """Shorten text with an ellipsis in the middle until it fits."""
        if self.width(text, f) <= max_width:
            return text
        keep = len(text)
        while keep > 4:
            keep -= 1
            short = text[: keep // 2] + "…" + text[len(text) - keep // 2:]
            if self.width(short, f) <= max_width:
                return short
        return text[:4] + "…"


def label(canvas, x, y, text, fill=MUTED):
    canvas.text(x, y, text.upper(), sans(12, 500), fill, tracking=0.16)


def classification_report(*, name, image_info, model_input, cam, result, exported=None):
    """PNG bytes. model_input: the 384 x 384 letterboxed uint8 slice; cam: [0, 1] Grad-CAM at the same size."""
    c = Canvas()
    right = WIDTH - PAD
    exported = exported or datetime.now(timezone.utc)

    # Header: what was classified.
    label(c, PAD, PAD + 12, "Tumour-type classification · result")
    size = result["letterbox"]["size"]
    c.text(right, PAD + 12 + 10 + 32, f"{image_info['width']} × {image_info['height']} · "
                                      f"{'grayscale' if image_info.get('grayscale') else 'colour'}",
           mono(14), SECONDARY, anchor="rs")
    c.text(right, PAD + 12 + 10 + 32 + 21, f"model input {size} × {size}", mono(14), SECONDARY, anchor="rs")
    meta_width = max(c.width(f"model input {size} × {size}", mono(14)), 220)
    title = c.fit(name or "uploaded image", sans(40, 700), right - PAD - meta_width - 32)
    c.text(PAD, PAD + 12 + 10 + 32, title, sans(40, 700), TEXT)
    header_bottom = PAD + 12 + 10 + 40 + 8

    # Footer and the prediction block are laid out from the bottom up.
    footer_top = HEIGHT - PAD - 16 - 36
    grid_bottom = footer_top - 40
    grid_top = grid_bottom - 152
    panel_top, panel_bottom = header_bottom + 40, grid_top - 40

    # Image panel: the model input with the heatmap at 85% opacity, as on the result screen.
    c.rect(PAD, panel_top, right, panel_bottom, PANEL)
    side = min(right - PAD, panel_bottom - panel_top) - 48
    px = round((PAD + (right - PAD - side) / 2) * SCALE)
    py = round((panel_top + (panel_bottom - panel_top - side) / 2) * SCALE)
    target = (round(side * SCALE),) * 2
    base = Image.fromarray(model_input).convert("RGB").resize(target, Image.BICUBIC)
    heat = Image.fromarray(heat_rgba(cam)).resize(target, Image.BILINEAR)
    heat.putalpha(heat.getchannel("A").point(lambda a: round(a * 0.85)))
    base.paste(heat, (0, 0), heat)
    c.image.paste(base, (px, py))

    # Prediction.
    predicted = result["predicted_class"]
    label(c, PAD, grid_top + 12, "Prediction")
    c.text(PAD, grid_top + 12 + 10 + 32, CLASS_NAMES.get(predicted, predicted), sans(40, 700), TEXT)
    c.text(PAD, grid_bottom - 10, f"{result['confidence']:.3f}", mono(64, 300), TEXT, tracking=-0.04)

    # All four probabilities, highest first.
    col2 = PAD + (right - PAD - 48) / 2.2 + 48
    rows = sorted(result["probabilities"].items(), key=lambda kv: -kv[1])
    row_h = 32
    top = grid_bottom - row_h * len(rows) - 1
    for i, (cls, p) in enumerate(rows):
        y = top + i * row_h
        c.hline(col2, right, y)
        lead = cls == predicted
        colour = TEXT if lead else MUTED
        mid = y + row_h / 2
        c.text(col2, mid + 5, CLASS_NAMES.get(cls, cls), sans(14), colour)
        bar_x0, bar_x1 = col2 + 110 + 12, right - 64 - 12
        c.rect(bar_x0, mid - 1, bar_x1, mid + 1, TRACK)
        c.rect(bar_x0, mid - 1, bar_x0 + (bar_x1 - bar_x0) * max(p, 0.005), mid + 1, FINDING if lead else BAR)
        c.text(right, mid + 4, f"{p:.4f}", mono(12), colour, anchor="rs")
    c.hline(col2, right, top + row_h * len(rows))

    # Footer.
    c.hline(PAD, right, footer_top)
    epoch = result.get("model", {}).get("checkpoint_epoch")
    c.text(PAD, footer_top + 16 + 13, "Brain Tumour MRI · EfficientNetV2-S · red: Grad-CAM for the predicted class, "
                                      "from a 12 × 12 feature map", sans(12), MUTED)
    c.text(PAD, footer_top + 16 + 31, "Research use only. Not a medical device; must not inform diagnosis or treatment.",
           sans(12), MUTED)
    c.text(right, footer_top + 16 + 13, f"exported {exported:%Y-%m-%d}", mono(12), MUTED, anchor="rs")
    if epoch is not None:
        c.text(right, footer_top + 16 + 31, f"checkpoint epoch {epoch}", mono(12), MUTED, anchor="rs")

    buffer = io.BytesIO()
    c.image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def is_grayscale(img):
    """True for single-channel images and colour images whose channels are identical."""
    if img.mode in ("L", "LA", "I", "I;16", "1"):
        return True
    rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
    return bool(np.abs(rgb[..., 0] - rgb[..., 1]).max() <= 2 and np.abs(rgb[..., 1] - rgb[..., 2]).max() <= 2)
