"""Display slices of a segmented study for the web client's viewer and the PNG export.

For each plane (axial, coronal, sagittal) one slice is kept: the one through the largest tumour
cross-section. Each plane has seven layers:
- the four sequences, as grayscale images that are transparent outside the brain;
- the three BraTS labels (edema, necrotic core, enhancing tumour), each as a red-family layer.

The client stacks the layers and changes sequence, labels and opacity without another request.
Only the small 2D slices are kept per result; PNGs are drawn on first request and cached.
"""
import io

import numpy as np
from PIL import Image

PLANES = {"axial": 2, "coronal": 1, "sagittal": 0}  # plane -> volume axis held fixed
SEQUENCES = ("flair", "t1", "t1ce", "t2")
# BraTS label -> layer name and colour (design-system finding colours, frontend/src/design/tokens.ts)
LABEL_LAYERS = {
    "edema": (2, (122, 17, 27), 0.62),
    "necrotic": (1, (215, 38, 61), 0.78),
    "enhancing": (4, (255, 194, 202), 0.78),
}
LAYERS = SEQUENCES + tuple(LABEL_LAYERS)
UPSCALE = 4  # slices are ~150-200 px; drawn at 4x with smooth edges for display
CROP_PAD = 6


def orient(a2d):
    """Volume slice (indexed [i, j]) -> display orientation, as in the training visualisations."""
    return np.flipud(a2d.T)


def take(volume, axis, index):
    s = [slice(None)] * 3
    s[axis] = index
    return orient(volume[tuple(s)])


def tumour_slice(labels, axis):
    """Index of the slice with the most tumour voxels along `axis` (the centre if there is no tumour)."""
    tumour = labels > 0
    if not tumour.any():
        return labels.shape[axis] // 2
    other = tuple(a for a in range(3) if a != axis)
    return int(tumour.sum(axis=other).argmax())


def extract(image, labels):
    """image: (4, X, Y, Z) in SEQUENCES order; labels: (X, Y, Z) BraTS label map.

    Returns {plane: {"index", "count", "brain", "labels", "flair", ...}} with each 2D slice cropped to
    the brain (plus a small margin), sequences already windowed to [0, 1] by the whole-brain intensity
    range so the same sequence looks the same in every plane.
    """
    brain = np.any(image != 0, axis=0)
    windows = []
    for vol in image:
        values = vol[brain]
        lo, hi = np.percentile(values, [0.5, 99.5]) if values.size else (0.0, 1.0)
        windows.append((float(lo), float(max(hi, lo + 1e-6))))
    views = {}
    for plane, axis in PLANES.items():
        index = tumour_slice(labels, axis)
        mask = take(brain, axis, index)
        rows, cols = np.nonzero(mask)
        if rows.size:
            r0, r1 = max(rows.min() - CROP_PAD, 0), min(rows.max() + CROP_PAD + 1, mask.shape[0])
            c0, c1 = max(cols.min() - CROP_PAD, 0), min(cols.max() + CROP_PAD + 1, mask.shape[1])
        else:
            r0, r1, c0, c1 = 0, mask.shape[0], 0, mask.shape[1]
        crop = (slice(r0, r1), slice(c0, c1))
        view = {"index": index, "count": int(labels.shape[axis]), "brain": mask[crop].astype(np.float32),
                "labels": take(labels, axis, index)[crop]}
        for name, vol, (lo, hi) in zip(SEQUENCES, image, windows):
            view[name] = np.clip((take(vol, axis, index)[crop] - lo) / (hi - lo), 0, 1).astype(np.float32)
        views[plane] = view
    return views


def _up(a, size, resample):
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize(size, resample))


def layer_rgba(view, layer):
    """One layer of one plane as an RGBA uint8 array at UPSCALE x."""
    h, w = view["brain"].shape
    size = (w * UPSCALE, h * UPSCALE)
    if layer in SEQUENCES:
        alpha = np.clip(_up(view["brain"], size, Image.BILINEAR) * 1.6 - 0.3, 0, 1)
        gray = np.clip(_up(view[layer], size, Image.BICUBIC), 0, 1) ** 0.9
        return np.dstack([255 * gray] * 3 + [255 * alpha]).round().astype(np.uint8)
    value, colour, opacity = LABEL_LAYERS[layer]
    a = np.clip((_up((view["labels"] == value).astype(np.float32), size, Image.BILINEAR) - 0.25) / 0.5, 0, 1)
    a = a * a * (3 - 2 * a) * opacity  # smoothstep: soft but well-defined edges
    rgba = np.zeros((size[1], size[0], 4), np.uint8)
    rgba[..., :3] = colour
    rgba[..., 3] = (255 * a).round().astype(np.uint8)
    return rgba


def layer_png(view, layer):
    buffer = io.BytesIO()
    Image.fromarray(layer_rgba(view, layer)).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def composite(view, sequence, labels=tuple(LABEL_LAYERS), opacity=1.0, background=(0, 0, 0)):
    """The slice as shown on screen: one sequence with the chosen label layers at `opacity`, as RGB."""
    base = Image.new("RGBA", (view["brain"].shape[1] * UPSCALE, view["brain"].shape[0] * UPSCALE), (*background, 255))
    base.alpha_composite(Image.fromarray(layer_rgba(view, sequence)))
    for name in LABEL_LAYERS:
        if name in labels:
            layer = layer_rgba(view, name)
            layer[..., 3] = (layer[..., 3] * opacity).round().astype(np.uint8)
            base.alpha_composite(Image.fromarray(layer))
    return base.convert("RGB")
