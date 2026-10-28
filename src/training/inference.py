"""Whole-volume sliding-window inference and Dice scoring for the 3D U-Net.

The network is trained on 128^3 patches, but preprocessed BraTS volumes are
larger (about 137 x 171 x 139 on average after cropping to the brain). The
volume is covered with overlapping windows; each window's probabilities are
multiplied by a Gaussian importance map (centre weight 1, edges small) and
accumulated, then divided by the accumulated weights. Predictions near a
window's border, where the network sees less context, therefore count less,
which suppresses seams between windows (the nnU-Net blending scheme).

Volumes are cropped to the brain bounding box during preprocessing and all
tumour voxels lie inside it, so Dice on the cropped volume equals Dice on the
original 240 x 240 x 155 grid as long as nothing is predicted outside the brain.
"""
import itertools
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from src.training.losses import REGIONS


def gaussian_importance_map(patch_size, sigma_scale=0.125, device=None):
    """(P, P, P) weights: Gaussian with sigma = patch_size * sigma_scale, max 1.

    The minimum is clamped to the smallest positive value so every voxel of a
    window keeps a non-zero weight (no division by zero at the volume border).
    """
    coords = torch.arange(patch_size, dtype=torch.float32, device=device) - (patch_size - 1) / 2
    g = torch.exp(-0.5 * (coords / (patch_size * sigma_scale)) ** 2)
    weights = g[:, None, None] * g[None, :, None] * g[None, None, :]
    weights /= weights.max()
    return weights.clamp_min(weights[weights > 0].min())


def window_starts(length, patch_size, overlap):
    """Evenly spaced window starts covering [0, length) with at least `overlap` fractional overlap."""
    if length <= patch_size:
        return [0]
    step = patch_size * (1 - overlap)
    n = math.ceil((length - patch_size) / step) + 1
    return [round(i * (length - patch_size) / (n - 1)) for i in range(n)]


@torch.no_grad()
def sliding_window_predict(model, image, patch_size=128, overlap=0.5, batch_size=2, mixed_precision=True):
    """Probabilities (R, X, Y, Z) on the device of `image` (C, X, Y, Z).

    Volumes smaller than the patch along an axis are zero-padded (zero is the
    background value after z-scoring) and the padding is removed afterwards.
    """
    model.eval()
    shape = image.shape[1:]
    pad = [max(0, patch_size - s) for s in shape]
    if any(pad):
        image = torch.nn.functional.pad(image, [p for axis in reversed(pad) for p in (0, axis)])
    padded = image.shape[1:]

    importance = gaussian_importance_map(patch_size, device=image.device)
    starts = list(itertools.product(*(window_starts(s, patch_size, overlap) for s in padded)))
    accum = weights = None
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        windows = [tuple(slice(s, s + patch_size) for s in start) for start in chunk]
        batch = torch.stack([image[(slice(None), *w)] for w in windows])
        with torch.autocast("cuda", dtype=torch.float16, enabled=mixed_precision and image.is_cuda):
            out = model(batch)
        logits = (out[-1] if isinstance(out, (list, tuple)) else out).float()
        probs = torch.sigmoid(logits) * importance
        if accum is None:
            accum = image.new_zeros((logits.shape[1], *padded), dtype=torch.float32)
            weights = image.new_zeros(padded, dtype=torch.float32)
        for w, p in zip(windows, probs):
            accum[(slice(None), *w)] += p
            weights[w] += importance
    probs = accum / weights
    return probs[(slice(None), *(slice(0, s) for s in shape))]


def dice_scores(pred, target):
    """Per-region Dice for binary (R, ...) tensors -> list of floats.

    An empty ground truth with an empty prediction scores 1 and an empty ground
    truth with any predicted voxel scores 0 (the BraTS convention for cases
    without enhancing tumour).
    """
    dims = tuple(range(1, pred.ndim))
    pred, target = pred.bool(), target.bool()
    intersection = (pred & target).sum(dims).double()
    denominator = (pred.sum(dims) + target.sum(dims)).double()
    dice = torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), torch.ones_like(denominator))
    return dice.tolist()


def prefetch(load, items, depth=2):
    """Yield (item, load(item)) in order, loading up to `depth` items ahead in threads."""
    items = list(items)
    with ThreadPoolExecutor(depth) as pool:
        futures = [pool.submit(load, item) for item in items[:depth]]
        for i, item in enumerate(items):
            result = futures[i].result()
            futures[i] = None
            if i + depth < len(items):
                futures.append(pool.submit(load, items[i + depth]))
            yield item, result


def evaluate_volumes(model, cases, load_case, device, patch_size=128, overlap=0.5, batch_size=2,
                     mixed_precision=True, threshold=0.5, keep=()):
    """Dice per case and region over full volumes.

    load_case(case_id) -> (image (C, X, Y, Z) array, targets (R, X, Y, Z) array).
    The next volumes are decompressed in background threads while the GPU works.
    Returns (metrics dict with mean Dice per region, {case_id: [Dice per region]},
    {case_id: (image, targets, probs)} for cases in `keep`).
    """
    per_case, kept = {}, {}
    for case, (image, targets) in prefetch(load_case, cases):
        image_t = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).to(device, non_blocking=True)
        target_t = torch.from_numpy(np.ascontiguousarray(targets)).to(device, non_blocking=True)
        probs = sliding_window_predict(model, image_t, patch_size, overlap, batch_size, mixed_precision)
        per_case[case] = dice_scores(probs > threshold, target_t)
        if case in keep:
            kept[case] = (image, targets, probs.cpu().numpy())
    scores = np.array(list(per_case.values()))  # (cases, regions)
    metrics = {f"dice_score/{r}": float(scores[:, i].mean()) for i, r in enumerate(REGIONS)}
    metrics["dice_score/mean"] = float(scores.mean())
    return metrics, per_case, kept
