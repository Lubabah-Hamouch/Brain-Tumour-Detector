"""Connected-component post-processing for predicted enhancing tumour (ET).

ET is the smallest region and the one most affected by isolated false
positives: a few stray voxels in a case without enhancing tumour turn an ET
Dice of 1 into 0. Removing predicted ET components below a minimum size
targets exactly those errors. Components are 26-connected (any shared face,
edge or corner), so a thin but continuous structure is one component.

The size threshold must be chosen on validation predictions, never on the test
set; see src.training.evaluate_segmentation.
"""
import numpy as np
from scipy import ndimage

STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)


def components(mask):
    """(labels, sizes) of the 26-connected components of a binary 3D mask; sizes[i] is component i + 1."""
    labels, n = ndimage.label(mask, structure=STRUCTURE_26)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    return labels, sizes


def remove_small_components(mask, min_voxels):
    """Copy of `mask` without the components smaller than `min_voxels`."""
    if min_voxels <= 1 or not mask.any():
        return mask.copy()
    labels, sizes = components(mask)
    keep = np.concatenate([[False], sizes >= min_voxels])
    return keep[labels]


def component_stats(pred, target):
    """Per predicted component: (size, voxels overlapping target), as an (n, 2) int array.

    A component with zero overlap is a pure false positive. Together with the
    target volume this is enough to recompute Dice for any size threshold
    without relabelling (see dice_after_removal).
    """
    labels, sizes = components(pred)
    overlap = np.bincount(labels[target.astype(bool)], minlength=len(sizes) + 1)[1:]
    return np.stack([sizes, overlap], axis=1).astype(np.int64)


def dice_after_removal(stats, target_voxels, min_voxels):
    """Dice of the prediction once components smaller than `min_voxels` are removed.

    Uses the BraTS convention for empty masks: empty truth and empty
    prediction score 1, empty truth with any prediction scores 0.
    """
    kept = stats[stats[:, 0] >= min_voxels] if len(stats) else stats
    pred_voxels = int(kept[:, 0].sum()) if len(kept) else 0
    intersection = int(kept[:, 1].sum()) if len(kept) else 0
    denominator = pred_voxels + target_voxels
    return 1.0 if denominator == 0 else 2 * intersection / denominator
