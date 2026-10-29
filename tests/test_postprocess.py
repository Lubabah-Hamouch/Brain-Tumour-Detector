import numpy as np
import pytest

from src.training.postprocess import component_stats, components, dice_after_removal, remove_small_components


def two_blobs():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[2:6, 2:6, 2:6] = True    # 64 voxels
    mask[15, 15, 15] = True       # 1 voxel
    return mask


def test_components_use_26_connectivity():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1, 1, 1] = mask[2, 2, 2] = True  # touch only at a corner
    _, sizes = components(mask)
    assert sizes.tolist() == [2]


def test_remove_small_components_keeps_large_blob_only():
    mask = two_blobs()
    out = remove_small_components(mask, 10)
    assert out.sum() == 64 and not out[15, 15, 15]
    assert remove_small_components(mask, 0).sum() == 65  # threshold 0 is a no-op


def test_dice_after_removal_matches_direct_computation():
    pred = two_blobs()
    target = np.zeros_like(pred)
    target[2:6, 2:6, 3:7] = True  # overlaps the large blob in 48 voxels
    stats = component_stats(pred, target)
    for threshold in (0, 2, 10, 100):
        kept = remove_small_components(pred, threshold)
        direct = 2 * (kept & target).sum() / (kept.sum() + target.sum())
        assert dice_after_removal(stats, int(target.sum()), threshold) == pytest.approx(direct)


def test_removal_rescues_case_without_target():
    pred = np.zeros((10, 10, 10), dtype=bool)
    pred[0, 0, 0] = True
    stats = component_stats(pred, np.zeros_like(pred))
    assert dice_after_removal(stats, 0, 0) == 0.0
    assert dice_after_removal(stats, 0, 5) == 1.0
