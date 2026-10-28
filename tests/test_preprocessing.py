import numpy as np
import pytest
import torch
from PIL import Image

from src.data.classification_preprocess import letterbox, to_grayscale, to_tensor
from src.data.segmentation_preprocess import (
    crop_to_mask,
    extract,
    patch_starts,
    region_targets,
    zscore_in_mask,
)


def test_region_targets_follow_brats_definitions():
    seg = np.array([0, 1, 2, 4], dtype=np.uint8).reshape(1, 1, 4)
    wt, tc, et = region_targets(seg)
    assert wt.ravel().tolist() == [0, 1, 1, 1]
    assert tc.ravel().tolist() == [0, 1, 0, 1]
    assert et.ravel().tolist() == [0, 0, 0, 1]


def test_zscore_uses_brain_voxels_only():
    rng = np.random.default_rng(0)
    image = np.zeros((2, 20, 20, 20), dtype=np.float32)
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[5:15, 5:15, 5:15] = True
    image[:, mask] = rng.normal(500, 80, size=(2, mask.sum()))
    out = zscore_in_mask(image, mask)
    assert np.allclose(out[:, mask].mean(axis=1), 0, atol=1e-4)
    assert np.allclose(out[:, mask].std(axis=1), 1, atol=1e-4)
    assert np.all(out[:, ~mask] == 0)


def test_crop_pads_up_to_patch_size():
    image = np.ones((1, 50, 50, 50), dtype=np.float32)
    targets = np.zeros((3, 50, 50, 50), dtype=np.uint8)
    mask = np.zeros((50, 50, 50), dtype=bool)
    mask[10:20, 10:40, 5:45] = True
    cropped, _, offset = crop_to_mask(image, targets, mask, min_size=32)
    assert cropped.shape == (1, 32, 32, 40)
    assert offset == [10, 10, 5]


@pytest.mark.parametrize("tumor_ratio, expected_biased", [(0.75, 3), (1.0, 4), (0.0, 0)])
def test_tumor_biased_patches_always_contain_tumor(tumor_ratio, expected_biased):
    rng = np.random.default_rng(1)
    targets = np.zeros((3, 60, 70, 80), dtype=np.uint8)
    targets[0, 55:58, 2:4, 70:75] = 1  # small tumour near a corner
    for _ in range(50):
        starts = patch_starts(targets, 4, tumor_ratio, 32, rng)
        assert sum(biased for _, biased in starts) == expected_biased
        for start, biased in starts:
            patch = extract(targets, start, 32)
            assert patch.shape == (3, 32, 32, 32)
            if biased:
                assert patch[0].sum() > 0


def test_patch_starts_without_tumor_falls_back_to_random():
    targets = np.zeros((3, 40, 40, 40), dtype=np.uint8)
    starts = patch_starts(targets, 4, 1.0, 32, np.random.default_rng(0))
    assert not any(biased for _, biased in starts)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "P"])
def test_grayscale_conversion_handles_all_source_modes(mode):
    img = Image.new("RGB", (30, 20), (90, 90, 90)).convert(mode)
    gray = to_grayscale(img)
    assert gray.mode == "L"
    assert abs(np.asarray(gray).mean() - 90) < 2


def test_letterbox_preserves_aspect_and_pads_black():
    img = Image.new("L", (200, 100), 255)
    out = letterbox(img, 64)
    arr = np.asarray(out)
    assert out.size == (64, 64)
    assert arr[0].max() == 0 and arr[-1].max() == 0  # padding rows
    assert arr[32].min() == 255  # content row spans full width


def test_tensor_is_three_channel_imagenet_normalised():
    x = to_tensor(Image.new("L", (8, 8), 0))
    assert x.shape == (3, 8, 8) and x.dtype == torch.float32
    assert torch.allclose(x[:, 0, 0], torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]))


def test_colour_spread_flags_coloured_overlays_only(tmp_path):
    from src.data.make_split import colour_spread

    gray_rgb = Image.new("RGB", (20, 20), (80, 80, 80))
    gray_rgb.save(tmp_path / "gray.png")
    tinted = gray_rgb.copy()
    tinted.putpixel((3, 3), (40, 40, 200))  # a blue watermark pixel
    tinted.save(tmp_path / "tinted.png")
    Image.new("L", (20, 20), 80).save(tmp_path / "l.png")
    assert colour_spread(tmp_path / "gray.png") == 0
    assert colour_spread(tmp_path / "l.png") == 0
    assert colour_spread(tmp_path / "tinted.png") > 30
