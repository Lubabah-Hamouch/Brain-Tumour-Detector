import json

import numpy as np
import pytest
import torch
from torch import nn

from src.config import PROJECT_ROOT
from src.data.make_brats_folds import MANIFEST, assign, load_fold
from src.data.segmentation_preprocess import BratsPatchDataset
from src.training.inference import dice_scores, gaussian_importance_map, sliding_window_predict, window_starts
from src.training.train_classification import classification_metrics
from src.training.utils import warmup_cosine


class FirstChannels(nn.Module):
    """Returns the first three input channels as logits, so stitching can be checked exactly."""

    def forward(self, x):
        return [x[:, :3] * 0.5, x[:, :3]]  # deep-supervision style: last output is the final one


def test_window_starts_cover_volume_with_overlap():
    for length in (128, 137, 171, 208, 255):
        starts = window_starts(length, 128, 0.5)
        assert starts[0] == 0 and starts[-1] + 128 == length
        assert all(b - a <= 64 for a, b in zip(starts, starts[1:]))
    assert window_starts(100, 128, 0.5) == [0]


def test_gaussian_map_peaks_in_centre_and_stays_positive():
    g = gaussian_importance_map(16)
    assert g.max() == 1 and g.min() > 0
    assert g[8, 8, 8] > g[0, 8, 8] > g[0, 0, 0]


def test_sliding_window_reconstructs_position_dependent_output_exactly():
    torch.manual_seed(0)
    image = torch.randn(4, 40, 52, 33)  # smaller than the patch on one axis: exercises padding
    probs = sliding_window_predict(FirstChannels(), image, patch_size=32, overlap=0.5, batch_size=3,
                                   mixed_precision=False)
    assert probs.shape == (3, 40, 52, 33)
    torch.testing.assert_close(probs, torch.sigmoid(image[:3]))


def test_dice_scores_conventions():
    target = torch.zeros(3, 4, 4, 4, dtype=torch.bool)
    target[0, :2] = True                       # WT present
    pred = target.clone()
    pred[1, 0, 0, 0] = True                    # TC: empty truth, one false positive -> 0
    wt, tc, et = dice_scores(pred, target)     # ET: both empty -> 1
    assert (wt, tc, et) == (1.0, 0.0, 1.0)
    half = target.clone()
    half[0, 1] = False
    assert dice_scores(half, target)[0] == pytest.approx(2 * 16 / (16 + 32))


def test_fold_manifest_is_patient_disjoint_and_complete():
    data = json.loads(MANIFEST.read_text())
    index_path = PROJECT_ROOT / data["source"] / "index.json"  # local data, absent on a fresh clone
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    folds = [set(v) for v in data["folds"].values()]
    test = set(data["test"])
    everything = set().union(*folds, test)
    assert sum(map(len, folds)) + len(test) == len(everything)  # no case in two places
    assert len(everything) == len(data["case_voxels"]) - 1     # minus the "regions" key
    if index is not None:
        assert everything == {item["case"] for item in index["items"]}
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 1
    train, val = load_fold(1)
    assert set(val) == folds[0] and not set(train) & set(val) and not (set(train) | set(val)) & test


def test_assign_is_deterministic_and_stratified():
    labels = {f"c{i:03d}": ("ET-absent" if i < 20 else f"WT-q{i % 4 + 1}") for i in range(200)}
    a = assign(labels, 0.15, 5, seed=1)
    assert a == assign(labels, 0.15, 5, seed=1)
    test_absent = sum(1 for c, s in a.items() if s == "test" and labels[c] == "ET-absent")
    assert test_absent == 3  # 15% of the 20 ET-absent cases


def test_warmup_cosine_shape():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = warmup_cosine(opt, warmup_steps=10, total_steps=110, min_lr_ratio=0.01)
    lrs = []
    for _ in range(110):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))
    assert lrs[-1] == pytest.approx(0.01, abs=1e-3)


def test_classification_metrics():
    labels = np.array([0, 0, 1, 1, 2, 3])
    preds = np.array([0, 1, 1, 1, 2, 2])
    cm, precision, recall, f1 = classification_metrics(labels, preds, 4)
    assert cm[0, 1] == 1 and cm[3, 2] == 1 and cm.sum() == 6
    assert precision[1] == pytest.approx(2 / 3) and recall[0] == pytest.approx(0.5)
    assert f1[3] == 0


def test_patch_flip_keeps_image_and_targets_aligned(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.normal(size=(4, 20, 20, 20)).astype(np.float16)
    targets = (image[:3] > 0).astype(np.uint8)  # targets are a function of the image
    np.savez(tmp_path / "case.npz", image=image, targets=targets)
    ds = BratsPatchDataset(tmp_path, ["case"], patch_size=16, flip=True, seed=3)
    for epoch in range(4):
        ds.set_epoch(epoch)
        x, y = ds[0]
        np.testing.assert_array_equal(y, (x[:3] > 0).astype(np.float32))
