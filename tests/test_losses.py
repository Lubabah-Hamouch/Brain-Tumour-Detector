import pytest
import torch

from src.training.losses import (
    DeepSupervisionDiceBCELoss,
    WeightedCrossEntropyLoss,
    inverse_frequency_weights,
)


def make_target(size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    wt = (torch.rand(1, 1, size, size, size, generator=g) > 0.6).float()
    tc = wt * (torch.rand(1, 1, size, size, size, generator=g) > 0.5).float()
    et = tc * (torch.rand(1, 1, size, size, size, generator=g) > 0.5).float()
    return torch.cat([wt, tc, et], dim=1)


def logits_from_mask(mask, scale=20.0):
    return (mask * 2 - 1) * scale


def pyramid(target):
    """Deep-supervision style outputs (coarsest first) that exactly match the target."""
    sizes = [target.shape[-1] // 4, target.shape[-1] // 2, target.shape[-1]]
    return [logits_from_mask(torch.nn.functional.interpolate(target, size=(s,) * 3, mode="nearest-exact")) for s in sizes]


def test_perfect_prediction_gives_near_zero_loss():
    target = make_target()
    total, _ = DeepSupervisionDiceBCELoss()(pyramid(target), target)
    assert total.item() < 1e-3


def test_wrong_prediction_gives_high_loss():
    target = make_target()
    total, _ = DeepSupervisionDiceBCELoss()(pyramid(1 - target), target)
    assert total.item() > 5.0  # BCE alone is ~20 per voxel at these logits; Dice loss ~1


def test_breakdown_sums_to_total():
    target = make_target()
    outputs = [torch.randn_like(o) for o in pyramid(target)]
    total, parts = DeepSupervisionDiceBCELoss()(outputs, target)
    by_level = sum(v for k, v in parts.items() if k.startswith("level/"))
    by_region = sum(v for k, v in parts.items() if k.startswith(("dice/", "bce/")))
    assert by_level == pytest.approx(total.item(), rel=1e-5)
    assert by_region == pytest.approx(total.item(), rel=1e-5)
    assert set(parts) >= {"dice/WT", "dice/TC", "dice/ET", "bce/WT", "bce/TC", "bce/ET", "level/0", "level/1", "level/2"}


def test_level_weights_favour_the_final_output():
    target = make_target()
    good = pyramid(target)
    loss = DeepSupervisionDiceBCELoss()
    final_wrong, _ = loss(good[:2] + [logits_from_mask(1 - target)], target)
    coarse_wrong, _ = loss([logits_from_mask(1 - torch.nn.functional.interpolate(target, size=(4,) * 3))] + good[1:], target)
    assert final_wrong > 3 * coarse_wrong  # weight 1.0 vs 0.25


def test_regions_are_independent():
    """An error confined to ET must not change the WT and TC terms."""
    target = make_target()
    outputs = pyramid(target)
    broken = [o.clone() for o in outputs]
    for o in broken:
        o[:, 2] = -o[:, 2]
    _, clean = DeepSupervisionDiceBCELoss()(outputs, target)
    _, parts = DeepSupervisionDiceBCELoss()(broken, target)
    for key in ("dice/WT", "dice/TC", "bce/WT", "bce/TC"):
        assert parts[key] == pytest.approx(clean[key], abs=1e-6)
    assert parts["bce/ET"] > clean["bce/ET"] + 1


def test_inverse_frequency_weights_balance_classes():
    w = inverse_frequency_weights([100, 300])
    assert torch.allclose(w, torch.tensor([2.0, 2 / 3]))
    assert (w * torch.tensor([100.0, 300.0])).sum() == pytest.approx(400.0)  # mean sample weight 1


def test_minority_misclassification_costs_more():
    loss = WeightedCrossEntropyLoss([1112, 1258, 1247, 1258], reduction="sum")  # train counts
    wrong = torch.tensor([[0.0, 2.0, 0.0, 0.0]])  # confidently predicts class 1
    minority = loss(wrong, torch.tensor([0]))  # true class 0 has the fewest samples
    majority_logits = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    majority = loss(majority_logits, torch.tensor([1]))  # same mistake, true class is a majority one
    assert minority > majority
    assert minority / majority == pytest.approx(1258 / 1112, rel=1e-4)
