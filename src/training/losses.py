"""Loss functions for segmentation (multi-label regions) and classification."""
import torch
import torch.nn.functional as F
from torch import nn

REGIONS = ("WT", "TC", "ET")


def soft_dice_loss(probs, target, smooth=1.0):
    """1 - soft Dice per channel, pooled over batch and space: (N, C, ...) -> (C,).

    Pooling over the batch ("batch Dice") keeps the loss defined when a region is
    absent from one sample. smooth=1 makes an absent region with a near-empty
    prediction score close to a perfect Dice instead of 0/0.
    """
    dims = (0, *range(2, probs.ndim))
    intersection = (probs * target).sum(dims)
    denominator = probs.sum(dims) + target.sum(dims)
    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)


class DeepSupervisionDiceBCELoss(nn.Module):
    """Dice + BCE per region (WT, TC, ET), applied to every deep-supervision output.

    Regions are independent sigmoid outputs, so Dice and BCE are computed per
    channel rather than over a softmax. Each supervision level's loss is the
    mean over regions of (dice_weight * Dice loss + bce_weight * BCE); levels are
    combined with level_weights (given lowest resolution first, like the U-Net
    outputs) normalised to sum to 1. The defaults 0.25, 0.5, 1.0 therefore become
    0.14, 0.29, 0.57; normalising keeps the loss on the same scale as a single
    output, so learning rates do not depend on the number of auxiliary heads.

    Targets are downsampled to each output's resolution with nearest-neighbour
    interpolation, which keeps them binary.

    forward() returns (total, breakdown). In the breakdown, the per-region terms
    (dice/<R>, bce/<R>) and the per-level terms (level/<i>) are weighted
    contributions: each family sums to the total.
    """

    def __init__(self, level_weights=(0.25, 0.5, 1.0), dice_weight=1.0, bce_weight=1.0, smooth=1.0):
        super().__init__()
        total = sum(level_weights)
        self.level_weights = [w / total for w in level_weights]
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, outputs, target):
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        if len(outputs) != len(self.level_weights):
            raise ValueError(f"Expected {len(self.level_weights)} outputs, got {len(outputs)}")
        target = target.float()
        n_regions = target.shape[1]

        total = target.new_zeros(())
        breakdown = {f"{kind}/{REGIONS[r]}": 0.0 for kind in ("dice", "bce") for r in range(n_regions)}
        for level, (logits, weight) in enumerate(zip(outputs, self.level_weights)):
            logits = logits.float()  # losses in fp32 under mixed precision
            t = target if logits.shape[2:] == target.shape[2:] else F.interpolate(target, size=logits.shape[2:], mode="nearest-exact")
            dice = soft_dice_loss(torch.sigmoid(logits), t, self.smooth)  # (C,)
            bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none").mean(dim=(0, *range(2, logits.ndim)))
            per_region = self.dice_weight * dice + self.bce_weight * bce
            level_loss = weight * per_region.mean()
            total = total + level_loss
            breakdown[f"level/{level}"] = level_loss.item()
            for r in range(n_regions):
                breakdown[f"dice/{REGIONS[r]}"] += (weight * self.dice_weight * dice[r] / n_regions).item()
                breakdown[f"bce/{REGIONS[r]}"] += (weight * self.bce_weight * bce[r] / n_regions).item()
        breakdown["total"] = total.item()
        return total, breakdown


def inverse_frequency_weights(class_counts):
    """Class weights w_c = N / (K * n_c), so each class contributes equally in expectation.

    Inverse frequency (sklearn's "balanced") is used because the imbalance here
    is mild (largest/smallest about 1.13): it gives weights close to 1 that
    exactly offset the count differences. Smoothed variants such as the
    effective number of samples are meant for long-tailed data and would change
    almost nothing at this ratio. The mean weight over samples is 1, so the loss
    scale matches unweighted cross-entropy.
    """
    counts = torch.as_tensor(list(class_counts), dtype=torch.float32)
    if (counts <= 0).any():
        raise ValueError("Every class needs at least one sample")
    return counts.sum() / (len(counts) * counts)


class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross-entropy with inverse-frequency class weights from training-split counts."""

    def __init__(self, class_counts, **kwargs):
        super().__init__(weight=inverse_frequency_weights(class_counts), **kwargs)
