"""EfficientNetV2-S classifier for 4-class brain MRI (glioma, meningioma, notumor, pituitary).

Wraps torchvision's EfficientNetV2-S with ImageNet weights and replaces the
1000-class head with a dropout + linear layer for the target classes. Expects
3 x 384 x 384 ImageNet-normalised input (see src.data.classification_preprocess).

The backbone and head are exposed as separate parameter groups so a training
script can apply discriminative learning rates, e.g. a small rate for the
pretrained backbone and a larger one for the freshly initialised head.
"""
from torch import nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

NUM_CLASSES = 4


class EfficientNetV2Classifier(nn.Module):
    """
    Args:
        num_classes: output classes.
        pretrained: load ImageNet-1k weights for the backbone.
        freeze_backbone: if True, only the head is trainable.
        dropout: dropout before the final linear layer (torchvision default 0.2).
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False, dropout=0.2):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_v2_s(weights=weights)
        in_features = net.classifier[-1].in_features  # 1280
        self.features = net.features
        self.pool = net.avgpool
        self.head = nn.Sequential(nn.Dropout(dropout, inplace=True), nn.Linear(in_features, num_classes))
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)
        self.set_backbone_trainable(not freeze_backbone)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.head(x)  # logits, shape (N, num_classes)

    def set_backbone_trainable(self, trainable):
        """Freeze or unfreeze the backbone (e.g. head-only warm-up, then full fine-tuning)."""
        for p in self.features.parameters():
            p.requires_grad = trainable

    def backbone_parameters(self):
        return list(self.features.parameters())

    def head_parameters(self):
        return list(self.head.parameters())

    def parameter_groups(self, backbone_lr, head_lr):
        """Optimizer param groups with separate learning rates; frozen parameters are left out."""
        groups = [
            {"name": "backbone", "params": [p for p in self.backbone_parameters() if p.requires_grad], "lr": backbone_lr},
            {"name": "head", "params": self.head_parameters(), "lr": head_lr},
        ]
        return [g for g in groups if g["params"]]
