import pytest
import torch

from src.models.classifier import EfficientNetV2Classifier
from src.models.unet3d import UNet3D, count_parameters

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


def assert_all_gradients_flow(model):
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"
            assert p.grad.abs().sum() > 0, f"zero gradient for {name}"


@cuda
def test_unet_forward_shapes_at_full_patch_size():
    model = UNet3D().cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        outputs = model(torch.randn(1, 4, 128, 128, 128, device="cuda"))
    assert [tuple(o.shape) for o in outputs] == [(1, 3, 32, 32, 32), (1, 3, 64, 64, 64), (1, 3, 128, 128, 128)]


@cuda
def test_unet_backward_reaches_every_parameter():
    torch.manual_seed(0)
    model = UNet3D().cuda().train()
    outputs = model(torch.randn(2, 4, 64, 64, 64, device="cuda"))
    # Dummy loss over all outputs, so the auxiliary heads receive gradients too.
    loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(o, torch.rand_like(o)) for o in outputs)
    loss.backward()
    assert torch.isfinite(loss)
    assert_all_gradients_flow(model)


def test_unet_is_configurable():
    model = UNet3D(widths=(16, 32, 64), bottleneck=128, deep_supervision=1)
    outputs = model(torch.randn(1, 4, 32, 32, 32))
    assert [tuple(o.shape) for o in outputs] == [(1, 3, 16, 16, 16), (1, 3, 32, 32, 32)]
    assert count_parameters(model) < count_parameters(UNet3D())

    plain = UNet3D(widths=(8, 16), bottleneck=16, deep_supervision=0)
    assert tuple(plain(torch.randn(1, 4, 16, 16, 16)).shape) == (1, 3, 16, 16, 16)
    with pytest.raises(ValueError):
        UNet3D(widths=(8, 16), deep_supervision=2)


def test_unet_predict_returns_independent_probabilities():
    model = UNet3D(widths=(8, 16), bottleneck=16, deep_supervision=1).eval()
    probs = model.predict(torch.randn(1, 4, 16, 16, 16))
    assert probs.shape == (1, 3, 16, 16, 16)
    assert probs.min() >= 0 and probs.max() <= 1
    # Sigmoid, not softmax: channels are not constrained to sum to 1.
    assert not torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]))


@cuda
def test_classifier_forward_and_backward():
    model = EfficientNetV2Classifier(pretrained=True).cuda().train()
    logits = model(torch.randn(2, 3, 384, 384, device="cuda"))
    assert tuple(logits.shape) == (2, 4)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3], device="cuda")).backward()
    assert_all_gradients_flow(model)


def test_classifier_parameter_groups_and_freezing():
    model = EfficientNetV2Classifier(pretrained=False)
    backbone, head = model.backbone_parameters(), model.head_parameters()
    assert {id(p) for p in backbone}.isdisjoint(id(p) for p in head)
    assert len(backbone) + len(head) == len(list(model.parameters()))

    groups = model.parameter_groups(backbone_lr=1e-4, head_lr=1e-3)
    assert [(g["name"], g["lr"]) for g in groups] == [("backbone", 1e-4), ("head", 1e-3)]

    frozen = EfficientNetV2Classifier(pretrained=False, freeze_backbone=True)
    assert not any(p.requires_grad for p in frozen.backbone_parameters())
    assert all(p.requires_grad for p in frozen.head_parameters())
    assert [g["name"] for g in frozen.parameter_groups(1e-4, 1e-3)] == ["head"]
    frozen.set_backbone_trainable(True)
    assert all(p.requires_grad for p in frozen.backbone_parameters())
