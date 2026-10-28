"""Report parameter counts, peak VRAM and step time for both models on the GPU.

For the U-Net, measures inference (forward only) and a training step
(forward + backward with a dummy loss over all deep-supervision outputs),
in fp32 and with fp16 autocast (mixed precision), at several batch sizes.

On Windows the NVIDIA driver can spill into shared system memory instead of
raising out-of-memory, which makes training very slow without failing. Peaks
are therefore compared with the VRAM free when the run started.

Usage:
    python -m src.models.profile_models
"""
import argparse
import time

import torch

from src.models.classifier import EfficientNetV2Classifier
from src.models.unet3d import UNet3D, count_parameters

GIB = 2**30


def measure(fn, device, repeats=3):
    """Run fn once to warm up, then report peak allocated/reserved memory and mean time."""
    torch.cuda.empty_cache()
    fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize(device)
    return {
        "allocated": torch.cuda.max_memory_allocated(device) / GIB,
        "reserved": torch.cuda.max_memory_reserved(device) / GIB,
        "seconds": (time.perf_counter() - start) / repeats,
    }


def unet_step(model, x, train, amp):
    def step():
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp), torch.set_grad_enabled(train):
            outputs = model(x)
            if train:
                loss = sum(o.float().mean() for o in outputs)
        if train:
            loss.backward()
    return step


def classifier_step(model, x, amp):
    def step():
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model(x).float().mean()
        loss.backward()
    return step


def report(label, fn, device, free_gib):
    try:
        r = measure(fn, device)
    except torch.cuda.OutOfMemoryError:
        print(f"  {label:<42} out of memory")
        torch.cuda.empty_cache()
        return
    flag = "  EXCEEDS FREE VRAM" if r["reserved"] > free_gib else ""
    print(f"  {label:<42} peak {r['allocated']:5.2f} GiB (reserved {r['reserved']:5.2f})  {r['seconds'] * 1000:7.0f} ms{flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--unet-batches", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--classifier-batches", type=int, nargs="+", default=[16, 32])
    args = parser.parse_args()

    device = torch.device("cuda")
    free, total = (v / GIB for v in torch.cuda.mem_get_info(device))
    print(f"GPU: {torch.cuda.get_device_name(device)}  total {total:.2f} GiB, free at start {free:.2f} GiB\n")

    unet = UNet3D().to(device)
    print(f"UNet3D: {count_parameters(unet):,} parameters ({count_parameters(unet) * 4 / 2**20:.1f} MiB fp32)")
    for bs in args.unet_batches:
        x = torch.randn(bs, 4, args.patch, args.patch, args.patch, device=device)
        unet.eval()
        report(f"inference      bs={bs} fp32", unet_step(unet, x, False, False), device, free)
        report(f"inference      bs={bs} mixed precision", unet_step(unet, x, False, True), device, free)
        unet.train()
        report(f"training step  bs={bs} fp32", unet_step(unet, x, True, False), device, free)
        report(f"training step  bs={bs} mixed precision", unet_step(unet, x, True, True), device, free)
        del x
    del unet
    torch.cuda.empty_cache()

    clf = EfficientNetV2Classifier().to(device).train()
    n_backbone = sum(p.numel() for p in clf.backbone_parameters())
    n_head = sum(p.numel() for p in clf.head_parameters())
    print(f"\nEfficientNetV2Classifier: {count_parameters(clf):,} parameters (backbone {n_backbone:,}, head {n_head:,})")
    for bs in args.classifier_batches:
        x = torch.randn(bs, 3, 384, 384, device=device)
        report(f"training step  bs={bs} fp32", classifier_step(clf, x, False), device, free)
        report(f"training step  bs={bs} mixed precision", classifier_step(clf, x, True), device, free)
        del x


if __name__ == "__main__":
    main()
