"""Train the 3D U-Net on preprocessed BraTS2021 volumes (single holdout split).

Each BraTS2021 case is a distinct patient, so a split over case IDs is a
patient-level split. Patches are sampled on the fly (BratsPatchDataset).

Usage:
    python -m src.training.train_segmentation --max-cases 8 --max-steps 25   # smoke test
    python -m src.training.train_segmentation
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.data.segmentation_preprocess import BratsPatchDataset
from src.models.unet3d import UNet3D
from src.training.config import SegmentationTrainConfig
from src.training.losses import DeepSupervisionDiceBCELoss
from src.training.utils import StepTimer, gpu_memory, init_wandb, log, seed_everything

VOLUME_DIR = PROCESSED_DIR / "brats2021_preprocessed"


def holdout_split(case_ids, val_fraction, seed):
    """Deterministic patient-level split: (train_ids, val_ids)."""
    ids = sorted(case_ids)
    random.Random(seed).shuffle(ids)
    n_val = max(1, round(len(ids) * val_fraction))
    return sorted(ids[n_val:]), sorted(ids[:n_val])


def list_cases(volume_dir=VOLUME_DIR):
    index = json.loads((Path(volume_dir) / "index.json").read_text())
    return [item["case"] for item in index["items"]]


def make_loader(cfg, case_ids, shuffle, seed):
    dataset = BratsPatchDataset(VOLUME_DIR, case_ids, cfg.patch_size, cfg.tumor_ratio, seed=seed)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers,
        pin_memory=True, persistent_workers=cfg.num_workers > 0, drop_last=shuffle,
    )
    return dataset, loader


@torch.no_grad()
def evaluate(model, loader, loss_fn, cfg, device, max_batches=None):
    model.eval()
    totals, n = {}, 0
    for i, (image, target) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.mixed_precision):
            outputs = model(image.to(device, non_blocking=True))
        _, parts = loss_fn(outputs, target.to(device, non_blocking=True))
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    model.train()
    return {f"val/{k}": v / n for k, v in totals.items()}


def train(cfg, max_cases=0, max_steps=None, max_val_batches=None, use_wandb=True, run_name=None):
    seed_everything(cfg.seed)
    device = torch.device("cuda")

    cases = list_cases()
    train_ids, val_ids = holdout_split(cases, cfg.val_fraction, cfg.seed)
    if max_cases:
        n_val = max(1, round(max_cases * cfg.val_fraction))
        train_ids, val_ids = train_ids[: max_cases - n_val], val_ids[:n_val]
    train_set, train_loader = make_loader(cfg, train_ids, shuffle=True, seed=cfg.seed)
    _, val_loader = make_loader(cfg, val_ids, shuffle=False, seed=cfg.seed + 1)

    model = UNet3D(widths=cfg.widths, bottleneck=cfg.bottleneck, deep_supervision=cfg.deep_supervision).to(device)
    loss_fn = DeepSupervisionDiceBCELoss(level_weights=cfg.level_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)
    timer = StepTimer(cfg.max_step_seconds)

    run = init_wandb(cfg, "train", use_wandb, run_name, {
        "train_cases": len(train_ids), "val_cases": len(val_ids), "max_steps": max_steps,
    })
    print(f"train cases {len(train_ids)}, val cases {len(val_ids)}, effective batch {cfg.effective_batch_size}")

    torch.cuda.reset_peak_memory_stats()
    step, history = 0, []
    model.train()
    for epoch in range(cfg.epochs if max_steps is None else 10**9):
        train_set.set_epoch(epoch)
        accum, parts_sum = 0, {}
        for image, target in train_loader:
            if accum == 0:
                timer.start()
                optimizer.zero_grad(set_to_none=True)
            image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.mixed_precision):
                outputs = model(image)
            loss, parts = loss_fn(outputs, target)
            scaler.scale(loss / cfg.grad_accum_steps).backward()
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v / cfg.grad_accum_steps
            accum += 1
            if accum < cfg.grad_accum_steps:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            seconds = timer.stop()
            step += 1
            history.append(parts_sum["total"])
            metrics = {f"train/{k}": v for k, v in parts_sum.items()}
            metrics.update({"train/grad_norm": grad_norm.item(), "train/lr": optimizer.param_groups[0]["lr"],
                            "train/step_seconds": seconds, "epoch": epoch, **gpu_memory()})
            log(run, metrics, step)
            print(f"step {step:4d}  loss {parts_sum['total']:.4f}  {seconds * 1000:5.0f} ms  "
                  f"peak {metrics['gpu/peak_allocated_gib']:.2f} GiB", flush=True)
            accum, parts_sum = 0, {}
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    val = evaluate(model, val_loader, loss_fn, cfg, device, max_val_batches)
    log(run, val, step)
    print("validation:", {k: round(v, 4) for k, v in val.items() if k in ("val/total", "val/dice/WT", "val/dice/TC", "val/dice/ET")})
    summary = {"steps": step, "loss_history": history, **val, **gpu_memory()}
    if run is not None:
        summary["wandb_url"] = run.url
        run.finish()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train_segmentation.json")
    parser.add_argument("--max-cases", type=int, default=0, help="use only N cases in total (smoke tests)")
    parser.add_argument("--max-steps", type=int, help="stop after N optimizer steps")
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = SegmentationTrainConfig.from_json(args.config)
    summary = train(cfg, args.max_cases, args.max_steps, args.max_val_batches, not args.no_wandb, args.run_name)
    if "wandb_url" in summary:
        print("wandb run:", summary["wandb_url"])


if __name__ == "__main__":
    main()
