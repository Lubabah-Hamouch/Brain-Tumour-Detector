"""Fine-tune EfficientNetV2-S on the cleaned 4-class MRI split.

Uses the existing cluster-aware train/val split (data/manifests): the source
dataset has no patient IDs, so near-duplicate clusters stand in for patients
and never cross splits.

Usage:
    python -m src.training.train_classification --max-train 96 --max-val 48 --max-steps 20   # smoke test
    python -m src.training.train_classification
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.config import PROJECT_ROOT
from src.data.classification_preprocess import CLASSES, BrainMRIClassificationDataset
from src.models.classifier import EfficientNetV2Classifier
from src.training.config import ClassificationTrainConfig
from src.training.losses import WeightedCrossEntropyLoss, inverse_frequency_weights
from src.training.utils import StepTimer, gpu_memory, init_wandb, log, seed_everything


def subset(dataset, n, seed):
    if not n or n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    return Subset(dataset, torch.randperm(len(dataset), generator=g)[:n].tolist())


@torch.no_grad()
def evaluate(model, loader, loss_fn, cfg, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for image, label in loader:
        image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.mixed_precision):
            logits = model(image)
        loss_sum += loss_fn(logits.float(), label).item() * len(label)
        correct += (logits.argmax(1) == label).sum().item()
        n += len(label)
    model.train()
    return {"val/loss": loss_sum / n, "val/accuracy": correct / n}


def train(cfg, max_train=0, max_val=0, max_steps=None, use_wandb=True, run_name=None):
    seed_everything(cfg.seed)
    device = torch.device("cuda")

    full_train = BrainMRIClassificationDataset("train", image_size=cfg.image_size)
    # Class weights always come from the full training split, not a smoke-test subset.
    class_counts = list(full_train.class_counts().values())
    train_set = subset(full_train, max_train, cfg.seed)
    val_set = subset(BrainMRIClassificationDataset("val", image_size=cfg.image_size), max_val, cfg.seed)
    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True,
                       persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = EfficientNetV2Classifier(pretrained=cfg.pretrained, freeze_backbone=cfg.freeze_backbone).to(device)
    loss_fn = WeightedCrossEntropyLoss(class_counts).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(cfg.backbone_lr, cfg.head_lr), weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)
    timer = StepTimer(cfg.max_step_seconds)

    weights = inverse_frequency_weights(class_counts).tolist()
    run = init_wandb(cfg, "train", use_wandb, run_name, {
        "train_images": len(train_set), "val_images": len(val_set), "max_steps": max_steps,
        "class_weights": dict(zip(CLASSES, weights)),
    })
    print(f"train images {len(train_set)}, val images {len(val_set)}, class weights "
          + ", ".join(f"{c} {w:.3f}" for c, w in zip(CLASSES, weights)))

    torch.cuda.reset_peak_memory_stats()
    step, history = 0, []
    model.train()
    for epoch in range(cfg.epochs if max_steps is None else 10**9):
        accum, loss_sum, correct, seen = 0, 0.0, 0, 0
        for image, label in train_loader:
            if accum == 0:
                timer.start()
                optimizer.zero_grad(set_to_none=True)
            image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.mixed_precision):
                logits = model(image)
            loss = loss_fn(logits.float(), label)
            scaler.scale(loss / cfg.grad_accum_steps).backward()
            loss_sum += loss.item() / cfg.grad_accum_steps
            correct += (logits.argmax(1) == label).sum().item()
            seen += len(label)
            accum += 1
            if accum < cfg.grad_accum_steps:
                continue

            scaler.step(optimizer)
            scaler.update()
            seconds = timer.stop()
            step += 1
            history.append(loss_sum)
            metrics = {"train/loss": loss_sum, "train/accuracy": correct / seen, "train/step_seconds": seconds,
                       "epoch": epoch, **gpu_memory()}
            metrics.update({f"train/lr_{g['name']}": g["lr"] for g in optimizer.param_groups})
            log(run, metrics, step)
            print(f"step {step:4d}  loss {loss_sum:.4f}  acc {correct / seen:.2f}  {seconds * 1000:5.0f} ms  "
                  f"peak {metrics['gpu/peak_allocated_gib']:.2f} GiB", flush=True)
            accum, loss_sum, correct, seen = 0, 0.0, 0, 0
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    val = evaluate(model, val_loader, loss_fn, cfg, device)
    log(run, val, step)
    print("validation:", {k: round(v, 4) for k, v in val.items()})
    summary = {"steps": step, "loss_history": history, **val, **gpu_memory()}
    if run is not None:
        summary["wandb_url"] = run.url
        run.finish()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train_classification.json")
    parser.add_argument("--max-train", type=int, default=0, help="random subset of N training images")
    parser.add_argument("--max-val", type=int, default=0, help="random subset of N validation images")
    parser.add_argument("--max-steps", type=int, help="stop after N optimizer steps")
    parser.add_argument("--run-name")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = ClassificationTrainConfig.from_json(args.config)
    summary = train(cfg, args.max_train, args.max_val, args.max_steps, not args.no_wandb, args.run_name)
    if "wandb_url" in summary:
        print("wandb run:", summary["wandb_url"])


if __name__ == "__main__":
    main()
