"""Fine-tune EfficientNetV2-S on the cleaned 4-class MRI split.

Uses the existing cluster-aware train/val split (data/manifests): the source
dataset has no patient IDs, so near-duplicate clusters stand in for patients
and never cross splits. The test split is not loaded here.

Each epoch trains on the full training split with on-the-fly augmentation
(the pre-augmented copies in the source dataset were removed during cleaning),
then evaluates on the full validation split: loss, accuracy, per-class
precision/recall/F1, macro F1 and the confusion matrix. Misclassified
validation images are logged every `misclassified_every` epochs.

Training stops early when validation loss has not improved for `patience`
epochs. The checkpoint kept (<checkpoint_dir>/best.pt, gitignored) is the
epoch with the best validation macro F1, which weighs every class equally.

Usage:
    python -m src.training.train_classification
    python -m src.training.train_classification --max-train 96 --max-val 48 --epochs 2   # smoke test
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.config import PROJECT_ROOT
from src.data.classification_preprocess import CLASSES, BrainMRIClassificationDataset
from src.models.classifier import EfficientNetV2Classifier
from src.training.config import ClassificationTrainConfig
from src.training.losses import WeightedCrossEntropyLoss, inverse_frequency_weights
from src.training.utils import (StepTimer, gpu_memory, init_wandb, keep_awake, log, save_checkpoint,
                                seed_everything, warmup_cosine)


def train_augmentation():
    """Mild geometric and intensity changes that keep the anatomy plausible.

    Left-right flips are valid for axial brain MRI. Rotations, shifts and
    zoom mimic head position and field-of-view differences; brightness and
    contrast jitter mimics scanner differences. Padding is black, matching
    the letterbox background.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
    ])


def subset(dataset, n, seed):
    if not n or n >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    return Subset(dataset, torch.randperm(len(dataset), generator=g)[:n].tolist())


def base_dataset(dataset):
    """(dataset, index map) so a Subset index can be traced back to the source image."""
    if isinstance(dataset, Subset):
        return dataset.dataset, list(dataset.indices)
    return dataset, list(range(len(dataset)))


def classification_metrics(labels, preds, n_classes):
    """Confusion matrix (rows true, columns predicted), per-class precision/recall/F1, macro F1."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(n_classes), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(n_classes), where=cm.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n_classes),
                   where=(precision + recall) > 0)
    return cm, precision, recall, f1


@torch.no_grad()
def evaluate(model, loader, loss_fn, cfg, device):
    """Validation loss and metrics, plus per-sample (label, prediction, confidence) arrays."""
    model.eval()
    loss_sum, labels, preds, confidence = 0.0, [], [], []
    for image, label in loader:
        image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.mixed_precision):
            logits = model(image)
        logits = logits.float()
        loss_sum += loss_fn(logits, label).item() * len(label)
        probs = logits.softmax(1)
        labels.append(label.cpu())
        preds.append(probs.argmax(1).cpu())
        confidence.append(probs.max(1).values.cpu())
    model.train()
    labels, preds, confidence = (torch.cat(x).numpy() for x in (labels, preds, confidence))
    cm, precision, recall, f1 = classification_metrics(labels, preds, len(CLASSES))
    metrics = {"val/loss": loss_sum / len(labels), "val/accuracy": float((labels == preds).mean()),
               "val/macro_f1": float(f1.mean())}
    for i, c in enumerate(CLASSES):
        metrics.update({f"val/precision/{c}": precision[i], f"val/recall/{c}": recall[i], f"val/f1/{c}": f1[i]})
    return metrics, cm, labels, preds, confidence


def misclassified_table(dataset, labels, preds, confidence, limit):
    """wandb.Table of the most confidently wrong validation images (the most informative errors)."""
    import wandb

    source, index_map = base_dataset(dataset)
    wrong = np.flatnonzero(labels != preds)
    wrong = wrong[np.argsort(-confidence[wrong])][:limit]
    table = wandb.Table(columns=["image", "file", "true", "predicted", "confidence"])
    for i in wrong:
        j = index_map[i]
        table.add_data(wandb.Image(source.load_image(j)), source.samples[j][0].name, CLASSES[labels[i]],
                       CLASSES[preds[i]], float(confidence[i]))
    return table


def train(cfg, max_train=0, max_val=0, use_wandb=True, run_name=None):
    seed_everything(cfg.seed)
    device = torch.device("cuda")

    augment = train_augmentation() if cfg.augment else None
    full_train = BrainMRIClassificationDataset("train", image_size=cfg.image_size, augment=augment)
    # Class weights always come from the full training split, not a smoke-test subset.
    class_counts = list(full_train.class_counts().values())
    train_set = subset(full_train, max_train, cfg.seed)
    val_set = subset(BrainMRIClassificationDataset("val", image_size=cfg.image_size), max_val, cfg.seed)
    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True,
                       persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    out_dir = PROJECT_ROOT / cfg.checkpoint_dir / ("smoke" if max_train or max_val else "")

    model = EfficientNetV2Classifier(pretrained=cfg.pretrained, freeze_backbone=cfg.freeze_backbone).to(device)
    loss_fn = WeightedCrossEntropyLoss(class_counts).to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(cfg.backbone_lr, cfg.head_lr), weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine(optimizer, cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch,
                              cfg.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)
    timer = StepTimer(cfg.max_step_seconds, patience=cfg.slow_step_patience)

    weights = inverse_frequency_weights(class_counts).tolist()
    run = init_wandb(cfg, "train", use_wandb, run_name, extra_config={
        "train_images": len(train_set), "val_images": len(val_set), "steps_per_epoch": steps_per_epoch,
        "class_weights": dict(zip(CLASSES, weights)),
    })
    print(f"train images {len(train_set)}, val images {len(val_set)}, {steps_per_epoch} steps/epoch, "
          f"up to {cfg.epochs} epochs, class weights " + ", ".join(f"{c} {w:.3f}" for c, w in zip(CLASSES, weights)))

    torch.cuda.reset_peak_memory_stats()
    step, best_f1, best_loss, stale, best_epoch = 0, -1.0, float("inf"), 0, -1
    model.train()
    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        accum, loss_sum, correct, seen = 0, 0.0, 0, 0
        epoch_loss, epoch_correct, epoch_seen = 0.0, 0, 0
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
            scheduler.step()
            seconds = timer.stop()
            step += 1
            epoch_loss += loss_sum * seen
            epoch_correct += correct
            epoch_seen += seen
            metrics = {"train/loss": loss_sum, "train/accuracy": correct / seen, "train/step_seconds": seconds,
                       "epoch": epoch, **gpu_memory()}
            metrics.update({f"train/lr_{g['name']}": g["lr"] for g in optimizer.param_groups})
            log(run, metrics, step)
            accum, loss_sum, correct, seen = 0, 0.0, 0, 0

        train_seconds = time.perf_counter() - epoch_start
        val, cm, labels, preds, confidence = evaluate(model, val_loader, loss_fn, cfg, device)
        epoch_metrics = {"epoch": epoch, "epoch/train_loss": epoch_loss / epoch_seen,
                         "epoch/train_accuracy": epoch_correct / epoch_seen, "epoch/train_seconds": train_seconds,
                         **val}

        improved_f1 = val["val/macro_f1"] > best_f1
        if improved_f1:
            best_f1, best_epoch = val["val/macro_f1"], epoch
            save_checkpoint(out_dir / "best.pt", model=model.state_dict(), epoch=epoch, metrics=val,
                            confusion_matrix=cm.tolist(), classes=list(CLASSES), config=cfg.to_dict())
        if val["val/loss"] < best_loss:
            best_loss, stale = val["val/loss"], 0
        else:
            stale += 1
        epoch_metrics.update({"val/best_macro_f1": best_f1, "val/best_loss": best_loss, "val/epochs_without_improvement": stale})

        if run is not None:
            import wandb

            epoch_metrics["val/confusion_matrix"] = wandb.plot.confusion_matrix(
                y_true=labels.tolist(), preds=preds.tolist(), class_names=list(CLASSES),
                title=f"Validation confusion matrix, epoch {epoch}")
            if epoch % cfg.misclassified_every == 0 or stale >= cfg.patience or epoch + 1 == cfg.epochs:
                epoch_metrics["val/misclassified"] = misclassified_table(val_set, labels, preds, confidence,
                                                                          cfg.misclassified_max)
        log(run, epoch_metrics, step)
        print(f"epoch {epoch:2d}  train loss {epoch_metrics['epoch/train_loss']:.4f} acc "
              f"{epoch_metrics['epoch/train_accuracy']:.3f}  |  val loss {val['val/loss']:.4f} acc "
              f"{val['val/accuracy']:.4f} macro F1 {val['val/macro_f1']:.4f}{'  *saved' if improved_f1 else ''}  "
              f"{(time.perf_counter() - epoch_start) / 60:.1f} min", flush=True)
        if stale >= cfg.patience:
            print(f"early stop: validation loss has not improved for {stale} epochs (best {best_loss:.4f})")
            break

    summary = {"steps": step, "epochs_run": epoch + 1, "best_epoch": best_epoch, "best_macro_f1": best_f1,
               "best_val_loss": best_loss, "checkpoint": str(out_dir / "best.pt"), **gpu_memory()}
    if run is not None:
        run.summary.update({"best_macro_f1": best_f1, "best_epoch": best_epoch, "best_val_loss": best_loss})
        summary["wandb_url"] = run.url
        run.finish()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train_classification.json")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train", type=int, default=0, help="random subset of N training images (smoke tests)")
    parser.add_argument("--max-val", type=int, default=0, help="random subset of N validation images (smoke tests)")
    parser.add_argument("--run-name")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = ClassificationTrainConfig.from_json(args.config, **({"epochs": args.epochs} if args.epochs else {}))
    with keep_awake():
        summary = train(cfg, args.max_train, args.max_val, not args.no_wandb, args.run_name)
    print(f"best macro F1 {summary['best_macro_f1']:.4f} at epoch {summary['best_epoch']}, "
          f"checkpoint {summary['checkpoint']}")
    if "wandb_url" in summary:
        print("wandb run:", summary["wandb_url"])


if __name__ == "__main__":
    main()
