"""Train the 3D U-Net on one cross-validation fold of BraTS2021.

Folds come from data/manifests/brats_5fold_splits.json (src.data.make_brats_folds):
fold k validates on folds[k] and trains on the other folds; the test cases are
never loaded here. Each BraTS2021 case is a distinct patient, so the folds are
patient-level.

One epoch draws `samples_per_case` random patches (tumour-biased, mirrored)
from every training case. Every `val_every` epochs and after the last epoch the
model segments every validation volume with sliding-window inference and
real Dice scores are computed per region. Metric names separate the two kinds
of number: train/*_loss/* are loss terms, val/dice_score/* are Dice scores.

Checkpoints go to <checkpoint_dir>/fold<k>/ (gitignored): best.pt has the
highest mean validation Dice, last.pt the latest epoch and everything needed
to resume (--resume continues the same wandb run).

Usage:
    python -m src.training.train_segmentation --fold 1
    python -m src.training.train_segmentation --fold 1 --resume
    python -m src.training.train_segmentation --max-cases 10 --epochs 2 --val-every 1   # smoke test
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.data.make_brats_folds import load_fold
from src.data.segmentation_preprocess import BratsPatchDataset
from src.models.unet3d import UNet3D, count_parameters
from src.training.config import SegmentationTrainConfig
from src.training.inference import evaluate_volumes
from src.training.losses import REGIONS, DeepSupervisionDiceBCELoss
from src.training.utils import (StepTimer, gpu_memory, init_wandb, keep_awake, load_checkpoint, log,
                                save_checkpoint, seed_everything, warmup_cosine)

VOLUME_DIR = PROCESSED_DIR / "brats2021_preprocessed"
MODALITIES = ("flair", "t1", "t1ce", "t2")  # channel order written by segmentation_preprocess
# Overlay colours for the nested regions: edema (WT only), necrotic core (TC only), enhancing tumour.
OVERLAY = ListedColormap(["#2ca02c", "#d62728", "#ffd700"])


def load_volume(case):
    with np.load(VOLUME_DIR / f"{case}.npz") as data:
        return data["image"], data["targets"]


def pick_vis_cases(val_ids, manifest, n):
    """n validation cases spread evenly over whole-tumour size (small to large)."""
    voxels = json.loads(Path(manifest).read_text())["case_voxels"]
    ranked = sorted(val_ids, key=lambda c: voxels[c][0])
    if n <= 0:
        return []
    return [ranked[round(i * (len(ranked) - 1) / max(1, n - 1))] for i in range(min(n, len(ranked)))]


def nested_labels(regions):
    """(3, ...) binary WT/TC/ET -> label map: 0 none, 1 edema, 2 necrotic core, 3 enhancing."""
    wt, tc, et = (regions[i] > 0.5 for i in range(3))
    return np.select([et, tc, wt], [3, 2, 1], 0)


def slice_figure(kept, per_case, epoch):
    """One row per case: FLAIR, T1ce, ground truth overlay, prediction overlay, on the axial slice of largest WT."""
    fig, axes = plt.subplots(len(kept), 4, figsize=(12, 3 * len(kept)), squeeze=False)
    for row, (case, (image, targets, probs)) in zip(axes, kept.items()):
        z = int(targets[0].sum(axis=(0, 1)).argmax())
        flair = image[MODALITIES.index("flair"), :, :, z].astype(np.float32).T
        t1ce = image[MODALITIES.index("t1ce"), :, :, z].astype(np.float32).T
        truth = nested_labels(targets[:, :, :, z]).T
        pred = nested_labels(probs[:, :, :, z]).T
        dice = ", ".join(f"{r} {d:.2f}" for r, d in zip(REGIONS, per_case[case]))
        panels = [(flair, None, f"{case}  z={z}\nFLAIR"), (t1ce, None, "T1ce"),
                  (flair, truth, "ground truth"), (flair, pred, f"prediction\n{dice}")]
        for ax, (base, labels, title) in zip(row, panels):
            ax.imshow(base, cmap="gray", origin="lower")
            if labels is not None:
                ax.imshow(np.ma.masked_equal(labels, 0), cmap=OVERLAY, vmin=1, vmax=3, alpha=0.5,
                          origin="lower", interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle(f"epoch {epoch}: green edema, red necrotic core, yellow enhancing tumour", fontsize=9)
    fig.tight_layout()
    return fig


def validate(model, val_ids, vis_ids, cfg, device, epoch, run):
    start = time.perf_counter()
    metrics, per_case, kept = evaluate_volumes(
        model, val_ids, load_volume, device, cfg.patch_size, cfg.sw_overlap, cfg.sw_batch_size,
        cfg.mixed_precision, keep=set(vis_ids))
    metrics = {f"val/{k}": v for k, v in metrics.items()}
    metrics["val/seconds"] = time.perf_counter() - start
    if run is not None and kept:
        import wandb

        fig = slice_figure({c: kept[c] for c in vis_ids if c in kept}, per_case, epoch)
        metrics["val/slices"] = wandb.Image(fig)
        plt.close(fig)
    model.train()
    return metrics, per_case


def train(cfg, max_cases=0, use_wandb=True, run_name=None, resume=False):
    seed_everything(cfg.seed)
    device = torch.device("cuda")

    manifest = PROJECT_ROOT / cfg.folds_manifest
    train_ids, val_ids = load_fold(cfg.fold, manifest)
    if max_cases:
        n_val = max(1, round(max_cases * 0.2))
        train_ids, val_ids = train_ids[: max_cases - n_val], val_ids[:n_val]
    vis_ids = pick_vis_cases(val_ids, manifest, cfg.vis_cases)
    out_dir = PROJECT_ROOT / cfg.checkpoint_dir / (f"fold{cfg.fold}" + ("-smoke" if max_cases else ""))

    train_set = BratsPatchDataset(VOLUME_DIR, train_ids, cfg.patch_size, cfg.tumor_ratio, cfg.samples_per_case,
                                  seed=cfg.seed, flip=cfg.flip)
    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0)
    steps_per_epoch = len(loader) // cfg.grad_accum_steps

    model = UNet3D(widths=cfg.widths, bottleneck=cfg.bottleneck, deep_supervision=cfg.deep_supervision).to(device)
    loss_fn = DeepSupervisionDiceBCELoss(level_weights=cfg.level_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = warmup_cosine(optimizer, cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch,
                              cfg.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)
    timer = StepTimer(cfg.max_step_seconds)

    start_epoch, step, best, run_id = 0, 0, -1.0, None
    if resume:
        state = load_checkpoint(out_dir / "last.pt", device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, step, best, run_id = state["epoch"] + 1, state["step"], state["best_dice"], state["wandb_run_id"]
        print(f"resuming after epoch {state['epoch']} (step {step}, best mean Dice {best:.4f})")

    run = init_wandb(cfg, "train", use_wandb, run_name or f"fold{cfg.fold}", group="unet3d-cv", run_id=run_id,
                     extra_config={"train_cases": len(train_ids), "val_cases": len(val_ids), "vis_cases": vis_ids,
                                   "steps_per_epoch": steps_per_epoch, "parameters": count_parameters(model)})
    print(f"fold {cfg.fold}: train cases {len(train_ids)}, val cases {len(val_ids)}, {steps_per_epoch} optimizer "
          f"steps/epoch, {cfg.epochs} epochs, checkpoints in {out_dir}")

    def checkpoint_state(epoch):
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict(), epoch=epoch, step=step, best_dice=best, config=cfg.to_dict(),
                    fold=cfg.fold, train_cases=train_ids, val_cases=val_ids,
                    wandb_run_id=run.id if run is not None else None)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.perf_counter()
        train_set.set_epoch(epoch)
        accum, parts_sum, epoch_sums, epoch_steps = 0, {}, {}, 0
        for image, target in loader:
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
            scheduler.step()
            seconds = timer.stop()
            step += 1
            epoch_steps += 1
            for k, v in parts_sum.items():
                epoch_sums[k] = epoch_sums.get(k, 0.0) + v
            metrics = {("train/loss" if k == "total" else f"train/{k}"): v for k, v in parts_sum.items()}
            metrics.update({"train/grad_norm": grad_norm.item(), "train/lr": optimizer.param_groups[0]["lr"],
                            "train/step_seconds": seconds, "epoch": epoch, **gpu_memory()})
            log(run, metrics, step)
            if step % 50 == 0:
                print(f"epoch {epoch:3d} step {step:6d}  loss {parts_sum['total']:.4f}  {seconds * 1000:5.0f} ms  "
                      f"peak {metrics['gpu/peak_allocated_gib']:.2f} GiB", flush=True)
            accum, parts_sum = 0, {}

        epoch_metrics = {("epoch/train_loss" if k == "total" else f"epoch/train_{k}"): v / epoch_steps
                         for k, v in epoch_sums.items()}
        epoch_metrics.update({"epoch": epoch, "epoch/train_seconds": time.perf_counter() - epoch_start})
        line = f"epoch {epoch:3d}  train loss {epoch_metrics['epoch/train_loss']:.4f}  " \
               f"{epoch_metrics['epoch/train_seconds'] / 60:.1f} min"

        if (epoch + 1) % cfg.val_every == 0 or epoch + 1 == cfg.epochs:
            val, _ = validate(model, val_ids, vis_ids, cfg, device, epoch, run)
            epoch_metrics.update(val)
            score = val["val/dice_score/mean"]
            if score > best:
                best = score
                save_checkpoint(out_dir / "best.pt", **checkpoint_state(epoch))
            epoch_metrics["val/best_dice_score_mean"] = best
            line += "  val Dice " + "  ".join(f"{r} {val[f'val/dice_score/{r}']:.4f}" for r in REGIONS) + \
                    f"  mean {score:.4f} (best {best:.4f})  {val['val/seconds'] / 60:.1f} min"
        log(run, epoch_metrics, step)
        save_checkpoint(out_dir / "last.pt", **checkpoint_state(epoch))
        print(line, flush=True)

    summary = {"steps": step, "best_dice_score_mean": best, "checkpoint_dir": str(out_dir), **gpu_memory()}
    if run is not None:
        run.summary["best_dice_score_mean"] = best
        summary["wandb_url"] = run.url
        run.finish()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train_segmentation.json")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--val-every", type=int)
    parser.add_argument("--max-cases", type=int, default=0, help="use only N cases in total (smoke tests)")
    parser.add_argument("--resume", action="store_true", help="continue from <checkpoint_dir>/fold<k>/last.pt")
    parser.add_argument("--run-name")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    overrides = {k: v for k, v in {"fold": args.fold, "epochs": args.epochs, "val_every": args.val_every}.items()
                 if v is not None}
    cfg = SegmentationTrainConfig.from_json(args.config, **overrides)
    with keep_awake():
        summary = train(cfg, args.max_cases, not args.no_wandb, args.run_name, args.resume)
    print(f"best mean Dice {summary['best_dice_score_mean']:.4f}, checkpoints in {summary['checkpoint_dir']}")
    if "wandb_url" in summary:
        print("wandb run:", summary["wandb_url"])


if __name__ == "__main__":
    main()
