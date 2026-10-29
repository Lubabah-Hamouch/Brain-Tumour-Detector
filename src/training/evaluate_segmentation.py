"""Held-out test evaluation of the five cross-validation U-Nets on BraTS2021.

Two stages, run in this order so the test set influences no decision:

1. tune (validation data only): every fold model segments its own validation
   fold, giving out-of-fold predictions for all 1,062 non-test cases. Predicted
   ET components are split into true positives (overlapping ground-truth ET)
   and pure false positives, and the ET size threshold for post-processing is
   the candidate with the highest mean out-of-fold ET Dice.

2. test: each fold model and the 5-fold ensemble (mean of the five sigmoid
   probability maps, thresholded at 0.5) segment the 189 test cases. Dice per
   region is reported for every model with and without ET post-processing.
   The single best model is chosen beforehand by cross-validation score.

Outputs (summary JSON and per-case CSV) go to reports/results/; a wandb run in
the segmentation project holds the same tables plus test-case slice overlays.

Usage:
    python -m src.training.evaluate_segmentation
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.config import PROJECT_ROOT
from src.data.make_brats_folds import MANIFEST
from src.models.unet3d import UNet3D
from src.training.inference import dice_scores, prefetch, sliding_window_predict
from src.training.losses import REGIONS
from src.training.postprocess import component_stats, dice_after_removal, remove_small_components
from src.training.train_segmentation import load_volume, pick_vis_cases, slice_figure
from src.training.utils import keep_awake, load_checkpoint

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
CANDIDATE_THRESHOLDS = (0, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000)
ET = REGIONS.index("ET")


def load_models(checkpoint_dir, folds, device):
    models, cv_scores = {}, {}
    for k in folds:
        state = load_checkpoint(Path(checkpoint_dir) / f"fold{k}" / "best.pt", device)
        cfg = state["config"]
        model = UNet3D(widths=tuple(cfg["widths"]), bottleneck=cfg["bottleneck"],
                       deep_supervision=cfg["deep_supervision"]).to(device)
        model.load_state_dict(state["model"])
        models[k] = model.eval()
        cv_scores[k] = state["best_dice"]
    return models, cv_scores, cfg


def predict(model, image, cfg):
    return sliding_window_predict(model, image, cfg["patch_size"], cfg["sw_overlap"], cfg["sw_batch_size"],
                                  cfg["mixed_precision"])


def to_device(image, targets, device):
    return (torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).to(device),
            torch.from_numpy(np.ascontiguousarray(targets)).to(device))


def tune_threshold(models, manifest, cfg, device):
    """Out-of-fold ET component statistics and the ET size threshold that maximises mean ET Dice."""
    folds = manifest["folds"]
    cases = [(k, c) for k in models for c in folds[str(k)]]
    per_case = []
    start = time.perf_counter()
    for i, ((k, case), (image, targets)) in enumerate(prefetch(lambda kc: load_volume(kc[1]), cases)):
        image_t, _ = to_device(image, targets, device)
        et_pred = (predict(models[k], image_t, cfg)[ET] > 0.5).cpu().numpy()
        et_true = targets[ET].astype(bool)
        per_case.append({"case": case, "fold": k, "target": int(et_true.sum()),
                         "stats": component_stats(et_pred, et_true)})
        if (i + 1) % 100 == 0:
            print(f"  tune {i + 1}/{len(cases)}  {time.perf_counter() - start:.0f}s", flush=True)

    curve = {t: float(np.mean([dice_after_removal(c["stats"], c["target"], t) for c in per_case]))
             for t in CANDIDATE_THRESHOLDS}
    best = max(curve.values())
    threshold = min(t for t, d in curve.items() if d >= best - 1e-9)  # smallest threshold at the maximum

    all_stats = np.concatenate([c["stats"] for c in per_case if len(c["stats"])])
    fp_sizes, tp_sizes = all_stats[all_stats[:, 1] == 0, 0], all_stats[all_stats[:, 1] > 0, 0]
    pct = (10, 25, 50, 75, 90)
    no_et = [c for c in per_case if c["target"] == 0]
    report = {
        "cases": len(per_case),
        "threshold": threshold,
        "mean_et_dice_by_threshold": {str(t): round(d, 5) for t, d in curve.items()},
        "false_positive_components": {"count": int(len(fp_sizes)),
                                      "size_percentiles": {str(p): float(np.percentile(fp_sizes, p)) for p in pct}},
        "true_positive_components": {"count": int(len(tp_sizes)),
                                     "size_percentiles": {str(p): float(np.percentile(tp_sizes, p)) for p in pct}},
        "true_positive_components_removed_at_threshold": int((tp_sizes < threshold).sum()),
        "false_positive_components_removed_at_threshold": int((fp_sizes < threshold).sum()),
        "cases_without_et": len(no_et),
        "cases_without_et_scored_1": {
            "raw": sum(dice_after_removal(c["stats"], 0, 0) == 1 for c in no_et),
            "post_processed": sum(dice_after_removal(c["stats"], 0, threshold) == 1 for c in no_et)},
    }
    return report, fp_sizes, tp_sizes


def evaluate_test(models, test_ids, threshold, cfg, device, vis_ids):
    """Per-case Dice for every fold model and the ensemble, raw and with ET post-processing."""
    names = [f"fold{k}" for k in models] + ["ensemble"]
    rows, kept = [], {}
    start = time.perf_counter()
    for i, (case, (image, targets)) in enumerate(prefetch(load_volume, test_ids)):
        image_t, target_t = to_device(image, targets, device)
        et_true = targets[ET].astype(bool)
        ensemble = None
        for name in names:
            if name == "ensemble":
                probs = ensemble / len(models)
            else:
                probs = predict(models[int(name[4:])], image_t, cfg)
                ensemble = probs if ensemble is None else ensemble + probs
            wt, tc, et = dice_scores(probs > 0.5, target_t)
            et_pp = dice_after_removal(component_stats((probs[ET] > 0.5).cpu().numpy(), et_true),
                                       int(et_true.sum()), threshold)
            rows.append({"case": case, "model": name, "WT": wt, "TC": tc, "ET": et, "ET_pp": et_pp})
            if name == "ensemble" and case in vis_ids:
                probs_np = probs.cpu().numpy()
                probs_np[ET] = remove_small_components(probs_np[ET] > 0.5, threshold)
                kept[case] = (image, targets, probs_np)
        if (i + 1) % 25 == 0:
            print(f"  test {i + 1}/{len(test_ids)}  {time.perf_counter() - start:.0f}s", flush=True)
    return rows, kept


def summarise(rows, names):
    summary = {}
    for name in names:
        r = [x for x in rows if x["model"] == name]
        wt, tc, et, et_pp = (float(np.mean([x[k] for x in r])) for k in ("WT", "TC", "ET", "ET_pp"))
        summary[name] = {
            "raw": {"WT": wt, "TC": tc, "ET": et, "mean": (wt + tc + et) / 3},
            "post_processed": {"WT": wt, "TC": tc, "ET": et_pp, "mean": (wt + tc + et_pp) / 3},
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "models" / "segmentation")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--vis-cases", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=0, help="limit each validation fold and the test set (smoke tests)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    manifest = json.loads(MANIFEST.read_text())
    test_ids = manifest["test"]
    if args.max_cases:
        test_ids = test_ids[: args.max_cases]
        manifest["folds"] = {k: v[: args.max_cases] for k, v in manifest["folds"].items()}
    models, cv_scores, cfg = load_models(args.checkpoint_dir, args.folds, device)
    best_fold = max(cv_scores, key=cv_scores.get)
    print(f"loaded folds {list(models)}; best by cross-validation: fold {best_fold} ({cv_scores[best_fold]:.4f})")
    results_dir = RESULTS_DIR / "smoke" if args.max_cases else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    with keep_awake():
        print("stage 1: ET threshold from out-of-fold validation predictions")
        tuning, fp_sizes, tp_sizes = tune_threshold(models, manifest, cfg, device)
        threshold = tuning["threshold"]
        print(f"  ET threshold {threshold} voxels; mean OOF ET Dice by threshold {tuning['mean_et_dice_by_threshold']}")

        print(f"stage 2: test set ({len(test_ids)} cases)")
        vis_ids = pick_vis_cases(test_ids, MANIFEST, args.vis_cases)
        rows, kept = evaluate_test(models, test_ids, threshold, cfg, device, vis_ids)

    names = [f"fold{k}" for k in models] + ["ensemble"]
    summary = summarise(rows, names)
    result = {"test_cases": len(test_ids), "cv_best_dice": {f"fold{k}": v for k, v in cv_scores.items()},
              "single_best_model": f"fold{best_fold}", "et_threshold_voxels": threshold,
              "et_threshold_tuning": tuning, "test": summary}
    (results_dir / "segmentation_test.json").write_text(json.dumps(result, indent=1) + "\n")
    with open(results_dir / "segmentation_test_per_case.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "model", "WT", "TC", "ET", "ET_pp"])
        writer.writeheader()
        writer.writerows({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    print(f"\n{'model':<10} {'post-proc':<9} {'WT':>7} {'TC':>7} {'ET':>7} {'mean':>7}")
    for name in names:
        for kind in ("raw", "post_processed"):
            s = summary[name][kind]
            print(f"{name:<10} {'yes' if kind != 'raw' else 'no':<9} " + " ".join(f"{s[k]:7.4f}" for k in ("WT", "TC", "ET", "mean")))

    if not args.no_wandb:
        import matplotlib.pyplot as plt
        import wandb

        run = wandb.init(project=cfg["wandb_project"], job_type="eval", name="test-eval", group="unet3d-cv",
                         config={"folds": list(models), "test_cases": len(test_ids), "et_threshold_voxels": threshold,
                                 "single_best_model": f"fold{best_fold}", "candidate_thresholds": CANDIDATE_THRESHOLDS})
        table = wandb.Table(columns=["model", "post_processing", "WT", "TC", "ET", "mean"])
        for name in names:
            for kind in ("raw", "post_processed"):
                s = summary[name][kind]
                table.add_data(name, kind != "raw", s["WT"], s["TC"], s["ET"], s["mean"])
                run.summary.update({f"test/{name}/{kind}/dice_score/{r}": s[r] for r in ("WT", "TC", "ET", "mean")})
        curve = wandb.Table(columns=["min_component_voxels", "mean_oof_et_dice"],
                            data=[[int(t), d] for t, d in tuning["mean_et_dice_by_threshold"].items()])
        bins = np.logspace(0, np.log10(max(fp_sizes.max(), tp_sizes.max()) + 1), 40)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(tp_sizes, bins=bins, alpha=0.6, label=f"overlaps true ET ({len(tp_sizes)})")
        ax.hist(fp_sizes, bins=bins, alpha=0.6, label=f"false positive ({len(fp_sizes)})")
        ax.axvline(threshold, color="k", linestyle="--", label=f"threshold {threshold}")
        ax.set(xscale="log", xlabel="predicted ET component size (voxels)", ylabel="components",
               title="Out-of-fold predicted ET components")
        ax.legend()
        per_case = {r["case"]: [r["WT"], r["TC"], r["ET_pp"]] for r in rows if r["model"] == "ensemble"}
        slices = slice_figure({c: kept[c] for c in vis_ids}, per_case, "test set, 5-fold ensemble + ET post-processing")
        run.log({"test/results": table,
                 "tuning/et_threshold_curve": wandb.plot.line(curve, "min_component_voxels", "mean_oof_et_dice",
                                                              title="Mean out-of-fold ET Dice vs threshold"),
                 "tuning/et_component_sizes": wandb.Image(fig), "test/slices": wandb.Image(slices)})
        plt.close(fig)
        plt.close(slices)
        print("wandb run:", run.url)
        run.finish()


if __name__ == "__main__":
    main()
