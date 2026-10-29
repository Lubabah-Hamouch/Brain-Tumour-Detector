"""Held-out test evaluation of the EfficientNetV2-S classifier.

Loads the checkpoint chosen by validation macro F1 (models/classification/best.pt)
and evaluates it once on the test split of the cleaned 4-class dataset, which
was not used for training, early stopping or checkpoint selection.

Reports accuracy, macro F1, per-class precision/recall/F1 and the confusion
matrix; writes reports/results/classification_test.json and logs a wandb run
with the confusion matrix and every misclassified test image.

Usage:
    python -m src.training.evaluate_classification
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT
from src.data.classification_preprocess import CLASSES, BrainMRIClassificationDataset
from src.models.classifier import EfficientNetV2Classifier
from src.training.config import ClassificationTrainConfig
from src.training.evaluate_segmentation import RESULTS_DIR
from src.training.losses import WeightedCrossEntropyLoss
from src.training.train_classification import evaluate, misclassified_table
from src.training.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "classification" / "best.pt")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    state = load_checkpoint(args.checkpoint, device)
    cfg = ClassificationTrainConfig(**{k: v for k, v in state["config"].items() if k != "effective_batch_size"})
    model = EfficientNetV2Classifier(pretrained=False).to(device)
    model.load_state_dict(state["model"])

    test_set = BrainMRIClassificationDataset("test", image_size=cfg.image_size)
    loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True)
    # Same loss weighting as training (training-split counts), so the test loss is comparable.
    train_counts = list(BrainMRIClassificationDataset("train").class_counts().values())
    loss_fn = WeightedCrossEntropyLoss(train_counts).to(device)

    metrics, cm, labels, preds, confidence = evaluate(model, loader, loss_fn, cfg, device)
    metrics = {k.replace("val/", "test/"): float(v) for k, v in metrics.items()}
    errors = [{"file": test_set.samples[i][0].name, "true": CLASSES[labels[i]], "predicted": CLASSES[preds[i]],
               "confidence": round(float(confidence[i]), 4)} for i in (labels != preds).nonzero()[0]]
    result = {"checkpoint_epoch": state["epoch"] + 1, "validation_metrics": state["metrics"],
              "test_images": len(test_set), "test_class_counts": test_set.class_counts(), "test": metrics,
              "confusion_matrix": {"classes": list(CLASSES), "rows_true_columns_predicted": cm.tolist()},
              "misclassified": errors}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "classification_test.json").write_text(json.dumps(result, indent=1) + "\n")

    print(f"test images {len(test_set)}: accuracy {metrics['test/accuracy']:.4f}, macro F1 {metrics['test/macro_f1']:.4f}, "
          f"loss {metrics['test/loss']:.4f}")
    print(f"{'class':<11} {'precision':>9} {'recall':>7} {'F1':>7}")
    for c in CLASSES:
        print(f"{c:<11} {metrics[f'test/precision/{c}']:9.4f} {metrics[f'test/recall/{c}']:7.4f} {metrics[f'test/f1/{c}']:7.4f}")
    print("confusion matrix (rows true, columns predicted):", list(CLASSES))
    for c, row in zip(CLASSES, cm):
        print(f"  {c:<11} {row.tolist()}")
    print(f"{len(errors)} misclassified")

    if not args.no_wandb:
        import wandb

        run = wandb.init(project=cfg.wandb_project, job_type="eval", name="test-eval",
                         config={"checkpoint_epoch": state["epoch"] + 1, "test_images": len(test_set)})
        run.summary.update(metrics)
        run.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
                     y_true=labels.tolist(), preds=preds.tolist(), class_names=list(CLASSES),
                     title="Test confusion matrix"),
                 "test/misclassified": misclassified_table(test_set, labels, preds, confidence, len(errors))})
        print("wandb run:", run.url)
        run.finish()


if __name__ == "__main__":
    main()
