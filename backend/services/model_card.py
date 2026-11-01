"""The model card's data: specification, test tables, paired comparisons, stress test, limitations.

Every number is read from the evaluation outputs (reports/results/), the data manifests
(data/manifests/) or the loaded models (parameter counts, patch and image sizes). The limitations
are the bullets of the "Limitations" section of reports/final_results.md. The only fixed text here
is what each classifier variant changed in training (configs/, and "Models considered" in
reports/final_results.md) and the display labels.
"""
import csv
import io
import json
import re
import statistics
from collections import Counter

from backend.services.errors import ApiError, relative_name
from src.config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "reports" / "results"
MANIFESTS = PROJECT_ROOT / "data" / "manifests"
FINAL_RESULTS = PROJECT_ROOT / "reports" / "final_results.md"

# key, name, training change, test report, checkpoint (the stress test is keyed by checkpoint)
CLASSIFIERS = (
    ("A", "Original", "Flips, affine, intensity", "classification_test.json", "models/classification/best.pt"),
    ("B", "Crop", "+ random crop before letterbox", "classification_test_crop.json",
     "models/classification_crop/best.pt"),
    ("C", "Crop + resolution", "+ down-sampling and JPEG re-encoding", "classification_test_crop_resdegrade.json",
     "models/classification_crop_resdegrade/best.pt"),
    ("D", "C, trained longer", "Early-stopping patience 10", "classification_test_crop_resdegrade_long.json",
     "models/classification_crop_resdegrade_long/best.pt"),
)


def _read(path, label):
    try:
        text = path.read_text(encoding="utf-8")
        return json.loads(text) if path.suffix == ".json" else text
    except (OSError, ValueError) as exc:
        raise ApiError(503, "results_unavailable", f"Could not read {label} ({exc}).") from exc


def result(name):
    return _read(RESULTS / name, f"reports/results/{name}")


def manifest(name):
    return _read(MANIFESTS / name, f"data/manifests/{name}")


def _paired(entry):
    return {"difference": entry["mean_dice_difference"], "ci95": entry["ci95"],
            "patients_better": entry["cases_better"], "patients_worse": entry["cases_worse"]}


def _sentences(text, n):
    return " ".join(re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())[:n])


def limitations():
    """Title and opening (at most two sentences) of each bullet under "## Limitations"."""
    text = _read(FINAL_RESULTS, "reports/final_results.md")
    match = re.search(r"^## Limitations\s*$(.*?)^(?:---|## )", text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        raise ApiError(503, "results_unavailable", "reports/final_results.md has no Limitations section.")
    items = []
    for block in re.split(r"^- ", match.group(1), flags=re.MULTILINE)[1:]:
        head = re.match(r"\*\*(.+?)\*\*\s*(.*)", block, flags=re.DOTALL)
        if not head:
            continue
        opening = re.split(r"\n\s*\n|\n\s+- ", head.group(2))[0]   # up to a blank line or nested list
        opening = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", opening)  # links -> their text
        opening = re.sub(r"\s*\(see [^)]*\)", "", opening)           # cross-references to other sections
        opening = re.sub(r"\*\*|`", "", " ".join(opening.split()))
        items.append({"title": head.group(1).rstrip("."), "body": _sentences(opening, 2)})
    return items


def _parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


def model_card(app):
    served = relative_name(app.state.settings.classifier_checkpoint)
    seg = result("segmentation_test.json")
    ci = result("test_confidence_intervals.json")
    stress = result("format_stress_test.json")
    best = seg["single_best_model"]

    # Segmentation test table: the single best fold and the ensemble, each without and with ET clean-up.
    seg_rows = []
    for model in (best, "ensemble"):
        for variant in ("raw", "post_processed"):
            scores = seg["test"][model][variant]
            seg_rows.append({"model": model, "post_processed": variant == "post_processed",
                             "served": model == "ensemble" and variant == "post_processed",
                             **{k: scores[k] for k in ("WT", "TC", "ET", "mean")}})
    cv = list(seg["cv_best_dice"].values())

    # Test patients without enhancing tumour, and how many the ensemble scores ET Dice 1 (no false ET).
    splits = manifest("brats_5fold_splits.json")
    et_index = splits["case_voxels"]["regions"].index("ET")
    no_et = {c for c in splits["test"] if splits["case_voxels"][c][et_index] == 0}
    scored_1 = {"raw": 0, "post_processed": 0}
    with open(RESULTS / "segmentation_test_per_case.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] == "ensemble" and row["case"] in no_et:
                scored_1["raw"] += float(row["ET"]) == 1.0
                scored_1["post_processed"] += float(row["ET_pp"]) == 1.0

    # Classification: the four variants, their stress-test results, and the served-vs-original test.
    clf_rows, chart = [], []
    for key, name, change, report, checkpoint in CLASSIFIERS:
        r = result(report)
        st = stress["models"][checkpoint]
        clf_rows.append({"key": key, "name": name, "change": change, "checkpoint": checkpoint,
                         "served": checkpoint == served, "checkpoint_epoch": r["checkpoint_epoch"],
                         "test_images": r["test_images"], "accuracy": r["test"]["test/accuracy"],
                         "macro_f1": r["test"]["test/macro_f1"], "accuracy_at_128px": st["tiny_square"]["accuracy"]})
        chart.append({"key": key, "name": name, "served": checkpoint == served,
                      "values": [st[c]["tumour_images_called_no_tumour"] for c in stress["conditions"]],
                      "accuracy": [st[c]["accuracy"] for c in stress["conditions"]]})
    original = clf_rows[0]
    served_row = next((r for r in clf_rows if r["served"]), None)
    if served_row is None:
        raise ApiError(503, "results_unavailable", f"No model card entry for the served classifier {served}.")
    served_report = next(c[3] for c in CLASSIFIERS if c[4] == served)
    comparison = None
    for path in RESULTS.glob("classification_comparison*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if {data["baseline"], data["candidate"]} == {CLASSIFIERS[0][3], served_report}:
            comparison = data

    split_rows = csv.DictReader(io.StringIO(manifest("figshare_4class_split.csv")))
    split_counts = Counter(row["split"] for row in split_rows if not row["excluded_reason"])
    kept = sum(split_counts.values())

    clf_service, seg_service = app.state.classifier, app.state.segmenter
    return {
        "headline": {
            "served_accuracy_at_128px": served_row["accuracy_at_128px"],
            "original_accuracy_at_128px": original["accuracy_at_128px"],
            "served_accuracy": served_row["accuracy"],
            "original_accuracy": original["accuracy"],
            "mcnemar_p": comparison["mcnemar_exact"]["p_value"] if comparison else None,
        },
        "specification": {
            "segmentation": {
                "patients": sum(s["cases"] for s in splits["summary"].values()),
                "test_patients": seg["test_cases"],
                "folds": len(cv),
                "parameters": _parameters(next(iter(seg_service.models.values()))) if seg_service else None,
                "patch_size": seg_service.cfg["patch_size"] if seg_service else None,
                "et_min_voxels": seg["et_threshold_voxels"],
            },
            "classification": {
                "images": kept,
                "split": {k: split_counts[k] / kept for k in ("train", "val", "test")},
                "parameters": _parameters(clf_service.model) if clf_service else None,
                "image_size": clf_service.image_size if clf_service else None,
            },
        },
        "segmentation": {
            "test_patients": seg["test_cases"],
            "single_best_fold": best,
            "rows": seg_rows,
            "ensemble_vs_single_best": _paired(ci["segmentation_paired_differences"][f"ensemble - {best} (post_processed)"]),
            "et_cleanup_on_ensemble": _paired(ci["segmentation_paired_differences"]["ensemble: post_processed - raw"]),
            "patients_without_et": len(no_et),
            "patients_without_et_scored_1": scored_1,
            "cross_validation": {"mean": statistics.mean(cv), "std": statistics.stdev(cv), "folds": len(cv)},
        },
        "classification": {"test_images": served_row["test_images"], "models": clf_rows},
        "stress_test": {
            "tumour_images": stress["tumour_images"],
            "conditions": [{"key": k, "label": "unmodified" if v is None else f"{v[0]} × {v[1]}"}
                           for k, v in stress["conditions"].items()],
            "models": chart,
        },
        "limitations": limitations(),
        "sources": [
            "reports/results/segmentation_test.json", "reports/results/segmentation_test_per_case.csv",
            "reports/results/test_confidence_intervals.json", "reports/results/format_stress_test.json",
            *(f"reports/results/{c[3]}" for c in CLASSIFIERS),
            *(f"reports/results/{p.name}" for p in sorted(RESULTS.glob("classification_comparison*.json"))
              if comparison and json.loads(p.read_text(encoding="utf-8")) == comparison),
            "data/manifests/brats_5fold_splits.json", "data/manifests/figshare_4class_split.csv",
            "reports/final_results.md (Limitations)", "the loaded models (parameter counts, patch and image size)",
        ],
    }
