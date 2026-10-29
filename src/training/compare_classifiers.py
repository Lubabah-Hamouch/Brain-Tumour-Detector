"""Compare two classifier test evaluations, focusing on the image-format shortcut.

Reads two result files written by src.training.evaluate_classification and
reports, for the same 1,037 test images:

- headline metrics side by side;
- error rate per class for images in the dominant 512 x 512 format versus any
  other format (the shortcut found in the error analysis);
- the tracked errors (by default the three gliomas the original model called
  no tumour), fixed or not;
- errors that are new in the second model and errors it fixed;
- an exact McNemar test on the images where exactly one model is wrong.

Writes reports/results/classification_comparison.json.

Usage:
    python -m src.training.compare_classifiers classification_test.json classification_test_crop.json
"""
import argparse
import json
from math import comb

from PIL import Image

from src.data.classification_preprocess import CLASSES, BrainMRIClassificationDataset
from src.training.evaluate_segmentation import RESULTS_DIR

STANDARD_FORMAT = (512, 512)
TRACKED = ("Te-gl_350.jpg", "Te-gl_97.jpg", "Te-gl_318.jpg")


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def format_breakdown(images, wrong):
    out = {}
    for cls in CLASSES:
        for fmt in ("512x512", "other"):
            names = [n for n, (c, size) in images.items() if c == cls and (size == STANDARD_FORMAT) == (fmt == "512x512")]
            errors = sum(n in wrong for n in names)
            out[f"{cls}/{fmt}"] = {"images": len(names), "errors": errors,
                                   "error_rate": errors / len(names) if names else None}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", help="result file name under reports/results/")
    parser.add_argument("candidate", help="result file name under reports/results/")
    parser.add_argument("--output", default="classification_comparison.json")
    args = parser.parse_args()

    runs = {name: json.loads((RESULTS_DIR / name).read_text()) for name in (args.baseline, args.candidate)}
    test = BrainMRIClassificationDataset("test")
    images = {}
    for path, label in test.samples:
        with Image.open(path) as im:
            images[path.name] = (CLASSES[label], im.size)

    wrong = {name: {e["file"]: e for e in r["misclassified"]} for name, r in runs.items()}
    base, cand = wrong[args.baseline], wrong[args.candidate]
    only_base, only_cand = sorted(set(base) - set(cand)), sorted(set(cand) - set(base))
    result = {
        "baseline": args.baseline, "candidate": args.candidate,
        "metrics": {name: {k.removeprefix("test/"): v for k, v in r["test"].items()} for name, r in runs.items()},
        "checkpoint_epochs": {name: r["checkpoint_epoch"] for name, r in runs.items()},
        "format_breakdown": {name: format_breakdown(images, wrong[name]) for name in runs},
        "tracked_errors": {f: {"baseline": base[f]["predicted"] if f in base else "correct",
                               "candidate": cand[f]["predicted"] if f in cand else "correct",
                               "size": images[f][1]} for f in TRACKED},
        "fixed_by_candidate": [{"file": f, "true": base[f]["true"], "baseline_predicted": base[f]["predicted"],
                                "size": images[f][1]} for f in only_base],
        "new_in_candidate": [{**cand[f], "size": images[f][1]} for f in only_cand],
        "both_wrong": sorted(set(base) & set(cand)),
        "mcnemar_exact": {"baseline_only_wrong": len(only_base), "candidate_only_wrong": len(only_cand),
                          "p_value": mcnemar_exact(len(only_base), len(only_cand))},
    }
    (RESULTS_DIR / args.output).write_text(json.dumps(result, indent=1, default=list) + "\n")

    print(f"{'metric':<26}" + "".join(f"{n:>32}" for n in runs))
    for key in ("accuracy", "macro_f1", "loss", *(f"f1/{c}" for c in CLASSES), *(f"recall/{c}" for c in CLASSES)):
        print(f"{key:<26}" + "".join(f"{result['metrics'][n][key]:>32.4f}" for n in runs))
    print("\nerror rate by format (errors / images):")
    for cls in CLASSES:
        cells = []
        for n in runs:
            b = result["format_breakdown"][n]
            cells.append(f"512: {b[f'{cls}/512x512']['errors']}/{b[f'{cls}/512x512']['images']}  "
                         f"other: {b[f'{cls}/other']['errors']}/{b[f'{cls}/other']['images']}")
        print(f"  {cls:<11}" + "".join(f"{c:>32}" for c in cells))
    print("\ntracked:", json.dumps(result["tracked_errors"]))
    print("fixed by candidate:", [(e["file"], e["baseline_predicted"]) for e in result["fixed_by_candidate"]])
    print("new in candidate:", [(e["file"], e["true"], e["predicted"], e["confidence"], e["size"]) for e in result["new_in_candidate"]])
    print("McNemar:", result["mcnemar_exact"])


if __name__ == "__main__":
    main()
