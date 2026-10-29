"""Confidence intervals for the test results (reads reports/results/, writes test_confidence_intervals.json).

Segmentation: 95% percentile-bootstrap intervals over test patients (10,000
resamples, seed 0) for mean Dice per region, plus paired differences
(ensemble vs the single best fold, post-processed vs raw). The same resampled
patients are used for both sides of a difference, so case-to-case difficulty
cancels out.

Classification: Wilson score interval for test accuracy.

Usage:
    python -m src.training.bootstrap_results
"""
import csv
import json

import numpy as np

from src.training.evaluate_segmentation import RESULTS_DIR

RESAMPLES = 10_000


def interval(samples):
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def wilson(p, n, z=1.96):
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def main():
    seg = json.loads((RESULTS_DIR / "segmentation_test.json").read_text())
    rows = list(csv.DictReader(open(RESULTS_DIR / "segmentation_test_per_case.csv")))
    cases = sorted({r["case"] for r in rows})
    lookup = {(r["case"], r["model"]): r for r in rows}

    def scores(model, post_processed):
        et = "ET_pp" if post_processed else "ET"
        a = np.array([[float(lookup[c, model][k]) for k in ("WT", "TC", et)] for c in cases])
        return np.column_stack([a, a.mean(1)])  # WT, TC, ET, mean

    idx = np.random.default_rng(0).integers(0, len(cases), (RESAMPLES, len(cases)))
    names = ("WT", "TC", "ET", "mean")
    best = seg["single_best_model"]
    out = {"method": f"percentile bootstrap over {len(cases)} test patients, {RESAMPLES} resamples, seed 0",
           "segmentation": {}, "segmentation_paired_differences": {}}
    for model in (best, "ensemble"):
        for pp in (False, True):
            s = scores(model, pp)
            out["segmentation"][f"{model}/{'post_processed' if pp else 'raw'}"] = {
                n: {"mean": float(s[:, i].mean()), "ci95": interval(s[idx, i].mean(1))} for i, n in enumerate(names)}
    comparisons = {
        f"ensemble - {best} (post_processed)": (scores("ensemble", True), scores(best, True)),
        f"ensemble - {best} (raw)": (scores("ensemble", False), scores(best, False)),
        "ensemble: post_processed - raw": (scores("ensemble", True), scores("ensemble", False)),
        f"{best}: post_processed - raw": (scores(best, True), scores(best, False)),
    }
    for label, (a, b) in comparisons.items():
        d = a[:, 3] - b[:, 3]
        out["segmentation_paired_differences"][label] = {
            "mean_dice_difference": float(d.mean()), "ci95": interval(d[idx].mean(1)),
            "cases_better": int((d > 1e-9).sum()), "cases_worse": int((d < -1e-9).sum())}

    clf = json.loads((RESULTS_DIR / "classification_test.json").read_text())
    acc, n = clf["test"]["test/accuracy"], clf["test_images"]
    out["classification"] = {"accuracy": acc, "ci95_wilson": wilson(acc, n), "images": n}

    (RESULTS_DIR / "test_confidence_intervals.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
