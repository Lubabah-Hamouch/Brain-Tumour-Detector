"""Patient-level test split and 5-fold cross-validation folds for BraTS2021.

Each BraTS2021 case is a distinct patient, so splitting by case ID is a
patient-level split. A test set is carved out first and is never used for
training, validation or model selection; the remaining cases are divided into
cross-validation folds (fold k validates on folds[k] and trains on the rest).

Assignment is stratified so every split has a similar mix of tumours:
    - cases without enhancing tumour (ET absent) form their own stratum, since
      ET Dice on them is all-or-nothing and they would otherwise cluster by chance;
    - cases with ET are binned by whole-tumour volume quartile.
Within each stratum cases are shuffled with a fixed seed, the test share is
taken first, and the rest are dealt round-robin into folds.

Per-case region voxel counts (on the preprocessed 1 mm grid) are stored in the
manifest so the stratification can be audited.

Usage:
    python -m src.data.make_brats_folds
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.config import PROCESSED_DIR, PROJECT_ROOT
from src.data.segmentation_preprocess import REGIONS

VOLUME_DIR = PROCESSED_DIR / "brats2021_preprocessed"
MANIFEST = PROJECT_ROOT / "data" / "manifests" / "brats_5fold_splits.json"


def region_voxels(path):
    with np.load(path) as data:
        targets = data["targets"]
    return path.stem, [int(targets[r].sum()) for r in range(len(REGIONS))]


def collect_voxels(volume_dir, workers):
    index = json.loads((Path(volume_dir) / "index.json").read_text())
    paths = [Path(volume_dir) / f"{item['case']}.npz" for item in index["items"]]
    with ProcessPoolExecutor(workers) as pool:
        return dict(pool.map(region_voxels, paths, chunksize=8))


def strata(voxels):
    """case -> stratum label: 'ET-absent' or 'WT-q1'..'WT-q4' (cases with ET)."""
    with_et = {c: v for c, v in voxels.items() if v[2] > 0}
    edges = np.quantile([v[0] for v in with_et.values()], [0.25, 0.5, 0.75])
    labels = {}
    for case, v in voxels.items():
        labels[case] = "ET-absent" if v[2] == 0 else f"WT-q{int(np.searchsorted(edges, v[0], side='right')) + 1}"
    return labels


def assign(labels, test_fraction, n_folds, seed):
    """case -> 'test' or fold number 1..n_folds."""
    by_stratum = defaultdict(list)
    for case, label in sorted(labels.items()):
        by_stratum[label].append(case)
    rng = random.Random(seed)
    assignment, offset = {}, 0
    for label in sorted(by_stratum):
        cases = by_stratum[label]
        rng.shuffle(cases)
        n_test = round(len(cases) * test_fraction)
        for case in cases[:n_test]:
            assignment[case] = "test"
        # Continue the round-robin across strata so fold sizes differ by at most one.
        for i, case in enumerate(cases[n_test:]):
            assignment[case] = (offset + i) % n_folds + 1
        offset += len(cases) - n_test
    return assignment


def summarise(assignment, labels, voxels):
    groups = defaultdict(list)
    for case, split in assignment.items():
        groups["test" if split == "test" else f"fold{split}"].append(case)
    summary = {}
    for name in sorted(groups):
        cases = groups[name]
        summary[name] = {
            "cases": len(cases),
            "strata": dict(sorted(Counter(labels[c] for c in cases).items())),
            "median_voxels": {r: int(np.median([voxels[c][i] for c in cases])) for i, r in enumerate(REGIONS)},
        }
    return summary


def build(volume_dir=VOLUME_DIR, test_fraction=0.15, n_folds=5, seed=42, workers=8):
    voxels = collect_voxels(volume_dir, workers)
    labels = strata(voxels)
    assignment = assign(labels, test_fraction, n_folds, seed)
    folds = {str(k): sorted(c for c, s in assignment.items() if s == k) for k in range(1, n_folds + 1)}
    return {
        "source": str(Path(volume_dir).relative_to(PROJECT_ROOT).as_posix()),
        "seed": seed,
        "test_fraction": test_fraction,
        "n_folds": n_folds,
        "stratification": "ET-absent cases as one stratum; cases with ET binned by whole-tumour volume quartile",
        "usage": "Fold k: validate on folds[k], train on all other folds. The test cases are used only for final reporting.",
        "summary": summarise(assignment, labels, voxels),
        "test": sorted(c for c, s in assignment.items() if s == "test"),
        "folds": folds,
        "case_voxels": {"regions": list(REGIONS), **{c: voxels[c] for c in sorted(voxels)}},
    }


def load_fold(fold, manifest=MANIFEST):
    """(train_ids, val_ids) for a 1-based fold number."""
    data = json.loads(Path(manifest).read_text())
    folds = data["folds"]
    if str(fold) not in folds:
        raise ValueError(f"Fold {fold} not in manifest (folds {sorted(folds)})")
    train = sorted(c for k, cases in folds.items() if k != str(fold) for c in cases)
    return train, list(folds[str(fold)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=MANIFEST)
    args = parser.parse_args()

    manifest = build(test_fraction=args.test_fraction, n_folds=args.folds, seed=args.seed, workers=args.workers)
    args.output.write_text(json.dumps(manifest, indent=1) + "\n")
    for name, s in manifest["summary"].items():
        print(f"{name:<6} {s['cases']:4d}  {s['strata']}  median voxels {s['median_voxels']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
