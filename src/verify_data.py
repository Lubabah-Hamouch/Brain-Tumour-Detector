"""Check that downloaded raw datasets are complete and readable, and print stats.

Usage:
    python -m src.verify_data classification
    python -m src.verify_data brats
    python -m src.verify_data figshare3
"""
import argparse
import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import h5py
import nibabel as nib
import numpy as np
from PIL import Image

from src.config import BRATS_DIR, CLASSIFICATION_DIR, FIGSHARE_3CLASS_DIR

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

BRATS_MODALITIES = ("flair", "t1", "t1ce", "t2", "seg")
FIGSHARE_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def dir_size_gb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def _check_brats_case(case):
    """Fully decode every volume of one case; gzip CRC errors surface as exceptions."""
    result = {"missing": [], "errors": [], "shapes": [], "seg_labels": []}
    for modality in BRATS_MODALITIES:
        path = case / f"{case.name}_{modality}.nii.gz"
        if not path.exists():
            result["missing"].append(modality)
            continue
        try:
            img = nib.load(path)
            data = np.asarray(img.dataobj)
            if not np.isfinite(data).all():
                raise ValueError("non-finite voxel values")
            result["shapes"].append(img.shape)
            if modality == "seg":
                result["seg_labels"] = np.unique(data).astype(int).tolist()
        except Exception as exc:
            result["errors"].append((path.name, repr(exc)))
    return case.name, result


def verify_brats(workers=8):
    cases = sorted(p for p in BRATS_DIR.glob("BraTS2021_*") if p.is_dir())
    incomplete, unreadable = [], []
    label_counts = Counter()
    shapes = Counter()

    # Decompression is zlib-bound and releases the GIL, so threads parallelise well.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (name, res) in enumerate(pool.map(_check_brats_case, cases), 1):
            if res["missing"]:
                incomplete.append((name, res["missing"]))
            unreadable.extend(res["errors"])
            shapes.update(res["shapes"])
            label_counts.update(res["seg_labels"])
            if done % 100 == 0 or done == len(cases):
                print(f"  checked {done}/{len(cases)} cases, {len(unreadable)} errors so far", flush=True)

    print(f"BraTS2021 root:    {BRATS_DIR}")
    print(f"Size on disk:      {dir_size_gb(BRATS_DIR):.2f} GB")
    print(f"Patient cases:     {len(cases)}")
    print(f"Incomplete cases:  {len(incomplete)} {incomplete[:5]}")
    print(f"Unreadable files:  {len(unreadable)} {unreadable[:5]}")
    print(f"Volume shapes:     {dict(shapes)}")
    print(f"Cases containing each seg label: {dict(sorted(label_counts.items()))}")


def verify_classification():
    # Layout: <split>/<class>/<image>
    images = sorted(p for p in CLASSIFICATION_DIR.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    counts = defaultdict(Counter)
    sizes, modes = Counter(), Counter()
    hashes = defaultdict(list)
    unreadable = []

    for path in images:
        split, label = path.parent.parent.name, path.parent.name
        try:
            with Image.open(path) as img:
                img.load()  # full decode, not just the header
                sizes[img.size] += 1
                modes[img.mode] += 1
        except Exception as exc:
            unreadable.append((str(path.relative_to(CLASSIFICATION_DIR)), repr(exc)))
            continue
        counts[split][label] += 1
        hashes[hashlib.md5(path.read_bytes()).hexdigest()].append((split, label))

    print(f"Classification root: {CLASSIFICATION_DIR}")
    print(f"Size on disk:        {dir_size_gb(CLASSIFICATION_DIR):.3f} GB")
    print(f"Images:              {len(images)}")
    print(f"Unreadable files:    {len(unreadable)} {unreadable[:5]}")
    print(f"Image modes:         {dict(modes)}")
    print(f"Distinct sizes:      {len(sizes)}; most common {sizes.most_common(5)}")

    labels = sorted({label for c in counts.values() for label in c})
    totals = Counter()
    print(f"\n{'class':<12}" + "".join(f"{s:>10}" for s in sorted(counts)) + f"{'total':>10}")
    for label in labels:
        row = [counts[s][label] for s in sorted(counts)]
        totals[label] = sum(row)
        print(f"{label:<12}" + "".join(f"{n:>10}" for n in row) + f"{sum(row):>10}")
    if totals:
        print(f"Imbalance ratio (largest/smallest class): {max(totals.values()) / min(totals.values()):.2f}")

    dupes = [v for v in hashes.values() if len(v) > 1]
    cross_split = [v for v in dupes if len({s for s, _ in v}) > 1]
    cross_label = [v for v in dupes if len({l for _, l in v}) > 1]
    print("\nExact duplicates (identical bytes):")
    print(f"  duplicate groups:           {len(dupes)} ({sum(len(v) - 1 for v in dupes)} redundant files)")
    print(f"  groups spanning splits:     {len(cross_split)}  (train/test leakage)")
    print(f"  groups with conflicting labels: {len(cross_label)}")


def verify_figshare_3class():
    # Images are numbered <n>.mat; cvind.mat holds the cross-validation fold indices.
    mats = sorted(p for p in FIGSHARE_3CLASS_DIR.rglob("*.mat") if p.stem.isdigit())
    classes = Counter()
    patients = set()
    shapes = Counter()
    unreadable = []

    for path in mats:
        try:
            with h5py.File(path, "r") as f:
                cj = f["cjdata"]
                label = int(np.array(cj["label"]).item())
                pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).ravel())
                image = np.array(cj["image"])
                np.array(cj["tumorMask"])
            classes[FIGSHARE_LABELS.get(label, f"unknown({label})")] += 1
            patients.add(pid)
            shapes[image.shape] += 1
        except Exception as exc:
            unreadable.append((path.name, repr(exc)))

    print(f"Figshare root:     {FIGSHARE_3CLASS_DIR}")
    print(f"Size on disk:      {dir_size_gb(FIGSHARE_3CLASS_DIR):.2f} GB")
    print(f".mat images:       {len(mats)}")
    print(f"Unreadable files:  {len(unreadable)} {unreadable[:5]}")
    print(f"Unique patients:   {len(patients)}")
    print(f"Image shapes:      {dict(shapes)}")
    print("Class breakdown:")
    for name, count in sorted(classes.items()):
        print(f"  {name:<12} {count:>5}  ({count / max(len(mats), 1):.1%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["classification", "brats", "figshare3"])
    args = parser.parse_args()
    {
        "classification": verify_classification,
        "brats": verify_brats,
        "figshare3": verify_figshare_3class,
    }[args.dataset]()


if __name__ == "__main__":
    main()
