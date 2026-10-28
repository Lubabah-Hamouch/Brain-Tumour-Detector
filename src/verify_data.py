"""Check that downloaded raw datasets are complete and readable, and print stats.

Usage:
    python -m src.verify_data brats
    python -m src.verify_data figshare
"""
import argparse
from collections import Counter

import h5py
import nibabel as nib
import numpy as np

from src.config import BRATS_DIR, FIGSHARE_DIR

BRATS_MODALITIES = ("flair", "t1", "t1ce", "t2", "seg")
FIGSHARE_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def dir_size_gb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def verify_brats():
    cases = sorted(p for p in BRATS_DIR.rglob("BraTS2021_*") if p.is_dir())
    incomplete, unreadable = [], []
    label_counts = Counter()
    shapes = Counter()

    for case in cases:
        files = {m: case / f"{case.name}_{m}.nii.gz" for m in BRATS_MODALITIES}
        missing = [m for m, f in files.items() if not f.exists()]
        if missing:
            incomplete.append((case.name, missing))
            continue
        for modality, path in files.items():
            try:
                img = nib.load(path)
                if modality == "seg":
                    # Full decompression of the mask catches truncated archives.
                    seg = np.asarray(img.dataobj)
                    label_counts.update(np.unique(seg).astype(int).tolist())
                shapes[img.shape] += 1
            except Exception as exc:
                unreadable.append((path.name, repr(exc)))

    print(f"BraTS2021 root:    {BRATS_DIR}")
    print(f"Size on disk:      {dir_size_gb(BRATS_DIR):.2f} GB")
    print(f"Patient cases:     {len(cases)}")
    print(f"Incomplete cases:  {len(incomplete)} {incomplete[:5]}")
    print(f"Unreadable files:  {len(unreadable)} {unreadable[:5]}")
    print(f"Volume shapes:     {dict(shapes)}")
    print(f"Cases containing each seg label: {dict(sorted(label_counts.items()))}")


def verify_figshare():
    mats = sorted(FIGSHARE_DIR.rglob("*.mat"))
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

    print(f"Figshare root:     {FIGSHARE_DIR}")
    print(f"Size on disk:      {dir_size_gb(FIGSHARE_DIR):.2f} GB")
    print(f".mat images:       {len(mats)}")
    print(f"Unreadable files:  {len(unreadable)} {unreadable[:5]}")
    print(f"Unique patients:   {len(patients)}")
    print(f"Image shapes:      {dict(shapes)}")
    print("Class breakdown:")
    for name, count in sorted(classes.items()):
        print(f"  {name:<12} {count:>5}  ({count / max(len(mats), 1):.1%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["brats", "figshare"])
    args = parser.parse_args()
    {"brats": verify_brats, "figshare": verify_figshare}[args.dataset]()


if __name__ == "__main__":
    main()
