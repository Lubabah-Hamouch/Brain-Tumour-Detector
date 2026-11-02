"""Download the trained weights served by the API from the project's GitHub release.

Saves them where the API looks by default (models/segmentation/fold<k>/best.pt and
models/classification_crop_resdegrade_long/best.pt) and checks each file's SHA-256. Files that
are already present and match are skipped. Uses only the standard library, so it runs before any
requirements are installed.

Usage:
    python -m src.download_weights            # keeps any checkpoint you trained yourself
    python -m src.download_weights --force    # replaces them with the release files

The weights were trained on BraTS 2021 and the Brain Tumor MRI Dataset and are shared for
research use under those datasets' terms (see "Data and terms of use" in the README).
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RELEASE_URL = "https://github.com/Lubabah-Hamouch/Brain-Tumour-Detector/releases/download/weights-v1"

# Release asset -> (path under the project root, SHA-256)
WEIGHTS = {
    "segmentation_fold1.pt": ("models/segmentation/fold1/best.pt",
                              "9ab290d2be93a0908077e9b37558240dbd867a896ad20b9b418a5af755cb59b3"),
    "segmentation_fold2.pt": ("models/segmentation/fold2/best.pt",
                              "55ce26fb318e9733c55b1972984645794d142e00a2ba88b469bfd346a43fe3c2"),
    "segmentation_fold3.pt": ("models/segmentation/fold3/best.pt",
                              "0598ec1bbd1f64224b89ea5774e0a73c433c18c2a57221373bf1cf7899c4aba0"),
    "segmentation_fold4.pt": ("models/segmentation/fold4/best.pt",
                              "3682f7b55b07e473523d9d61585431fa5e9d70e831e637da1556c67c71d2cb13"),
    "segmentation_fold5.pt": ("models/segmentation/fold5/best.pt",
                              "a43b14dae71801f10a5e893dc82f47942da9c9ebe7644ef41e514d0e6bd4bdc1"),
    "classifier_crop_resdegrade_long.pt": ("models/classification_crop_resdegrade_long/best.pt",
                                           "2c6a7cb431ef97876de88ddc6eb3e8e7c26bc9a5db8cc1f38f439d8c61cb4fe2"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, dest):
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while block := response.read(1 << 20):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="replace existing checkpoints that differ from the release (e.g. your own training runs)")
    args = parser.parse_args()
    failed = []
    for asset, (target, expected) in WEIGHTS.items():
        dest = PROJECT_ROOT / target
        if dest.exists():
            if sha256(dest) == expected:
                print(f"{target}: already present")
                continue
            if not args.force:
                print(f"{target}: exists and is not the release file (kept; use --force to replace it)")
                continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"{target} <- {asset}")
        download(f"{RELEASE_URL}/{asset}", dest)
        if sha256(dest) != expected:
            dest.unlink()
            failed.append(target)
            print("  SHA-256 mismatch, file removed")
    if failed:
        sys.exit(f"Download failed for: {', '.join(failed)}")
    print("All weights in place.")


if __name__ == "__main__":
    main()
