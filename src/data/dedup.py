"""Perceptual-hash near-duplicate detection for image classification datasets.

Expects the layout <root>/<split>/<label>/<image>. Read-only with respect to
the dataset: it reports clusters, cross-split leakage, label conflicts and
augmented files, and optionally writes the findings to a JSON file.

Usage:
    python -m src.data.dedup data/raw/figshare_classification_4class \
        --max-phash 4 --max-dhash 4 --out reports/dedup_strict.json
"""
import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
# Byte-wise popcount table for vectorised Hamming distances on packed hashes.
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


@dataclass(frozen=True)
class ImageRecord:
    path: str  # relative to the dataset root, forward slashes
    split: str
    label: str


def collect_images(root, suffixes=IMAGE_SUFFIXES):
    root = Path(root)
    records = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in suffixes:
            rel = p.relative_to(root)
            records.append(ImageRecord(rel.as_posix(), rel.parts[0], p.parent.name))
    return records


def compute_hashes(root, records, hash_size=8):
    """Return packed (n, hash_size**2 / 8) uint8 arrays of pHash and dHash bits."""
    root = Path(root)
    phashes, dhashes = [], []
    for rec in records:
        with Image.open(root / rec.path) as img:
            gray = img.convert("L")
            phashes.append(np.packbits(imagehash.phash(gray, hash_size=hash_size).hash.ravel()))
            dhashes.append(np.packbits(imagehash.dhash(gray, hash_size=hash_size).hash.ravel()))
    return np.stack(phashes), np.stack(dhashes)


def _hamming_to_rest(packed, i):
    """Hamming distance from row i to every row after it."""
    return _POPCOUNT[np.bitwise_xor(packed[i + 1:], packed[i])].sum(axis=1)


def find_clusters(phashes, dhashes, max_phash, max_dhash, ignore_pairs=()):
    """Group images whose pHash AND dHash distances are within the thresholds.

    Matching is transitive (union-find), so a cluster can contain pairs that are
    individually further apart than the thresholds. ignore_pairs is a collection
    of (i, j) index pairs confirmed by review to be false matches. Returns
    clusters of size >= 2 as lists of indices, plus the number of matching pairs.
    """
    ignore = {frozenset(p) for p in ignore_pairs}
    n = len(phashes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = 0
    for i in range(n - 1):
        match = (_hamming_to_rest(phashes, i) <= max_phash) & (_hamming_to_rest(dhashes, i) <= max_dhash)
        for j in np.nonzero(match)[0] + i + 1:
            if frozenset((i, int(j))) in ignore:
                continue
            pairs += 1
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri

    groups = defaultdict(list)
    for k in range(n):
        groups[find(k)].append(k)
    clusters = sorted((g for g in groups.values() if len(g) > 1), key=lambda g: (-len(g), g[0]))
    return clusters, pairs


def find_augmented(records, marker="-aug-"):
    return [r for r in records if marker in Path(r.path).name]


def summarize(records, clusters, pairs, test_split="Testing"):
    def splits(c):
        return {records[k].split for k in c}

    def labels(c):
        return {records[k].label for k in c}

    cross_split = [c for c in clusters if len(splits(c)) > 1]
    conflicts = [c for c in clusters if len(labels(c)) > 1]
    leaked_test = {k for c in cross_split for k in c if records[k].split == test_split}
    return {
        "images": len(records),
        "matching_pairs": pairs,
        "clusters": len(clusters),
        "images_in_clusters": sum(len(c) for c in clusters),
        "redundant_images": sum(len(c) - 1 for c in clusters),
        "largest_cluster": max((len(c) for c in clusters), default=0),
        "cluster_size_histogram": dict(sorted(Counter(len(c) for c in clusters).items())),
        "clusters_spanning_splits": len(cross_split),
        "test_images_with_cross_split_copy": len(leaked_test),
        "leaked_test_images_by_label": dict(sorted(Counter(records[k].label for k in leaked_test).items())),
        "clusters_with_label_conflicts": len(conflicts),
    }


def cluster_details(records, clusters):
    return [
        {
            "id": cid,
            "size": len(c),
            "splits": sorted({records[k].split for k in c}),
            "labels": sorted({records[k].label for k in c}),
            "members": [records[k].path for k in c],
        }
        for cid, c in enumerate(clusters)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="dataset root laid out as <split>/<label>/<image>")
    parser.add_argument("--max-phash", type=int, default=4, help="max pHash Hamming distance")
    parser.add_argument("--max-dhash", type=int, default=4, help="max dHash Hamming distance")
    parser.add_argument("--hash-size", type=int, default=8, help="hash side length (bits = hash_size**2)")
    parser.add_argument("--test-split", default="Testing", help="name of the held-out split folder")
    parser.add_argument("--aug-marker", default="-aug-", help="filename marker for augmented images")
    parser.add_argument("--out", type=Path, help="write summary and clusters to this JSON file")
    args = parser.parse_args()

    records = collect_images(args.root)
    phashes, dhashes = compute_hashes(args.root, records, args.hash_size)
    clusters, pairs = find_clusters(phashes, dhashes, args.max_phash, args.max_dhash)
    details = cluster_details(records, clusters)
    summary = summarize(records, clusters, pairs, args.test_split)
    augmented = find_augmented(records, args.aug_marker)

    print(f"Thresholds: pHash <= {args.max_phash}, dHash <= {args.max_dhash} ({args.hash_size**2}-bit)")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    aug_counts = Counter((r.split, r.label) for r in augmented)
    print(f"  augmented files ('{args.aug_marker}'): {len(augmented)} {dict(sorted(aug_counts.items()))}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "root": str(args.root),
            "params": {"max_phash": args.max_phash, "max_dhash": args.max_dhash, "hash_size": args.hash_size},
            "summary": summary,
            "augmented_files": [r.path for r in augmented],
            "label_conflicts": [d for d in details if len(d["labels"]) > 1],
            "clusters": details,
        }, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
