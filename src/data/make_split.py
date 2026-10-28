"""Build a cleaned, cluster-aware train/val/test split of a classification dataset.

Reads a cleaning config (exclusions, false-match overrides, thresholds, ratios),
re-clusters near-duplicates with src.data.dedup, assigns every cluster wholly to
one split with per-class stratification, then writes:

    <manifest_dir>/<name>_split.csv     one row per source image (split or exclusion reason)
    <manifest_dir>/<name>_summary.json  per-split class counts and parameters
    <output>/<split>/<label>/<image>    copies of the images (raw data is never modified)

Usage:
    python -m src.data.make_split data/manifests/figshare_4class_cleaning.json --dry-run
    python -m src.data.make_split data/manifests/figshare_4class_cleaning.json
"""
import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from src.config import PROJECT_ROOT
from src.data.dedup import collect_images, compute_hashes, find_clusters

SPLITS = ("train", "val", "test")


def load_config(path):
    config = json.loads(Path(path).read_text())
    config["name"] = Path(path).stem.removesuffix("_cleaning")
    return config


def apply_exclusions(records, config):
    """Return (kept records, {path: reason}) according to the config."""
    excluded = {}
    marker = config.get("exclude_filename_marker")
    if marker:
        for rec in records:
            if marker["marker"] in Path(rec.path).name:
                excluded[rec.path] = marker["reason"]
    known = {r.path for r in records}
    for group in config.get("exclude_files", []):
        for path in group["files"]:
            if path not in known:
                raise FileNotFoundError(f"Excluded file not in dataset: {path}")
            excluded.setdefault(path, group["reason"])
    return [r for r in records if r.path not in excluded], excluded


def assign_groups(groups, records, ratios, seed):
    """Greedy stratified assignment of whole groups to splits.

    Within each class, groups are shuffled (seeded) then placed largest-first
    into whichever split is furthest below its target share of that class.
    """
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for group in groups:
        labels = Counter(records[k].label for k in group)
        if len(labels) > 1:
            raise ValueError(f"Group spans labels {dict(labels)}: {[records[k].path for k in group]}")
        by_label[next(iter(labels))].append(group)

    assignment = {}
    for label in sorted(by_label):
        label_groups = by_label[label]
        rng.shuffle(label_groups)
        label_groups.sort(key=len, reverse=True)  # stable: ties keep shuffled order
        total = sum(len(g) for g in label_groups)
        counts = dict.fromkeys(SPLITS, 0)
        for group in label_groups:
            split = max(SPLITS, key=lambda s: ratios[s] * total - counts[s])
            counts[split] += len(group)
            for k in group:
                assignment[k] = split
    return assignment


def build(config, dry_run=False):
    source = PROJECT_ROOT / config["source"]
    records = collect_images(source)
    kept, excluded = apply_exclusions(records, config)

    params = config["dedup"]
    phashes, dhashes = compute_hashes(source, kept, params["hash_size"])
    index = {r.path: i for i, r in enumerate(kept)}
    ignore = [(index[a], index[b]) for a, b in (p["files"] for p in config.get("ignore_pairs", []))]
    clusters, _ = find_clusters(phashes, dhashes, params["max_phash"], params["max_dhash"], ignore)

    cluster_of = {}
    for cid, cluster in enumerate(clusters):
        for k in cluster:
            cluster_of[k] = cid
    singletons = [[k] for k in range(len(kept)) if k not in cluster_of]
    assignment = assign_groups(clusters + singletons, kept, config["split"]["ratios"], config["split"]["seed"])

    counts = {s: Counter() for s in SPLITS}
    for k, split in assignment.items():
        counts[split][kept[k].label] += 1
    leaks = sum(1 for c in clusters if len({assignment[k] for k in c}) > 1)

    summary = {
        "source": config["source"],
        "source_images": len(records),
        "excluded": len(excluded),
        "excluded_by_reason": dict(Counter(excluded.values())),
        "kept": len(kept),
        "clusters": len(clusters),
        "images_in_clusters": sum(len(c) for c in clusters),
        "clusters_split_across_splits": leaks,
        "dedup": params,
        "split": config["split"],
        "counts": {s: dict(sorted(counts[s].items())) for s in SPLITS},
        "totals": {s: sum(counts[s].values()) for s in SPLITS},
    }
    if dry_run:
        return summary

    manifest_dir = Path(config.get("manifest_dir", PROJECT_ROOT / "data" / "manifests"))
    with open(manifest_dir / f"{config['name']}_split.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_path", "label", "original_split", "split", "cluster_id", "cluster_size", "excluded_reason"])
        for rec in records:
            if rec.path in excluded:
                writer.writerow([rec.path, rec.label, rec.split, "", "", "", excluded[rec.path]])
                continue
            k = index[rec.path]
            cid = cluster_of.get(k)
            size = len(clusters[cid]) if cid is not None else 1
            writer.writerow([rec.path, rec.label, rec.split, assignment[k], "" if cid is None else cid, size, ""])
    (manifest_dir / f"{config['name']}_summary.json").write_text(json.dumps(summary, indent=2))

    output = PROJECT_ROOT / config["output"]
    if output.exists():
        raise FileExistsError(f"{output} already exists; remove it before rebuilding")
    for k, split in assignment.items():
        rec = kept[k]
        dest = output / split / rec.label / Path(rec.path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rec.path, dest)
    return summary


def print_summary(summary):
    print(f"Source images: {summary['source_images']}  excluded: {summary['excluded']}  kept: {summary['kept']}")
    for reason, n in summary["excluded_by_reason"].items():
        print(f"  {n:4d}  {reason}")
    print(f"Clusters: {summary['clusters']} ({summary['images_in_clusters']} images); "
          f"split across splits: {summary['clusters_split_across_splits']}")
    labels = sorted({l for c in summary["counts"].values() for l in c})
    print(f"\n{'class':<12}" + "".join(f"{s:>14}" for s in SPLITS) + f"{'total':>8}")
    for label in labels:
        row = [summary["counts"][s].get(label, 0) for s in SPLITS]
        total = sum(row)
        print(f"{label:<12}" + "".join(f"{n:>7} ({n / total:4.0%})" for n in row) + f"{total:>8}")
    totals = [summary["totals"][s] for s in SPLITS]
    grand = sum(totals)
    print(f"{'total':<12}" + "".join(f"{n:>7} ({n / grand:4.0%})" for n in totals) + f"{grand:>8}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="cleaning config JSON")
    parser.add_argument("--dry-run", action="store_true", help="compute and print the split without writing anything")
    args = parser.parse_args()
    print_summary(build(load_config(args.config), dry_run=args.dry_run))


if __name__ == "__main__":
    main()
