"""BraTS2021 preprocessing: normalisation, region targets and tumour-biased 3D patches.

Per case:
1. Load the selected modalities (default all four: FLAIR, T1, T1ce, T2) and the mask.
2. Brain mask = voxels non-zero in any modality (BraTS volumes are skull-stripped,
   so background is exactly 0).
3. Z-score each modality using only brain voxels, then set background to 0.
   MRI intensities have no absolute scale (unlike CT Hounsfield units) and vary
   by scanner and protocol, so each volume is standardised on its own. The brain
   is only ~16% of the volume; including background would collapse the statistics.
4. Crop to the brain bounding box, padding up to at least the patch size.
5. Targets as three binary channels, the BraTS evaluation regions:
   WT = labels {1, 2, 4}, TC = labels {1, 4}, ET = label {4}.
6. Sample patches: round(n * tumor_ratio) are centred on a random tumour voxel
   (guaranteed to contain tumour); the rest start at a uniformly random position.

Modes:
    patches  write a fixed set of patches per case to data/processed/brats2021_patches/
    volumes  write each preprocessed (cropped, normalised) case once to
             data/processed/brats2021_preprocessed/ for on-the-fly patch sampling
             with BratsPatchDataset

Usage:
    python -m src.data.segmentation_preprocess --mode patches --max-cases 5 --preview
    python -m src.data.segmentation_preprocess --mode volumes --workers 6
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

from src.config import BRATS_DIR, PROCESSED_DIR, PROJECT_ROOT

ALL_MODALITIES = ("flair", "t1", "t1ce", "t2")
REGIONS = ("WT", "TC", "ET")
REGION_LABELS = {"WT": (1, 2, 4), "TC": (1, 4), "ET": (4,)}


@dataclass
class PreprocessConfig:
    modalities: tuple = ALL_MODALITIES
    patch_size: int = 128
    patches_per_volume: int = 4
    tumor_ratio: float = 0.75
    seed: int = 42
    mode: str = "patches"
    max_cases: int = 0  # 0 = all cases
    source: str = str(BRATS_DIR)
    output: str = field(default="")

    def __post_init__(self):
        unknown = set(self.modalities) - set(ALL_MODALITIES)
        if unknown:
            raise ValueError(f"Unknown modalities: {unknown}")
        if not 0.0 <= self.tumor_ratio <= 1.0:
            raise ValueError("tumor_ratio must be in [0, 1]")
        if not self.output:
            name = "brats2021_patches" if self.mode == "patches" else "brats2021_preprocessed"
            self.output = str(PROCESSED_DIR / name)


def load_modality(path):
    """One NIfTI volume as float32 (X, Y, Z) in native orientation (header scaling applied)."""
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


def load_case(case_dir, modalities=ALL_MODALITIES):
    """Return image (C, X, Y, Z) float32 and label map (X, Y, Z) uint8 in native orientation."""
    case_dir = Path(case_dir)
    image = np.stack([load_modality(case_dir / f"{case_dir.name}_{m}.nii.gz") for m in modalities])
    seg = np.asarray(nib.load(case_dir / f"{case_dir.name}_seg.nii.gz").dataobj).astype(np.uint8)
    return image, seg


def brain_mask(image):
    return np.any(image != 0, axis=0)


def zscore_in_mask(image, mask):
    """Per-channel z-score over brain voxels; background set to 0."""
    out = np.zeros_like(image)
    for c in range(image.shape[0]):
        voxels = image[c][mask]
        std = voxels.std()
        out[c][mask] = (voxels - voxels.mean()) / (std if std > 0 else 1.0)
    return out


def region_targets(seg):
    """Label map -> (3, X, Y, Z) uint8 binary masks in REGIONS order."""
    return np.stack([np.isin(seg, REGION_LABELS[r]) for r in REGIONS]).astype(np.uint8)


def crop_box(mask, min_size):
    """Geometry of crop_to_mask: (lo, hi, pad) with pad[i] = (before, after) zero padding per axis."""
    idx = np.nonzero(mask)
    lo = [int(i.min()) for i in idx]
    hi = [int(i.max()) + 1 for i in idx]
    pad = [((max(min_size - (b - a), 0)) // 2, (max(min_size - (b - a), 0) + 1) // 2) for a, b in zip(lo, hi)]
    return lo, hi, pad


def crop_to_mask(image, targets, mask, min_size):
    """Crop to the mask bounding box, then zero-pad each axis up to min_size."""
    lo, hi, pad = crop_box(mask, min_size)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    image, targets = image[(slice(None), *sl)], targets[(slice(None), *sl)]
    return np.pad(image, [(0, 0)] + pad), np.pad(targets, [(0, 0)] + pad), lo


def preprocess_image(image, patch_size=128):
    """Inference-time preprocessing of (C, X, Y, Z) modalities, identical to training.

    Brain mask, z-score inside it, crop to the brain and pad to patch_size.
    Returns (image, box) where box = crop_box(...) maps predictions back with uncrop().
    """
    mask = brain_mask(image)
    if not mask.any():
        raise ValueError("volume is empty: no non-zero voxels in any modality")
    image = zscore_in_mask(image, mask)
    box = crop_box(mask, patch_size)
    lo, hi, pad = box
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return np.pad(image[(slice(None), *sl)], [(0, 0)] + pad), box


def uncrop(array, box, shape):
    """Inverse of the crop and pad: place (..., X', Y', Z') back into a zero array of spatial `shape`."""
    lo, hi, pad = box
    inner = tuple(slice(before, before + (b - a)) for (before, _), a, b in zip(pad, lo, hi))
    out = np.zeros((*array.shape[:-3], *shape), dtype=array.dtype)
    out[(..., *(slice(a, b) for a, b in zip(lo, hi)))] = array[(..., *inner)]
    return out


def preprocess_case(case_dir, modalities=ALL_MODALITIES, patch_size=128):
    image, seg = load_case(case_dir, modalities)
    mask = brain_mask(image)
    image = zscore_in_mask(image, mask)
    image, targets, offset = crop_to_mask(image, region_targets(seg), mask, patch_size)
    return image, targets, offset


def patch_starts(targets, n, tumor_ratio, patch_size, rng):
    """Return a list of (start, is_tumor_biased) for n patches.

    Tumour-biased patches are centred on a random whole-tumour voxel; clamping the
    window into the volume keeps that voxel inside the patch, so every such patch
    contains tumour. Falls back to random sampling if the case has no tumour.
    """
    shape = targets.shape[1:]
    tumor_voxels = np.argwhere(targets[0] > 0)
    n_tumor = round(n * tumor_ratio) if len(tumor_voxels) else 0
    starts = []
    for k in range(n):
        if k < n_tumor:
            center = tumor_voxels[rng.integers(len(tumor_voxels))]
            start = [int(np.clip(c - patch_size // 2, 0, s - patch_size)) for c, s in zip(center, shape)]
        else:
            start = [int(rng.integers(0, s - patch_size + 1)) for s in shape]
        starts.append((tuple(start), k < n_tumor))
    return starts


def extract(array, start, patch_size):
    return array[(slice(None), *(slice(s, s + patch_size) for s in start))]


def list_cases(source, max_cases=0):
    cases = sorted(p for p in Path(source).glob("BraTS2021_*") if p.is_dir())
    return cases[:max_cases] if max_cases else cases


def _write_volume(case_dir, out, modalities, patch_size):
    """Preprocess one case and save it; skips cases already written (resumable)."""
    path = out / f"{case_dir.name}.npz"
    if not path.exists():
        image, targets, offset = preprocess_case(case_dir, modalities, patch_size)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, image=image.astype(np.float16), targets=targets, crop_offset=np.array(offset))
        tmp.replace(path)  # atomic: a partial file never looks finished
    with np.load(path) as data:
        return {"case": case_dir.name, "shape": list(data["image"].shape[1:]),
                "crop_offset": data["crop_offset"].tolist() if "crop_offset" in data else None}


def run_volumes(cfg, workers):
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    cases = list_cases(cfg.source, cfg.max_cases)
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_write_volume, c, out, cfg.modalities, cfg.patch_size) for c in cases]
        for done, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if done % 50 == 0 or done == len(cases):
                print(f"  {done}/{len(cases)} cases", flush=True)
    records.sort(key=lambda r: r["case"])
    (out / "index.json").write_text(json.dumps({"config": asdict(cfg), "items": records}, indent=2))
    return records


def run(cfg, workers=1):
    if cfg.mode == "volumes":
        return run_volumes(cfg, workers)
    out = Path(cfg.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    records = []
    for case_dir in list_cases(cfg.source, cfg.max_cases):
        image, targets, offset = preprocess_case(case_dir, cfg.modalities, cfg.patch_size)
        for k, (start, biased) in enumerate(patch_starts(targets, cfg.patches_per_volume, cfg.tumor_ratio, cfg.patch_size, rng)):
            img_p, tgt_p = extract(image, start, cfg.patch_size), extract(targets, start, cfg.patch_size)
            name = f"{case_dir.name}_p{k}"
            np.savez_compressed(out / f"{name}.npz", image=img_p.astype(np.float16), targets=tgt_p)
            records.append({
                "patch": name, "case": case_dir.name, "start": list(start), "crop_offset": offset,
                "tumor_biased": biased, **{f"{r}_voxels": int(tgt_p[i].sum()) for i, r in enumerate(REGIONS)},
            })
        print(f"{case_dir.name}: {image.shape[1:]} -> {cfg.patches_per_volume} patches", flush=True)
    (out / "index.json").write_text(json.dumps({"config": asdict(cfg), "items": records}, indent=2))
    return records


class BratsPatchDataset(Dataset):
    """On-the-fly tumour-biased patch sampling from preprocessed volumes (mode=volumes).

    Each item draws a fresh patch, so every epoch sees different crops (call
    set_epoch() each epoch). Randomness is derived from (seed, epoch, index),
    so DataLoader worker processes, which each receive a copy of the dataset,
    do not draw identical patches.
    With flip=True each spatial axis is mirrored with probability 0.5 (the
    mirroring augmentation used by nnU-Net); image and targets flip together.
    Returns (image float32 (C, P, P, P), targets float32 (3, P, P, P)).
    """

    def __init__(self, volume_dir, case_ids, patch_size=128, tumor_ratio=0.75, samples_per_case=1, seed=0,
                 flip=False):
        self.paths = [Path(volume_dir) / f"{c}.npz" for c in case_ids]
        self.patch_size = patch_size
        self.tumor_ratio = tumor_ratio
        self.samples_per_case = samples_per_case
        self.seed = seed
        self.flip = flip
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.paths) * self.samples_per_case

    def __getitem__(self, index):
        rng = np.random.default_rng((self.seed, self.epoch, index))
        with np.load(self.paths[index // self.samples_per_case]) as data:
            image, targets = data["image"], data["targets"]
        biased = rng.random() < self.tumor_ratio
        (start, _), = patch_starts(targets, 1, 1.0 if biased else 0.0, self.patch_size, rng)
        image = extract(image, start, self.patch_size).astype(np.float32)
        targets = extract(targets, start, self.patch_size).astype(np.float32)
        if self.flip:
            axes = [axis for axis in (1, 2, 3) if rng.random() < 0.5]
            if axes:
                image, targets = np.flip(image, axes).copy(), np.flip(targets, axes).copy()
        return image, targets


def save_patch_preview(patch_path, out_path, modalities, n_slices=5):
    """Axial slice grid: one row per modality plus a FLAIR row with WT/TC/ET contours."""
    import matplotlib.pyplot as plt

    with np.load(patch_path) as data:
        image, targets = data["image"].astype(np.float32), data["targets"]
    depth = image.shape[3]
    tumor_per_slice = targets[0].sum(axis=(0, 1))
    zs = np.linspace(8, depth - 9, n_slices).round().astype(int).tolist()
    if tumor_per_slice.any():
        zs[n_slices // 2] = int(tumor_per_slice.argmax())  # always include the most tumour-rich slice
    zs = sorted(set(zs))

    rows = len(modalities) + 1
    fig, axes = plt.subplots(rows, len(zs), figsize=(2.4 * len(zs), 2.4 * rows))
    flair = modalities.index("flair") if "flair" in modalities else 0
    for col, z in enumerate(zs):
        for row, name in enumerate(modalities):
            ax = axes[row, col]
            ax.imshow(np.rot90(image[row, :, :, z]), cmap="gray", vmin=-2, vmax=4)
            ax.set_title(f"{name}  z={z}" if row == 0 else name, fontsize=8)
        ax = axes[-1, col]
        ax.imshow(np.rot90(image[flair, :, :, z]), cmap="gray", vmin=-2, vmax=4)
        for i, color in enumerate(("lime", "yellow", "red")):
            if targets[i, :, :, z].any():
                ax.contour(np.rot90(targets[i, :, :, z]), levels=[0.5], colors=color, linewidths=0.8)
        ax.set_title("WT green / TC yellow / ET red", fontsize=7)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(Path(patch_path).stem, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="JSON file with PreprocessConfig fields; CLI flags override it")
    parser.add_argument("--mode", choices=["patches", "volumes"])
    parser.add_argument("--modalities", nargs="+", choices=ALL_MODALITIES)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--patches-per-volume", type=int)
    parser.add_argument("--tumor-ratio", type=float, help="fraction of patches centred on tumour")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-cases", type=int, help="process only the first N cases (0 = all)")
    parser.add_argument("--output", type=str)
    parser.add_argument("--workers", type=int, default=1, help="parallel processes (volumes mode)")
    parser.add_argument("--preview", action="store_true", help="write slice-grid previews to reports/")
    args = parser.parse_args()

    values = json.loads(args.config.read_text()) if args.config else {}
    for key in ("mode", "modalities", "patch_size", "patches_per_volume", "tumor_ratio", "seed", "max_cases", "output"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    if "modalities" in values:
        values["modalities"] = tuple(values["modalities"])
    return PreprocessConfig(**values), args.preview, args.workers


def main():
    cfg, preview, workers = parse_args()
    records = run(cfg, workers)
    print(f"Wrote {len(records)} items to {cfg.output}")
    if preview and cfg.mode == "patches":
        reports = PROJECT_ROOT / "reports"
        biased = [r for r in records if r["tumor_biased"]][:2]
        random_ = [r for r in records if not r["tumor_biased"]][:1]
        for r in biased + random_:
            out = reports / f"seg_patch_{r['patch']}.png"
            save_patch_preview(Path(cfg.output) / f"{r['patch']}.npz", out, list(cfg.modalities))
            print(f"Preview: {out}")


if __name__ == "__main__":
    main()
