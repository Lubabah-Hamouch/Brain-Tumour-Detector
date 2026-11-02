"""Brain tumour segmentation of a BraTS-style multimodal MRI study with the 5-fold U-Net ensemble.

Every step is the evaluation pipeline (src.training.evaluate_segmentation), imported rather
than reimplemented:
    load_modality / preprocess_image  (src.data.segmentation_preprocess): z-score inside the
        brain mask, crop to the brain, pad to the patch size - the training preprocessing
    load_models / predict              (src.training.evaluate_segmentation): fold checkpoints and
        Gaussian-weighted sliding-window inference with 128^3 windows
    mean of the fold probabilities, threshold 0.5, then ET small-component removal
        (src.training.postprocess) with the threshold tuned on out-of-fold validation data
The region masks are combined into a BraTS label map (1 necrotic core, 2 edema, 4 enhancing)
and placed back into the uploaded volume's grid (uncrop), so the returned NIfTI overlays the
input directly.

Each result is kept in memory (the most recent `cache_size`) with the label map, one display
slice per plane (backend/services/slices.py) and the brain mask (bit-packed), for the label-map
download, the viewer's slice images, the PNG report and the 3D surfaces
(backend/services/meshes.py, built on first request).
"""
import base64
import gzip
import re
import tempfile
import threading
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from backend.services.cache import ResultCache
from backend.services.errors import ApiError, relative_name
from backend.services import meshes, slices
from backend.services.imaging import label_overlay, png_base64
from backend.services.report import segmentation_report
from src.data.segmentation_preprocess import ALL_MODALITIES, load_modality, preprocess_image, uncrop
from src.training.evaluate_segmentation import load_models, predict
from src.training.inference import dice_scores
from src.training.losses import REGIONS
from src.training.postprocess import components, remove_small_components

ET = REGIONS.index("ET")
BRATS_LABELS = {"necrotic_core": 1, "edema": 2, "enhancing_tumour": 4}
NIFTI_SUFFIXES = (".nii", ".nii.gz")
EXPECTED_SPACING_MM = 1.0


def nested_to_brats(regions):
    """(3, ...) binary WT/TC/ET -> BraTS label map: 4 where ET, else 1 where TC, else 2 where WT."""
    wt, tc, et = (regions[i].astype(bool) for i in range(3))
    return np.select([et, tc, wt], [4, 1, 2], 0).astype(np.uint8)


def brats_regions(labels):
    """BraTS label map -> {region: bool mask}, the inverse used for volumes and statistics."""
    return {"WT": labels > 0, "TC": np.isin(labels, (1, 4)), "ET": labels == 4}


def study_name(names):
    """A name for the study from its file names: the FLAIR name without extension and sequence suffix."""
    name = Path(names.get("flair", "") or next(iter(names.values()), "")).name
    name = re.sub(r"\.nii(\.gz)?$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[_.\- ]?(flair|t2f|fl)$", "", name, flags=re.IGNORECASE)
    return name or "study"


def read_nifti(name, data, max_mb):
    """Uploaded bytes -> (float32 volume, nibabel image). Raises ApiError with a precise reason."""
    if not name or not name.lower().endswith(NIFTI_SUFFIXES):
        raise ApiError(415, "unsupported_file_type", f"'{name}' is not a NIfTI file; expected .nii or .nii.gz.")
    if len(data) == 0:
        raise ApiError(422, "empty_file", f"'{name}' is empty.")
    if len(data) > max_mb * 2**20:
        raise ApiError(413, "file_too_large", f"'{name}' is {len(data) / 2**20:.0f} MB; the limit is {max_mb:.0f} MB.")
    suffix = ".nii.gz" if name.lower().endswith(".gz") else ".nii"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        try:
            img = nib.load(path)
            volume = load_modality(path)
        except Exception as exc:  # nibabel raises many types for truncated or invalid files
            raise ApiError(422, "corrupt_nifti", f"'{name}' could not be read as NIfTI ({type(exc).__name__}: {exc}).") from exc
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ApiError(422, "not_3d", f"'{name}' has shape {list(volume.shape)}; each modality must be one 3D volume.")
    if not np.isfinite(volume).all():
        raise ApiError(422, "non_finite_values", f"'{name}' contains NaN or infinite values.")
    return volume, img


class SegmenterService:
    def __init__(self, checkpoint_dir, folds, device, et_min_voxels, cache_size, max_volume_mb):
        self.device = device
        self.models, self.cv_scores, self.cfg = load_models(checkpoint_dir, folds, device)
        self.cfg = dict(self.cfg, mixed_precision=self.cfg["mixed_precision"] and device.type == "cuda")
        self.et_min_voxels = et_min_voxels
        self.max_volume_mb = max_volume_mb
        self.cache = ResultCache(cache_size)
        self.checkpoint_dir = relative_name(Path(checkpoint_dir))
        self._lock = threading.Lock()  # one volume at a time on the GPU
        self._mesh_lock = threading.Lock()  # build each result's surfaces once

    def load_study(self, files):
        """files: {modality: (filename, bytes)} -> (image (4, X, Y, Z), reference nibabel image, warnings)."""
        volumes, images, warnings = {}, {}, []
        for modality in ALL_MODALITIES:
            name, data = files[modality]
            volumes[modality], images[modality] = read_nifti(name, data, self.max_volume_mb)

        shapes = {m: list(v.shape) for m, v in volumes.items()}
        if len({tuple(s) for s in shapes.values()}) != 1:
            raise ApiError(422, "shape_mismatch", f"All four modalities must have the same voxel grid; got {shapes}. "
                                                  "They must be co-registered to one grid, as in BraTS.")
        zooms = {m: [round(float(z), 4) for z in img.header.get_zooms()[:3]] for m, img in images.items()}
        if any(not np.allclose(z, zooms["flair"], atol=1e-3) for z in zooms.values()):
            raise ApiError(422, "spacing_mismatch", f"All four modalities must have the same voxel spacing; got {zooms}.")
        ref = images["flair"]
        if any(not np.allclose(img.affine, ref.affine, atol=1e-3) for img in images.values()):
            warnings.append("The modalities' affines differ although their grids match; check that they are co-registered.")
        if any(abs(z - EXPECTED_SPACING_MM) > 0.1 for z in zooms["flair"]):
            warnings.append(f"Voxel spacing is {zooms['flair']} mm; the model was trained on 1 mm isotropic BraTS data "
                            "and accuracy may drop at other resolutions.")
        image = np.stack([volumes[m] for m in ALL_MODALITIES])
        if not np.any(image != 0, axis=0).any():
            raise ApiError(422, "empty_volume", "All four volumes are zero everywhere.")
        brain_fraction = float(np.any(image != 0, axis=0).mean())
        if brain_fraction > 0.5:
            warnings.append(f"{brain_fraction:.0%} of voxels are non-zero; the volumes look as if they are not "
                            "skull-stripped. The model expects skull-stripped BraTS-style input.")
        return image, ref, warnings

    def prepare(self, files):
        """Read and validate the uploaded study. Fast (well under a second) and raises ApiError on bad input,
        so it runs inside the request and the client gets a 4xx immediately."""
        t0 = time.perf_counter()
        image, ref, warnings = self.load_study(files)
        names = {m: name for m, (name, _) in files.items()}
        return {"image": image, "ref": ref, "warnings": warnings, "files": names, "study": study_name(names),
                "load_seconds": time.perf_counter() - t0}

    def segment(self, files, inline_mask=False, base_url=""):
        """Synchronous convenience: prepare + run."""
        return self.run(self.prepare(files), inline_mask, base_url)

    def run(self, study, inline_mask=False, base_url="", progress=None, result_id=None):
        """Preprocess, infer with every fold model, post-process. Slow: seconds on a GPU, minutes on a CPU.

        progress(stage, fraction, detail) is called as real work completes: stage is one of
        'preprocessing', 'inference', 'postprocessing'; fraction is the share of that stage done.
        result_id: the key to keep the result under; a new random one by default.
        """
        report = progress or (lambda stage, fraction, detail="": None)
        image, ref, warnings = study["image"], study["ref"], study["warnings"]
        shape = list(image.shape[1:])
        spacing = [float(z) for z in ref.header.get_zooms()[:3]]
        t1 = time.perf_counter()

        report("preprocessing", 0.0, "Z-score inside the brain, crop to the brain")
        x, box = preprocess_image(image, self.cfg["patch_size"])
        x_t = torch.from_numpy(x).to(self.device)
        t2 = time.perf_counter()

        with self._lock:
            total, fold_masks = None, []
            n_models = len(self.models)
            report("inference", 0.0, f"Model 1 of {n_models}")
            for i, model in enumerate(self.models.values()):
                probs = predict(model, x_t, self.cfg)
                total = probs if total is None else total + probs
                fold_masks.append(probs > 0.5)
                done = i + 1
                report("inference", done / n_models,
                       f"Model {min(done + 1, n_models)} of {n_models}" if done < n_models else f"{n_models} models done")
            ensemble = total / len(self.models)
            masks = ensemble > 0.5
            agreement = np.mean([dice_scores(m, masks) for m in fold_masks], axis=0)
            ensemble_np = ensemble.cpu().numpy()
            masks_np = masks.cpu().numpy()
            del x_t, total, fold_masks, ensemble, masks
        t3 = time.perf_counter()

        report("postprocessing", 0.0, "ET clean-up, label map, viewer slices")
        et_raw = masks_np[ET]
        _, et_sizes = components(et_raw)
        masks_np[ET] = remove_small_components(et_raw, self.et_min_voxels)
        labels_crop = nested_to_brats(masks_np)
        regions = brats_regions(labels_crop)
        voxel_ml = float(np.prod(spacing)) / 1000.0
        region_results = {}
        for i, r in enumerate(REGIONS):
            n = int(regions[r].sum())
            region_results[r] = {"voxels": n, "volume_ml": round(n * voxel_ml, 3),
                                 "mean_probability": round(float(ensemble_np[i][regions[r]].mean()), 4) if n else None,
                                 "model_agreement": round(float(agreement[i]), 4)}

        labels = uncrop(labels_crop, box, shape)
        nifti = nib.Nifti1Image(labels, ref.affine)
        nifti.header.set_xyzt_units("mm")
        mask_bytes = gzip.compress(nifti.to_bytes(), compresslevel=6)
        views = slices.extract(image, labels)
        # Non-zero voxels only: filling holes for the 3D brain surface waits until a mesh is requested.
        nonzero = np.any(image != 0, axis=0)
        entry = {"mask": mask_bytes, "views": views, "png": {}, "study": study["study"], "shape": shape,
                 "brain_bits": np.packbits(nonzero), "glb": None,
                 "spacing": spacing, "regions": region_results}
        result_id = self.cache.put(entry, result_id)
        entry["result_id"] = result_id
        t4 = time.perf_counter()

        segment_url = f"{base_url}/api/segment/{result_id}"
        return {
            "result_id": result_id,
            "study": study["study"],
            "files": study.get("files", {}),
            "mask_url": f"{segment_url}/mask.nii.gz",
            "report_url": f"{segment_url}/report.png",
            "mesh_url": f"{segment_url}/mesh.glb",
            "mask_nifti_base64": base64.b64encode(mask_bytes).decode("ascii") if inline_mask else None,
            "labels": BRATS_LABELS,
            "regions": region_results,
            "shape": shape,
            "spacing_mm": spacing,
            "tumour_bbox": self._bbox(labels > 0),
            "preview": self._previews(image[ALL_MODALITIES.index("flair")], labels),
            "slices": {plane: {"index": v["index"], "count": v["count"],
                               "width": int(v["brain"].shape[1] * slices.UPSCALE),
                               "height": int(v["brain"].shape[0] * slices.UPSCALE),
                               "layers": {layer: f"{segment_url}/slices/{plane}/{layer}.png" for layer in slices.LAYERS}}
                       for plane, v in views.items()},
            "post_processing": {"et_min_component_voxels": self.et_min_voxels,
                                "et_components_predicted": int(len(et_sizes)),
                                "et_components_removed": int((et_sizes < self.et_min_voxels).sum()),
                                "et_voxels_removed": int(et_sizes[et_sizes < self.et_min_voxels].sum())},
            "model": {"checkpoint_dir": self.checkpoint_dir, "folds": list(self.models),
                      "cross_validation_mean_dice": {f"fold{k}": round(v, 4) for k, v in self.cv_scores.items()},
                      "patch_size": self.cfg["patch_size"], "sliding_window_overlap": self.cfg["sw_overlap"]},
            "timing_seconds": {"load": round(study["load_seconds"], 2), "preprocess": round(t2 - t1, 2),
                               "inference": round(t3 - t2, 2), "postprocess_and_encode": round(t4 - t3, 2)},
            "warnings": warnings,
        }

    def mask_nifti(self, result_id):
        entry = self.cache.get(result_id)
        return entry["mask"] if entry else None

    def slice_png(self, result_id, plane, layer):
        """One viewer layer as PNG, or None if the result is no longer kept."""
        entry = self.cache.get(result_id)
        if entry is None:
            return None
        key = (plane, layer)
        if key not in entry["png"]:
            entry["png"][key] = slices.layer_png(entry["views"][plane], layer)
        return entry["png"][key]

    def mesh_glb(self, result_id):
        """The result's 3D surfaces as binary glTF, built on first request; None if no longer kept."""
        entry = self.cache.get(result_id)
        if entry is None:
            return None
        with self._mesh_lock:
            if entry["glb"] is None:
                shape = tuple(entry["shape"])
                nonzero = np.unpackbits(entry["brain_bits"], count=int(np.prod(shape))).reshape(shape).astype(bool)
                brain = meshes.fill(nonzero)
                labels = np.asarray(nib.Nifti1Image.from_bytes(gzip.decompress(entry["mask"])).dataobj)
                entry["glb"], entry["mesh_summary"] = meshes.build_glb(brain, labels, entry["spacing"])
        return entry["glb"]

    def report_png(self, result_id, plane, sequence, labels, opacity):
        """The slice on screen with the overlay burned in, plus volumes and identifiers, as one PNG."""
        entry = self.cache.get(result_id)
        if entry is None:
            return None
        view = entry["views"][plane]
        return segmentation_report(
            study=entry["study"], result_id=result_id, plane=plane, sequence=sequence,
            index=view["index"], count=view["count"], shape=entry["shape"], spacing=entry["spacing"],
            regions=entry["regions"], image=slices.composite(view, sequence, labels, opacity))

    @staticmethod
    def _bbox(mask):
        if not mask.any():
            return None
        idx = np.nonzero(mask)
        return {"min": [int(i.min()) for i in idx], "max": [int(i.max()) for i in idx]}

    @staticmethod
    def _previews(flair, labels):
        """Axial, coronal and sagittal slices through the largest tumour cross-section (or the volume centre)."""
        tumour = labels > 0
        out = []
        for plane, axis in (("axial", 2), ("coronal", 1), ("sagittal", 0)):
            other = tuple(a for a in range(3) if a != axis)
            index = int(tumour.sum(axis=other).argmax()) if tumour.any() else flair.shape[axis] // 2
            take = [slice(None)] * 3
            take[axis] = index
            out.append({"plane": plane, "index": index,
                        "png": png_base64(label_overlay(flair[tuple(take)], labels[tuple(take)]))})
        return out
