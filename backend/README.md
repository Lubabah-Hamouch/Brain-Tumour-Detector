# Backend API

FastAPI service for the two trained models:

- **`POST /api/classify`**: tumour type from one 2D MRI image, with a Grad-CAM heatmap. The model
  is the primary classifier in `reports/final_results.md`.
- **`POST /api/segment`**: 3D tumour segmentation (WT/TC/ET) from the four BraTS MRI sequences.
  The model is the 5-fold U-Net ensemble with enhancing-tumour post-processing. It runs as a
  background job, polled with **`GET /api/jobs/{job_id}`**.
- **`GET /api/results/summary`**: the headline held-out test results, read from
  `reports/results/`, for the frontend's Overview.

Both endpoints call the training and evaluation code in `src/` directly (preprocessing, models,
sliding-window inference, post-processing, Grad-CAM) rather than reimplementing it.
`tests/test_api.py` checks the API's predictions against the recorded evaluation results.

## Running

From the project root, inside the project virtual environment:

```bash
pip install -r requirements-backend.txt
uvicorn backend.main:app --port 8010
```

- **Interactive docs:** http://localhost:8010/docs. Every endpoint can be tried from the browser.
- **Startup:** about 5 s on a GPU. All six models (1 classifier, 5 segmentation folds) load once
  and stay in memory.
- **GPU memory:** 0.4 GiB with the models loaded. Segmentation inference peaks at about 2.4 GiB
  (sliding-window batch of 2, mixed precision).
- **Port:** any free port works. 8010 is used here because 8000 is often taken by local Docker
  services.

Settings come from environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `BT_DEVICE` | `auto` | `auto` (CUDA if available), `cuda` (fail if unavailable) or `cpu` |
| `BT_CLASSIFIER_CHECKPOINT` | `models/classification_crop_resdegrade_long/best.pt` | classifier checkpoint |
| `BT_SEGMENTATION_DIR` | `models/segmentation` | folder with `fold<k>/best.pt` |
| `BT_SEGMENTATION_FOLDS` | `1,2,3,4,5` | folds in the ensemble |
| `BT_ET_MIN_VOXELS` | tuned value (50) | enhancing-tumour component size threshold |
| `BT_CORS_ORIGINS` | `localhost` / `127.0.0.1` on ports 3000 and 5173 | allowed frontend origins, comma-separated |
| `BT_MAX_IMAGE_MB` / `BT_MAX_VOLUME_MB` | 20 / 200 | upload limits |
| `BT_RESULT_CACHE_SIZE` | 16 | segmentation masks kept for download |

The default `BT_ET_MIN_VOXELS` is read from `reports/results/segmentation_test.json`, where it was
tuned on validation data.

**CPU mode** (`BT_DEVICE=cpu`), measured on a 16-core laptop CPU:

- classification takes about 1 s per image;
- segmentation takes about 3.3 minutes per BraTS study (195 s of inference);
- results match GPU mode apart from rounding: CPU runs in full precision, the GPU in mixed
  precision. On the test case, WT differed by 1 voxel.

## `GET /api/health`

Returns the device, the GPU name, and which checkpoints are loaded, with their validation metrics.

## `POST /api/classify`

`multipart/form-data` with one field, `file`, containing a `.jpg`, `.jpeg` or `.png` image.

```bash
curl -F file=@slice.jpg http://localhost:8010/api/classify
```

**Preprocessing:** the image is converted to grayscale, letterboxed to 384 × 384 and
ImageNet-normalised. This is exactly what validation and test images received.

**Response:**

```json
{
  "predicted_class": "pituitary",
  "confidence": 0.9999,
  "probabilities": {"glioma": 0.0, "meningioma": 0.0, "notumor": 0.0001, "pituitary": 0.9999},
  "image": {"width": 512, "height": 512, "mode": "L"},
  "letterbox": {"size": 384, "scale": 0.75, "offset_x": 0, "offset_y": 0, "resized_width": 384, "resized_height": 384},
  "gradcam": {"target_class": "pituitary", "heatmap_png": "<base64>", "overlay_png": "<base64>", "resolution_note": "..."},
  "model": {"checkpoint": "models/classification_crop_resdegrade_long/best.pt", "checkpoint_epoch": 22, "validation": {}},
  "warnings": []
}
```

- **Grad-CAM images:** both are 384 × 384 PNGs in the coordinates of the letterboxed model input.
  - `heatmap_png` is 8-bit grayscale, so a frontend can apply its own colormap.
  - `overlay_png` is ready to display.
- **Mapping back to the original image:** use the `letterbox` fields. A heatmap pixel (u, v)
  corresponds to original-image pixel ((u − offset_x) / scale, (v − offset_y) / scale).
- **Resolution:** the heatmap comes from a 12 × 12 feature map, so it is coarse.

**Scope:** the model was trained on single 2D slices, mostly contrast-enhanced T1. T2, FLAIR and CT
images are poorly covered (see `reports/final_results.md`).

## `POST /api/segment`

`multipart/form-data` with **four required file fields**: `flair`, `t1`, `t1ce` and `t2`. Each is
one 3D NIfTI volume (`.nii` or `.nii.gz`).

```bash
curl -F flair=@case_flair.nii.gz -F t1=@case_t1.nii.gz -F t1ce=@case_t1ce.nii.gz -F t2=@case_t2.nii.gz \
     http://localhost:8010/api/segment
# {"job_id": "9b1e...", "status": "queued", "status_url": "http://localhost:8010/api/jobs/9b1e..."}
curl http://localhost:8010/api/jobs/9b1e...
```

**Asynchronous:** segmentation takes seconds on a GPU and minutes on a CPU, which is too long to
hold an HTTP request open. The request is handled in two steps:

1. **Validation, synchronous.** The four volumes are read and checked (file types, sizes, NIfTI
   headers, matching grids) before the response. Any problem is returned immediately as a normal
   4xx error, and no job is created.
2. **Inference, in the background.** The response is `202 Accepted` with a `job_id`, a
   `status_url` and a `Location` header. Poll `GET /api/jobs/{job_id}` until `status` is
   `complete` or `failed`.

Jobs run in the API process, one at a time (a single worker thread, so the GPU is never shared).
Further jobs wait in a queue and report their `queue_position`.

**Job status** (`GET /api/jobs/{job_id}`):

```json
{
  "job_id": "9b1e...",
  "kind": "segmentation",
  "status": "running",
  "queue_position": null,
  "stage": "inference",
  "stage_label": "Inference",
  "stage_detail": "Model 3 of 5",
  "progress": 0.61,
  "stages": [
    {"name": "preprocessing", "label": "Preprocess", "status": "complete", "fraction": 1.0, "detail": "", "seconds": 0.7},
    {"name": "inference", "label": "Inference", "status": "running", "fraction": 0.6, "detail": "Model 3 of 5", "seconds": 2.1},
    {"name": "postprocessing", "label": "Post-process and encode", "status": "pending", "fraction": 0.0, "detail": "", "seconds": null}
  ],
  "created_at": 1793530000.1,
  "started_at": 1793530000.2,
  "finished_at": null,
  "elapsed_seconds": 2.9,
  "result": null,
  "error": null
}
```

- **`status`:** `queued`, `running`, `complete` or `failed`.
- **Stages:** `preprocessing`, then `inference` (which advances once per fold model), then
  `postprocessing`. The pipeline reports each stage as it runs; nothing is simulated.
- **`progress`:** overall 0-1, weighting each stage by its typical share of the run time.
- **`result`:** once `complete`, the segmentation result described below.
- **`error`:** once `failed`, `{"code", "message"}` in the same shape as every other API error.
- **Retention:** the 16 most recent finished jobs are kept (`BT_RESULT_CACHE_SIZE`); older ones
  answer `404 job_not_found`. Jobs are lost when the server restarts.

**Why all four sequences:** the network takes the four sequences as its four input channels.
There is no way to run it on fewer, so a request with any field missing is rejected.

**Input requirements:**

- **Co-registered:** all four volumes share one voxel grid (same shape and spacing), as in BraTS.
  A shape or spacing mismatch is rejected.
- **Skull-stripped:** the brain is found as the non-zero voxels. Input where more than half the
  voxels are non-zero gets a warning that it looks not skull-stripped.
- **1 mm isotropic spacing,** e.g. 240 × 240 × 155. Other spacings are processed but return a
  warning.

Raw DICOM, or volumes that are not co-registered or not skull-stripped, need the standard BraTS
preprocessing first (for example with the BraTS Toolkit or CaPTk).

**Pipeline:** identical to the test evaluation.

1. z-score normalisation inside the brain mask, then crop to the brain;
2. sliding-window inference with 128³ windows, 50% overlap and Gaussian blending, for each of the
   5 fold models;
3. the mean of the five probability maps, thresholded at 0.5;
4. removal of enhancing-tumour components smaller than 50 voxels;
5. conversion to a BraTS label map, placed back into the uploaded grid.

A 240 × 240 × 155 study takes about 5-6 s on an RTX 4060 Laptop GPU, 3.5 s of which is inference.

**Result** (the job's `result` field):

```json
{
  "result_id": "3f0c...",
  "mask_url": "http://localhost:8010/api/segment/3f0c.../mask.nii.gz",
  "mask_nifti_base64": null,
  "labels": {"necrotic_core": 1, "edema": 2, "enhancing_tumour": 4},
  "regions": {
    "WT": {"voxels": 97337, "volume_ml": 97.337, "mean_probability": 0.9868, "model_agreement": 0.9936},
    "TC": {"voxels": 41519, "volume_ml": 41.519, "mean_probability": 0.9893, "model_agreement": 0.9954},
    "ET": {"voxels": 24924, "volume_ml": 24.924, "mean_probability": 0.9806, "model_agreement": 0.9883}
  },
  "shape": [240, 240, 155],
  "spacing_mm": [1.0, 1.0, 1.0],
  "tumour_bbox": {"min": [x0, y0, z0], "max": [x1, y1, z1]},
  "preview": [{"plane": "axial", "index": 110, "png": "<base64>"}, "... coronal, sagittal"],
  "post_processing": {"et_min_component_voxels": 50, "et_components_predicted": 3, "et_components_removed": 2, "et_voxels_removed": 2},
  "model": {"folds": [1, 2, 3, 4, 5], "cross_validation_mean_dice": {}},
  "timing_seconds": {"load": 0.7, "preprocess": 0.7, "inference": 3.5, "postprocess_and_encode": 0.4},
  "warnings": []
}
```

**The label map** is the main output.

- **Format:** a gzipped NIfTI in the same voxel grid and affine as the uploaded volumes, with values
  0, 1 (necrotic core), 2 (edema) and 4 (enhancing tumour). It overlays the input directly in any
  NIfTI viewer, such as NiiVue, Papaya or ITK-SNAP.
- **Size:** about 20-30 KB for a BraTS study. The uncompressed array is 8.9 MB.
- **Retrieval:** download it from `mask_url`, or add `?inline_mask=true` to the `POST` to receive it
  base64-encoded in the job result.
- **Retention:** downloads stay available for the 16 most recent results and are lost when the
  server restarts.

**Region definitions:**

| Region | Labels |
|---|---|
| WT (whole tumour) | 1, 2, 4 |
| TC (tumour core) | 1, 4 |
| ET (enhancing tumour) | 4 |

Volumes use the voxel spacing from the NIfTI header.

**Label map vs the evaluation's region masks:** the network predicts the three regions
independently. The label map assigns ET first, then the rest of TC, then the rest of WT. A voxel
predicted as ET but not TC therefore counts towards TC in the label map. This changes Dice by about
0.001 relative to the evaluation's direct region masks.

**Region statistics:**

- `mean_probability`: the mean ensemble probability inside the region.
- `model_agreement`: the mean Dice between each fold model's own mask and the ensemble mask. This
  measures how consistently the five models agree, which flags uncertain cases. It is **not** an
  estimate of accuracy against ground truth.

**Previews:** three PNG slices (axial, coronal, sagittal) through the largest tumour cross-section.
Each shows FLAIR with the labels overlaid: red for necrotic core, green for edema, yellow for
enhancing tumour. They are for quick display only; render the NIfTI for anything interactive.

## `GET /api/results/summary`

The headline held-out test results, computed from the evaluation reports rather than hard-coded:

- **Segmentation:** ensemble Dice after post-processing (mean, WT, TC, ET) and the number of test
  patients, from `segmentation_test.json`.
- **Classification:** accuracy with a 95% Wilson interval, macro F1 and the number of test images,
  from the `classification_test*.json` report of the checkpoint the API is serving, plus the
  accuracy on images shrunk to 128 px from `format_stress_test.json`.

The response lists the report files it used under `sources`. If a report is missing it returns
`503 results_unavailable`.

## Errors

Every error has the same shape and an HTTP status that matches the problem:

```json
{"error": {"code": "shape_mismatch", "message": "All four modalities must have the same voxel grid; got {...}"}}
```

| Status | Codes |
|---|---|
| 404 | `result_not_found`, `job_not_found` |
| 413 | `file_too_large` |
| 415 | `unsupported_file_type` |
| 422 | `missing_fields`, `empty_file`, `unreadable_image`, `image_too_small`, `corrupt_nifti`, `not_3d`, `non_finite_values`, `shape_mismatch`, `spacing_mismatch`, `empty_volume`, `invalid_request` |
| 503 | `model_unavailable`, `results_unavailable` |

## Tests

```bash
pytest tests/test_api.py tests/test_jobs.py
```

`tests/test_jobs.py` covers the job queue on its own, without models: progress reporting, results,
failures, serial execution and eviction.

The tests need the checkpoints and local datasets; they are skipped without them. They check:

- the response shapes;
- that every one of the classifier's 18 recorded test errors is reproduced with the same class and
  confidence (±0.01);
- that 40 sampled correctly classified test images stay correct;
- that the segmentation label map for a held-out BraTS test case gives the recorded ensemble Dice
  (±0.005);
- that a segmentation job passes through the real stages in order;
- that the results summary matches the report files;
- every error path listed above.
