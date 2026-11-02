# Backend API

FastAPI service for the two trained models:

- **`POST /api/classify`**: tumour type from one 2D MRI image, with a Grad-CAM heatmap. The model
  is the primary classifier in `reports/final_results.md`. Each result can be downloaded as one
  PNG from **`GET /api/classify/{result_id}/report.png`**.
- **`GET /api/samples`**: five held-out test images for trying the classifier without a scan.
- **`POST /api/segment`**: 3D tumour segmentation (WT/TC/ET) from the four BraTS MRI sequences.
  The model is the 5-fold U-Net ensemble with enhancing-tumour post-processing. It runs as a
  background job, polled with **`GET /api/jobs/{job_id}`**. Each result has viewer slices
  (**`GET /api/segment/{result_id}/slices/{plane}/{layer}.png`**), 3D surfaces
  (**`GET /api/segment/{result_id}/mesh.glb`**), the label map download and a PNG report
  (**`GET /api/segment/{result_id}/report.png`**).
- **`POST /api/sample-study/segment`**: the pre-computed segmentation of a held-out BraTS test
  study, for trying the segmentation without data. **`GET /api/sample-study`** serves that study's
  files, except on the public deployment.
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
- **Weights:** `python -m src.download_weights` puts the published checkpoints where the defaults
  below expect them, or train your own (root README).

**Docker:** `backend/Dockerfile` builds this service from the repository root.

- **PyTorch build:** the `TORCH_VARIANT` build argument selects `cpu` (default) or a CUDA build
  such as `cu118`. The CPU image needs no GPU or CUDA libraries.
- **Contents:** the image holds the code, `reports/results/`, `reports/final_results.md` and
  `data/manifests/`, so the results and model card endpoints work.
- **Weights:** on start, the container runs `python -m src.download_weights`, which downloads any
  missing release weights into `/app/models`. A container with nothing mounted therefore works on
  its own. Mount `models/` there to keep them between containers. With your own checkpoints
  mounted, the download keeps them.
- **Data:** mount it at `/app/data` for the sample buttons.
- **Compose:** `docker-compose.yml` at the root wires this up with the web client; see "Running
  with Docker" in the root README.
- **Port:** `$PORT` when set, else 8010.
- **VM deployment:** the same Dockerfile runs behind Caddy on a VM; see `DEPLOYMENT.md`.

```bash
docker build -f backend/Dockerfile -t brain-tumour-api .
docker run --rm -p 8010:8010 brain-tumour-api                                # downloads the weights
docker run --rm -p 8010:8010 -v "$PWD/models:/app/models" -v "$PWD/data:/app/data:ro" brain-tumour-api
```

Settings come from environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `BT_DEVICE` | `auto` | `auto` (CUDA if available), `cuda` (fail if unavailable) or `cpu` |
| `BT_CLASSIFIER_CHECKPOINT` | `models/classification_crop_resdegrade_long/best.pt` | classifier checkpoint |
| `BT_SEGMENTATION_DIR` | `models/segmentation` | folder with `fold<k>/best.pt` |
| `BT_SEGMENTATION_FOLDS` | `1,2,3,4,5` | folds in the ensemble |
| `BT_ET_MIN_VOXELS` | tuned value (50) | enhancing-tumour component size threshold |
| `BT_CORS_ORIGINS` | `localhost` / `127.0.0.1` on ports 3000 and 5173 | allowed frontend origins, comma-separated |
| `ALLOWED_ORIGIN` | none | further allowed origins, comma-separated, added to `BT_CORS_ORIGINS`: the deployed web client, e.g. `https://your-app.vercel.app` |
| `BT_MAX_IMAGE_MB` / `BT_MAX_VOLUME_MB` | 20 / 200 | upload limits |
| `BT_RESULT_CACHE_SIZE` | 16 | recent results kept for download: segmentation masks, viewer slices and 3D surfaces, classification reports, finished jobs |
| `BT_SAMPLE_RESULTS` | `models/sample_results.npz` | where the pre-computed sample results are saved and loaded; `off` disables them |
| `BT_SAMPLE_RESULTS_REQUIRED` | `0` | `1`: refuse to start unless that file holds results for the served models (the public deployment, which cannot recompute them) |
| `BT_SAMPLE_STUDY_FILES` | `1` | `0`: no `GET /api/sample-study` endpoints, so the sample study's raw NIfTI files are never served (the public deployment) |
| `BT_DOWNLOAD_WEIGHTS` | `1` | Docker only: `0` skips downloading missing weights on start |

The default `BT_ET_MIN_VOXELS` is read from `reports/results/segmentation_test.json`, where it was
tuned on validation data.

**Pre-computed sample results** (`backend/services/samples.py`): the sample study and the five
sample images are known in advance, so their results are computed once, not on every click.

- **Computing them:** on start, the API loads them from `BT_SAMPLE_RESULTS`. If the file is
  missing, it computes them as one background job and saves them. The file is only used when the
  served checkpoints, the ET threshold and the sample files match the ones it was computed with;
  otherwise it is computed again.
- **Matching an upload:** an upload is answered from them only if it is byte for byte a sample:
  same field, file name, size and SHA-256. The web client uploads the samples unchanged, so its
  sample buttons always match.
- **The response:** the classification response, or the segmentation job, is complete at once,
  with `precomputed: true`. Every download URL in it (report, slices, mesh, label map) is served
  from the same stored result. These stored results are never evicted.
- **Other uploads** run the models exactly as before. Their only extra step is a set lookup of
  name and size.

`GET /api/health` reports `sample_results`: `ready`, `computing` or `off`.

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
  "result_id": "1d006ce7...",
  "report_url": "http://localhost:8010/api/classify/1d006ce7.../report.png",
  "predicted_class": "pituitary",
  "confidence": 0.9999,
  "probabilities": {"glioma": 0.0, "meningioma": 0.0, "notumor": 0.0001, "pituitary": 0.9999},
  "image": {"width": 512, "height": 512, "mode": "L", "grayscale": true},
  "letterbox": {"size": 384, "scale": 0.75, "offset_x": 0, "offset_y": 0, "resized_width": 384, "resized_height": 384},
  "model_input_png": "<base64>",
  "gradcam": {"target_class": "pituitary", "heatmap_png": "<base64>", "heat_png": "<base64>", "overlay_png": "<base64>",
              "contour": [[[201.5, 140.0], [188.2, 151.0], "..."]], "contour_level": 0.5, "resolution_note": "..."},
  "model": {"checkpoint": "models/classification_crop_resdegrade_long/best.pt", "checkpoint_epoch": 22, "validation": {}},
  "warnings": []
}
```

- **Model input:** `model_input_png` is the letterboxed 384 × 384 grayscale image the model saw.
- **Grad-CAM images:** all are 384 × 384 PNGs in the coordinates of the letterboxed model input.
  - `heatmap_png` is 8-bit grayscale, so a client can apply its own colormap.
  - `heat_png` is RGBA on the web client's red ramp (dark red to pale pink), transparent where
    attribution is weak. Lay it over `model_input_png`; the web client does this with an opacity
    slider.
  - `overlay_png` is a ready-made blend with the jet colormap, as in the error-inspection figures.
- **Outline:** `contour` is where the heatmap reaches `contour_level` (half) of its peak, as closed
  polylines of `[x, y]` in model-input pixels. The web client draws it as a vector line over the
  heatmap, so the region of interest reads clearly at any size.
- **Mapping back to the original image:** use the `letterbox` fields. A heatmap pixel (u, v)
  corresponds to original-image pixel ((u − offset_x) / scale, (v − offset_y) / scale).
- **Resolution:** the heatmap comes from a 12 × 12 feature map, so it is coarse. For display it is
  smoothed with a Gaussian of half a feature-map cell (16 px). That removes the cross-shaped ridges
  that bilinear upsampling leaves along the cell edges; it adds no detail the map does not have.
- **"No tumour" predictions:** the heatmap still shows what the model looked at to decide, which
  is not a lesion. The web client starts on the plain image for these and says so, and the report
  leaves the heatmap out.

**Scope:** the model was trained on single 2D slices, mostly contrast-enhanced T1. T2, FLAIR and CT
images are poorly covered (see `reports/final_results.md`).

## `GET /api/classify/{result_id}/report.png`

One PNG of a classification result, for saving or sharing: the file name and size, the model input
with the Grad-CAM overlay and its half-peak outline, the prediction with its probability, all four
probabilities as bars, and the model, export date and research-use notice. For a "no tumour"
prediction the image is shown without the heatmap, and the footer says why.

- **Size:** 2400 × 3000 (a 1200 × 1500 layout at 2×), about 0.8 MB.
- **Rendered on the server** with Pillow when first requested, then kept with the result. It uses the
  web client's fonts and colours, so it matches the screen in every browser. The fonts (Geist and
  JetBrains Mono, SIL Open Font License) are bundled in `backend/assets/fonts/` with their licences.
- **Retention:** like segmentation masks, the 16 most recent results are kept in memory
  (`BT_RESULT_CACHE_SIZE`); older ones answer `404 result_not_found`.

## `GET /api/samples`

Five held-out test images, so the classifier can be tried without a scan of one's own:
`Tr-me_1065.jpg` (meningioma), `Te-pi_102.jpg` (pituitary), `Te-no_169.jpg` (no tumour),
`Te-gl_12.jpg` (glioma) and `Te-gl_350.jpg` (glioma, one of the recorded test errors: called
meningioma). The correct glioma is listed before the error, so the failure can be compared with a
normal result of the same class. The no-tumour image is 512 × 512, the format of most tumour images,
so its correct result does not come from the file-format cue. Each entry gives the true class, the
size, whether the served classifier gets it wrong on the test set (`known_error`, from the test
report) and a `url`.

`GET /api/samples/{name}` returns the image. Only these five files are served, read from the local
cleaned test set (`data/processed/`); any other name answers `404 sample_not_found`. The images are
never committed to the repository.

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
hold an HTTP request open. The request is handled in two steps (the sample study is the exception:
its job is `complete` in the `202` response, from the pre-computed result):

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
  "study": "BraTS2021_00003",
  "files": {"flair": "BraTS2021_00003_flair.nii.gz", "t1": "...", "t1ce": "...", "t2": "..."},
  "mask_url": "http://localhost:8010/api/segment/3f0c.../mask.nii.gz",
  "report_url": "http://localhost:8010/api/segment/3f0c.../report.png",
  "mesh_url": "http://localhost:8010/api/segment/3f0c.../mesh.glb",
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
  "slices": {
    "axial": {"index": 110, "count": 155, "width": 540, "height": 664,
              "layers": {"flair": "http://localhost:8010/api/segment/3f0c.../slices/axial/flair.png", "...": "t1, t1ce, t2, edema, necrotic, enhancing"}},
    "coronal": {"index": 111, "count": 240, "...": "..."},
    "sagittal": {"index": 131, "count": 240, "...": "..."}
  },
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

**Viewer slices:** for each plane (axial, coronal, sagittal), the slice through the largest tumour
cross-section, cropped to the brain and drawn at 4× as seven PNG layers of the same size:

- `flair`, `t1`, `t1ce`, `t2`: grayscale, windowed to the 0.5–99.5 percentile of the brain,
  transparent outside the brain;
- `edema`, `necrotic`, `enhancing`: one label each, on the web client's red ramp, to stack on top.

A client stacks them and switches sequence, labels and opacity without another request. Only the
small 2D slices are kept per result; each PNG is drawn on first request and cached with the result.
`GET /api/segment/{result_id}/slices/{plane}/{layer}.png` serves them.

**Previews:** three PNG slices (axial, coronal, sagittal) through the largest tumour cross-section.
Each shows FLAIR with the labels overlaid: red for necrotic core, green for edema, yellow for
enhancing tumour. They are for quick display only; render the NIfTI for anything interactive.

## `GET /api/segment/{result_id}/mesh.glb`

The brain and the predicted tumour regions as 3D surfaces, in one binary glTF file (`.glb`). The
web client's 3D view loads it, and any glTF viewer or 3D tool can open it.

- **Nodes:**
  - `brain`: the non-zero region of the four sequences, holes filled;
  - `edema`: the whole-tumour outer surface (labels 1, 2 and 4);
  - `necrotic`: the necrotic core (label 1);
  - `enhancing`: the enhancing tumour (label 4).
- **Coordinates:** millimetres from the NIfTI voxel spacing, centred on the brain, Y up. The scene's
  `extras` hold the brain's radius and a default viewing direction.
- **How it is built** (`backend/services/meshes.py`), the same way as the Overview's static render:
  - each mask is Gaussian-smoothed and turned into a surface with marching cubes (scikit-image);
  - each surface is smoothed with Taubin's method, which does not shrink it;
  - vertex normals are computed.
- **Size:** each surface has a triangle budget (brain 90,000, tumour layers 30,000-50,000) for smooth
  rendering on an ordinary laptop. When marching cubes at full resolution exceeds the budget, its
  sampling step is coarsened. For a BraTS study the file is about 2.5 MB and about 140,000
  triangles.
- **When:** built on the first request, in about a second, then cached with the result. The cached
  result keeps the label map and a bit-packed brain mask for this, not the volumes.
- **Failure:** returns the usual error JSON. A failure is not cached, so the next request tries
  again.

## `GET /api/segment/{result_id}/report.png`

One slice with the segmentation burned in, for slides and reports, laid out like the classification
report:

- the study name, plane, slice index, sequence, grid and spacing;
- the slice with the chosen labels;
- the WT, TC and ET volumes over the whole study, with what each contains;
- the export date and result ID.

Query parameters choose what the viewer shows: `plane` (`axial`, `coronal` or `sagittal`),
`sequence` (`flair`, `t1`, `t1ce` or `t2`), `labels` (comma-separated, from `edema`, `necrotic` and
`enhancing`) and `opacity` (0-100). The image is 2400 × 3000. It is rendered on the server with
the same fonts and colours as the classification report.

## `POST /api/sample-study/segment`

The held-out test patient `BraTS2021_00003`, so the segmentation can be tried without data of
one's own. This is what the web client's "Run the sample patient" button calls.

- **Response:** `202` with a job that is already `complete`, as for `POST /api/segment`, plus the
  `case`. The job's result is the pre-computed segmentation (`precomputed: true`).
- **No files:** nothing is uploaded, and the study's volumes are not sent.
- **Not available:** `503 sample_result_unavailable` while the result is still being computed,
  `404` where there is none.

## `GET /api/sample-study`

The same study's four sequences, for local use: it lists them with their size and a `url`.
`GET /api/sample-study/{name}` returns one of those four files, read from the local BraTS data
(`data/raw/`).

- Nothing else is served, the case's ground-truth segmentation included; other names answer
  `404 sample_not_found`.
- The web client falls back to these files only while a local API is still computing the
  pre-computed result.
- With `BT_SAMPLE_STUDY_FILES=0` (the public deployment) neither endpoint exists: the BraTS terms
  do not allow redistributing the raw volumes.

## `GET /api/results/summary`

The headline held-out test results, computed from the evaluation reports rather than hard-coded:

- **Segmentation:** ensemble Dice after post-processing (mean, WT, TC, ET) and the number of test
  patients, from `segmentation_test.json`.
- **Classification:** accuracy with a 95% Wilson interval, macro F1 and the number of test images,
  from the `classification_test*.json` report of the checkpoint the API is serving, plus the
  accuracy on images shrunk to 128 px from `format_stress_test.json`. Also the number of test
  errors and how many fall on each pair of classes (`confusions`), used to put a prediction's
  runner-up in context.

The response lists the report files it used under `sources`. If a report is missing it returns
`503 results_unavailable`.

## `GET /api/results/model-card`

Everything the web client's Model card shows, computed from the same files rather than hard-coded:

- **Headline:** accuracy of the served and the original classifier on test images shrunk to 128 px
  and on unmodified images, and the McNemar p-value between them (`format_stress_test.json`, the
  `classification_test*.json` reports, the matching `classification_comparison*.json`).
- **Specification:** patient and image counts and the classification split fractions
  (`data/manifests/`), parameter counts and patch and image sizes (the loaded models), the ET
  clean-up threshold (`segmentation_test.json`).
- **Segmentation:** test Dice for the best single fold and the ensemble, each without and with ET
  clean-up; the paired bootstrap comparisons (`test_confidence_intervals.json`); the test patients
  without enhancing tumour and how many score ET Dice 1 before and after clean-up
  (`segmentation_test_per_case.csv` and the split manifest); cross-validation mean ± standard deviation.
- **Classification:** accuracy, macro F1 and accuracy at 128 px for the four variants considered.
- **Stress test:** tumour images called "no tumour" per variant and size condition.
- **Limitations:** the title and opening of each bullet under "Limitations" in
  `reports/final_results.md`.

The only fixed text is each classifier variant's label and what it changed in training. The
response lists the files it read under `sources`.

## Errors

Every error has the same shape and an HTTP status that matches the problem:

```json
{"error": {"code": "shape_mismatch", "message": "All four modalities must have the same voxel grid; got {...}"}}
```

| Status | Codes |
|---|---|
| 404 | `result_not_found`, `job_not_found`, `sample_not_found` |
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
- that the classification report renders, and that only the listed samples are served;
- that the 3D surfaces are a valid glTF file with the four named surfaces, within their triangle
  budgets, centred and in millimetres, and served from the cache the second time;
- that every segmentation viewer layer is served with matching sizes and follows the label map, that
  the segmentation report renders with the requested plane and sequence, and that only the sample
  study's four files are served;
- that the results summary matches the report files;
- every error path listed above.
