# Brain Tumour MRI: segmentation and tumour-type classification

Two deep-learning models for brain MRI, trained and evaluated on public data, served by an API and
a web application:

- **Segmentation.** A 5-fold 3D U-Net ensemble outlines glioma sub-regions (whole tumour, tumour
  core, enhancing tumour) from the four standard MRI sequences. It is trained on **BraTS 2021**:
  1,251 patients, of which 189 are held out for testing.
- **Classification.** A fine-tuned EfficientNetV2-S names the tumour type in a single 2D slice:
  glioma, meningioma, pituitary tumour or no tumour. It is trained on a **cleaned version of the
  public 4-class Brain Tumor MRI dataset**: 6,951 images after de-duplication, label-conflict
  removal and a near-duplicate-aware 70/15/15 split.

Every reported number comes from test sets that were never used for training, early stopping,
checkpoint selection or tuning. Full methods, tables and confidence intervals are in
**[reports/final_results.md](reports/final_results.md)**.

> [!IMPORTANT]
> **Not a medical device.** This is a research and portfolio demonstration. Its outputs must not
> inform diagnosis or treatment, and it has not been validated on data from outside its training
> sources.

## Results

| Task | Evaluation | Result |
|---|---|---|
| Segmentation | 5-fold cross-validation (each fold on its own validation patients) | mean Dice **0.898 ± 0.004** |
| Segmentation | Held-out test, 189 patients, ensemble + enhancing-tumour clean-up | mean Dice **0.900** (95% CI 0.883–0.914): WT 0.928, TC 0.894, ET 0.877 |
| Classification | Held-out test, 1,037 images | accuracy **98.3%** (95% CI 97.3–98.9%), macro F1 0.983 |
| Classification | Same test images shrunk to 128 × 128 | accuracy **98.2%** (the first model: 74.9%; see below) |

Ensembling the five fold models beats the best single fold by +0.009 mean Dice (95% CI +0.003 to
+0.016), better for 141 of 189 test patients. Intervals are bootstrap over test patients for
segmentation and Wilson intervals for classification accuracy.

## Highlight: finding and fixing a shortcut in the classifier

The first classifier reached 98.3% test accuracy. Error analysis showed that part of that score
came from **reading the image file format rather than the anatomy**.

1. **The errors pointed to it.** All three gliomas the model called "no tumour" show a clearly
   visible lesion. Grad-CAM placed the evidence on normal-looking brain, not on the lesion. What
   the three shared was format: none was a 512 × 512 image.
2. **The dataset explains it.** The tumour classes come mostly from sources that publish
   512 × 512 images (95% of glioma and pituitary training images), while the no-tumour class
   comes largely from a separate collection of mixed-size images (1.3% are 512 × 512, 68% are
   smaller than 256 px). Small files blur when upsampled and non-square ones get letterbox bars,
   so format leaves visible traces that predict the label. On the test set, gliomas outside the
   512 × 512 format were misclassified 56% of the time (5 of 9) against 1.5% for 512 × 512
   gliomas.
3. **A stress test measures it.** Each test image keeps its content but is resized to a typical
   no-tumour file size. A model that reads anatomy should not react. Shrunk to 128 × 128, the
   original model calls **250 of 778 tumour images "no tumour"** and its accuracy falls to 74.9%.
4. **Two fixes were tried and measured.** A random crop before letterboxing removed the
   aspect-ratio cue but not the resolution cue (still 213 of 778). Adding random down-sampling and
   JPEG re-encoding to training removed both. With a longer schedule, the final model calls
   **0 of 778** tumour images "no tumour" at 128 × 128 and keeps 98.2% accuracy.
5. **The choice rests on that evidence, not on headline accuracy.** On natural test images the
   final model and the original are statistically identical (both 98.26%, exact McNemar p = 1.0).
   The stress test is what separates them.

The stress test uses synthetic down-sampling, which is related to the training augmentation, so
robustness to images from a genuinely different source still needs an external dataset. That and
the other known weaknesses are listed under [Limitations](reports/final_results.md#limitations):
T2 and CT images are poorly covered, and normal slices through the sella can be called
"pituitary". The full analysis is in
[Models considered](reports/final_results.md#models-considered).

## The application

Screenshots are not included because every screen shows patient images from the datasets, which
are not redistributed here. Run the app locally to see it (instructions below). It has four
screens:

- **Overview.** The headline test results, read live from the evaluation outputs, with a 3D render
  of a held-out test patient's brain and predicted tumour.
- **Segment.** Drop the four NIfTI files of a study (or load the held-out sample study).
  - Each file is matched to its sequence from the file name, can be reassigned, and has its header
    checked in the browser.
  - Segmentation runs as a background job. The screen shows the real stages as the server reports
    them (preprocessing, inference model by model, post-processing).
  - The result opens in an axial, coronal and sagittal viewer with sequence switching, label
    toggles and overlay opacity, next to region volumes and model agreement.
  - Downloads: the label map (`.nii.gz`) and a PNG of the slice on screen.
- **Classify.** Upload a JPEG or PNG slice (or pick a test-set sample) and see the predicted type,
  all four probabilities and a Grad-CAM heatmap. The result downloads as one PNG.
- **Model card.** The specification, both test tables with paired comparisons, the format stress
  test chart and the limitations. Every number comes from the evaluation files.

## Architecture

```
Next.js web client ──HTTP──▶ FastAPI service
(Overview, Segment,          ├─ POST /api/classify ── synchronous ──────────▶ EfficientNetV2-S + Grad-CAM
 Classify, Model card)       ├─ POST /api/segment ── 202 + job id
                             │    └─ background worker, one job at a time ──▶ 3D U-Net, 5-fold ensemble
                             ├─ GET  /api/jobs/{id} ── stage, progress, result
                             ├─ GET  /api/results/… ── read from reports/results/
                             └─ GET  …/report.png ── PNG reports drawn on the server
```

- **Segmentation model.** A 3D U-Net (16.5 M parameters) with instance normalisation, deep
  supervision and one sigmoid output per region. It is trained on 128³ patches with Dice + BCE
  loss.
  - **Inference:** a sliding window with Gaussian blending. The five fold models' probabilities are
    averaged.
  - **Clean-up:** enhancing-tumour components under 50 voxels are removed. That threshold was tuned
    on out-of-fold validation data, never on the test set.
- **Classification model.** EfficientNetV2-S (20.2 M parameters) from ImageNet weights, fine-tuned
  end to end on grayscale slices letterboxed to 384 × 384.
  - **Augmentation:** random crops, down-sampling and JPEG re-encoding, the shortcut fix described
    above.
  - **Explanation:** Grad-CAM on the predicted class.
- **API** ([backend/README.md](backend/README.md)).
  - **Model loading:** the models load once at start-up.
  - **Classification:** answers synchronously in about a second.
  - **Segmentation:** takes seconds on a GPU and minutes on a CPU, so it runs as an in-process
    background job. The upload is validated synchronously, so bad files fail immediately with a
    specific error. The job then reports real stage progress to `GET /api/jobs/{id}`.
  - **Results endpoints:** they read `reports/results/` directly, so the web client never
    duplicates a number.
  - **PNG reports:** rendered on the server with the web client's fonts and colours.
- **Web client** ([frontend/README.md](frontend/README.md)).
  - **Stack:** Next.js (App Router) and React in TypeScript.
  - **Design system:** a small token-based design system with one colour rule: the interface is
    greyscale, and red is reserved for what the models find.
- **Tests.** `pytest` checks the API against the recorded evaluation results.
  - **Classifier:** every recorded test error is reproduced, with the same class and confidence.
  - **Segmenter:** a held-out case gives the recorded ensemble Dice.
  - **Also covered:** the job queue, every error path, and that the model card matches the results
    files.

## Running it locally

**Requirements:**

- Python 3.12;
- Node.js 20.9 or later;
- an NVIDIA GPU with CUDA for training (8 GB was enough). The API also runs on a CPU, where
  segmentation takes about 3 minutes;
- a Kaggle account for the data;
- about 25 GB of disk for the raw and preprocessed data, and about 38 GB at the peak, while the
  13 GB BraTS archive is being extracted.

The trained weights are not distributed, so the steps below rebuild them from the public data. All
commands run from the repository root. They are shown for PowerShell; in a POSIX shell use
`source .venv/Scripts/activate` (Windows) or `source .venv/bin/activate`.

**1. Environment**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt -r requirements-backend.txt
```

**2. Data.** The datasets are downloaded from Kaggle and never committed (see
[data/README.md](data/README.md)).

- **Credentials:** copy `.env.example` to `.env` and add your Kaggle username and API key. `.env`
  is gitignored.
- **Splits:** the split manifests are committed, so the test sets and folds come out exactly as
  reported.

```powershell
python -m src.download_data all                    # BraTS 2021 (~13 GB archive) and the classification dataset
python -m src.verify_data brats
python -m src.verify_data classification
python -m src.data.make_split data/manifests/figshare_4class_cleaning.json   # cleaned 70/15/15 classification split
python -m src.data.segmentation_preprocess --config configs/segmentation_preprocess.json --workers 3
```

**3. Train and evaluate.** Runs log to Weights & Biases (`python -m wandb login`). Add `--no-wandb`
to any training or evaluation command to skip that.

```powershell
foreach ($k in 1..5) { python -m src.training.train_segmentation --fold $k }   # ~3 h per fold on an RTX 4060 Laptop GPU
python -m src.training.train_classification --config configs/train_classification_crop_resdegrade_long.json
python -m src.training.evaluate_segmentation
python -m src.training.evaluate_classification --checkpoint models/classification_crop_resdegrade_long/best.pt `
    --output classification_test_crop_resdegrade_long.json
```

The commands for every other model and analysis in the report, including the format stress test,
are listed under [Reproducing](reports/final_results.md#reproducing).

**4. API**

```powershell
uvicorn backend.main:app --port 8010               # interactive docs at http://localhost:8010/docs
```

**5. Web client** (in a second terminal, with the API running)

```powershell
cd frontend
npm install
npm run dev                                        # http://localhost:3000
```

The Overview's 3D image is rendered from a test patient's data, so it is not committed either. To
generate it, see `src/visualization/render_hero.py`. Without it, the Overview shows a note in its
place. The tab icons are drawn by `src/visualization/render_icons.py`.

**6. Tests**

```powershell
python -m pytest tests
```

The integration tests need the trained checkpoints and the local data, and are skipped without them.

## Data and terms of use

- **BraTS 2021 Task 1** (RSNA-ASNR-MICCAI Brain Tumor Segmentation Challenge), obtained through
  the Kaggle mirror `dschettler8845/brats-2021-task1`. Use of the data is governed by the BraTS
  challenge's terms. Work using it should cite:
  - U. Baid et al., "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and
    Radiogenomic Classification", arXiv:2107.02314 (2021);
  - B. H. Menze et al., IEEE Transactions on Medical Imaging 34(10), 2015;
  - S. Bakas et al., Scientific Data 4, 170117 (2017).
- **Brain Tumor MRI Dataset** (Kaggle `masoudnickparvar/brain-tumor-mri-dataset`). It combines the
  Figshare release of Cheng et al., the SARTAJ dataset and Br35H. See the dataset page for its
  terms and the original sources.

No images, volumes, patient-derived renders or model weights are committed. The repository holds
the code, the split manifests and the evaluation results.

## License

The code in this repository is released under the [MIT License](LICENSE).

The licence covers this repository's own contents only. It does not extend to the BraTS 2021 data
or the Brain Tumor MRI Dataset, or to anything derived from them, such as preprocessed volumes,
renders or weights trained on them. Those remain under each dataset's own terms, described above.

## Repository layout

```
backend/     FastAPI service: models, async jobs, results and report endpoints   (backend/README.md)
frontend/    Next.js web client: Overview, Segment, Classify, Model card          (frontend/README.md)
src/         Data preparation, training, evaluation, error analysis, rendering
configs/     Training and preprocessing settings, one file per reported model
data/        Local datasets (gitignored) and the committed split manifests        (data/README.md)
reports/     final_results.md and machine-readable results in reports/results/
tests/       API and job-queue tests
models/      Trained checkpoints (gitignored)
```
