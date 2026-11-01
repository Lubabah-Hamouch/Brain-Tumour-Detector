# Final results

Results for both tasks on held-out test sets that were never used for training, early stopping,
checkpoint selection, fold selection or post-processing tuning. Every number below can be
regenerated from the trained checkpoints with the commands in [Reproducing](#reproducing).
Machine-readable versions are in [`results/`](results/).

| Task | Model | Test set | Headline result |
|---|---|---|---|
| Brain tumour segmentation (BraTS 2021) | 3D U-Net, 5-fold ensemble + ET post-processing | 189 patients | mean Dice **0.900** (WT 0.928, TC 0.894, ET 0.877) |
| Tumour type classification (4 classes) | EfficientNetV2-S, fine-tuned with crop + resolution-degradation augmentation | 1,037 images | accuracy **98.3%** (95% CI 97.3-98.9%), macro F1 0.983; stays at 98.2% when test images are shrunk to 128 × 128 (see [Models considered](#models-considered)) |

---

## 1. Segmentation

### Setup

- **Data:** BraTS 2021 Task 1, 1,251 patients, four MRI sequences (FLAIR, T1, T1ce, T2). Each volume
  is z-scored inside the brain mask and cropped to the brain. The targets are the three nested BraTS
  regions: whole tumour (WT), tumour core (TC) and enhancing tumour (ET).
- **Split** (`data/manifests/brats_5fold_splits.json`):
  - 189 patients (15%) are held out as the test set.
  - The remaining 1,062 are divided into 5 cross-validation folds of 212-213 patients.
  - Assignment is stratified by ET presence and whole-tumour volume quartile, and every split is
    patient-level.
- **Model:** 3D U-Net with 16.5 M parameters:
  - instance norm and LeakyReLU;
  - deep supervision at three resolutions;
  - one sigmoid output per region.
- **Training (per fold):**
  - 40 epochs; each epoch takes one tumour-biased 128³ patch per training patient, with random
    mirroring.
  - Loss: Dice + BCE per region.
  - Optimiser: AdamW at 3e-4 with 2 warmup epochs, then cosine decay. Batch 1 with 2-step gradient
    accumulation, mixed precision.
  - The checkpoint kept is the epoch with the best validation Dice.
  - Each fold took about 3 h on an RTX 4060 Laptop GPU (8 GB).
- **Inference:**
  - Sliding window of 128³ windows with 50% overlap and Gaussian-weighted blending.
  - Probabilities are thresholded at 0.5.
  - Dice is computed per patient on the full volume, then averaged. Under the BraTS convention, a
    region absent from both ground truth and prediction scores 1, and a region absent from ground
    truth but predicted scores 0.

### Cross-validation (validation folds)

Each fold's score is from its own validation fold at the selected epoch. These numbers were used
for checkpoint selection, so they are slightly optimistic.

| Fold | Val patients | Best epoch | WT | TC | ET | Mean |
|---|---|---|---|---|---|---|
| 1 | 213 | 36 | 0.9273 | 0.9038 | 0.8538 | 0.8950 |
| 2 | 213 | 32 | 0.9267 | 0.9050 | 0.8717 | 0.9011 |
| 3 | 212 | 40 | 0.9243 | 0.9030 | 0.8520 | 0.8931 |
| 4 | 212 | 36 | 0.9254 | 0.9056 | 0.8805 | 0.9039 |
| 5 | 212 | 40 | 0.9253 | 0.8971 | 0.8733 | 0.8986 |
| **Mean ± std** | | | **0.9258 ± 0.0012** | **0.9029 ± 0.0034** | **0.8663 ± 0.0127** | **0.8983 ± 0.0044** |

Standard deviation is the sample standard deviation across the 5 folds.

### Test set (189 patients)

"Single best" is fold 4, chosen by cross-validation score before the test set was evaluated. The
ensemble averages the five models' sigmoid probabilities before thresholding.

| Model | ET post-processing | WT | TC | ET | Mean |
|---|---|---|---|---|---|
| Single best (fold 4) | no | 0.9218 | 0.8835 | 0.8566 | 0.8873 |
| Single best (fold 4) | yes | 0.9218 | 0.8835 | 0.8672 | 0.8908 |
| 5-fold ensemble | no | 0.9278 | 0.8939 | 0.8673 | 0.8963 |
| **5-fold ensemble** | **yes** | **0.9278** | **0.8939** | **0.8773** | **0.8997** |

Post-processing changes only ET, so the WT and TC columns are identical with and without it.

95% bootstrap confidence intervals (10,000 resamples of the test patients):

| Model | WT | TC | ET | Mean |
|---|---|---|---|---|
| Ensemble + post-processing | 0.917-0.938 | 0.868-0.917 | 0.855-0.897 | 0.883-0.914 |
| Single best + post-processing | 0.907-0.934 | 0.856-0.908 | 0.843-0.888 | 0.873-0.907 |

The intervals are wide because per-patient Dice varies a lot. For the ensemble with
post-processing, the median per-patient Dice and interquartile range are:

| Region | Median | IQR |
|---|---|---|
| WT | 0.950 | 0.912-0.971 |
| TC | 0.954 | 0.906-0.977 |
| ET | 0.922 | 0.857-0.955 |

A minority of hard cases pulls the means down.

Each fold model on its own, for reference:

| Fold | WT | TC | ET (raw) | ET (post-processed) | Mean (raw) | Mean (post-processed) |
|---|---|---|---|---|---|---|
| 1 | 0.9208 | 0.8868 | 0.8531 | 0.8634 | 0.8869 | 0.8903 |
| 2 | 0.9213 | 0.8949 | 0.8696 | 0.8696 | 0.8953 | 0.8953 |
| 3 | 0.9229 | 0.8846 | 0.8613 | 0.8610 | 0.8896 | 0.8895 |
| 4 | 0.9218 | 0.8835 | 0.8566 | 0.8672 | 0.8873 | 0.8908 |
| 5 | 0.9198 | 0.8910 | 0.8714 | 0.8765 | 0.8940 | 0.8958 |
| Mean ± std | 0.9213 ± 0.0012 | 0.8881 ± 0.0047 | 0.8624 ± 0.0080 | 0.8676 ± 0.0060 | 0.8906 ± 0.0039 | 0.8923 ± 0.0029 |

### What the comparisons show

Paired differences in mean Dice over the same 189 test patients, with 95% bootstrap CIs:

| Comparison | Difference | 95% CI | Patients better / worse |
|---|---|---|---|
| Ensemble vs single best (both post-processed) | **+0.0088** | +0.0033 to +0.0156 | 141 / 48 |
| Ensemble vs single best (both raw) | +0.0090 | +0.0036 to +0.0157 | 143 / 46 |
| Post-processing on the ensemble | +0.0033 | -0.0004 to +0.0088 | 26 / 44 |
| Post-processing on the single best | +0.0036 | -0.0000 to +0.0089 | 38 / 40 |

- **Ensembling is a clear improvement.** It improves all three regions (raw: WT +0.006, TC +0.010,
  ET +0.011), the interval excludes zero, and three quarters of patients score higher.
- **ET post-processing is not a clear improvement in mean Dice.** Its intervals include zero. The
  +0.010 ET gain for the ensemble comes entirely from the 5 test patients without enhancing tumour:
  - Raw, 3 of the 5 score ET Dice 1. After small false-positive components are removed, all 5 do.
    Each patient moved from 0 to 1 adds 1/189 to the mean.
  - On the 184 patients who have ET, post-processing slightly lowers ET Dice (0.8745 to 0.8740). It
    removes small fragments that touch real enhancing tumour.

  Whether to keep it depends on what matters more for the use case: avoiding a spurious "enhancing
  tumour present" call, or the last fraction of overlap on true ET. Both versions are reported.
- **Test vs cross-validation:**
  - The single-model test mean (0.8906 ± 0.0039 across folds) is about 0.008 below the
    cross-validation mean (0.8983). That gap is consistent with the optimism of selecting
    checkpoints on the validation folds.
  - WT and ET land close to their CV values. TC drops the most, by about 0.015.

### ET post-processing: how the threshold was chosen

The rule: remove every predicted ET connected component (26-connectivity) smaller than **50
voxels** (50 mm³ at the 1 mm BraTS resolution). Only the ET channel is changed. The threshold was
fixed using validation data only, before any test inference:

1. Each fold model segmented its own validation fold, giving out-of-fold predictions for all 1,062
   non-test patients.
2. Every predicted ET component was classified as a true positive (overlapping ground-truth ET) or
   a pure false positive. Their sizes in voxels:

   | Component type | Count | 10th percentile | Median | 75th percentile | 90th percentile |
   |---|---|---|---|---|---|
   | Pure false positive | 597 | 1 | 4 | 20 | 85 |
   | Overlapping true ET | 1,789 | 2 | 4,861 | 21,482 | 37,698 |

   The false positives are overwhelmingly tiny. Most true components are thousands of voxels,
   although some true lesions also produce small fragments.
3. Mean out-of-fold ET Dice was computed for each candidate threshold:

   | Minimum component size (voxels) | 0 | 10 | 25 | **50** | 75 | 100 | 200 | 300 | 500 | 1000 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | Mean out-of-fold ET Dice | 0.8663 | 0.8663 | 0.8682 | **0.8691** | 0.8687 | 0.8680 | 0.8677 | 0.8657 | 0.8635 | 0.8526 |

   50 voxels maximises it. At that threshold, 507 of the 597 false-positive components are removed,
   along with 502 small true-positive fragments. Larger thresholds increasingly delete real small
   lesions, and Dice declines steadily.
4. On the validation folds, the rule raised the share of no-ET patients scored correctly from 16 of
   28 to 19 of 28.

---

## 2. Classification

### Setup

- **Data:** the 4-class Brain Tumor MRI dataset (glioma, meningioma, no tumour, pituitary) after
  cleaning:
  - Near-duplicate clusters were found by perceptual hashing.
  - Pre-augmented copies, label-conflict clusters and 34 watermarked images (a label shortcut)
    were removed.
  - The cleaned set has 6,951 images, split 70/15/15. Every near-duplicate cluster is assigned
    wholly to one split, stratified by class (see `data/README.md`).
  - The dataset has no patient identifiers, so near-duplicate clusters are the closest available
    proxy for patient-level separation.
- **Images:** grayscale, letterboxed to 384 × 384, replicated to 3 channels, ImageNet
  normalisation. Validation and test images get exactly this preprocessing and nothing else.
- **Model:** EfficientNetV2-S with ImageNet weights, fine-tuned end to end:
  - backbone learning rate 1e-4, head 1e-3, AdamW;
  - 1 warmup epoch, then cosine decay;
  - class-weighted cross-entropy (inverse frequency).
- **Training augmentation (primary model,
  `configs/train_classification_crop_resdegrade_long.json`):**
  - On the original image, before letterboxing:
    1. a random crop of 70-100% of the area, with width/height ratio 0.75-1.33;
    2. for half of the images, down-sampling to a shorter side of 150-320 px, then JPEG
       re-encoding at quality 30-95.
  - After letterboxing: flips, small affine transforms, brightness/contrast jitter.
  - The steps before letterboxing exist to remove an image-format shortcut found during error
    analysis (see [Models considered](#models-considered)).
- **Stopping and selection:**
  - Training stops when validation loss has not improved for 10 epochs; this happened after
    epoch 23.
  - The checkpoint kept is the epoch with the best validation macro F1, epoch 22.

### Primary model: test results

`models/classification_crop_resdegrade_long/best.pt`:

| Split | Images | Accuracy | Macro F1 |
|---|---|---|---|
| Validation (used for checkpoint selection) | 1,039 | 98.85% | 0.9883 |
| **Test** | **1,037** | **98.26%** (95% CI 97.27-98.90%, Wilson) | **0.9827** |

Test results per class:

| Class | Images | Precision | Recall | F1 |
|---|---|---|---|---|
| Glioma | 270 | 0.9814 | 0.9778 | 0.9796 |
| Meningioma | 238 | 0.9833 | 0.9874 | 0.9853 |
| No tumour | 259 | 0.9921 | 0.9691 | 0.9805 |
| Pituitary | 270 | 0.9746 | 0.9963 | 0.9853 |

Test confusion matrix (rows: true class, columns: predicted):

| | Glioma | Meningioma | No tumour | Pituitary |
|---|---|---|---|---|
| **Glioma** | 264 | 4 | 2 | 0 |
| **Meningioma** | 1 | 235 | 0 | 2 |
| **No tumour** | 3 | 0 | 251 | 5 |
| **Pituitary** | 1 | 0 | 0 | 269 |

**The 18 errors:**

- **Missed tumours: 2 gliomas called "no tumour".** Both are scan types the training data barely
  covers:
  - `Te-gl_97`, a T2 image;
  - `Te-gl_318`, a CT image.
- **No tumour called pituitary: 5 images.** All five (`Tr-no_173`, `_344`, `_539`, `_543`,
  `_674`) are skull-base slices through the sella turcica from what appears to be a single series.
  The model associates that anatomical level with the pituitary class.
- **No tumour called glioma: 3 images.** These include `Tr-no_261` at 1.00 confidence and
  `Tr-no_215` at 0.92.
- **Glioma and meningioma confused: 5 images.**
- **Other: 3 images.** Two meningiomas were called pituitary and one pituitary image was called
  glioma.

### Models considered

Four classifiers were trained on the same split with the same seed and schedule. Only the
augmentation and the early-stopping patience differ. Each was evaluated once on the test set.

| Model | Change | Test accuracy | Macro F1 | Test loss |
|---|---|---|---|---|
| Original | baseline augmentation, patience 5 | 98.26% | 0.9826 | 0.0766 |
| Crop | + random crop before letterboxing | 98.17% | 0.9815 | 0.0505 |
| Crop + resolution | + down-sampling and JPEG re-encoding | 97.78% | 0.9777 | 0.0808 |
| **Crop + resolution, patience 10 (primary)** | same, early-stopping patience 10 | **98.26%** | **0.9827** | 0.0687 |

**On natural test images, the primary model and the original are statistically identical.** Each
is wrong on 8 images the other gets right (exact McNemar p = 1.0), and no pair of models differs
significantly.

**The deciding evidence is the format stress test.** Error analysis showed that image format
(file size, aspect ratio, compression) predicts the class in this dataset:

- 95% of glioma and pituitary training images are 512 × 512;
- only 1.3% of no-tumour images are, and 68% of them are smaller than 256 px.

The stress test leaves each test image's content unchanged and resizes the file to a typical
no-tumour size. A model that reads anatomy should not react.

| Tumour images (of 778) called "no tumour" | Unmodified | 225 × 225 | 210 × 234 | 128 × 128 |
|---|---|---|---|---|
| Original | 3 | 12 | 10 | **250** |
| Crop | 1 | 14 | 16 | **213** |
| Crop + resolution | 2 | 3 | 3 | 9 |
| **Crop + resolution, patience 10 (primary)** | 2 | 1 | 1 | **0** |

| Overall test accuracy | Unmodified | 225 × 225 | 210 × 234 | 128 × 128 |
|---|---|---|---|---|
| Original | 98.26% | 97.30% | 97.49% | 74.93% |
| Crop | 98.17% | 97.49% | 97.11% | 77.53% |
| Crop + resolution | 97.78% | 97.78% | 97.97% | 97.11% |
| **Crop + resolution, patience 10 (primary)** | **98.26%** | **98.07%** | **98.26%** | **98.17%** |

**Reading the stress test:**

- **The original model's 98.3% depends partly on the file format.** Shrink tumour images to
  128 × 128 and it calls a third of them (250 of 778) "no tumour"; two thirds of gliomas are lost.
- **The primary model reaches the same accuracy without that dependence.** Its predictions barely
  move under any resize: 5-11 of 1,037 change, against 15-257 for the original.
- **This is why it was chosen** even though its accuracy on natural images is statistically the same
  as the original's.
- **Caveat:** the stress test uses synthetic down-sampling, which is related to the training
  augmentation. Robustness to images from a genuinely different source would need an external
  dataset to confirm.

The subsections below record how the models were developed, in order:

- the original model's results and error analysis, which uncovered the shortcut;
- the two mitigation attempts;
- the longer-training check.

### Original model: results

`models/classification/best.pt`, `configs/train_classification.json`:

- **Augmentation:** flips, affine transforms and intensity jitter only.
- **Stopping:** early-stopping patience 5; training stopped after epoch 12, and the checkpoint kept
  is epoch 7.
- **Scope of this subsection and the next two:** they describe this model. Its error analysis led to
  the mitigations below.

| Split | Images | Accuracy | Macro F1 |
|---|---|---|---|
| Validation (used for checkpoint selection) | 1,039 | 98.85% | 0.9881 |
| **Test** | **1,037** | **98.26%** (95% CI 97.27-98.90%, Wilson) | **0.9826** |

Test results per class:

| Class | Images | Precision | Recall | F1 |
|---|---|---|---|---|
| Glioma | 270 | 0.9924 | 0.9667 | 0.9794 |
| Meningioma | 238 | 0.9790 | 0.9790 | 0.9790 |
| No tumour | 259 | 0.9884 | 0.9846 | 0.9865 |
| Pituitary | 270 | 0.9712 | 1.0000 | 0.9854 |

Test confusion matrix (rows: true class, columns: predicted):

| | Glioma | Meningioma | No tumour | Pituitary |
|---|---|---|---|---|
| **Glioma** | 261 | 5 | 3 | 1 |
| **Meningioma** | 2 | 233 | 0 | 3 |
| **No tumour** | 0 | 0 | 255 | 4 |
| **Pituitary** | 0 | 0 | 0 | 270 |

### Original model: errors (18 of 1,037)

The full list, with file names and confidences, is in `results/classification_test.json`, and the
images are in the wandb run.

- **Missed tumours: 3 gliomas predicted as no tumour** (`Te-gl_350`, `Te-gl_97`, `Te-gl_318`). This
  is the clinically most serious error type, at 3 of the 778 tumour images (0.4%). Visual
  inspection (below) shows these are **genuine misses of visible lesions, not labelling errors**.
- **No tumour predicted as pituitary: 4 images.** These are false alarms, and they make up most of
  pituitary's precision loss. Inspection shows tumour-free slices through the sella region (below).
- **Glioma and meningioma confused: 7 images** (5 one way, 2 the other). This was also the main
  confusion on the validation set; the two can look similar on a single 2D slice.
- **Other: 4 images.** 3 meningioma and 1 glioma were predicted as pituitary.
- **Generalisation:** test accuracy is 0.6 points below validation accuracy. That is consistent
  with selecting the checkpoint on validation data, and the test result is the one to quote.

### Inspection of the most serious errors

The two error groups with clinical consequences were inspected image by image: tumours called
"no tumour", and normal brains called "pituitary". For each image, the review covered:

- the original file;
- the 384 × 384 input the model sees;
- the class probabilities;
- a Grad-CAM heatmap for the predicted class.

The figures are generated with `src.training.inspect_errors` (see [Reproducing](#reproducing)). They
are not committed because they contain dataset images. The hypothesis going in was that the three
"missed" gliomas might be tumour-free slices carrying a wrong source label, like the label
conflicts found during dataset cleaning. **That hypothesis was wrong.**

**Gliomas predicted as no tumour: all three show a clearly visible lesion.**

| Image | Original format | What it shows | P(no tumour) |
|---|---|---|---|
| `Te-gl_350` | 175 × 167, RGB | Axial FLAIR-type MRI with a large hyperintense frontal lesion | 0.9996 |
| `Te-gl_97` | 504 × 608, RGB | Axial T2 MRI with a large hyperintense lesion in the lateral hemisphere | 0.9998 |
| `Te-gl_318` | 197 × 229, RGB | **A CT scan, not MRI**, with a rounded lesion showing internal layering | 0.9077 |

- None of the three has a near-duplicate anywhere in the dataset, so there is no conflicting copy
  that would suggest a source labelling mistake.
- `Te-gl_318` is a data-quality problem in its own right: a CT image filed in an MRI dataset. Its
  tumour-type label cannot be verified from the image.
- **The Grad-CAM heatmaps agree that the lesion was not used.** For the predicted "no tumour"
  class, the attribution sits on normal-appearing brain away from the lesion, and the lesion
  itself contributes little. The model judged the rest of the slice to look normal and did not use
  the tumour.

**Why these were missed: an image-format shortcut.** The three images share one property: none is
in the 512 × 512 format that dominates the tumour classes. In this dataset, the tumour classes come
mainly from Figshare and SARTAJ, while the no-tumour class comes largely from Br35H, a separately collected
set of mixed-size images. Image format therefore predicts the label:

| Training split | 512 × 512 | Smaller side < 256 px | Square |
|---|---|---|---|
| Glioma | 95.3% | 1.0% | 95.9% |
| Meningioma | 79.9% | 8.7% | 82.7% |
| **No tumour** | **1.3%** | **68.2%** | **32.7%** |
| Pituitary | 95.2% | 1.0% | 97.5% |

The model can pick up this cue without looking at the anatomy. Format leaves clear traces in the
input:

- small images are blurred when upsampled to 384 px;
- non-square images get black letterbox bars;
- JPEG compression differs between sources.

The test errors match this explanation:

| Test glioma images | Misclassified |
|---|---|
| 512 × 512 (261 images) | 4 (1.5%) |
| **Any other format (9 images)** | **5 (56%)**, including all 3 predicted as no tumour |

Nine images is a small sample, but the gap is large, and it matches the direction of the
training-set imbalance. The headline 98.3% accuracy is measured on a test set with the same
format-label correlation as training. It therefore **overstates performance on images whose format
does not match the source each class came from**, for example tumour images from a different
scanner, export pipeline or modality.

**No tumour predicted as pituitary (`Tr-no_173`, `Tr-no_344`, `Tr-no_674`, `Tr-no_907`): the
labels look correct.**

- None of the four shows a tumour, so the "no tumour" label stands.
- Three are near-identical axial slices at the skull base, through the sella turcica where the
  pituitary gland sits, apparently from one patient. The fourth is a normal mid-sagittal T1 slice
  with the sella in full view.
- Pituitary-tumour images are dominated by exactly these views. The model has partly learned
  "slice through the sella" rather than "pituitary tumour present", which is an anatomical-level
  shortcut and not tumour detection.

### Mitigation attempt: random-crop augmentation

The classifier was retrained from scratch with one change, to try to break the format shortcut:
a **random crop of the original image before letterboxing** (`RandomAreaCrop`, enabled in
`configs/train_classification_crop.json`).

- **The crop:** each training image is cropped to 70-100% of its area, with a width/height ratio
  between 0.75 and 1.33, and then letterboxed to 384 × 384 as before.
- **Why before letterboxing:** tumour images, almost all 512 × 512 squares, then appear with the
  same varied shapes and padding bars as the mixed-format no-tumour images. Cropping after
  letterboxing would keep each source's original bars and proportions.
- **Everything else is identical:** the split, seed, augmentations, schedule, early stopping and
  checkpoint rule. Validation and test preprocessing are unchanged.
- **Training:** stopped early after epoch 12. The checkpoint kept is epoch 9, with validation
  accuracy 98.56% and macro F1 0.985.
- **Both checkpoints are kept:** `models/classification/` holds the original and
  `models/classification_crop/` the mitigated model.

**Headline metrics (1,037 test images):**

| | Original | Crop-mitigated |
|---|---|---|
| Accuracy | 98.26% (18 errors) | 98.17% (19 errors) |
| Macro F1 | 0.9826 | 0.9815 |
| Test loss (weighted cross-entropy) | 0.0766 | **0.0505** |
| F1 glioma / meningioma / no tumour / pituitary | 0.979 / 0.979 / 0.987 / 0.985 | 0.987 / 0.977 / 0.980 / 0.982 |
| Tumour images called "no tumour" (missed tumours, of 778) | 3 | **1** |
| No-tumour images called a tumour (false alarms, of 259) | **4** | 9 |

- **Accuracy:** the two models are statistically indistinguishable. Each is wrong on about ten
  images the other gets right (10 vs 11), and an exact McNemar test gives p = 1.0.
- **Loss:** the lower test loss means the mitigated model's errors are made with less confidence.
- **Error profile:** the models trade error types. The mitigated model misses fewer tumours (1 vs 3)
  but raises more than twice as many false alarms on no-tumour images (9 vs 4). With these
  counts, neither change is statistically meaningful on its own.

**The three tracked gliomas the original model called "no tumour":**

| Image | Original model | Crop-mitigated model |
|---|---|---|
| `Te-gl_350` (175 × 167) | no tumour, 0.9996 | **glioma (correct), 0.65** |
| `Te-gl_97` (504 × 608) | no tumour, 0.9998 | meningioma, 0.72 (no tumour 0.18) |
| `Te-gl_318` (197 × 229, CT) | no tumour, 0.91 | meningioma, 0.98 |

All three are now recognised as tumours, and one is classified correctly. The other two are given
the wrong tumour type. `Te-gl_318` is a CT image, for which the correct label can't be expected.

**Format breakdown (gliomas, test set):**

| Glioma images | Original | Crop-mitigated |
|---|---|---|
| 512 × 512 (261) | 4 errors (1.5%) | 3 errors (1.1%) |
| Other formats (9) | 5 errors (56%) | 3 errors (33%) |

On the natural test images the gap narrows, but with 9 images that change is not distinguishable
from noise.

**Format stress test.** This is a direct test of the shortcut, run on all test images
(`src.training.format_stress_test`). Each image's content is left unchanged; only its file format
is altered by resizing it to a typical no-tumour size before the usual preprocessing. A model that
reads tumour appearance should barely react.

| Tumour images (of 778) called "no tumour" | Unmodified | Resized to 225 × 225 | Resized to 210 × 234 |
|---|---|---|---|
| Original | 3 | 12 | 10 |
| Crop-mitigated | 1 | 14 | 16 |

| Overall accuracy | Unmodified | 225 × 225 | 210 × 234 |
|---|---|---|---|
| Original | 98.26% | 97.30% | 97.49% |
| Crop-mitigated | 98.17% | 97.49% | 97.11% |

- **Both models still shift towards "no tumour"** when a tumour image is given the low resolution
  typical of the no-tumour source. Down-sampling also removes some genuine detail, but random
  information loss would produce errors in all directions. Instead, most of the new errors go to
  "no tumour" specifically.
- **The mitigated model is no more robust here.** It reacts slightly more: 14-16 tumours lost
  against 10-12.
- **No-tumour images under the mitigated model:** they are classified more accurately once
  resized to 225 × 225 (98.5% against 96.5% unmodified). That is further evidence that low
  resolution itself still signals "no tumour" to it.

**Errors new in the mitigated model.** These were inspected with `src.training.inspect_errors`.

- **Pituitary called "no tumour":** `Tr-pi_322`, at 0.62. This is the one missed tumour, a 512 × 512
  sagittal slice with no conspicuous sellar mass at this resolution.
- **No tumour called a tumour type:** `Tr-no_59` and `Tr-no_383` called meningioma, `Tr-no_539` and
  `Tr-no_543` pituitary, `Tr-no_777` pituitary, and `Tr-no_261` glioma.
  - `Tr-no_539` and `Tr-no_543` are further sella-level slices from the same series as the original
    model's errors.
  - `Tr-no_777` contains a large cyst-like cavity. It is labelled no tumour and is plausibly not a
    tumour, but it is not a normal brain.
  - `Tr-no_261` sits on the decision boundary: glioma 0.51 in the evaluation, which uses
    mixed-precision inference. Under full-precision inference it flips to no tumour (0.52 against
    0.48).
- **Other new errors:**
  - three 512 × 512 gliomas called meningioma;
  - one meningioma called pituitary.
- **Interpretation:** the extra false alarms are what weakening the format cue looks like from the
  other side. Part of the original model's no-tumour accuracy came from recognising the no-tumour
  source format. With that cue partly removed, the model has to judge more of these images by
  content, and gets a few more wrong.

**Assessment: partially fixed.**

- **What changed:** the crop removed the aspect-ratio and framing part of the shortcut. All three
  missed gliomas are now recognised as tumours, and there are fewer missed tumours overall (1 vs 3).
  Overall accuracy is unchanged and calibration is better.
- **What didn't change:** the resolution part of the shortcut. The stress test shows both models
  still treat a low-resolution image as evidence for "no tumour". The crop could not address this,
  because it changes framing, not resolution.
- **Next step:** a resolution-targeted augmentation is the obvious follow-up: randomly down-sample
  training images (for example to 150-512 px) before upsampling, plus JPEG re-compression. The
  format stress test is the right yardstick for it.

### Mitigation attempt 2: crop + resolution degradation

A third model was trained from scratch with a resolution and compression degradation stacked on
top of the random crop (`RandomResolutionDegrade`, enabled in
`configs/train_classification_crop_resdegrade.json`, checkpoint in
`models/classification_crop_resdegrade/`).

- **The degradation:** half of all training images, in every class, are processed as follows, after
  the crop and before letterboxing:
  1. down-sample so the shorter side is a random 150-320 px, keeping the aspect ratio;
  2. re-encode as JPEG at a random quality between 30 and 95.

  The letterbox then upsamples the result. This is the same path a genuinely small, re-encoded
  no-tumour file takes.
- **Everything else is identical** to the other two runs, including the early-stopping rule.
- **Training:** validation loss bottomed out at epoch 5, so training stopped after epoch 10. The
  checkpoint kept is epoch 10, with validation accuracy 98.27% and macro F1 0.982.
  - Validation F1 was still rising when training stopped. The harder augmentation probably needs a
    longer schedule than the shared stopping rule allowed, so this model is likely somewhat
    under-trained relative to the others.

**All three models on the 1,037 test images:**

| | Original | Crop | Crop + resolution |
|---|---|---|---|
| Accuracy | **98.26%** (18 errors) | 98.17% (19) | 97.78% (23) |
| Macro F1 | **0.9826** | 0.9815 | 0.9777 |
| Test loss | 0.0766 | **0.0505** | 0.0808 |
| F1 glioma / meningioma / no tumour / pituitary | 0.979 / 0.979 / 0.987 / 0.985 | 0.987 / 0.977 / 0.980 / 0.982 | 0.967 / 0.973 / 0.984 / 0.987 |
| Tumours called "no tumour" (of 778) | 3 | **1** | 2 |
| No-tumour images called a tumour (of 259) | **4** | 9 | 6 |
| Glioma / meningioma confusions | 7 | **6** | 10 |

Exact McNemar tests find none of the accuracy differences significant:

| Comparison | Images only the first model gets wrong | Images only the second gets wrong | p |
|---|---|---|---|
| Crop + resolution vs original | 11 | 6 | 0.33 |
| Crop + resolution vs crop | 14 | 10 | 0.54 |

**The three tracked gliomas:**

| Image | Original | Crop | Crop + resolution |
|---|---|---|---|
| `Te-gl_350` (175 × 167, FLAIR-type) | no tumour, 1.00 | glioma (correct), 0.65 | meningioma, 0.99 |
| `Te-gl_97` (504 × 608, T2) | no tumour, 1.00 | meningioma, 0.72 | no tumour, 0.96 |
| `Te-gl_318` (197 × 229, CT) | no tumour, 0.91 | meningioma, 0.98 | no tumour, 0.85 |

These three are not the right measure of the shortcut for this model, because the stress test
below shows resolution no longer moves its predictions. What remains is a matter of content:

- **`Te-gl_350`:** the Grad-CAM heatmap now sits squarely on the lesion. The original model's
  attribution avoided it. The model sees the tumour but gives the wrong type, which is plausible on
  a FLAIR-type slice when tumour training images are mostly contrast-enhanced T1.
- **`Te-gl_97`:** a T2 image.
- **`Te-gl_318`:** a CT image.

The training data contains very few tumour images of either kind. These are now failures of
coverage across scan types, not of image format, and no augmentation of format can fix them.

**Format stress test, all three models.** Every test image is resized to a typical no-tumour file
size before preprocessing. The 128 × 128 condition is smaller than anything the degradation
augmentation produces.

| Tumour images (of 778) called "no tumour" | Unmodified | 225 × 225 | 210 × 234 | 128 × 128 |
|---|---|---|---|---|
| Original | 3 | 12 | 10 | **250** |
| Crop | 1 | 14 | 16 | **213** |
| Crop + resolution | 2 | 3 | 3 | **9** |

| Overall accuracy | Unmodified | 225 × 225 | 210 × 234 | 128 × 128 |
|---|---|---|---|---|
| Original | 98.26% | 97.30% | 97.49% | 74.93% (glioma 32%) |
| Crop | 98.17% | 97.49% | 97.11% | 77.53% (glioma 57%) |
| Crop + resolution | 97.78% | 97.78% | 97.97% | **97.11%** (glioma 97%) |

| Predictions changed vs unmodified (of 1,037) | 225 × 225 | 210 × 234 | 128 × 128 |
|---|---|---|---|
| Original | 15 | 15 | 257 |
| Crop | 23 | 27 | 237 |
| Crop + resolution | 3 | 4 | 16 |

- **The 128 × 128 condition shows the full size of the shortcut in the first two models.** Shrinking
  tumour images to a small file makes the original model call a third of them "no tumour" (250 of
  778), and two thirds of gliomas are lost. The crop-only model is barely better.
- **The crop + resolution model is essentially format-invariant.** It changes 3-4 predictions at
  the typical no-tumour sizes and 16 at 128 × 128, which is outside the size range it was trained
  on.
- **Caveat:** the stress test uses the same kind of operation as the augmentation (down-sampling),
  so some robustness is expected by construction. The size range alone does not prove that the
  model would cope with real images from another source; an external dataset would.

**Assessment: the resolution shortcut is fixed at a small cost in tumour-type discrimination.**

- **Gain:** the third model is the only one whose test accuracy cannot be explained partly by file
  format. Its accuracy on shrunk images is the same as on originals, against a 23-point drop for
  the original model.
- **Cost:** about 0.5 points of overall accuracy, which is not statistically significant. It comes
  mainly from more glioma/meningioma confusions on standard 512 × 512 images: 13 errors on 512 × 512
  images, against 7 for the crop model and 9 for the original.
- **Likely explanations for the cost:** learning fine texture is harder when half the training
  images are degraded, and training stopped while validation F1 was still rising. A longer schedule
  is the first thing to try. One option is early stopping on macro F1 with more patience, judged on
  validation data.

### Longer training for the crop + resolution model

The crop + resolution run stopped while validation F1 was still rising, so it was retrained with
one change: early-stopping patience raised from 5 to 10 epochs
(`configs/train_classification_crop_resdegrade_long.json`, checkpoint in
`models/classification_crop_resdegrade_long/`). The epoch limit (30) had not been reached, so it was
left unchanged.

**Validation curve.** Training is deterministic, so the first 10 epochs repeat the earlier run
exactly. With more patience, training continued:

| Epoch | Validation loss | Validation accuracy | Validation macro F1 |
|---|---|---|---|
| 13 | 0.054 (best loss) | 98.46% | 0.984 |
| **22** | 0.073 | **98.85%** | **0.988** (checkpoint kept) |
| 23 | 0.081 | 98.56% | 0.985 |

Early stopping ended training after epoch 23. Validation macro F1 rose from 0.982 (the earlier
checkpoint) to 0.988. **This confirms on validation data alone that the earlier run was
under-trained.**

**Test set (1,037 images):**

| | Original | Crop + resolution (patience 5) | **Crop + resolution (patience 10)** |
|---|---|---|---|
| Accuracy | 98.26% (18 errors) | 97.78% (23) | **98.26%** (18) |
| Macro F1 | 0.9826 | 0.9777 | **0.9827** |
| Test loss | 0.0766 | 0.0808 | **0.0687** |
| F1 glioma / meningioma / no tumour / pituitary | 0.979 / 0.979 / 0.987 / 0.985 | 0.967 / 0.973 / 0.984 / 0.987 | 0.980 / 0.985 / 0.981 / 0.985 |
| Tumours called "no tumour" (of 778) | 3 | 2 | 2 |
| No-tumour images called a tumour (of 259) | 4 | 6 | 8 |
| Glioma / meningioma confusions | 7 | 10 | **5** |
| 95% CI for accuracy (Wilson) | 97.3-98.9% | 96.7-98.5% | 97.3-98.9% |

Exact McNemar tests on the paired test predictions:

| Comparison | Wrong only for the patience-10 model | Wrong only for the other model | p |
|---|---|---|---|
| vs patience 5 | 4 | 9 | 0.27 |
| vs original | 8 | 8 | 1.0 |

**Format stress test** (tumour images called "no tumour", of 778, and overall accuracy):

| | Unmodified | 225 × 225 | 210 × 234 | 128 × 128 |
|---|---|---|---|---|
| Original | 3 (98.26%) | 12 (97.30%) | 10 (97.49%) | 250 (74.93%) |
| Crop | 1 (98.17%) | 14 (97.49%) | 16 (97.11%) | 213 (77.53%) |
| Crop + resolution, patience 5 | 2 (97.78%) | 3 (97.78%) | 3 (97.97%) | 9 (97.11%) |
| **Crop + resolution, patience 10** | 2 (98.26%) | 1 (98.07%) | 1 (98.26%) | **0 (98.17%)** |

Longer training kept the format robustness and slightly improved it: 5-11 of 1,037 predictions
change under resizing.

**What longer training changed:**

- **Recovered:** the accuracy and tumour-type discrimination given up in the shorter run. Glioma /
  meningioma confusions fell from 10 to 5, the fewest of any model.
- **Unchanged:** the three tracked gliomas.
  - `Te-gl_350` is still called meningioma.
  - `Te-gl_97` (T2) and `Te-gl_318` (CT) are still called "no tumour". These are the scan-type
    coverage gaps described above, and longer training does not address them.
- **Remaining weak spot:** false alarms on no-tumour images, 8 of 259.
  - Five are sella-level slices called pituitary: `Tr-no_173`, `_344` and `_674`, also errors of the
    original model, and `Tr-no_539` and `_543`, also errors of the crop model.
  - Three are called glioma: `Tr-no_261` at 1.00, `Tr-no_215` at 0.92 and `Tr-no_144` at 0.52.

### Why the patience-10 model is primary

**Crop + resolution, patience 10 (`models/classification_crop_resdegrade_long/best.pt`).**

- **Headline:** test accuracy **98.3%** (95% CI 97.3-98.9%, Wilson), macro F1 0.983.
- **Why it replaces the shorter run:**
  - The gain over the shorter run (+0.5 points, 5 fewer errors) is not statistically significant on
    its own (p = 0.27).
  - But the choice does not rest on the test set. The recipe is the same. Validation F1 shows the
    shorter run stopped before convergence, and the longer run is better or equal on every measure
    reported: accuracy, F1, test loss, type confusion and format robustness.
- **Why it is the model to showcase:**
  - It matches the original model's test accuracy exactly: 98.26%, with 8 errors unique to each
    model (p = 1.0).
  - It does so without the format shortcut. When test images are shrunk to 128 × 128, the original
    model's accuracy falls to 74.9%, while this model's stays at 98.2%.
- **Report alongside it:**
  - The false-alarm rate on normal brains: 8 of 259, partly from sella-level slices being called
    pituitary.
  - The coverage gap for non-T1 sequences and CT.
  - That format robustness has been shown against synthetic down-sampling, not on an external
    dataset.

---

## Limitations

- **Single test split per task.** The intervals above reflect patient sampling within that split,
  not variation between institutions or scanners. BraTS 2021 pools many institutions, but the test
  set comes from the same pool as the training data.
- **Classification has an image-format shortcut.** Image format (resolution, aspect ratio, source)
  is strongly correlated with class in this dataset, because the no-tumour class was assembled
  mainly from a different source than the tumour classes. On the test set, gliomas outside the
  dominant 512 × 512 format were misclassified 56% of the time (5 of 9), against 1.5% for
  512 × 512 gliomas. All 3 tumours called "no tumour" are visible lesions in non-standard formats,
  one of them a CT scan (see [Inspection of the most serious errors](#inspection-of-the-most-serious-errors)).
  How the mitigations fared:
  - Random-crop augmentation removed the framing and aspect-ratio part of the cue, but not the
    resolution part.
  - Adding resolution and JPEG degradation removed the resolution part as well. On test images
    shrunk to 128 × 128, 9 of 778 tumours are called "no tumour", against 250 for the original model
    (see [Mitigation attempt 2](#mitigation-attempt-2-crop--resolution-degradation)).

  The original and crop-only models' accuracies therefore hold only for images in their class's
  source format. Even for the crop + resolution model, format robustness is shown only against
  synthetic down-sampling. An external dataset is still needed to show robustness to real
  differences between sources.
- **Limited coverage of scan types.** Tumour training images are mostly contrast-enhanced T1. A T2
  glioma and a CT image in the test set are called "no tumour" by the format-robust model. The
  classifier should not be expected to handle sequences or modalities it has rarely seen.
- **The classifier also confuses anatomy with pathology at the sella.** Normal slices through the
  sella turcica can be called "pituitary" (all 5 of the primary model's no-tumour images called
  pituitary are axial skull-base slices at that level), because the pituitary class is dominated by
  those views.
- **Classification is slice-level.** The 4-class dataset is a collection of single 2D slices
  without patient identifiers. Near-duplicate clustering prevents the most direct leakage, but
  different slices of the same patient could still fall in different splits. The test accuracy
  should be read with this in mind.
- **Metric scope.** Segmentation is reported with Dice only. Boundary metrics such as the 95th
  percentile Hausdorff distance, also standard for BraTS, have not been computed.
- **No test-time augmentation.** Mirroring at inference is a common further improvement and was not
  used.

---

## Reproducing

From the trained checkpoints in `models/` (not tracked in git):

```bash
python -m src.training.evaluate_segmentation      # ET threshold on out-of-fold validation, then test (~25 min on an RTX 4060 Laptop)
python -m src.training.evaluate_classification    # test split (~1 min)
python -m src.training.bootstrap_results          # confidence intervals from the per-case results

# error inspection figures (written to reports/figures/, untracked because they contain dataset images)
python -m src.training.inspect_errors --files Te-gl_350.jpg Te-gl_97.jpg Te-gl_318.jpg \
    --out reports/figures/glioma_missed_as_notumor.png
python -m src.training.inspect_errors --files Tr-no_173.jpg Tr-no_344.jpg Tr-no_674.jpg Tr-no_907.jpg \
    --out reports/figures/notumor_called_pituitary.png

# crop-mitigated classifier: training, test evaluation, comparison, format stress test
python -m src.training.train_classification --config configs/train_classification_crop.json
python -m src.training.evaluate_classification --checkpoint models/classification_crop/best.pt \
    --output classification_test_crop.json
python -m src.training.compare_classifiers classification_test.json classification_test_crop.json

# crop + resolution-degradation classifier
python -m src.training.train_classification --config configs/train_classification_crop_resdegrade.json
python -m src.training.evaluate_classification --checkpoint models/classification_crop_resdegrade/best.pt \
    --output classification_test_crop_resdegrade.json
python -m src.training.compare_classifiers classification_test.json classification_test_crop_resdegrade.json \
    --output classification_comparison_resdegrade_vs_original.json
python -m src.training.compare_classifiers classification_test_crop.json classification_test_crop_resdegrade.json \
    --output classification_comparison_resdegrade_vs_crop.json

# crop + resolution with early-stopping patience 10 (recommended primary classifier)
python -m src.training.train_classification --config configs/train_classification_crop_resdegrade_long.json
python -m src.training.evaluate_classification --checkpoint models/classification_crop_resdegrade_long/best.pt \
    --output classification_test_crop_resdegrade_long.json
python -m src.training.compare_classifiers classification_test_crop_resdegrade.json \
    classification_test_crop_resdegrade_long.json --output classification_comparison_resdegrade_long_vs_resdegrade.json
python -m src.training.compare_classifiers classification_test.json classification_test_crop_resdegrade_long.json \
    --output classification_comparison_resdegrade_long_vs_original.json

# format stress test, all four classifiers
python -m src.training.format_stress_test models/classification/best.pt models/classification_crop/best.pt \
    models/classification_crop_resdegrade/best.pt models/classification_crop_resdegrade_long/best.pt
```

| File | Contents |
|---|---|
| `results/segmentation_test.json` | Per-model test Dice (raw and post-processed), CV scores, ET threshold tuning |
| `results/segmentation_test_per_case.csv` | Dice per test patient, per model |
| `results/classification_test.json` | Test metrics, confusion matrix, every misclassified image |
| `results/test_confidence_intervals.json` | Bootstrap and Wilson intervals, paired comparisons |
| `results/classification_test_crop.json` | Test metrics for the crop-mitigated classifier |
| `results/classification_comparison.json` | Original vs crop-mitigated: format breakdown, tracked errors, McNemar test |
| `results/classification_test_crop_resdegrade.json` | Test metrics for the crop + resolution classifier |
| `results/classification_comparison_resdegrade_vs_original.json`, `..._vs_crop.json` | Crop + resolution vs the other two classifiers |
| `results/classification_test_crop_resdegrade_long.json` | Test metrics for the recommended classifier (crop + resolution, patience 10) |
| `results/classification_comparison_resdegrade_long_vs_*.json` | Recommended classifier vs the shorter run and vs the original |
| `results/format_stress_test.json` | All four classifiers: accuracy and tumour-to-no-tumour flips when only image format is changed |

Training: `python -m src.training.train_segmentation --fold k` for k = 1..5, and
`python -m src.training.train_classification`. Settings are in `configs/`.

wandb runs (private to the project team):

- Segmentation test evaluation: https://forge.coreweave.com/wandb/lubabah-universit-mundiapolis-casablanca/brain-tumor-segmentation/runs/vr1kl5yz
- Classification test evaluation: https://forge.coreweave.com/wandb/lubabah-universit-mundiapolis-casablanca/brain-tumor-classification/runs/yb1ubrsw
- Segmentation training, folds 1-5: `h9opulfq`, `eph5gm7a`, `rugue12h`, `o9inkjww`, `iv9fbnnd`
- Classification training: `7lxvraje` (original), `0wlar3w4` (crop), `och4cbz1` (crop + resolution), `37cway31` (crop + resolution, patience 10)
- Classification test evaluation, crop model: https://forge.coreweave.com/wandb/lubabah-universit-mundiapolis-casablanca/brain-tumor-classification/runs/sqg9fe9j
- Classification test evaluation, crop + resolution model: https://forge.coreweave.com/wandb/lubabah-universit-mundiapolis-casablanca/brain-tumor-classification/runs/eqqz2ld2
- Classification test evaluation, crop + resolution, patience 10 (recommended): https://forge.coreweave.com/wandb/lubabah-universit-mundiapolis-casablanca/brain-tumor-classification/runs/u05clq4j
