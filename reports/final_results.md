# Final results

Results for both tasks on held-out test sets that were never used for training, early stopping,
checkpoint selection, fold selection or post-processing tuning. Every number below can be
regenerated from the trained checkpoints with the commands in [Reproducing](#reproducing).
Machine-readable versions are in [`results/`](results/).

| Task | Model | Test set | Headline result |
|---|---|---|---|
| Brain tumour segmentation (BraTS 2021) | 3D U-Net, 5-fold ensemble + ET post-processing | 189 patients | mean Dice **0.900** (WT 0.928, TC 0.894, ET 0.877) |
| Tumour type classification (4 classes) | EfficientNetV2-S, fine-tuned | 1,037 images | accuracy **98.3%** (95% CI 97.3-98.9%), macro F1 0.983; in-distribution only, see the [image-format shortcut](#inspection-of-the-most-serious-errors) |

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
  normalisation.
- **Model:** EfficientNetV2-S with ImageNet weights, fine-tuned end to end:
  - backbone learning rate 1e-4, head 1e-3, AdamW;
  - 1 warmup epoch, then cosine decay;
  - class-weighted cross-entropy (inverse frequency);
  - augmentation: flips, small affine transforms, brightness/contrast jitter.
- **Stopping and selection:** training stopped early when validation loss had not improved for 5
  epochs, which happened after epoch 12. The checkpoint kept is epoch 7, the best validation macro
  F1.

### Results

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

### Errors (18 of 1,037)

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
  Random-crop augmentation removed the framing and aspect-ratio part of the cue, but not the
  resolution part. In the format stress test, both models still move tumour images towards "no
  tumour" when the images are down-sampled (see
  [Mitigation attempt](#mitigation-attempt-random-crop-augmentation)). Remaining options:
  - resolution augmentation (random down-sampling and blur, JPEG re-compression);
  - cropping every image to the head bounding box before resizing;
  - testing on an external dataset.

  Until the stress-test gap closes, the classification accuracy should be read as in-distribution
  accuracy only.
- **The classifier also confuses anatomy with pathology at the sella.** Normal slices through the
  sella turcica can be called "pituitary" (4 test images), because the pituitary class is dominated
  by those views.
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
python -m src.training.format_stress_test models/classification/best.pt models/classification_crop/best.pt
```

| File | Contents |
|---|---|
| `results/segmentation_test.json` | Per-model test Dice (raw and post-processed), CV scores, ET threshold tuning |
| `results/segmentation_test_per_case.csv` | Dice per test patient, per model |
| `results/classification_test.json` | Test metrics, confusion matrix, every misclassified image |
| `results/test_confidence_intervals.json` | Bootstrap and Wilson intervals, paired comparisons |
| `results/classification_test_crop.json` | Test metrics for the crop-mitigated classifier |
| `results/classification_comparison.json` | Original vs crop-mitigated: format breakdown, tracked errors, McNemar test |
| `results/format_stress_test.json` | Accuracy and tumour-to-no-tumour flips when only image format is changed |

Training: `python -m src.training.train_segmentation --fold k` for k = 1..5, and
`python -m src.training.train_classification`. Settings are in `configs/`.

wandb runs (private to the project team):

- Segmentation test evaluation: https://forge.coreweave.com/wandb/lubabah-universit-mundiapolis-casablanca/brain-tumor-segmentation/runs/vr1kl5yz
- Classification test evaluation: https://forge.coreweave.com/wandb/lubabah-universit-mundiapolis-casablanca/brain-tumor-classification/runs/yb1ubrsw
- Segmentation training, folds 1-5: `h9opulfq`, `eph5gm7a`, `rugue12h`, `o9inkjww`, `iv9fbnnd`
- Classification training: `7lxvraje` (original), `0wlar3w4` (crop mitigation)
- Classification test evaluation, crop-mitigated model: https://forge.coreweave.com/wandb/lubabah-universit-mundiapolis-casablanca/brain-tumor-classification/runs/sqg9fe9j
