# Data

Raw and processed datasets are stored here locally but are **never committed**;
everything under `data/raw/` and `data/processed/` is gitignored.

| Path | Dataset | Use | Source |
|---|---|---|---|
| `raw/figshare_classification_4class/` | Brain Tumor MRI Dataset: glioma, meningioma, pituitary, no tumor (Figshare + SARTAJ + Br35H) | Classification (primary) | Kaggle `masoudnickparvar/brain-tumor-mri-dataset` |
| `raw/brats2021/` | BraTS 2021 Task 1 (glioma segmentation, multimodal MRI) | Segmentation | Kaggle `dschettler8845/brats-2021-task1` |
| `processed/brats2021_preprocessed/` | BraTS 2021, z-scored in brain mask, cropped to brain, WT/TC/ET targets (one `.npz` per case, ~12 GB) | Segmentation input for on-the-fly patch sampling | `python -m src.data.segmentation_preprocess --config configs/segmentation_preprocess.json --workers 3` |
| `raw/figshare_classification_3class/` | Original Figshare release (Cheng et al.): glioma, meningioma, pituitary | Comparison only | Kaggle `ashkhagan/figshare-brain-tumor-dataset` |

## Cleaned classification split

`processed/figshare_classification_4class_clean/{train,val,test}/<class>/` is built from the
raw 4-class dataset; `raw/` is never modified. The original Training/Testing split is not used
because it leaks near-duplicate images across splits.

- `manifests/figshare_4class_cleaning.json`: cleaning decisions (excluded files with reasons,
  reviewed false matches, dedup thresholds, split ratios, seed)
- `manifests/figshare_4class_split.csv`: one row per raw image with its new split, near-duplicate
  cluster, or exclusion reason
- `manifests/figshare_4class_summary.json`: per-split class counts

Steps: drop pre-augmented `-aug-` files and label-conflict clusters; group near-duplicates
(pHash and dHash Hamming distance <= 4) into clusters; assign each cluster wholly to one split,
stratified by class, 70/15/15. After assignment, images with coloured overlays (website
watermarks, all notumor, 34 images) are dropped; nothing else changes split.

```bash
python -m src.data.make_split data/manifests/figshare_4class_cleaning.json
```

## BraTS test set and cross-validation folds

`manifests/brats_5fold_splits.json` assigns every preprocessed BraTS2021 case (one case per
patient) to either the held-out test set (189 cases, 15%) or one of 5 cross-validation folds
(212-213 cases each). The test cases are never used for training, validation or model
selection, only for final reporting. Fold k validates on `folds[k]` and trains on the other
four folds.

Assignment is stratified (seed 42): cases without enhancing tumour form one stratum, and cases
with it are binned by whole-tumour volume quartile. The manifest also stores per-case WT/TC/ET
voxel counts, so the balance can be checked.

```bash
python -m src.data.make_brats_folds
```

## Pretrained weights

The classifier loads torchvision's ImageNet weights for EfficientNetV2-S. On first use they are
downloaded to the PyTorch cache, outside the repo:

- `%USERPROFILE%\.cache\torch\hub\checkpoints\efficientnet_v2_s-dd5fe13b.pth` (~87 MB)
- the location follows `torch.hub.get_dir()`; set `TORCH_HOME` to move it

They are not committed and are not part of this directory's data footprint; deleting them only
means they are downloaded again on the next run.

To recreate:

```bash
cp .env.example .env        # then fill in Kaggle credentials
python -m src.download_data all
python -m src.verify_data classification
python -m src.verify_data brats
python -m src.verify_data figshare3
```
