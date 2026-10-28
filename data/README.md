# Data

Raw and processed datasets are stored here locally but are **never committed**;
everything under `data/raw/` and `data/processed/` is gitignored.

| Path | Dataset | Use | Source |
|---|---|---|---|
| `raw/figshare_classification_4class/` | Brain Tumor MRI Dataset: glioma, meningioma, pituitary, no tumor (Figshare + SARTAJ + Br35H) | Classification (primary) | Kaggle `masoudnickparvar/brain-tumor-mri-dataset` |
| `raw/brats2021/` | BraTS 2021 Task 1 (glioma segmentation, multimodal MRI) | Segmentation | Kaggle `dschettler8845/brats-2021-task1` |
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
stratified by class, 70/15/15.

```bash
python -m src.data.make_split data/manifests/figshare_4class_cleaning.json
```

To recreate:

```bash
cp .env.example .env        # then fill in Kaggle credentials
python -m src.download_data all
python -m src.verify_data classification
python -m src.verify_data brats
python -m src.verify_data figshare3
```
