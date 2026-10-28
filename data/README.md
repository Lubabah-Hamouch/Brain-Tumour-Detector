# Data

Raw and processed datasets are stored here locally but are **never committed**;
everything under `data/raw/` and `data/processed/` is gitignored.

| Path | Dataset | Use | Source |
|---|---|---|---|
| `raw/figshare_classification_4class/` | Brain Tumor MRI Dataset: glioma, meningioma, pituitary, no tumor (Figshare + SARTAJ + Br35H) | Classification (primary) | Kaggle `masoudnickparvar/brain-tumor-mri-dataset` |
| `raw/brats2021/` | BraTS 2021 Task 1 (glioma segmentation, multimodal MRI) | Segmentation | Kaggle `dschettler8845/brats-2021-task1` |
| `raw/figshare_classification_3class/` | Original Figshare release (Cheng et al.): glioma, meningioma, pituitary | Comparison only | Kaggle `ashkhagan/figshare-brain-tumor-dataset` |

To recreate:

```bash
cp .env.example .env        # then fill in Kaggle credentials
python -m src.download_data all
python -m src.verify_data classification
python -m src.verify_data brats
python -m src.verify_data figshare3
```
