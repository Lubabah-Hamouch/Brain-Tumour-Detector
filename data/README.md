# Data

Raw and processed datasets are stored here locally but are **never committed**;
everything under `data/raw/` and `data/processed/` is gitignored.

| Path | Dataset | Source |
|---|---|---|
| `raw/brats2021/` | BraTS 2021 Task 1 (glioma segmentation, multimodal MRI) | Kaggle `dschettler8845/brats-2021-task1` |
| `raw/figshare_classification/` | Figshare brain tumor dataset (Cheng et al.), 3 classes | Kaggle `ashkhagan/figshare-brain-tumor-dataset` |

To recreate:

```bash
cp .env.example .env        # then fill in Kaggle credentials
python -m src.download_data all
python -m src.verify_data brats
python -m src.verify_data figshare
```
