# Brain Tumor Segmentation and Classification

Work in progress. Full documentation to follow.

## Layout

```
backend/     API service (planned)
frontend/    Web client (planned)
data/        Local datasets, gitignored (see data/README.md)
models/      Trained weights, gitignored
notebooks/   Exploration only
src/         Pipeline code: data loading, preprocessing, training, evaluation
reports/     Generated figures and metrics
tests/       Tests
docs/        Project documents
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add Kaggle credentials
python -m src.download_data all
```
