# Brain Tumor Segmentation and Classification

Work in progress. Full documentation to follow.

## Layout

```
backend/     FastAPI service for both models (see backend/README.md)
frontend/    Web client (planned)
data/        Local datasets, gitignored (see data/README.md)
models/      Trained weights, gitignored
notebooks/   Exploration only
src/         Pipeline code: data loading, preprocessing, training, evaluation
reports/     Generated figures and metrics
tests/       Tests
```

## Setup

All commands run inside a project-local virtual environment (Python 3.12):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Git Bash: source .venv/Scripts/activate
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
copy .env.example .env                # add Kaggle credentials
python -m wandb login                 # paste the API key at the prompt
python -m src.download_data all
python -m pytest tests
```

## API

```powershell
python -m pip install -r requirements-backend.txt
uvicorn backend.main:app --port 8010    # docs at http://localhost:8010/docs
```

Endpoints, input requirements and response formats are documented in `backend/README.md`.
