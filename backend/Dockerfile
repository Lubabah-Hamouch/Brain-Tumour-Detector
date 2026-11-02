# FastAPI service. Build from the repository root (docker-compose.yml does this):
#   docker build -f backend/Dockerfile -t brain-tumour-api .                          # CPU
#   docker build -f backend/Dockerfile --build-arg TORCH_VARIANT=cu118 -t brain-tumour-api:cu118 .
# Weights and data are not in the image: mount models/ (and optionally data/) at run time.
FROM python:3.12-slim

# PyTorch wheel index: cpu, or a CUDA build such as cu118 (needs an NVIDIA GPU and the
# NVIDIA Container Toolkit at run time).
ARG TORCH_VARIANT=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt requirements-backend.txt ./
RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install -r requirements-backend.txt

COPY src/ src/
COPY backend/ backend/
COPY reports/final_results.md reports/
COPY reports/results/ reports/results/
COPY data/manifests/ data/manifests/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8010
# Start-up imports PyTorch and loads six models: seconds when warm, a few minutes on the first start
# of the CUDA image. Failed checks during the start period do not count.
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/api/health')"
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8010"]
