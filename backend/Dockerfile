# FastAPI service. Build from the repository root (docker-compose.yml does this):
#   docker build -f backend/Dockerfile -t brain-tumour-api .                          # CPU
#   docker build -f backend/Dockerfile --build-arg TORCH_VARIANT=cu118 -t brain-tumour-api:cu118 .
# Weights are not in the image: on start the container downloads any that are missing from the
# weights-v1 release into /app/models (or mount models/ there). Data is optional: mount data/ for
# the "try a sample" buttons. The VM deployment builds this same file, from a bundle that also
# holds the pre-computed sample results (deploy/vm/build_bundle.py).
FROM python:3.12-slim

# PyTorch wheel index: cpu, or a CUDA build such as cu118 (needs an NVIDIA GPU and the
# NVIDIA Container Toolkit at run time).
ARG TORCH_VARIANT=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# Behind an HTTPS proxy (Caddy on the VM), take the scheme from the X-Forwarded-* headers, so the
# result URLs the API returns are https:// like the page that requests them.
ENV FORWARDED_ALLOW_IPS=*

WORKDIR /app

COPY requirements.txt requirements-backend.txt ./
RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install -r requirements-backend.txt

COPY src/ src/
COPY backend/ backend/
COPY reports/final_results.md reports/
COPY reports/results/ reports/results/
COPY data/ data/
# Empty from the repository; the deployment bundle adds the pre-computed sample results here.
COPY models/ models/

# models/ must be writable by the app user: the weights are downloaded there on start, and the
# pre-computed sample results are saved there.
RUN useradd --create-home --uid 1000 app && chown -R app:app models
USER app

# The port is $PORT when set (as some hosting platforms do), else 8010.
EXPOSE 8010
# Start-up downloads missing weights (about 413 MB, once per container), imports PyTorch and loads
# six models. Failed checks during the start period do not count.
HEALTHCHECK --interval=15s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8010') + '/api/health')"
# BT_DOWNLOAD_WEIGHTS=0 skips the download, for a models/ mount that is already complete.
CMD ["sh", "-c", "if [ \"${BT_DOWNLOAD_WEIGHTS:-1}\" != 0 ]; then python -m src.download_weights; fi && exec uvicorn backend.main:app --host 0.0.0.0 --port \"${PORT:-8010}\""]
