"""FastAPI application serving the brain tumour classifier and segmentation ensemble.

Run from the project root:
    uvicorn backend.main:app --port 8010
Interactive docs: http://localhost:8010/docs
Settings: environment variables listed in backend/config.py.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import Settings
from backend.routers import classification, health, results, segmentation
from backend.services.classifier import ClassifierService
from backend.services.device import resolve_device
from backend.services.errors import ApiError
from backend.services.jobs import JobStore
from backend.services.segmenter import SegmenterService

log = logging.getLogger("backend")


def error_body(code, message):
    return {"error": {"code": code, "message": message}}


def create_app(settings=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        device = resolve_device(settings.device)
        app.state.settings, app.state.device = settings, device
        log.info("loading models on %s", device)
        # Models load once at startup and stay in memory; a missing checkpoint is a startup error.
        app.state.classifier = ClassifierService(settings.classifier_checkpoint, device, settings.result_cache_size)
        app.state.segmenter = SegmenterService(settings.segmentation_dir, settings.segmentation_folds, device,
                                               settings.et_min_voxels, settings.result_cache_size,
                                               settings.max_volume_mb)
        app.state.jobs = JobStore(keep=settings.result_cache_size)
        log.info("models loaded")
        yield
        app.state.jobs.shutdown()
        app.state.classifier = app.state.segmenter = None

    app = FastAPI(title="Brain Tumour MRI API", version="1.0.0", lifespan=lifespan,
                  description="Tumour-type classification of 2D MRI slices and 3D tumour segmentation of "
                              "BraTS-style multimodal MRI. See /docs for each endpoint's input requirements.")
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status_code, content=error_body(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        missing = [str(e["loc"][-1]) for e in exc.errors() if e.get("type") == "missing"]
        if missing:
            message = f"Missing required field(s): {', '.join(missing)}."
            if request.url.path.endswith("/segment"):
                message += " Segmentation needs all four modalities: flair, t1, t1ce and t2."
            return JSONResponse(status_code=422, content=error_body("missing_fields", message))
        details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return JSONResponse(status_code=422, content=error_body("invalid_request", details))

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content=error_body("internal_error", f"{type(exc).__name__}: {exc}"))

    for router in (health.router, classification.router, segmentation.router, results.router):
        app.include_router(router)
    return app


app = create_app()
