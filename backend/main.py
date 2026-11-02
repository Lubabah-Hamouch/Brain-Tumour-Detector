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
from backend.services.samples import SampleResults
from backend.services.segmenter import SegmenterService

log = logging.getLogger("backend")
# uvicorn configures only its own loggers; this makes the API's start-up messages (models loading,
# sample results loaded or being computed) visible in the server log too. It does nothing when the
# root logger is already configured.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")


def error_body(code, message):
    return {"error": {"code": code, "message": message}}


def sample_results(settings, classifier, segmenter, jobs):
    """Results for the sample inputs: loaded from BT_SAMPLE_RESULTS, else computed once in the background.

    The computation is an ordinary job in the queue, so an upload sent while it runs waits its turn
    as behind any other job. Until it finishes, sample uploads run the models like any other.

    With BT_SAMPLE_RESULTS_REQUIRED (the Hugging Face Space, which has no sample study files to
    compute from) a missing or mismatched file stops the start-up instead."""
    required = settings.sample_results_required
    if settings.sample_results is None:
        if required:
            raise RuntimeError("BT_SAMPLE_RESULTS_REQUIRED=1 but BT_SAMPLE_RESULTS is off.")
        return None
    images, study = classification.sample_paths(), segmentation.sample_files()
    if not all(p.exists() for p in study.values()):
        study = {}
    if not images and not study and not required:
        log.info("no sample inputs in data/: nothing to pre-compute")
        return None
    checkpoints = [settings.classifier_checkpoint]
    checkpoints += [settings.segmentation_dir / f"fold{k}" / "best.pt" for k in settings.segmentation_folds]
    samples = SampleResults(settings.sample_results, classifier, segmenter, checkpoints, images, study)
    loaded = samples.load()
    if required and not (loaded and samples.has_study):
        raise RuntimeError(
            f"BT_SAMPLE_RESULTS_REQUIRED=1 but {settings.sample_results} is missing, unreadable, lacks the sample "
            "study, or was computed for other weights, settings or sample images (see the warning above). "
            "Rebuild it with python deploy/huggingface/build_space.py.")
    if not loaded:
        log.info("computing the sample results in the background")

        def work(progress):
            try:
                samples.compute(progress)
            except Exception:
                log.exception("computing the sample results failed; sample uploads run the models instead")
                raise

        jobs.submit("sample-results", segmentation.STAGES, work)
    return samples


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
        app.state.samples = sample_results(settings, app.state.classifier, app.state.segmenter, app.state.jobs)
        yield
        app.state.jobs.shutdown()
        app.state.classifier = app.state.segmenter = app.state.samples = None

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

    routers = [health.router, classification.router, segmentation.router, results.router]
    if settings.sample_study_files:  # off on the Space: the raw volumes are not redistributed
        routers.append(segmentation.sample_files_router)
    for router in routers:
        app.include_router(router)
    return app


app = create_app()
