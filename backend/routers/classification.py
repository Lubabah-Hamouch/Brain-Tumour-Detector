import json

from fastapi import APIRouter, File, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from backend.schemas.classification import ClassificationResponse, Sample
from backend.schemas.common import ErrorResponse
from backend.services.errors import ApiError, relative_name
from src.config import PROJECT_ROOT
from src.data.classification_preprocess import CLEAN_DIR, IMAGE_SUFFIXES

router = APIRouter(prefix="/api", tags=["classification"])

# Held-out test images offered to visitors without a scan of their own: a correctly classified
# meningioma and pituitary, and one of the recorded test errors.
SAMPLES = ("Tr-me_1065.jpg", "Te-pi_102.jpg", "Te-gl_350.jpg")

ERRORS = {413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
          503: {"model": ErrorResponse}}


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    responses=ERRORS,
    summary="Classify a 2D brain MRI image (glioma, meningioma, no tumour, pituitary)",
    description=(
        "Upload one 2D MRI slice as JPEG or PNG. It is preprocessed exactly as during training and "
        "evaluation (grayscale, letterbox to 384 x 384, ImageNet normalisation). The response gives "
        "the predicted class, all class probabilities and a Grad-CAM heatmap for the predicted class, "
        "in the coordinates of the letterboxed model input.\n\n"
        "The model was trained on single axial, coronal or sagittal slices from the Brain Tumor MRI "
        "dataset, mostly contrast-enhanced T1. Other sequences (T2, FLAIR) and CT are poorly covered; "
        "see reports/final_results.md."
    ),
)
async def classify(request: Request, file: UploadFile = File(..., description="JPEG or PNG image")):
    service = request.app.state.classifier
    if service is None:
        raise ApiError(503, "model_unavailable", "The classification model is not loaded.")
    name = file.filename or ""
    if not name.lower().endswith(tuple(IMAGE_SUFFIXES)):
        raise ApiError(415, "unsupported_file_type",
                       f"'{name}' is not a supported image; upload one of {sorted(IMAGE_SUFFIXES)}.")
    data = await file.read()
    limit = request.app.state.settings.max_image_mb
    if not data:
        raise ApiError(422, "empty_file", f"'{name}' is empty.")
    if len(data) > limit * 2**20:
        raise ApiError(413, "file_too_large", f"'{name}' is {len(data) / 2**20:.1f} MB; the limit is {limit:.0f} MB.")
    img = service.open_image(data)
    base_url = str(request.base_url).rstrip("/")
    return await run_in_threadpool(service.predict, img, name, base_url)


@router.get(
    "/classify/{result_id}/report.png",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}, 404: {"model": ErrorResponse}},
    summary="Download a classification result as one PNG",
    description=(
        "A 2400 x 3000 PNG with the slice as the model saw it, the Grad-CAM overlay, the prediction "
        "and all four probabilities. `result_id` comes from the classification response; results "
        "are kept in memory for the most recent requests only."
    ),
)
async def download_report(request: Request, result_id: str):
    service = request.app.state.classifier
    png = await run_in_threadpool(service.report_png, result_id) if service else None
    if png is None:
        raise ApiError(404, "result_not_found",
                       f"No classification result '{result_id}'. Results are kept in memory for the "
                       f"{service.results.size if service else 0} most recent requests and are lost on restart.")
    return Response(png, media_type="image/png",
                    headers={"Content-Disposition": f'attachment; filename="classification_{result_id[:8]}.png"'})


def _sample_paths():
    paths = {}
    for name in SAMPLES:
        found = sorted((CLEAN_DIR / "test").glob(f"*/{name}"))
        if found:
            paths[name] = found[0]
    return paths


def _recorded_errors(checkpoint):
    """file name -> predicted class, from the test report of the served checkpoint."""
    served = relative_name(checkpoint)
    for report in sorted((PROJECT_ROOT / "reports" / "results").glob("classification_test*.json")):
        data = json.loads(report.read_text())
        if data.get("checkpoint") == served:
            return {e["file"]: e["predicted"] for e in data["misclassified"]}
    return {}


@router.get("/samples", response_model=list[Sample], summary="Held-out test images to try the classifier on")
def list_samples(request: Request):
    base_url = str(request.base_url).rstrip("/")
    errors = _recorded_errors(request.app.state.settings.classifier_checkpoint)
    return [Sample(name=name, true_class=path.parent.name, size_bytes=path.stat().st_size,
                   known_error=name in errors, recorded_prediction=errors.get(name),
                   url=f"{base_url}/api/samples/{name}")
            for name, path in _sample_paths().items()]


@router.get("/samples/{name}", response_class=FileResponse, responses={404: {"model": ErrorResponse}},
            summary="One sample image (only the listed samples are served)")
def get_sample(name: str):
    path = _sample_paths().get(name)
    if path is None:
        raise ApiError(404, "sample_not_found", f"'{name}' is not one of the sample images {list(SAMPLES)}.")
    return FileResponse(path, media_type="image/jpeg", filename=name, content_disposition_type="inline")
