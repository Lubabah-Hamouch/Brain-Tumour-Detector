from fastapi import APIRouter, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.schemas.classification import ClassificationResponse
from backend.schemas.common import ErrorResponse
from backend.services.errors import ApiError
from src.data.classification_preprocess import IMAGE_SUFFIXES

router = APIRouter(prefix="/api", tags=["classification"])

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
    return await run_in_threadpool(service.predict, img)
