from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from backend.schemas.common import ErrorResponse
from backend.schemas.jobs import JobAccepted, JobStatus
from backend.services.errors import ApiError

router = APIRouter(prefix="/api", tags=["segmentation"])

ERRORS = {404: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse},
          422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}

DESCRIPTION = """
Segment a brain tumour from **all four BraTS MRI sequences**, uploaded as four separate NIfTI files
(`.nii` or `.nii.gz`) in the fields `flair`, `t1`, `t1ce` and `t2`. All four are required: the
network takes them as four input channels, and there is no single-sequence fallback.

**Input requirements (as in BraTS 2021):**
- the four volumes are co-registered to one voxel grid (same shape and spacing);
- skull-stripped (background exactly 0);
- 1 mm isotropic spacing (e.g. 240 x 240 x 155). Other spacings are accepted with a warning.

Raw scanner DICOM, or volumes that are not skull-stripped or co-registered, need the BraTS
preprocessing first.

**Pipeline:** the evaluation pipeline, unchanged. That is z-score normalisation inside the brain,
128³ sliding-window inference with Gaussian blending for each of the 5 cross-validation models,
the mean of their probabilities, and removal of enhancing-tumour components smaller than the
threshold tuned on validation data.

**Asynchronous:** inference takes about 5 s on a GPU and about 3 minutes on a CPU, so this endpoint
does not wait for it. The upload is read and validated first (bad files fail here with a 4xx), then
a background job is queued and the response is `202 Accepted` with a `job_id`. Poll
`GET /api/jobs/{job_id}` for the current stage and progress; the job's `result` holds the
segmentation once `status` is `complete`.

**Output (in the job result):** region volumes and statistics, three preview slices, and a label
map in BraTS convention (1 necrotic core, 2 edema, 4 enhancing tumour). The label map is in the
same grid and affine as the upload, so it overlays the input directly. Download it from
`mask_url`, or set `inline_mask=true` to embed it as base64.
"""

# Share of total job time per stage, from measured GPU runs (preprocess 0.73 s, inference 3.53 s,
# post-processing 0.42 s); used only to weight the overall progress value.
STAGES = (("preprocessing", "Preprocess", 0.16), ("inference", "Inference", 0.75),
          ("postprocessing", "Post-process and encode", 0.09))


@router.post("/segment", response_model=JobAccepted, status_code=202, responses=ERRORS,
             summary="Start segmenting a brain tumour (WT/TC/ET) from FLAIR, T1, T1ce and T2 NIfTI volumes",
             description=DESCRIPTION)
async def segment(
    request: Request,
    response: Response,
    flair: UploadFile = File(..., description="FLAIR volume (.nii / .nii.gz)"),
    t1: UploadFile = File(..., description="T1 volume (.nii / .nii.gz)"),
    t1ce: UploadFile = File(..., description="Contrast-enhanced T1 volume (.nii / .nii.gz)"),
    t2: UploadFile = File(..., description="T2 volume (.nii / .nii.gz)"),
    inline_mask: bool = Query(False, description="Also return the .nii.gz label map inline as base64"),
):
    service = request.app.state.segmenter
    if service is None:
        raise ApiError(503, "model_unavailable", "The segmentation models are not loaded.")
    files = {}
    for modality, upload in (("flair", flair), ("t1", t1), ("t1ce", t1ce), ("t2", t2)):
        files[modality] = (upload.filename or f"{modality} (no filename)", await upload.read())
    base_url = str(request.base_url).rstrip("/")
    # Reading and validating the volumes takes well under a second; errors are returned right away.
    study = await run_in_threadpool(service.prepare, files)
    job_id = request.app.state.jobs.submit(
        "segmentation", STAGES, lambda progress: service.run(study, inline_mask, base_url, progress))
    status_url = f"{base_url}/api/jobs/{job_id}"
    response.headers["Location"] = status_url
    return {"job_id": job_id, "status": "queued", "status_url": status_url}


@router.get("/segment/{result_id}/mask.nii.gz", responses={404: {"model": ErrorResponse}},
            summary="Download a segmentation label map (BraTS labels 0/1/2/4, gzipped NIfTI)",
            response_class=Response)
def download_mask(request: Request, result_id: str):
    service = request.app.state.segmenter
    data = service.cache.get(result_id) if service else None
    if data is None:
        raise ApiError(404, "result_not_found",
                       f"No mask for result '{result_id}'. Results are kept in memory for the "
                       f"{service.cache.size if service else 0} most recent requests and are lost on restart.")
    return Response(content=data, media_type="application/gzip",
                    headers={"Content-Disposition": f'attachment; filename="segmentation_{result_id}.nii.gz"'})


@router.get("/jobs/{job_id}", response_model=JobStatus, responses={404: {"model": ErrorResponse}}, tags=["jobs"],
            summary="Status, progress and (when complete) result of a background job")
def job_status(request: Request, job_id: str):
    snapshot = request.app.state.jobs.get(job_id)
    if snapshot is None:
        raise ApiError(404, "job_not_found",
                       f"No job '{job_id}'. Jobs are kept in memory for the {request.app.state.jobs.keep} most "
                       "recent submissions and are lost when the server restarts.")
    return snapshot
