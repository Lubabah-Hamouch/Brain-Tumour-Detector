from typing import Literal

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from backend.schemas.common import ErrorResponse
from backend.schemas.jobs import JobAccepted, JobStatus
from backend.schemas.segmentation import SampleStudy
from backend.services import slices
from backend.services.errors import ApiError
from src.config import BRATS_DIR

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

**Output (in the job result):** region volumes and statistics, viewer slices (one per plane: four
sequences and three label layers, each a PNG URL), three preview slices, and a label
map in BraTS convention (1 necrotic core, 2 edema, 4 enhancing tumour). The label map is in the
same grid and affine as the upload, so it overlays the input directly. Download it from
`mask_url`, or set `inline_mask=true` to embed it as base64.
"""

# Held-out test patient offered as a sample study (in the test split of data/manifests).
SAMPLE_CASE = "BraTS2021_00003"

Plane = Literal["axial", "coronal", "sagittal"]
Sequence = Literal["flair", "t1", "t1ce", "t2"]
Layer = Literal["flair", "t1", "t1ce", "t2", "edema", "necrotic", "enhancing"]

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
    data = service.mask_nifti(result_id) if service else None
    if data is None:
        raise _not_kept(service, result_id)
    return Response(content=data, media_type="application/gzip",
                    headers={"Content-Disposition": f'attachment; filename="segmentation_{result_id}.nii.gz"'})


def _not_kept(service, result_id):
    return ApiError(404, "result_not_found",
                    f"No segmentation result '{result_id}'. Results are kept in memory for the "
                    f"{service.cache.size if service else 0} most recent requests and are lost on restart.")


@router.get("/segment/{result_id}/slices/{plane}/{layer}.png", response_class=Response,
            responses={200: {"content": {"image/png": {}}}, 404: {"model": ErrorResponse}},
            summary="One layer of the result viewer: a sequence or a label, for one plane",
            description="Sequences are grayscale and transparent outside the brain; labels (edema, necrotic, "
                        "enhancing) are red-family layers to stack on top. All layers of a plane share one size; "
                        "the job result lists their URLs under `slices`.")
async def slice_layer(request: Request, result_id: str, plane: Plane, layer: Layer):
    service = request.app.state.segmenter
    png = await run_in_threadpool(service.slice_png, result_id, plane, layer) if service else None
    if png is None:
        raise _not_kept(service, result_id)
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@router.get("/segment/{result_id}/report.png", response_class=Response,
            responses={200: {"content": {"image/png": {}}}, 404: {"model": ErrorResponse}},
            summary="Download one slice with the segmentation burned in, with volumes, as a PNG",
            description="A 2400 x 3000 PNG of the chosen plane and sequence with the chosen labels at the chosen "
                        "overlay opacity (what the viewer shows), the WT/TC/ET volumes and the study and result IDs.")
async def download_report(
    request: Request,
    result_id: str,
    plane: Plane = "axial",
    sequence: Sequence = "flair",
    labels: str = Query("edema,necrotic,enhancing", description="Comma-separated label layers to draw"),
    opacity: int = Query(100, ge=0, le=100, description="Overlay opacity in percent"),
):
    service = request.app.state.segmenter
    chosen = tuple(name for name in labels.split(",") if name in slices.LABEL_LAYERS)
    png = None
    if service:
        png = await run_in_threadpool(service.report_png, result_id, plane, sequence, chosen, opacity / 100)
    if png is None:
        raise _not_kept(service, result_id)
    return Response(png, media_type="image/png", headers={
        "Content-Disposition": f'attachment; filename="segmentation_{result_id[:8]}_{plane}_{sequence}.png"'})


@router.get("/segment/{result_id}/mesh.glb", response_class=Response,
            responses={200: {"content": {"model/gltf-binary": {}}}, 404: {"model": ErrorResponse},
                       500: {"model": ErrorResponse}},
            summary="3D surfaces of a result (brain and tumour regions) as binary glTF",
            description="Nodes `brain`, `edema` (whole-tumour outer surface), `necrotic` and `enhancing`, in "
                        "millimetres, centred on the brain, Y up. Built with marching cubes and Taubin smoothing on "
                        "the first request (about a second), then cached with the result.")
async def mesh(request: Request, result_id: str):
    service = request.app.state.segmenter
    glb = await run_in_threadpool(service.mesh_glb, result_id) if service else None
    if glb is None:
        raise _not_kept(service, result_id)
    return Response(glb, media_type="model/gltf-binary", headers={"Cache-Control": "private, max-age=3600"})


def _sample_files():
    return {m: BRATS_DIR / SAMPLE_CASE / f"{SAMPLE_CASE}_{m}.nii.gz" for m in slices.SEQUENCES}


@router.get("/sample-study", response_model=SampleStudy, responses={404: {"model": ErrorResponse}},
            summary="A held-out test study (four NIfTI files) to try the segmentation on")
def sample_study(request: Request):
    files = _sample_files()
    if not all(p.exists() for p in files.values()):
        raise ApiError(404, "sample_not_found", f"The sample study {SAMPLE_CASE} is not in the local BraTS data.")
    base_url = str(request.base_url).rstrip("/")
    return {"case": SAMPLE_CASE, "split": "test",
            "files": [{"sequence": m, "name": p.name, "size_bytes": p.stat().st_size,
                       "url": f"{base_url}/api/sample-study/{p.name}"} for m, p in files.items()]}


@router.get("/sample-study/{name}", response_class=FileResponse, responses={404: {"model": ErrorResponse}},
            summary="One file of the sample study (only its four sequences are served)")
def sample_study_file(name: str):
    path = next((p for p in _sample_files().values() if p.name == name and p.exists()), None)
    if path is None:
        raise ApiError(404, "sample_not_found", f"'{name}' is not a file of the sample study {SAMPLE_CASE}.")
    return FileResponse(path, media_type="application/gzip", filename=name)


@router.get("/jobs/{job_id}", response_model=JobStatus, responses={404: {"model": ErrorResponse}}, tags=["jobs"],
            summary="Status, progress and (when complete) result of a background job")
def job_status(request: Request, job_id: str):
    snapshot = request.app.state.jobs.get(job_id)
    if snapshot is None:
        raise ApiError(404, "job_not_found",
                       f"No job '{job_id}'. Jobs are kept in memory for the {request.app.state.jobs.keep} most "
                       "recent submissions and are lost when the server restarts.")
    return snapshot
