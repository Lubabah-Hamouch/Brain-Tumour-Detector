import torch
from fastapi import APIRouter, Request

from backend.schemas.common import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service and model status")
def health(request: Request):
    state = request.app.state
    clf, seg, samples = state.classifier, state.segmenter, state.samples
    return {
        "status": "ok" if clf and seg else "degraded",
        "device": str(state.device),
        "cuda_device": torch.cuda.get_device_name(state.device) if state.device.type == "cuda" else None,
        "classifier": {"loaded": clf is not None, "checkpoint": clf.checkpoint_name if clf else "",
                       "detail": {"classes": list(clf.classes), **clf.info} if clf else {}},
        "segmentation": {"loaded": seg is not None, "checkpoint": seg.checkpoint_dir if seg else "",
                         "detail": {"folds": list(seg.models), "et_min_component_voxels": seg.et_min_voxels} if seg else {}},
        "sample_results": "off" if samples is None else "ready" if samples.ready else "computing",
    }
