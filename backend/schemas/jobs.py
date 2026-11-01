from pydantic import BaseModel, Field

from backend.schemas.common import ErrorDetail
from backend.schemas.segmentation import SegmentationResponse


class JobAccepted(BaseModel):
    job_id: str
    status: str = Field(description="Always 'queued' on submission")
    status_url: str = Field(description="Poll this URL (GET) for progress and the result")


class JobStage(BaseModel):
    name: str = Field(description="preprocessing | inference | postprocessing")
    label: str
    status: str = Field(description="pending | running | complete | failed")
    fraction: float = Field(description="Share of this stage done, 0-1")
    detail: str = Field(description="What the stage is doing now, e.g. 'Model 3 of 5'")
    seconds: float | None = Field(description="Time spent in the stage so far (or in total once complete)")


class JobStatus(BaseModel):
    job_id: str
    kind: str
    status: str = Field(description="queued | running | complete | failed")
    queue_position: int | None = Field(description="1 = next to run; only while queued")
    stage: str | None
    stage_label: str | None
    stage_detail: str | None
    progress: float = Field(description="Overall progress 0-1, weighted by typical stage durations")
    stages: list[JobStage]
    created_at: float
    started_at: float | None
    finished_at: float | None
    elapsed_seconds: float
    result: SegmentationResponse | None = Field(None, description="Present once status is 'complete'")
    error: ErrorDetail | None = Field(None, description="Present once status is 'failed'")
