from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code, e.g. 'unsupported_file_type'")
    message: str = Field(description="Human-readable explanation of what went wrong and how to fix it")


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ModelInfo(BaseModel):
    loaded: bool
    checkpoint: str
    detail: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when every model is loaded")
    device: str
    cuda_device: str | None = None
    classifier: ModelInfo
    segmentation: ModelInfo
    sample_results: str = Field("off", description="Pre-computed results for the sample inputs: 'ready', "
                                                   "'computing' (sample uploads run the models until then) or 'off'")
