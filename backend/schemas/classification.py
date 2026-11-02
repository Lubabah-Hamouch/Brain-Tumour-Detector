from pydantic import BaseModel, Field


class Letterbox(BaseModel):
    """How the uploaded image was placed in the square model input.

    The image is resized by `scale` (longer side to `size`) and pasted at
    (`offset_x`, `offset_y`) on a black `size` x `size` canvas. A heatmap pixel
    (u, v) maps to original-image coordinates ((u - offset_x) / scale, (v - offset_y) / scale).
    """

    size: int
    scale: float
    offset_x: int
    offset_y: int
    resized_width: int
    resized_height: int


class GradCam(BaseModel):
    target_class: str = Field(description="Class the heatmap explains (the predicted class)")
    heatmap_png: str = Field(description="Base64 PNG, 8-bit grayscale, size x size, 255 = strongest attribution")
    heat_png: str = Field(description="Base64 PNG, RGBA, size x size: the heatmap on the web client's red ramp, "
                                      "transparent where attribution is weak; lay it over model_input_png")
    overlay_png: str = Field(description="Base64 PNG, RGB: the model input with the heatmap in a jet colormap")
    contour: list[list[list[float]]] = Field(description="Outline where the heatmap reaches contour_level of its peak: "
                                                         "polylines of [x, y] in model-input pixels. Always present, for "
                                                         "every predicted class; empty when the heatmap is all zero")
    contour_level: float = Field(description="Fraction of the peak the contour is drawn at")
    resolution_note: str = Field(description="Grad-CAM comes from a 12 x 12 feature map; treat it as coarse")


class ClassificationResponse(BaseModel):
    result_id: str = Field(description="Identifies this result for the report download")
    report_url: str = Field(description="PNG of the result: slice, Grad-CAM overlay, prediction and probabilities")
    predicted_class: str
    confidence: float = Field(description="Probability of the predicted class")
    probabilities: dict[str, float] = Field(description="Softmax probability per class, summing to 1")
    image: dict = Field(description="Original upload: width, height, mode, grayscale")
    letterbox: Letterbox
    model_input_png: str = Field(description="Base64 PNG, 8-bit grayscale: the letterboxed size x size image the model saw")
    gradcam: GradCam
    model: dict = Field(description="Checkpoint and its validation metrics")
    warnings: list[str] = Field(default_factory=list)


class Sample(BaseModel):
    name: str
    true_class: str = Field(description="Ground-truth class (the test-set folder)")
    size_bytes: int
    known_error: bool = Field(description="True if the served classifier misclassifies it on the test set")
    recorded_prediction: str | None = Field(description="The recorded wrong prediction, for known errors")
    url: str
