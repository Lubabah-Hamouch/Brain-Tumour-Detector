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
    overlay_png: str = Field(description="Base64 PNG, RGB: the model input with the heatmap in a jet colormap")
    resolution_note: str = Field(description="Grad-CAM comes from a 12 x 12 feature map; treat it as coarse")


class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(description="Probability of the predicted class")
    probabilities: dict[str, float] = Field(description="Softmax probability per class, summing to 1")
    image: dict = Field(description="Original upload: width, height, mode")
    letterbox: Letterbox
    gradcam: GradCam
    model: dict = Field(description="Checkpoint and its validation metrics")
    warnings: list[str] = Field(default_factory=list)
