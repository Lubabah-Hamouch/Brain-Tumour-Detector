from pydantic import BaseModel, Field


class RegionResult(BaseModel):
    voxels: int = Field(description="Voxels in the region in the returned label map")
    volume_ml: float = Field(description="voxels x voxel volume from the NIfTI header, in millilitres (cm^3)")
    mean_probability: float | None = Field(
        description="Mean ensemble probability over the region's voxels (None if the region is empty)")
    model_agreement: float = Field(
        description="Mean Dice between each fold model's mask and the ensemble mask. 1.0 = all five models "
                    "agree exactly; lower values flag uncertain regions. An agreement measure, not accuracy.")


class SliceImage(BaseModel):
    plane: str = Field(description="axial | coronal | sagittal")
    index: int = Field(description="Slice index along that axis, in the uploaded volume's voxel grid")
    png: str = Field(description="Base64 PNG: FLAIR with the label map overlaid")


class ViewerPlane(BaseModel):
    index: int = Field(description="Slice index along the plane's axis (through the largest tumour cross-section)")
    count: int = Field(description="Number of slices along that axis")
    width: int = Field(description="Pixel width of every layer of this plane")
    height: int
    layers: dict[str, str] = Field(description="PNG URL per layer: flair, t1, t1ce, t2 (grayscale, transparent "
                                               "outside the brain) and edema, necrotic, enhancing (label overlays)")


class SegmentationResponse(BaseModel):
    result_id: str
    study: str = Field(description="Study name, from the FLAIR file name")
    files: dict[str, str] = Field(default_factory=dict, description="Uploaded file name per sequence")
    mask_url: str = Field(description="GET this URL for the label map as .nii.gz (same grid and affine as the upload)")
    report_url: str = Field(description="GET this URL (optionally with plane, sequence, labels, opacity) for a PNG "
                                        "of one slice with the overlay, the volumes and identifiers")
    mesh_url: str = Field(description="GET this URL for the brain and tumour surfaces as binary glTF (.glb)")
    mask_nifti_base64: str | None = Field(None, description="The same .nii.gz, inline, if inline_mask=true")
    labels: dict[str, int] = Field(description="Label values in the mask (BraTS convention)")
    regions: dict[str, RegionResult] = Field(description="WT (whole tumour), TC (tumour core), ET (enhancing tumour)")
    shape: list[int]
    spacing_mm: list[float]
    tumour_bbox: dict | None = Field(description="min/max voxel index of the whole tumour per axis, or None")
    slices: dict[str, ViewerPlane] = Field(description="axial, coronal and sagittal viewer layers")
    preview: list[SliceImage]
    post_processing: dict = Field(description="ET small-component removal applied, and what it removed")
    model: dict
    timing_seconds: dict
    warnings: list[str] = Field(default_factory=list)


class SampleFile(BaseModel):
    sequence: str
    name: str
    size_bytes: int
    url: str


class SampleStudy(BaseModel):
    case: str
    split: str = Field(description="The data split the case belongs to")
    files: list[SampleFile]
