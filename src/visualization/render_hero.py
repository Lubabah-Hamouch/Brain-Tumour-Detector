"""Render the Overview hero: 3D surfaces of a held-out test brain and its predicted tumour regions.

The surfaces come from the real data: the brain is the non-zero region of the four BraTS sequences
and the tumour regions come from the ensemble's label map for the same case. Download that label
map from the API first, for example:

    curl -F flair=@..._flair.nii.gz -F t1=@..._t1.nii.gz -F t1ce=@..._t1ce.nii.gz \
         -F t2=@..._t2.nii.gz http://localhost:8010/api/segment
    # poll the returned status_url until "complete", then download result.mask_url

    python -m src.visualization.render_hero --label-map path/to/mask.nii.gz

The output is patient-derived, so it is written to frontend/public/generated/, which is not
committed. Rendering needs the packages in requirements-render.txt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter

from src.config import BRATS_DIR, PROJECT_ROOT

DEFAULT_CASE = "BraTS2021_00003"  # held-out test patient
DEFAULT_OUT = PROJECT_ROOT / "frontend" / "public" / "generated" / "hero-3d.png"

# Design-system colours (frontend/src/design/tokens.ts).
BRAIN = "#D5D8DC"
BRAIN_EDGE = "#C9CDD2"
EDEMA = "#7A111B"
NECROTIC = "#D7263D"
ENHANCING = "#FFC2CA"

# Volumes are LPS: +x patient left, +y posterior, +z superior. View from front-left, above.
VIEW_DIRECTION = (0.42, -0.80, 0.46)
CAMERA_DISTANCE = 455
RENDER_SIZE = (1660, 2000)
OUTPUT_SIZE = (996, 1200)


def surface(mask: np.ndarray, sigma: float, smooth_iter: int = 40):
    """Smoothed, outward-oriented triangle surface of a binary mask (1 mm voxels)."""
    import pyvista as pv
    from skimage.measure import marching_cubes

    field = gaussian_filter(mask.astype(np.float32), sigma)
    verts, faces, _, _ = marching_cubes(field, level=0.5, spacing=(1.0, 1.0, 1.0))
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
    mesh = pv.PolyData(verts, cells).smooth_taubin(n_iter=smooth_iter, pass_band=0.05)
    return mesh.compute_normals(auto_orient_normals=True)


def render(brain, edema, necrotic, enhancing) -> np.ndarray:
    """Off-screen render with a transparent background; returns an RGBA array."""
    import pyvista as pv

    p = pv.Plotter(off_screen=True, window_size=RENDER_SIZE, lighting="none")
    p.set_background("black")
    p.enable_depth_peeling(number_of_peels=16, occlusion_ratio=0.0)
    p.enable_anti_aliasing("ssaa")
    p.add_mesh(brain, color=BRAIN, opacity=0.26, specular=0.9, specular_power=45, smooth_shading=True,
               ambient=0.14, diffuse=0.6,
               silhouette=dict(color=BRAIN_EDGE, line_width=2.0, opacity=0.35))
    p.add_mesh(edema, color=EDEMA, opacity=0.5, specular=0.3, smooth_shading=True, ambient=0.35, diffuse=0.7)
    p.add_mesh(necrotic, color=NECROTIC, opacity=0.95, specular=0.3, smooth_shading=True, ambient=0.5,
               diffuse=0.7)
    p.add_mesh(enhancing, color=ENHANCING, opacity=0.32, specular=0.45, specular_power=50, smooth_shading=True,
               ambient=0.22, diffuse=0.6)

    centre = np.array(brain.center)
    direction = np.array(VIEW_DIRECTION, dtype=float)
    direction /= np.linalg.norm(direction)
    p.camera.focal_point = centre
    p.camera.position = centre + direction * CAMERA_DISTANCE
    p.camera.up = (0, 0, 1)
    p.camera.view_angle = 27
    p.reset_camera_clipping_range()
    for position, intensity in (((1.0, 1.3, 1.0), 0.95), ((-1.3, 0.2, 0.7), 0.3), ((0.2, 0.5, -1.6), 0.75)):
        p.add_light(pv.Light(light_type="camera light", position=position, focal_point=(0, 0, 0),
                             intensity=intensity))
    image = p.screenshot(None, transparent_background=True, return_img=True)
    p.close()
    return image


def glow(image: np.ndarray) -> np.ndarray:
    """Soft red glow around the tumour, screen-blended over a lightly softened render (RGBA in and out)."""
    alpha = image[..., 3].astype(np.float32) / 255
    rgb = image[..., :3].astype(np.float32) / 255
    # Red-dominant pixels are the tumour surfaces; the grey brain contributes nothing.
    red = np.clip(rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) * 0.6, 0, 1) * alpha
    emit = rgb * red[..., None]
    k = image.shape[0] / 1200
    halo = np.stack([gaussian_filter(emit[..., c], 50 * k) * 1.05 + gaussian_filter(emit[..., c], 14 * k) * 0.25
                     for c in range(3)], -1)
    soft = np.stack([gaussian_filter(rgb[..., c] * alpha, 1.2 * k) for c in range(3)], -1)
    soft_alpha = gaussian_filter(alpha, 1.2 * k)
    premultiplied = 1 - (1 - np.clip(soft, 0, 1)) * (1 - np.clip(halo, 0, 1))
    out_alpha = np.clip(np.maximum(soft_alpha, halo.max(-1) * 1.3), 0, 1)
    colour = np.where(out_alpha[..., None] > 1e-4, premultiplied / np.maximum(out_alpha[..., None], 1e-4), 0)
    return np.dstack([np.clip(colour, 0, 1) * 255, out_alpha * 255]).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--label-map", type=Path, required=True,
                        help="BraTS label map (0/1/2/4) predicted for the case, e.g. from the API's mask_url")
    parser.add_argument("--case", default=DEFAULT_CASE, help=f"BraTS case ID (default {DEFAULT_CASE})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output PNG")
    args = parser.parse_args()

    from PIL import Image

    case_dir = BRATS_DIR / args.case
    volumes = [np.asarray(nib.load(case_dir / f"{args.case}_{m}.nii.gz").dataobj, dtype=np.float32)
               for m in ("flair", "t1", "t1ce", "t2")]
    labels = np.asarray(nib.load(args.label_map).dataobj)
    if labels.shape != volumes[0].shape:
        raise SystemExit(f"label map shape {labels.shape} does not match the case volumes {volumes[0].shape}")

    brain = binary_fill_holes(np.any(np.stack(volumes) != 0, axis=0))
    image = render(
        brain=surface(brain, 1.1, smooth_iter=30),
        edema=surface(labels > 0, 1.0),
        necrotic=surface(labels == 1, 0.9),
        enhancing=surface(labels == 4, 0.8),
    )
    hero = Image.fromarray(glow(image), "RGBA").resize(OUTPUT_SIZE, Image.LANCZOS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    hero.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
