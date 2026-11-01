"""3D surfaces of a segmentation result, as one binary glTF (.glb) for the web client's 3D view.

The approach is the one used for the Overview's static render (src/visualization/render_hero.py),
without its rendering dependencies: each region's mask is smoothed with a Gaussian, turned into a
surface with marching cubes, and smoothed again with Taubin's method (which, unlike plain
Laplacian smoothing, does not shrink the surface). Surfaces:

- brain: the non-zero region of the four sequences, holes filled; drawn translucent;
- edema: the whole-tumour outer surface (labels 1, 2 and 4), drawn translucent dark red;
- necrotic: the necrotic core (label 1);
- enhancing: the enhancing tumour (label 4).

The tumour layers carry the 2D viewer's label names, so its label toggles drive the 3D view too.

Each surface has a triangle budget for smooth rendering on an ordinary laptop. When marching cubes
at full resolution exceeds it, the sampling step is coarsened (the field is already smooth, so this
costs little detail). Coordinates are millimetres, centred on the brain, in a Y-up frame:
x = first voxel axis, y = third (superior in BraTS orientation), z = minus the second.
"""
import json
import struct
import warnings

import numpy as np
from scipy import sparse
from scipy.ndimage import binary_fill_holes, gaussian_filter

# name, Gaussian sigma (voxels), triangle budget, colour (sRGB), opacity
SURFACES = (
    ("brain", 1.1, 90_000, "#D5D8DC", 0.26),
    ("edema", 1.0, 50_000, "#7A111B", 0.5),
    ("necrotic", 0.9, 30_000, "#D7263D", 0.95),
    ("enhancing", 0.8, 40_000, "#FFC2CA", 0.32),
)
CROP_PAD = 4
# Taubin passes per surface: the brain as smooth as in the static render, the tumour with more detail.
TAUBIN_ITERATIONS = {"brain": 30}
TAUBIN_DEFAULT = 12
MAX_STEP = 4
# Default view, as in the static render: from the front-left, above (in the output frame).
CAMERA_DIRECTION = (0.42, 0.46, 0.80)


def region_masks(brain, labels):
    return {"brain": brain, "edema": labels > 0, "necrotic": labels == 1, "enhancing": labels == 4}


def fill(nonzero):
    """Non-zero voxels of the study -> the brain as a filled boolean mask."""
    return binary_fill_holes(nonzero)


def _taubin(verts, faces, iterations=TAUBIN_DEFAULT, lam=0.5, mu=-0.53):
    n = len(verts)
    i = faces[:, [0, 1, 2, 1, 2, 0]].ravel()
    j = faces[:, [1, 2, 0, 0, 1, 2]].ravel()
    adjacency = sparse.coo_matrix((np.ones(len(i), np.float32), (i, j)), shape=(n, n)).tocsr()
    adjacency.data[:] = 1.0  # duplicate edges (shared by two faces) count once
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1
    step = sparse.diags(1.0 / degree) @ adjacency
    v = verts.astype(np.float64)
    for _ in range(iterations):
        v = v + lam * (step @ v - v)
        v = v + mu * (step @ v - v)
    return v.astype(np.float32)


def _normals(verts, faces):
    tri = verts[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # area-weighted
    normals = np.zeros_like(verts)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(length, 1e-12)).astype(np.float32)


def surface(mask, spacing, sigma, max_triangles, smoothing=TAUBIN_DEFAULT):
    """Mask -> (vertices in mm, faces, normals, sampling step), or None if the mask is empty."""
    from skimage.measure import marching_cubes

    if not mask.any():
        return None
    idx = np.nonzero(mask)
    lo = np.maximum([a.min() - CROP_PAD for a in idx], 0)
    hi = np.minimum([a.max() + CROP_PAD + 1 for a in idx], mask.shape)
    field = gaussian_filter(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32), sigma)
    # A region of a few voxels can smooth to below 0.5 everywhere; keep it visible.
    level = min(0.5, 0.6 * float(field.max()))
    if level <= 0:
        return None
    step = 1
    while True:
        with warnings.catch_warnings():
            # scikit-image 0.26 sets array shapes in a way NumPy 2.5 deprecates; the result is unaffected.
            warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array", category=DeprecationWarning)
            verts, faces, _, _ = marching_cubes(field, level=level, spacing=tuple(spacing), step_size=step,
                                                allow_degenerate=False)
        if len(faces) <= max_triangles or step >= MAX_STEP:
            break
        step += 1
    verts = verts + lo * np.asarray(spacing)
    verts = _taubin(verts, faces, smoothing)
    normals = _normals(verts, faces)
    # Normals should point out of the region: away from its centre for most vertices.
    if np.mean(np.sum((verts - verts.mean(axis=0)) * normals, axis=1)) < 0:
        faces = faces[:, ::-1]
        normals = -normals
    return verts, faces.astype(np.uint32), normals, step


def _to_view(points, centre):
    """Voxel-axis millimetres -> Y-up view frame centred on the brain."""
    p = points - centre
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1).astype(np.float32)


def _srgb_to_linear(hex_colour):
    c = np.array([int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).tolist()


def build_glb(brain, labels, spacing):
    """brain: bool (X, Y, Z); labels: BraTS label map on the same grid; spacing: mm per voxel.

    Returns (glb bytes, summary dict).
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    idx = np.nonzero(brain)
    if not idx[0].size:
        raise ValueError("the brain mask is empty")
    lo = np.array([a.min() for a in idx]) * spacing
    hi = np.array([a.max() for a in idx]) * spacing
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2)

    masks = region_masks(brain, labels)
    gltf = {"asset": {"version": "2.0", "generator": "brain-tumour-mri backend"}, "scene": 0,
            "scenes": [{"nodes": [], "extras": {"radius_mm": radius, "camera_direction": CAMERA_DIRECTION}}],
            "nodes": [], "meshes": [], "materials": [], "accessors": [], "bufferViews": [], "buffers": []}
    binary = bytearray()
    summary = {}

    def add_view(data, target):
        while len(binary) % 4:
            binary.append(0)
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data), "target": target})
        binary.extend(data)
        return len(gltf["bufferViews"]) - 1

    for name, sigma, budget, colour, opacity in SURFACES:
        result = surface(masks[name], spacing, sigma, budget, TAUBIN_ITERATIONS.get(name, TAUBIN_DEFAULT))
        if result is None:
            summary[name] = {"triangles": 0}
            continue
        verts, faces, normals, step = result
        verts, normals = _to_view(verts, centre), _to_view(normals, np.zeros(3))
        index_type, index_dtype = (5123, np.uint16) if len(verts) < 65536 else (5125, np.uint32)
        accessors = gltf["accessors"]
        accessors.append({"bufferView": add_view(verts.tobytes(), 34962), "componentType": 5126,
                          "count": len(verts), "type": "VEC3",
                          "min": verts.min(axis=0).tolist(), "max": verts.max(axis=0).tolist()})
        accessors.append({"bufferView": add_view(normals.tobytes(), 34962), "componentType": 5126,
                          "count": len(normals), "type": "VEC3"})
        accessors.append({"bufferView": add_view(faces.astype(index_dtype).ravel().tobytes(), 34963),
                          "componentType": index_type, "count": faces.size, "type": "SCALAR"})
        gltf["materials"].append({
            "name": name, "doubleSided": True,
            "alphaMode": "BLEND" if opacity < 1 else "OPAQUE",
            "pbrMetallicRoughness": {"baseColorFactor": _srgb_to_linear(colour) + [opacity],
                                     "metallicFactor": 0.0, "roughnessFactor": 0.5}})
        gltf["meshes"].append({"name": name, "primitives": [{
            "attributes": {"POSITION": len(accessors) - 3, "NORMAL": len(accessors) - 2},
            "indices": len(accessors) - 1, "material": len(gltf["materials"]) - 1}]})
        gltf["nodes"].append({"name": name, "mesh": len(gltf["meshes"]) - 1})
        gltf["scenes"][0]["nodes"].append(len(gltf["nodes"]) - 1)
        summary[name] = {"triangles": int(len(faces)), "vertices": int(len(verts)), "sampling_step": step}

    while len(binary) % 4:
        binary.append(0)
    gltf["buffers"].append({"byteLength": len(binary)})
    body = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    body += b" " * (-len(body) % 4)
    glb = (struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(body) + 8 + len(binary))
           + struct.pack("<II", len(body), 0x4E4F534A) + body
           + struct.pack("<II", len(binary), 0x004E4942) + bytes(binary))
    return glb, summary
