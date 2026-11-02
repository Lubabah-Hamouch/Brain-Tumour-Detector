"""Integration tests for the FastAPI backend.

The prediction tests compare API output with the numbers the evaluation scripts recorded in
reports/results/, so a preprocessing or inference difference between the API and the evaluated
pipeline fails here. They need the trained checkpoints and the local datasets and are skipped
when those are absent.
"""
import base64
import csv
import gzip
import io
import json
import random
import time

import nibabel as nib
import numpy as np
import pytest
from PIL import Image

from src.config import PROJECT_ROOT, RAW_DIR
from src.data.classification_preprocess import CLASSES, CLEAN_DIR
from src.data.segmentation_preprocess import load_case, region_targets
from src.training.inference import dice_scores

RESULTS = PROJECT_ROOT / "reports" / "results"
CLASSIFIER = PROJECT_ROOT / "models" / "classification_crop_resdegrade_long" / "best.pt"
SEG_FOLDS = [PROJECT_ROOT / "models" / "segmentation" / f"fold{k}" / "best.pt" for k in range(1, 6)]
MODALITIES = ("flair", "t1", "t1ce", "t2")

pytestmark = pytest.mark.skipif(
    not (CLASSIFIER.exists() and all(p.exists() for p in SEG_FOLDS) and CLEAN_DIR.exists()
         and (RAW_DIR / "brats2021").exists()),
    reason="needs trained checkpoints and local datasets")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from backend.main import create_app

    with TestClient(create_app()) as c:
        yield c


def upload_image(client, path):
    return client.post("/api/classify", files={"file": (path.name, path.read_bytes(), "image/jpeg")})


def nifti_bytes(array, affine=None):
    return gzip.compress(nib.Nifti1Image(array.astype(np.float32), np.eye(4) if affine is None else affine).to_bytes())


def brats_files(case):
    d = RAW_DIR / "brats2021" / case
    return {m: (f"{case}_{m}.nii.gz", (d / f"{case}_{m}.nii.gz").read_bytes(), "application/gzip") for m in MODALITIES}


def test_health_reports_loaded_models(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["classifier"]["loaded"] and body["segmentation"]["loaded"]
    assert body["segmentation"]["detail"]["folds"] == [1, 2, 3, 4, 5]


def test_classify_response_shape(client):
    r = upload_image(client, CLEAN_DIR / "test" / "glioma" / "Te-gl_350.jpg")
    assert r.status_code == 200
    body = r.json()
    assert set(body["probabilities"]) == set(CLASSES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert body["confidence"] == max(body["probabilities"].values())
    heat = Image.open(io.BytesIO(base64.b64decode(body["gradcam"]["heatmap_png"])))
    assert heat.size == (384, 384) and heat.mode == "L"
    lb = body["letterbox"]
    assert lb["resized_width"] == 384 or lb["resized_height"] == 384
    model_input = Image.open(io.BytesIO(base64.b64decode(body["model_input_png"])))
    assert model_input.size == (384, 384) and model_input.mode == "L"
    red = Image.open(io.BytesIO(base64.b64decode(body["gradcam"]["heat_png"])))
    assert red.size == (384, 384) and red.mode == "RGBA"
    assert body["image"]["grayscale"] is True
    assert body["report_url"].endswith(f"/api/classify/{body['result_id']}/report.png")
    # The half-peak outline: closed polylines inside the model input, around the heatmap's peak.
    cam = np.asarray(heat, dtype=np.float32) / 255
    assert body["gradcam"]["contour_level"] == 0.5 and body["gradcam"]["contour"]
    for line in body["gradcam"]["contour"]:
        xy = np.asarray(line)
        assert len(xy) >= 3 and (xy >= -1).all() and (xy <= 384).all()
    py, px = np.unravel_index(cam.argmax(), cam.shape)
    xs, ys = zip(*(p for line in body["gradcam"]["contour"] for p in line))
    assert min(xs) <= px <= max(xs) and min(ys) <= py <= max(ys)


def test_classify_contour_present_for_every_outcome(client):
    """Every prediction, "no tumour" included, carries the outline as a list of closed polylines."""
    predicted = set()
    for cls in CLASSES:
        body = upload_image(client, sorted((CLEAN_DIR / "test" / cls).glob("*.jpg"))[0]).json()
        predicted.add(body["predicted_class"])
        contour = body["gradcam"]["contour"]
        assert isinstance(contour, list) and contour, body["predicted_class"]
        assert all(len(line) >= 3 and all(len(p) == 2 for p in line) for line in contour)
    assert "notumor" in predicted


def test_cam_contour_empty_for_blank_heatmap():
    """A heatmap with nothing in it gives an empty outline, not an error or a missing field."""
    from backend.services.imaging import cam_contour, smooth_cam

    cam = smooth_cam(np.zeros((384, 384), dtype=np.float32))
    assert np.isfinite(cam).all() and cam_contour(cam) == []


def test_classify_report_omits_heatmap_for_no_tumour(client):
    """A no-tumour report shows the plain slice: no Grad-CAM reds in the image panel."""
    images = sorted((CLEAN_DIR / "test" / "notumor").glob("*.jpg"))[:5]
    body = next(b for b in (upload_image(client, p).json() for p in images) if b["predicted_class"] == "notumor")
    report = np.asarray(Image.open(io.BytesIO(client.get(body["report_url"].replace("http://testserver", "")).content)).convert("RGB"))
    panel = report[300:1900]  # the image panel, between the header and the prediction block
    r, g, b = (panel[..., i].astype(int) for i in range(3))
    assert not ((r - np.maximum(g, b)) > 40).any()


def test_classify_report_png(client):
    """The downloadable report renders for a recent result, at 2x the 1200 x 1500 layout."""
    body = upload_image(client, CLEAN_DIR / "test" / "meningioma" / "Tr-me_1065.jpg").json()
    r = client.get(body["report_url"].replace("http://testserver", ""))
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.headers["content-disposition"].startswith("attachment;")
    report = Image.open(io.BytesIO(r.content))
    assert report.size == (2400, 3000)
    # The prediction bar is drawn in the finding red; the rest of the page is greyscale or the heatmap.
    assert (np.asarray(report.convert("RGB")) == (215, 38, 61)).all(axis=-1).any()


def test_classify_report_unknown_is_404(client):
    r = client.get("/api/classify/not-a-result/report.png")
    assert r.status_code == 404 and r.json()["error"]["code"] == "result_not_found"


def test_samples_are_listed_and_served(client):
    samples = client.get("/api/samples").json()
    assert [s["name"] for s in samples] == ["Tr-me_1065.jpg", "Te-pi_102.jpg", "Te-gl_350.jpg"]
    recorded = json.loads((RESULTS / "classification_test_crop_resdegrade_long.json").read_text())
    errors = {e["file"] for e in recorded["misclassified"]}
    for s in samples:
        path = CLEAN_DIR / "test" / s["true_class"] / s["name"]
        assert s["known_error"] == (s["name"] in errors)
        r = client.get(s["url"].replace("http://testserver", ""))
        assert r.status_code == 200 and r.content == path.read_bytes()
    assert [s["name"] for s in samples if s["known_error"]] == ["Te-gl_350.jpg"]


@pytest.mark.parametrize("name", ["Te-gl_106.jpg", "..%2F..%2F.env", "missing.jpg"])
def test_only_listed_samples_are_served(client, name):
    r = client.get(f"/api/samples/{name}")
    assert r.status_code == 404


def test_classify_matches_evaluation(client):
    """Every recorded test error is reproduced (same class, same confidence) and sampled correct images stay correct."""
    recorded = json.loads((RESULTS / "classification_test_crop_resdegrade_long.json").read_text())
    errors = {e["file"]: e for e in recorded["misclassified"]}
    paths = {p.name: p for p in (CLEAN_DIR / "test").glob("*/*")}
    for name, e in errors.items():
        body = upload_image(client, paths[name]).json()
        assert body["predicted_class"] == e["predicted"], name
        assert body["confidence"] == pytest.approx(e["confidence"], abs=0.01), name
    correct = sorted(n for n in paths if n not in errors)
    for name in random.Random(0).sample(correct, 40):
        body = upload_image(client, paths[name]).json()
        assert body["predicted_class"] == paths[name].parent.name, name


def wait_for_job(client, status_url, timeout=600):
    """Poll a job until it finishes; returns (final snapshot, list of statuses seen)."""
    seen, deadline = [], time.time() + timeout
    while time.time() < deadline:
        snap = client.get(status_url.replace("http://testserver", "")).json()
        seen.append(snap["status"])
        if snap["status"] in ("complete", "failed"):
            return snap, seen
        time.sleep(0.2)
    raise AssertionError(f"job did not finish in {timeout} s")


@pytest.fixture(scope="module")
def segmented(client):
    case = json.loads((PROJECT_ROOT / "data" / "manifests" / "brats_5fold_splits.json").read_text())["test"][0]
    r = client.post("/api/segment?inline_mask=true", files=brats_files(case))
    assert r.status_code == 202, r.text
    accepted = r.json()
    assert accepted["status"] == "queued" and r.headers["location"] == accepted["status_url"]
    job, seen = wait_for_job(client, accepted["status_url"])
    assert job["status"] == "complete", job["error"]
    return case, job["result"], job, seen


def test_segment_job_reports_real_stages(segmented):
    _, _, job, seen = segmented
    assert seen[-1] == "complete" and "failed" not in seen
    assert [s["name"] for s in job["stages"]] == ["preprocessing", "inference", "postprocessing"]
    assert all(s["status"] == "complete" and s["fraction"] == 1.0 and s["seconds"] is not None for s in job["stages"])
    assert job["progress"] == 1.0 and job["elapsed_seconds"] > 0 and job["error"] is None
    assert job["stages"][1]["detail"] == "5 models done"


def test_unknown_job_is_404(client):
    r = client.get("/api/jobs/not-a-job")
    assert r.status_code == 404 and r.json()["error"]["code"] == "job_not_found"


def test_results_summary_matches_report_files(client):
    body = client.get("/api/results/summary").json()
    seg = json.loads((RESULTS / "segmentation_test.json").read_text())["test"]["ensemble"]["post_processed"]
    assert body["segmentation"]["dice"]["mean"] == pytest.approx(seg["mean"])
    assert body["segmentation"]["test_patients"] == 189
    clf = body["classification"]
    assert clf["checkpoint"] == "models/classification_crop_resdegrade_long/best.pt"
    assert clf["accuracy"] == pytest.approx(1019 / 1037) and clf["test_images"] == 1037
    assert clf["accuracy_ci95"][0] < clf["accuracy"] < clf["accuracy_ci95"][1]
    assert clf["accuracy_at_128px"] == pytest.approx(0.9817, abs=1e-4)
    assert clf["test_errors"] == 18 == sum(c["errors"] for c in clf["confusions"])
    assert {"classes": ["glioma", "meningioma"], "errors": 5} in clf["confusions"]


def test_segment_response_shape(client, segmented):
    _, body, _, _ = segmented
    assert body["shape"] == [240, 240, 155] and body["spacing_mm"] == [1.0, 1.0, 1.0]
    assert set(body["regions"]) == {"WT", "TC", "ET"}
    for region in body["regions"].values():
        assert region["volume_ml"] == pytest.approx(region["voxels"] / 1000)
        assert 0 <= region["model_agreement"] <= 1
    assert body["regions"]["WT"]["voxels"] >= body["regions"]["TC"]["voxels"] >= body["regions"]["ET"]["voxels"]
    assert [p["plane"] for p in body["preview"]] == ["axial", "coronal", "sagittal"]
    downloaded = client.get(body["mask_url"].replace("http://testserver", ""))
    assert downloaded.status_code == 200
    assert downloaded.content == base64.b64decode(body["mask_nifti_base64"])


def test_segment_matches_evaluation(segmented):
    """Dice of the returned label map against ground truth equals the evaluated 5-fold ensemble + post-processing."""
    case, body, _, _ = segmented
    mask = nib.Nifti1Image.from_bytes(gzip.decompress(base64.b64decode(body["mask_nifti_base64"])))
    labels = np.asarray(mask.dataobj)
    _, seg = load_case(RAW_DIR / "brats2021" / case)
    truth = region_targets(seg).astype(bool)
    pred = np.stack([labels > 0, np.isin(labels, (1, 4)), labels == 4])
    wt, tc, et = dice_scores(_torch(pred), _torch(truth))
    rows = {r["model"]: r for r in csv.DictReader(open(RESULTS / "segmentation_test_per_case.csv")) if r["case"] == case}
    expected = rows["ensemble"]
    assert wt == pytest.approx(float(expected["WT"]), abs=0.005)
    assert tc == pytest.approx(float(expected["TC"]), abs=0.005)
    assert et == pytest.approx(float(expected["ET_pp"]), abs=0.005)
    assert np.allclose(mask.affine, nib.load(RAW_DIR / "brats2021" / case / f"{case}_flair.nii.gz").affine)


def _torch(array):
    import torch

    return torch.from_numpy(np.ascontiguousarray(array))


# --- error handling -------------------------------------------------------------------------

def test_classify_rejects_wrong_type(client):
    r = client.post("/api/classify", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_file_type"


def test_classify_rejects_corrupt_image(client):
    r = client.post("/api/classify", files={"file": ("broken.jpg", b"\xff\xd8\xff not really a jpeg", "image/jpeg")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "unreadable_image"


def test_classify_rejects_tiny_image(client):
    buffer = io.BytesIO()
    Image.new("L", (16, 16)).save(buffer, format="PNG")
    r = client.post("/api/classify", files={"file": ("tiny.png", buffer.getvalue(), "image/png")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "image_too_small"


def test_classify_requires_file(client):
    r = client.post("/api/classify")
    assert r.status_code == 422 and r.json()["error"]["code"] == "missing_fields"


def test_segment_requires_all_four_modalities(client):
    files = {m: (f"{m}.nii.gz", nifti_bytes(np.ones((8, 8, 8))), "application/gzip") for m in ("flair", "t1", "t1ce")}
    r = client.post("/api/segment", files=files)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "missing_fields" and "t2" in err["message"]


def test_segment_rejects_non_nifti(client):
    files = {m: (f"{m}.nii.gz", nifti_bytes(np.ones((8, 8, 8))), "application/gzip") for m in MODALITIES}
    files["t1"] = ("t1.png", b"\x89PNG", "image/png")
    r = client.post("/api/segment", files=files)
    assert r.status_code == 415 and "t1.png" in r.json()["error"]["message"]


def test_segment_rejects_corrupt_nifti(client):
    files = {m: (f"{m}.nii.gz", nifti_bytes(np.ones((8, 8, 8))), "application/gzip") for m in MODALITIES}
    files["t2"] = ("t2.nii.gz", gzip.compress(b"definitely not nifti"), "application/gzip")
    r = client.post("/api/segment", files=files)
    assert r.status_code == 422 and r.json()["error"]["code"] == "corrupt_nifti"


def test_segment_rejects_mismatched_shapes(client):
    files = {m: (f"{m}.nii.gz", nifti_bytes(np.ones((8, 8, 8))), "application/gzip") for m in MODALITIES}
    files["t1ce"] = ("t1ce.nii.gz", nifti_bytes(np.ones((9, 8, 8))), "application/gzip")
    r = client.post("/api/segment", files=files)
    assert r.status_code == 422 and r.json()["error"]["code"] == "shape_mismatch"


def test_segment_rejects_4d_volume(client):
    files = {m: (f"{m}.nii.gz", nifti_bytes(np.ones((8, 8, 8))), "application/gzip") for m in MODALITIES}
    files["flair"] = ("flair.nii.gz", nifti_bytes(np.ones((8, 8, 8, 2))), "application/gzip")
    r = client.post("/api/segment", files=files)
    assert r.status_code == 422 and r.json()["error"]["code"] == "not_3d"


def test_unknown_mask_is_404(client):
    r = client.get("/api/segment/does-not-exist/mask.nii.gz")
    assert r.status_code == 404 and r.json()["error"]["code"] == "result_not_found"


def test_unknown_segmentation_slices_and_report_are_404(client):
    for path in ("/api/segment/does-not-exist/slices/axial/flair.png", "/api/segment/does-not-exist/report.png"):
        r = client.get(path)
        assert r.status_code == 404 and r.json()["error"]["code"] == "result_not_found"


def test_segment_viewer_slices(client, segmented):
    """Every plane has seven same-sized layers, through the largest tumour cross-section."""
    case, body, _, _ = segmented
    assert body["study"] == case and body["files"]["t1ce"] == f"{case}_t1ce.nii.gz"
    labels = np.asarray(nib.Nifti1Image.from_bytes(gzip.decompress(base64.b64decode(body["mask_nifti_base64"]))).dataobj)
    for plane, axis in (("axial", 2), ("coronal", 1), ("sagittal", 0)):
        view = body["slices"][plane]
        assert view["count"] == labels.shape[axis]
        other = tuple(a for a in range(3) if a != axis)
        assert view["index"] == int((labels > 0).sum(axis=other).argmax())
        assert set(view["layers"]) == {"flair", "t1", "t1ce", "t2", "edema", "necrotic", "enhancing"}
        for layer, url in view["layers"].items():
            r = client.get(url.replace("http://testserver", ""))
            assert r.status_code == 200 and r.headers["content-type"] == "image/png", (plane, layer)
            img = Image.open(io.BytesIO(r.content))
            assert img.size == (view["width"], view["height"]) and img.mode == "RGBA"
    # The label layers carry the label map: no edema on screen where the slice has none, and vice versa.
    axial = body["slices"]["axial"]
    edema = np.asarray(Image.open(io.BytesIO(client.get(axial["layers"]["edema"].replace("http://testserver", "")).content)))
    assert (edema[..., 3] > 0).any() == (labels[:, :, axial["index"]] == 2).any()


def test_segment_report_png(client, segmented):
    _, body, _, _ = segmented
    url = body["report_url"].replace("http://testserver", "")
    r = client.get(url, params={"plane": "coronal", "sequence": "t1ce", "labels": "edema,enhancing", "opacity": 60})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.headers["content-disposition"].startswith("attachment;") and "coronal_t1ce" in r.headers["content-disposition"]
    assert Image.open(io.BytesIO(r.content)).size == (2400, 3000)
    assert client.get(url, params={"plane": "oblique"}).status_code == 422


def test_sample_study_is_listed_and_served(client):
    body = client.get("/api/sample-study").json()
    assert body["case"] == "BraTS2021_00003" and body["split"] == "test"
    assert [f["sequence"] for f in body["files"]] == ["flair", "t1", "t1ce", "t2"]
    for f in body["files"]:
        r = client.get(f["url"].replace("http://testserver", ""))
        assert r.status_code == 200 and r.content == (RAW_DIR / "brats2021" / body["case"] / f["name"]).read_bytes()
        assert len(r.content) == f["size_bytes"]
    for name in ("BraTS2021_00003_seg.nii.gz", "BraTS2021_00009_flair.nii.gz", "..%2F..%2F.env"):
        assert client.get(f"/api/sample-study/{name}").status_code == 404


def test_model_card_matches_results_files(client):
    """Every model card number traces to reports/results/ or reports/final_results.md."""
    card = client.get("/api/results/model-card").json()
    seg = json.loads((RESULTS / "segmentation_test.json").read_text())
    stress = json.loads((RESULTS / "format_stress_test.json").read_text())
    final = (PROJECT_ROOT / "reports" / "final_results.md").read_text(encoding="utf-8")

    # Segmentation table and paired comparisons, as printed in final_results.md.
    rows = {(r["model"], r["post_processed"]): r for r in card["segmentation"]["rows"]}
    assert rows[("ensemble", True)]["served"] and rows[("ensemble", True)]["mean"] == seg["test"]["ensemble"]["post_processed"]["mean"]
    assert f"{rows[('fold4', False)]['mean']:.4f}" == "0.8873" and "| Single best (fold 4) | no | 0.9218 | 0.8835 | 0.8566 | 0.8873 |" in final
    ens = card["segmentation"]["ensemble_vs_single_best"]
    assert f"+{ens['difference']:.4f}" == "+0.0088" and (ens["patients_better"], ens["patients_worse"]) == (141, 48)
    cleanup = card["segmentation"]["et_cleanup_on_ensemble"]
    assert cleanup["ci95"][0] < 0 < cleanup["ci95"][1]  # not significant, as the report says
    assert card["segmentation"]["patients_without_et"] == 5
    assert card["segmentation"]["patients_without_et_scored_1"] == {"raw": 3, "post_processed": 5}
    cv = card["segmentation"]["cross_validation"]
    assert f"{cv['mean']:.4f} ± {cv['std']:.4f}" == "0.8983 ± 0.0044" and "**0.8983 ± 0.0044**" in final

    # Classification variants and the stress test.
    models = card["classification"]["models"]
    assert [m["key"] for m in models] == ["A", "B", "C", "D"] and [m["served"] for m in models] == [False, False, False, True]
    assert [f"{m['accuracy'] * 100:.2f}%" for m in models] == ["98.26%", "98.17%", "97.78%", "98.26%"]
    assert [f"{m['accuracy_at_128px'] * 100:.2f}%" for m in models] == ["74.93%", "77.53%", "97.11%", "98.17%"]
    assert [m["values"] for m in card["stress_test"]["models"]] == [[3, 12, 10, 250], [1, 14, 16, 213], [2, 3, 3, 9], [2, 1, 1, 0]]
    assert card["stress_test"]["tumour_images"] == stress["tumour_images"] == 778
    assert [c["label"] for c in card["stress_test"]["conditions"]] == ["unmodified", "225 × 225", "210 × 234", "128 × 128"]
    assert card["headline"]["mcnemar_p"] == 1.0

    # Specification: dataset counts from the manifests, parameter counts from the loaded models.
    spec = card["specification"]
    assert spec["segmentation"]["patients"] == 1251 and spec["classification"]["images"] == 6951
    assert f"{spec['segmentation']['parameters'] / 1e6:.1f}" == "16.5" and "3D U-Net with 16.5 M parameters" in final
    assert round(spec["classification"]["parameters"] / 1e6, 1) == 20.2
    assert [round(spec["classification"]["split"][k] * 100) for k in ("train", "val", "test")] == [70, 15, 15]

    # Limitations: the bullets of final_results.md, in order.
    titles = [line.split("**")[1].rstrip(".") for line in final.split("## Limitations")[1].split("## Reproducing")[0].splitlines()
              if line.startswith("- **")]
    assert [x["title"] for x in card["limitations"]] == titles and len(titles) == 7


def test_segment_mesh_glb(client, segmented):
    """The 3D surfaces: a valid binary glTF with the four named surfaces, within budget, cached."""
    import struct

    from backend.services.meshes import SURFACES

    _, body, _, _ = segmented
    url = body["mesh_url"].replace("http://testserver", "")
    r = client.get(url)
    assert r.status_code == 200 and r.headers["content-type"] == "model/gltf-binary"
    glb = r.content
    magic, version, length = struct.unpack_from("<III", glb, 0)
    assert (magic, version, length) == (0x46546C67, 2, len(glb))
    json_length, json_type = struct.unpack_from("<II", glb, 12)
    assert json_type == 0x4E4F534A
    gltf = json.loads(glb[20:20 + json_length])
    assert [n["name"] for n in gltf["nodes"]] == ["brain", "edema", "necrotic", "enhancing"]
    budgets = {name: budget for name, _, budget, _, _ in SURFACES}
    for mesh in gltf["meshes"]:
        prim = mesh["primitives"][0]
        triangles = gltf["accessors"][prim["indices"]]["count"] // 3
        assert 0 < triangles <= budgets[mesh["name"]], mesh["name"]
        assert gltf["accessors"][prim["attributes"]["POSITION"]]["count"] == gltf["accessors"][prim["attributes"]["NORMAL"]]["count"]
    # Millimetres, centred on the brain: a BraTS brain spans roughly 120-180 mm.
    brain = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    extent = np.array(brain["max"]) - np.array(brain["min"])
    assert ((extent > 100) & (extent < 200)).all() and np.allclose(np.array(brain["max"]) + np.array(brain["min"]), 0, atol=5)
    assert client.get(url).content == glb  # served from the cache the second time
    assert client.get("/api/segment/does-not-exist/mesh.glb").status_code == 404
