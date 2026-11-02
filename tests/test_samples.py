"""Unit tests for the pre-computed sample results, the pinned result cache and the CORS settings (no models needed).

tests/test_api.py covers the same path end to end with the real models.
"""
import base64

import numpy as np
import pytest

from backend.config import DEV_ORIGINS, cors_origins
from backend.services.cache import ResultCache
from backend.services.jobs import JobStore
from backend.services.samples import SampleResults, check_release_file, upload_key, with_base_url

STAGES = (("prep", "Prepare", 0.2), ("work", "Work", 0.8))


def test_pinned_results_survive_eviction():
    cache = ResultCache(2)
    cache.pin("sample", {"x": 1})
    keys = [cache.put({"n": i}) for i in range(5)]
    assert cache.get("sample") == {"x": 1}
    assert cache.get(keys[0]) is None and cache.get(keys[-1]) == {"n": 4}
    assert cache.put({"n": 9}, "chosen") == "chosen" and cache.get("chosen") == {"n": 9}


def test_add_complete_records_a_finished_job_without_running_anything():
    store = JobStore(keep=4)
    try:
        snap = store.get(store.add_complete("segmentation", STAGES, {"answer": 1}))
    finally:
        store.shutdown()
    assert snap["status"] == "complete" and snap["result"] == {"answer": 1} and snap["progress"] == 1.0
    assert snap["queue_position"] is None and snap["error"] is None and snap["elapsed_seconds"] == 0
    assert [(s["status"], s["fraction"], s["seconds"]) for s in snap["stages"]] == [("complete", 1.0, 0.0)] * 2


def test_cors_origins():
    assert cors_origins({}) == DEV_ORIGINS
    assert cors_origins({"ALLOWED_ORIGIN": "https://app.vercel.app/"}) == DEV_ORIGINS + ("https://app.vercel.app",)
    assert cors_origins({"ALLOWED_ORIGIN": "https://a.app, https://b.app", "BT_CORS_ORIGINS": "http://localhost:3000"}) \
        == ("http://localhost:3000", "https://a.app", "https://b.app")
    assert cors_origins({"ALLOWED_ORIGIN": " "}) == DEV_ORIGINS


def test_upload_key_depends_on_every_part_but_not_field_order():
    a = upload_key([("t1", "a.nii.gz", b"1"), ("t2", "b.nii.gz", b"2")])
    assert a == upload_key([("t2", "b.nii.gz", b"2"), ("t1", "a.nii.gz", b"1")])
    assert a != upload_key([("t1", "a.nii.gz", b"1"), ("t2", "b.nii.gz", b"3")])
    assert a != upload_key([("t1", "c.nii.gz", b"1"), ("t2", "b.nii.gz", b"2")])
    assert a != upload_key([("t2", "a.nii.gz", b"1"), ("t1", "b.nii.gz", b"2")])


def test_with_base_url_only_touches_api_paths():
    result = {"url": "/api/x", "nested": [{"u": "/api/y"}], "name": "api/z", "n": 3, "png": "iVBOR"}
    assert with_base_url(result, "https://h") == {"url": "https://h/api/x", "nested": [{"u": "https://h/api/y"}],
                                                  "name": "api/z", "n": 3, "png": "iVBOR"}
    assert result["url"] == "/api/x"  # the stored result is not changed


class FakeClassifier:
    def __init__(self, checkpoint):
        self.checkpoint, self.results, self.calls = checkpoint, ResultCache(2), 0

    @staticmethod
    def open_image(data):
        return data

    def predict(self, img, name="", base_url="", result_id=None):
        self.calls += 1
        result = {"predicted_class": "glioma", "report_url": f"{base_url}/api/classify/{result_id}/report.png"}
        self.results.put({"result": result, "cam": np.zeros(3)}, result_id)
        return {"result_id": result_id, **result}


class FakeSegmenter:
    et_min_voxels = 50

    def __init__(self):
        self.cache, self.calls = ResultCache(2), 0

    def prepare(self, files):
        return files

    def run(self, study, inline_mask=False, base_url="", progress=None, result_id=None):
        self.calls += 1
        self.cache.put({"mask": b"mask-bytes", "png": {}, "glb": None}, result_id)
        return {"result_id": result_id, "mask_url": f"{base_url}/api/segment/{result_id}/mask.nii.gz",
                "slices": {}, "mask_nifti_base64": None}

    def slice_png(self, result_id, plane, layer):
        return b""

    def mesh_glb(self, result_id):
        self.cache.get(result_id)["glb"] = b"glb"

    def mask_nifti(self, result_id):
        return self.cache.get(result_id)["mask"]


@pytest.fixture
def setup(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"weights")
    image = tmp_path / "Te-gl_12.jpg"
    image.write_bytes(b"jpeg bytes")
    study = {}
    for m in ("flair", "t1", "t1ce", "t2"):
        study[m] = tmp_path / f"case_{m}.nii.gz"
        study[m].write_bytes(f"volume {m}".encode())

    def make(study_files=study):
        """study_files={}: the public deployment, where the sample study's files are absent."""
        clf, seg = FakeClassifier(ckpt), FakeSegmenter()
        return (SampleResults(tmp_path / "out" / "samples.npz", clf, seg, [ckpt], {image.name: image}, study_files),
                clf, seg)

    return make, ckpt, image, study


def study_upload(study):
    return {m: (p.name, p.read_bytes()) for m, p in study.items()}


def test_samples_are_served_by_content_and_survive_a_restart(setup):
    make, ckpt, image, study = setup
    samples, clf, seg = make()
    assert not samples.load() and not samples.ready and samples.study_result("") is None
    assert samples.classification(image.name, image.read_bytes(), "http://h") is None
    samples.compute()
    assert samples.ready and samples.has_study and clf.calls == 1 and seg.calls == 1

    hit = samples.classification(image.name, image.read_bytes(), "http://h")
    assert hit["precomputed"] and hit["report_url"] == f"http://h/api/classify/{hit['result_id']}/report.png"
    assert clf.results.get(hit["result_id"]) is not None
    # Anything but the exact sample runs the model: other bytes, another name.
    assert samples.classification(image.name, b"jpeg bytez", "http://h") is None
    assert samples.classification("other.jpg", image.read_bytes(), "http://h") is None

    seg_hit = samples.segmentation(study_upload(study), "https://h", inline_mask=True)
    assert seg_hit["precomputed"] and seg_hit["mask_url"].startswith("https://h/api/segment/")
    assert base64.b64decode(seg_hit["mask_nifti_base64"]) == b"mask-bytes"
    assert seg.cache.get(seg_hit["result_id"])["glb"] == b"glb"
    changed = study_upload(study)
    changed["t2"] = ("case_t2.nii.gz", b"volume T2")
    assert samples.segmentation(changed, "https://h") is None
    # The sample button's request gets the same result, with no files involved.
    direct = samples.study_result("https://h")
    assert direct["result_id"] == seg_hit["result_id"] and direct["mask_nifti_base64"] is None

    # A new process loads the saved results instead of computing them...
    again, clf2, seg2 = make()
    assert again.load() and again.ready and clf2.calls == seg2.calls == 0
    assert again.classification(image.name, image.read_bytes(), "")["result_id"] == hit["result_id"]
    assert seg2.cache.get(seg_hit["result_id"]) is not None
    # Without the study's files (the public deployment) the saved study result still loads and is served.
    space, _, seg3 = make(study_files={})
    assert space.load() and space.has_study and seg3.calls == 0
    assert space.study_result("https://space")["mask_url"].startswith("https://space/api/segment/")
    # Not if a sample file present here changed...
    image.write_bytes(b"other jpeg")
    assert not make()[0].load() and not make(study_files={})[0].load()
    image.write_bytes(b"jpeg bytes")
    assert make()[0].load()
    # ...or the weights changed.
    ckpt.write_bytes(b"retrained weights")
    stale, _, _ = make()
    assert not stale.load()


def test_unwritable_results_file_still_serves(setup, tmp_path):
    make, _, image, _ = setup
    samples, _, _ = make()
    (tmp_path / "out").write_text("a file where the folder should be")
    samples.compute()
    assert samples.ready and samples.classification(image.name, image.read_bytes(), "") is not None


def test_release_check_rejects_other_weights_and_missing_files(setup, tmp_path):
    make, _, image, _ = setup
    samples, _, _ = make()
    samples.compute()
    problems = check_release_file(tmp_path / "out" / "samples.npz", [image.name, "Te-pi_102.jpg"], 50)
    assert any("other weights" in p for p in problems)
    assert any("Te-pi_102.jpg" in p for p in problems)
    assert not any("sample study" in p for p in problems)
    assert check_release_file(tmp_path / "missing.npz", [], 50) == [f"{tmp_path / 'missing.npz'} is missing"]
