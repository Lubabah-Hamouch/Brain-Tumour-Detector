"""Pre-computed results for the sample inputs offered by the web client.

The sample study and the five sample images (GET /api/samples) are known in advance, so their
results are computed once and kept for the life of the process, pinned in the same result caches
the download, slice, mesh and report endpoints read. The web client's sample buttons are answered
from there at once; every other upload runs the models exactly as before.

- The sample study's result is returned by POST /api/sample-study/segment, without any files
  changing hands. On the public deployment that is the only way: the study's raw volumes are not
  deployed (BraTS terms), so the result comes from the saved file and cannot be recomputed there.
- An upload of a sample is also answered from them, matched on content: field, file name, size and
  SHA-256 of the bytes must all match. A file that differs in any way runs the models. The size
  check comes first, so other uploads are not even hashed.

The results are saved to one file (BT_SAMPLE_RESULTS) with a fingerprint of the served checkpoints
and settings and the hashes of the sample files. A restart loads that file in about a second
instead of recomputing; a file written for other weights, settings or sample files present here is
ignored and the results are computed again. The file is an .npz
read with allow_pickle=False, like the checkpoints' weights_only loading: arrays and bytes are
stored as arrays, everything else as JSON, so loading it never runs code.
"""
import base64
import hashlib
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np

from backend.services import slices
from src.download_weights import WEIGHTS, sha256

log = logging.getLogger("backend.samples")

FORMAT = 1
INDEX = "index"  # the npz member holding the JSON


def pack(obj, arrays):
    """Nested results -> JSON-compatible structure; arrays and bytes go to `arrays` (a list)."""
    if isinstance(obj, np.ndarray):
        arrays.append(obj)
        return {"$array": len(arrays) - 1}
    if isinstance(obj, (bytes, bytearray)):
        arrays.append(np.frombuffer(obj, dtype=np.uint8))
        return {"$bytes": len(arrays) - 1}
    if isinstance(obj, dict):  # keys may be tuples, e.g. (plane, layer)
        return {"$dict": [[pack(k, arrays), pack(v, arrays)] for k, v in obj.items()]}
    if isinstance(obj, tuple):
        return {"$tuple": [pack(v, arrays) for v in obj]}
    if isinstance(obj, list):
        return [pack(v, arrays) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def unpack(obj, arrays):
    if isinstance(obj, list):
        return [unpack(v, arrays) for v in obj]
    if isinstance(obj, dict):
        (tag, value), = obj.items()
        if tag == "$array":
            return arrays[f"a{value}"]
        if tag == "$bytes":
            return arrays[f"a{value}"].tobytes()
        if tag == "$tuple":
            return tuple(unpack(v, arrays) for v in value)
        return {unpack(k, arrays): unpack(v, arrays) for k, v in value}
    return obj


def upload_key(files):
    """[(field, file name, bytes)] -> hex digest identifying exactly this upload."""
    digest = hashlib.sha256()
    for field, name, data in sorted(files, key=lambda f: f[0]):
        digest.update(f"{field}\0{name}\0{len(data)}\0".encode())
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def with_base_url(result, base_url):
    """A copy of a stored result with its API paths ("/api/...") made absolute for this request."""
    if isinstance(result, dict):
        return {k: with_base_url(v, base_url) for k, v in result.items()}
    if isinstance(result, list):
        return [with_base_url(v, base_url) for v in result]
    if isinstance(result, str) and result.startswith("/api/"):
        return base_url + result
    return result


def read(path):
    """The saved results file -> {"fingerprint", "inputs", "classification", "segmentation"}."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return unpack(json.loads(arrays.pop(INDEX).tobytes()), arrays)


class SampleResults:
    def __init__(self, path, classifier, segmenter, checkpoints, classification_inputs, segmentation_inputs):
        """checkpoints: the served checkpoint files; classification_inputs: {file name: path};
        segmentation_inputs: {modality: path}, or {} where the sample study's files are not present
        (the public deployment), in which case its result can only come from the saved file."""
        self.path = Path(path) if path else None
        self.checkpoints = [Path(p) for p in checkpoints]
        self.classifier, self.segmenter = classifier, segmenter
        self.classification_inputs = dict(classification_inputs)
        self.segmentation_inputs = dict(segmentation_inputs)
        self._classification = {}  # upload key -> result with relative URLs
        self._segmentation = {}
        self._study = None  # the sample study's result, for POST /api/sample-study/segment
        self._sizes = set()  # (field, name, size) of every sample file: the cheap first check
        self._lock = threading.Lock()
        self.ready = False
        self.fingerprint = {"format": FORMAT, "et_min_voxels": segmenter.et_min_voxels,
                            "checkpoints": [sha256(p) for p in self.checkpoints]}
        self.input_hashes = {f"file:{name}": sha256(p) for name, p in self.classification_inputs.items()}
        self.input_hashes.update({f"{m}:{Path(p).name}": sha256(p) for m, p in self.segmentation_inputs.items()})

    # ---- serving ----

    def classification(self, name, data, base_url):
        if ("file", name, len(data)) not in self._sizes:
            return None
        result = self._classification.get(upload_key([("file", name, data)]))
        return with_base_url(result, base_url) if result else None

    def segmentation(self, files, base_url, inline_mask=False):
        """files: {modality: (name, bytes)}."""
        if any((m, name, len(data)) not in self._sizes for m, (name, data) in files.items()):
            return None
        return self._served(self._segmentation.get(upload_key([(m, name, data) for m, (name, data) in files.items()])),
                            base_url, inline_mask)

    def study_result(self, base_url, inline_mask=False):
        """The sample study's result, without its files: what the sample button shows."""
        return self._served(self._study, base_url, inline_mask)

    def _served(self, result, base_url, inline_mask):
        if result is None:
            return None
        result = with_base_url(result, base_url)
        if inline_mask:
            mask = self.segmenter.mask_nifti(result["result_id"])
            result["mask_nifti_base64"] = base64.b64encode(mask).decode("ascii")
        return result

    @property
    def has_study(self):
        return self._study is not None

    # ---- loading, computing, saving ----

    def load(self):
        """Pin the saved results if the file matches the served checkpoints and settings, and every sample
        file present here. Sample files that are absent (the study on the public deployment) cannot be checked and are
        taken as saved. Returns True if it loaded the file."""
        if self.path is None or not self.path.exists():
            return False
        try:
            saved = read(self.path)
        except Exception as exc:  # unreadable or from an incompatible version: compute again
            log.warning("ignoring %s (%s: %s)", self.path, type(exc).__name__, exc)
            return False
        if saved.get("fingerprint") != self.fingerprint:
            log.warning("ignoring %s: it was computed for other weights or settings", self.path)
            return False
        stale = [k for k, h in self.input_hashes.items() if saved.get("inputs", {}).get(k) != h]
        if stale:
            log.warning("ignoring %s: it was computed for other sample files (%s)", self.path, ", ".join(stale))
            return False
        for item in saved["classification"]:
            self._add(self._classification, self.classifier.results, item)
        for item in saved["segmentation"]:
            self._add(self._segmentation, self.segmenter.cache, item)
        self.ready = True
        log.info("loaded pre-computed sample results from %s", self.path)
        return True

    def compute(self, progress=None):
        """Run every sample through the models, pin the results and save them. Slow on a CPU (minutes)."""
        report = progress or (lambda stage, fraction, detail="": None)
        stored = {"classification": [], "segmentation": []}
        t0 = time.perf_counter()
        for name, path in self.classification_inputs.items():
            data = Path(path).read_bytes()
            key = upload_key([("file", name, data)])
            result_id = key[:32]
            result = self.classifier.predict(self.classifier.open_image(data), name, "", result_id=result_id)
            stored["classification"].append({"key": key, "sizes": [("file", name, len(data))], "result": result,
                                             "entry": self.classifier.results.get(result_id)})
        if self.segmentation_inputs:
            files = {m: (Path(p).name, Path(p).read_bytes()) for m, p in self.segmentation_inputs.items()}
            key = upload_key([(m, name, data) for m, (name, data) in files.items()])
            result_id = key[:32]
            result = self.segmenter.run(self.segmenter.prepare(files), False, "", report, result_id=result_id)
            # Everything the result screen asks for, so nothing is built on the first visit either.
            for plane in result["slices"]:
                for layer in slices.LAYERS:
                    self.segmenter.slice_png(result_id, plane, layer)
            self.segmenter.mesh_glb(result_id)
            stored["segmentation"].append({"key": key, "sizes": [(m, name, len(data)) for m, (name, data) in files.items()],
                                           "result": result, "entry": self.segmenter.cache.get(result_id)})
        log.info("computed sample results in %.1f s", time.perf_counter() - t0)
        # Saved before they are served, so no request touches the entries while they are written.
        self._save(stored)
        for item in stored["classification"]:
            self._add(self._classification, self.classifier.results, item)
        for item in stored["segmentation"]:
            self._add(self._segmentation, self.segmenter.cache, item)
        self.ready = True

    def _add(self, lookup, cache, item):
        result = dict(item["result"], precomputed=True)
        cache.pin(result["result_id"], item["entry"])
        with self._lock:
            self._sizes.update(tuple(s) for s in item["sizes"])
            lookup[item["key"]] = result
            if lookup is self._segmentation:
                self._study = result

    def _save(self, stored):
        if self.path is None:
            return
        tmp = self.path.with_suffix(".part")
        try:
            arrays = []
            index = json.dumps(pack({"fingerprint": self.fingerprint, "inputs": self.input_hashes, **stored},
                                    arrays)).encode()
            members = {f"a{i}": a for i, a in enumerate(arrays)}
            members[INDEX] = np.frombuffer(index, dtype=np.uint8)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                np.savez(f, **members)
            tmp.replace(self.path)
            log.info("saved sample results to %s", self.path)
        except (OSError, TypeError, ValueError) as exc:  # e.g. a read-only models/ mount: the results still serve until restart
            log.warning("could not save sample results to %s (%s); they will be computed again on restart",
                        self.path, exc)


def check_release_file(path, image_names, et_min_voxels):
    """Problems that stop a saved results file from serving with the release weights (empty: it will).

    The public deployment has no sample study files to recompute from, so its image build runs this
    (python -m backend.services.samples) and fails on any problem, rather than start without results."""
    if not Path(path).exists():
        return [f"{path} is missing"]
    try:
        saved = read(path)
    except Exception as exc:
        return [f"{path} cannot be read ({type(exc).__name__}: {exc})"]
    fp = saved.get("fingerprint", {})
    # Checkpoint order as the API loads them: the classifier, then the folds.
    release = [WEIGHTS["classifier_crop_resdegrade_long.pt"][1]]
    release += [WEIGHTS[f"segmentation_fold{k}.pt"][1] for k in range(1, 6)]
    problems = []
    if fp.get("format") != FORMAT:
        problems.append(f"format {fp.get('format')}, expected {FORMAT}")
    if fp.get("checkpoints") != release:
        problems.append("computed with other weights than the weights-v1 release")
    if fp.get("et_min_voxels") != et_min_voxels:
        problems.append(f"ET threshold {fp.get('et_min_voxels')}, expected {et_min_voxels}")
    if len(saved.get("segmentation", [])) != 1:
        problems.append("no result for the sample study")
    missing = set(image_names) - {item["sizes"][0][1] for item in saved.get("classification", [])}
    if missing:
        problems.append(f"no result for the sample images {sorted(missing)}")
    return problems


if __name__ == "__main__":
    import sys

    from backend.config import tuned_et_threshold
    from backend.routers.classification import SAMPLES

    target = sys.argv[1] if len(sys.argv) > 1 else "models/sample_results.npz"
    found = check_release_file(target, SAMPLES, tuned_et_threshold())
    if found:
        sys.exit(f"Sample results unusable: {'; '.join(found)}. Rebuild the bundle with "
                 "python deploy/vm/build_bundle.py.")
    print(f"{target}: results for the sample study and {len(SAMPLES)} sample images, release weights")
