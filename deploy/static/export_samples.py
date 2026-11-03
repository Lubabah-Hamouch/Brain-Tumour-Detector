"""Export the samples and test results to static files, for a web client deployed without the full API.

Writes frontend/public/demo/: the five sample images and their classification results, the sample
patient's segmentation (viewer slices, 3D surfaces, label map, report PNGs), the results summary,
the model card and, if it has been rendered, the Overview's 3D render. A web client built without NEXT_PUBLIC_API_URL (frontend/src/lib/api.ts)
reads them instead of calling the API.

Nothing is recomputed or re-implemented: the script starts the API in-process with the release
weights and the samples' results that deploy/vm/build_bundle.py computed for them (as the VM
serves them), asks it what the web client would ask, and saves the answers. Every API URL in an
answer is fetched and saved too, and rewritten to the file's path under /demo/.

What becomes public is what the Oracle VM deployment already serves (DEPLOYMENT.md, "What becomes
public"): the sample images (CC BY 4.0) and results derived from the sample patient. No raw BraTS
volumes: the script stops if a NIfTI file other than the predicted label map ends up in the output.

The segmentation report PNG is drawn on request for any plane, sequence, set of labels and overlay
opacity. The static copy has one per plane and sequence, with all labels at full opacity.

Run from the repository root, in the project environment, with the weights in models/ and the
sample images in data/ (as for the "try a sample" buttons locally), after build_bundle.py:
    python deploy/vm/build_bundle.py          # once: computes build/vm/models/sample_results.npz
    python deploy/static/export_samples.py
"""
import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "frontend" / "public" / "demo"
PREFIX = "/demo"
BASE = "http://testserver"
RELEASE_PREFIX = "build/release-weights/models/"

# API path -> path under OUT; anything else under /api/ is an error, so nothing is missed.
ROUTES = (
    (r"/api/samples/(?P<name>[^/]+)", "classify/images/{name}"),
    (r"/api/classify/(?P<id>\w+)/report\.png", "classify/reports/{sample}.png"),
    (r"/api/segment/(?P<id>\w+)/slices/(?P<plane>\w+)/(?P<layer>\w+)\.png", "segment/slices/{plane}/{layer}.png"),
    (r"/api/segment/(?P<id>\w+)/mask\.nii\.gz", "segment/mask.nii.gz"),
    (r"/api/segment/(?P<id>\w+)/mesh\.glb", "segment/mesh.glb"),
    # Fetched below for each plane and sequence; the web client adds /<plane>_<sequence>.png.
    (r"/api/segment/(?P<id>\w+)/report\.png", "segment/reports"),
)
PLANES = ("axial", "coronal", "sagittal")
SEQUENCES = ("flair", "t1", "t1ce", "t2")


def get(client, path, **kwargs):
    response = client.get(path, **kwargs)
    if response.status_code != 200:
        sys.exit(f"GET {path}: {response.status_code} {response.text[:300]}")
    return response


def write(rel, data):
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else json.dumps(data, indent=1).encode())


def localise(client, obj, sample=None):
    """Copy of a response with every API URL saved under OUT and replaced by its static path."""
    if isinstance(obj, dict):
        return {k: localise(client, v, sample) for k, v in obj.items()}
    if isinstance(obj, list):
        return [localise(client, v, sample) for v in obj]
    if not isinstance(obj, str):
        return obj
    # The results were computed from build/release-weights/models/ (deploy/vm/build_bundle.py); the
    # API serves the same files from models/.
    if obj.startswith(RELEASE_PREFIX):
        return "models/" + obj[len(RELEASE_PREFIX):]
    if not obj.startswith(BASE + "/api/"):
        return obj
    path = obj[len(BASE):]
    for pattern, target in ROUTES:
        match = re.fullmatch(pattern, path)
        if match:
            rel = target.format(sample=sample, **match.groupdict())
            if rel != "segment/reports":
                write(rel, get(client, path).content)
            return f"{PREFIX}/{rel}"
    sys.exit(f"No static path for {path}: add it to ROUTES (and to the web client if it is new).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-results", type=Path, default=ROOT / "build" / "vm" / "models" / "sample_results.npz",
                        help="the samples' results for the release weights, as deploy/vm/build_bundle.py makes them")
    parser.add_argument("--weights-cache", type=Path, default=ROOT / "build" / "release-weights",
                        help="where release weights not found locally are downloaded (as build_bundle.py)")
    args = parser.parse_args()

    os.environ.setdefault("BT_DEVICE", "cpu")
    from fastapi.testclient import TestClient

    from backend.config import Settings, tuned_et_threshold
    from backend.main import create_app
    from backend.routers.classification import SAMPLES
    from backend.services.samples import check_release_file
    from deploy.vm.build_bundle import release_weights

    problems = check_release_file(args.sample_results, SAMPLES, tuned_et_threshold())
    if problems:
        sys.exit(f"{args.sample_results}: {'; '.join(problems)}. Make it with python deploy/vm/build_bundle.py.")
    if OUT.exists():
        shutil.rmtree(OUT)
    settings = Settings.from_env()

    # The test results describe the served classifier, which the API finds by its path under models/;
    # they do not depend on the weights themselves.
    with TestClient(create_app(replace(settings, sample_results=None)), base_url=BASE) as client:
        write("summary.json", get(client, "/api/results/summary").json())
        write("model-card.json", get(client, "/api/results/model-card").json())

    # The samples, from the results computed with the release weights, as on the VM.
    models = release_weights(args.weights_cache)
    release = replace(settings, classifier_checkpoint=models / "classification_crop_resdegrade_long" / "best.pt",
                      segmentation_dir=models / "segmentation", sample_results=args.sample_results,
                      sample_results_required=True)
    with TestClient(create_app(release), base_url=BASE) as client:
        samples = get(client, "/api/samples").json()
        if [s["name"] for s in samples] != list(SAMPLES):
            sys.exit(f"Sample images missing from data/: found {[s['name'] for s in samples]}, expected {list(SAMPLES)}")
        for sample in samples:
            image = get(client, sample["url"]).content
            result = client.post("/api/classify", files={"file": (sample["name"], image, "image/jpeg")}).json()
            if not result.get("precomputed"):
                sys.exit(f"{sample['name']}: no pre-computed result; {args.sample_results} is out of date.")
            stem = Path(sample["name"]).stem
            write(f"classify/results/{stem}.json", localise(client, result, stem))
        write("classify/samples.json", localise(client, samples))

        accepted = client.post("/api/sample-study/segment").json()
        result = get(client, accepted["status_url"][len(BASE):]).json()["result"]
        for plane in PLANES:
            for sequence in SEQUENCES:
                png = get(client, result["report_url"][len(BASE):], params={"plane": plane, "sequence": sequence})
                write(f"segment/reports/{plane}_{sequence}.png", png.content)
        write("segment/result.json", localise(client, result))

    hero = ROOT / "frontend" / "public" / "generated" / "hero-3d.png"
    if hero.exists():
        shutil.copyfile(hero, OUT / "hero-3d.png")
    else:
        print("No Overview render (python -m src.visualization.render_hero): the demo's Overview shows none.")

    nifti = [p for p in OUT.rglob("*.nii*") if p != OUT / "segment" / "mask.nii.gz"]
    if nifti:
        sys.exit(f"NIfTI files other than the label map in the output: {nifti}. The raw BraTS volumes must not be published.")
    files = [p for p in OUT.rglob("*") if p.is_file()]
    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
