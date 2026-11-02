"""Assemble the API's deployment bundle for a VM in build/vm/, and the same as one archive,
build/brain-tumour-api.tar.gz, to copy to the VM. It holds:

    docker-compose.yml, Caddyfile, .env.example
                              the API and Caddy (HTTPS) as one compose stack, from deploy/vm/
    Dockerfile                backend/Dockerfile plus the deployment's settings (DEPLOY_SETTINGS)
    backend/ src/ reports/ data/manifests/ requirements*.txt
                              the files the image needs, committed or new, as for docker compose
    data/processed/figshare_classification_4class_clean/test/<class>/
                              the five sample images (Brain Tumor MRI Dataset, CC BY 4.0)
    data/hero/hero-3d.png     the Overview render, if generated; the API serves it at /api/hero.png
    models/sample_results.npz the samples' results, computed here with the release weights

It never holds the sample study's raw NIfTI volumes: the BraTS terms do not allow redistributing
them. Only results derived from them are deployed (label map, viewer slices, 3D surfaces, volumes),
so the server cannot recompute the sample study and the results file is mandatory there. The
image build checks it against the release weights and fails if it does not match; the API then
refuses to start without it, and does not serve the sample study's files at all.

The weights are not included either: the container downloads them from the weights-v1 release on
its first start. The sample results are computed with exactly those release files, which are taken
from models/release/ or models/ when their SHA-256 matches, and downloaded into
build/release-weights/ otherwise. Computing takes a few minutes on a CPU (--device cuda is faster;
results match to within a voxel). It needs the sample study and images in data/, as for the local
"try a sample" buttons. Run from the repository root, in the project environment:

    python deploy/vm/build_bundle.py

Then copy the archive to the VM and start it there (DEPLOYMENT.md has the full steps):

    docker compose up -d --build
"""
import argparse
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.download_weights import WEIGHTS, sha256  # noqa: E402

# The files the image needs: what .dockerignore lets into the build for docker compose.
CODE = ["backend", "src", "reports/final_results.md", "reports/results", "data/manifests",
        "requirements.txt", "requirements-backend.txt", "models/.gitkeep"]
# The compose stack, from deploy/vm/, copied with LF line endings since they are read on Linux.
STACK = ["docker-compose.yml", "Caddyfile", ".env.example"]
RAW_SUFFIXES = (".nii", ".nii.gz", ".mat", ".h5", ".npy")
ARCHIVE_ROOT = "brain-tumour-api"

# Appended to backend/Dockerfile for the deployment.
DEPLOY_SETTINGS = """
# ---- Deployment (added by deploy/vm/build_bundle.py) ----
# The sample study's raw volumes are not deployed (BraTS terms): its files are not served, and its
# result must come from models/sample_results.npz. Stop the build here if that file is missing or
# was not computed with the release weights the container downloads.
ENV BT_SAMPLE_STUDY_FILES=0 \\
    BT_SAMPLE_RESULTS_REQUIRED=1
RUN python -m backend.services.samples models/sample_results.npz
"""

# Marks the folder as made by this script, so a rebuild may replace it.
MARKER = ".bundle"
MARKER_TEXT = "The API's deployment bundle, made by deploy/vm/build_bundle.py. Rebuilding replaces this folder.\n"


def copy(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def code_files():
    """The working tree's files under CODE, committed or not, minus anything .gitignore excludes
    (caches, data, weights)."""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *CODE],
                         cwd=ROOT, capture_output=True, check=True)
    return [Path(p) for p in out.stdout.decode().split("\0") if p and (ROOT / p).is_file()]


def release_weights(cache):
    """A folder laid out like models/ holding exactly the release checkpoints."""
    missing = []
    for asset, (target, expected) in WEIGHTS.items():
        dest = cache / target
        if dest.exists() and sha256(dest) == expected:
            continue
        for candidate in (ROOT / target, ROOT / "models" / "release" / asset):
            if candidate.exists() and sha256(candidate) == expected:
                print(f"{target} <- {candidate.relative_to(ROOT)}")
                copy(candidate, dest)
                break
        else:
            missing.append(target)
    if missing:
        print(f"Downloading {len(missing)} release file(s) into {cache.relative_to(ROOT)}")
        subprocess.run([sys.executable, "-m", "src.download_weights", "--root", str(cache)], cwd=ROOT, check=True)
    return cache / "models"


def compute_results(out, images, study, device_name, cache):
    from backend.config import Settings
    from backend.services.classifier import ClassifierService
    from backend.services.device import resolve_device
    from backend.services.samples import SampleResults
    from backend.services.segmenter import SegmenterService

    models = release_weights(cache)
    settings = Settings.from_env()
    device = resolve_device(device_name)
    classifier_ckpt = models / "classification_crop_resdegrade_long" / "best.pt"
    seg_dir = models / "segmentation"
    print(f"Computing the sample results on {device} (a few minutes on a CPU)")
    classifier = ClassifierService(classifier_ckpt, device)
    segmenter = SegmenterService(seg_dir, settings.segmentation_folds, device, settings.et_min_voxels,
                                 settings.result_cache_size, settings.max_volume_mb)
    checkpoints = [classifier_ckpt] + [seg_dir / f"fold{k}" / "best.pt" for k in settings.segmentation_folds]
    # The images are read from the bundle, as the service will; the study from data/, since it stays here.
    samples = SampleResults(out / "models" / "sample_results.npz", classifier, segmenter, checkpoints,
                            {name: out / p.relative_to(ROOT) for name, p in images.items()}, study)
    samples.compute()


def check(out):
    """Stop if the bundle holds raw volumes or unusable sample results."""
    from backend.config import tuned_et_threshold
    from backend.routers.classification import SAMPLES
    from backend.services.samples import check_release_file

    raw = [p for p in out.rglob("*") if p.is_file() and p.name.lower().endswith(RAW_SUFFIXES)]
    if raw or (out / "data" / "raw").exists():
        sys.exit(f"The bundle must not contain raw volumes, found: {[str(p.relative_to(out)) for p in raw]}")
    problems = check_release_file(out / "models" / "sample_results.npz", SAMPLES, tuned_et_threshold())
    if problems:
        sys.exit(f"Sample results unusable: {'; '.join(problems)}")


def archive(out, dest):
    """out as a .tar.gz with one top folder, ARCHIVE_ROOT; ownership and modes as on Linux."""
    def normalise(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mode = 0o755 if info.isdir() else 0o644
        return info

    with tarfile.open(dest, "w:gz") as tar:
        tar.add(out, arcname=ARCHIVE_ROOT, filter=normalise)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "vm", help="output folder")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda, for computing the sample results")
    parser.add_argument("--weights-cache", type=Path, default=ROOT / "build" / "release-weights",
                        help="where release weights not found locally are downloaded")
    args = parser.parse_args()
    out = args.out.resolve()
    tarball = out.parent / f"{ARCHIVE_ROOT}.tar.gz"

    from backend.routers.classification import SAMPLES, sample_paths
    from backend.routers.segmentation import SAMPLE_CASE, sample_files

    images, study = sample_paths(), sample_files()
    if len(images) != len(SAMPLES):
        sys.exit(f"Sample images missing from the local test set: {sorted(set(SAMPLES) - set(images))}. "
                 "See data/README.md for preparing the data.")
    if not all(p.exists() for p in study.values()):
        sys.exit(f"The sample study {SAMPLE_CASE} is missing from data/raw/brats2021/; its result is computed "
                 "from it here (the files themselves are not bundled).")

    if out.exists():
        # Only ever replace an earlier bundle, recognised by its marker.
        marker = out / MARKER
        if any(out.iterdir()) and not (marker.exists() and marker.read_text(encoding="utf-8") == MARKER_TEXT):
            sys.exit(f"{out} exists and is not a bundle made by this script; choose another --out.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for path in code_files():
        copy(ROOT / path, out / path)
    dockerfile = (ROOT / "backend" / "Dockerfile").read_text(encoding="utf-8")
    (out / "Dockerfile").write_text(dockerfile.rstrip("\n") + "\n" + DEPLOY_SETTINGS, encoding="utf-8",
                                    newline="\n")
    for name in STACK:
        text = (ROOT / "deploy" / "vm" / name).read_text(encoding="utf-8")
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    (out / MARKER).write_text(MARKER_TEXT, encoding="utf-8", newline="\n")
    for p in images.values():
        copy(p, out / p.relative_to(ROOT))

    hero = ROOT / "frontend" / "public" / "generated" / "hero-3d.png"
    if hero.exists():
        copy(hero, out / "data" / "hero" / "hero-3d.png")
    else:
        print("No hero render (python -m src.visualization.render_hero); /api/hero.png will answer 404.")

    compute_results(out, images, study, args.device, args.weights_cache.resolve())
    check(out)
    archive(out, tarball)

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**20
    print(f"\nBundle ready: {out} ({size:.0f} MB); no raw volumes, sample results match the release weights")
    print(f"Archive to copy to the VM: {tarball} ({tarball.stat().st_size / 2**20:.0f} MB)")
    print("On the VM, in the extracted folder:  docker compose up -d --build  (DEPLOYMENT.md)")


if __name__ == "__main__":
    main()
