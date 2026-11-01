"""Headline test results for the web client, read from reports/results/ (the evaluation outputs).

Nothing here is hard-coded: segmentation numbers come from segmentation_test.json, and the
classification numbers from the classification_test_*.json whose checkpoint is the one the
API is serving, so the page always describes the model that answers requests.
"""
import json
from collections import Counter

from fastapi import APIRouter, Request

from backend.schemas.common import ErrorResponse
from backend.services.errors import ApiError, relative_name
from src.config import PROJECT_ROOT
from src.training.bootstrap_results import wilson

router = APIRouter(prefix="/api", tags=["results"])
RESULTS = PROJECT_ROOT / "reports" / "results"


def _load(name):
    try:
        return json.loads((RESULTS / name).read_text())
    except (OSError, ValueError) as exc:
        raise ApiError(503, "results_unavailable", f"Could not read reports/results/{name} ({exc}).") from exc


def summary(classifier_checkpoint):
    seg = _load("segmentation_test.json")
    ens = seg["test"]["ensemble"]["post_processed"]
    served = relative_name(classifier_checkpoint)
    clf_file = next((p.name for p in sorted(RESULTS.glob("classification_test*.json"))
                     if json.loads(p.read_text()).get("checkpoint") == served), None)
    if clf_file is None:
        raise ApiError(503, "results_unavailable", f"No test results in reports/results/ for the served classifier {served}.")
    clf = _load(clf_file)
    accuracy, n = clf["test"]["test/accuracy"], clf["test_images"]
    stress = _load("format_stress_test.json")["models"].get(served, {})
    # Test errors per unordered class pair (glioma read as meningioma and the reverse count together).
    pairs = Counter(tuple(sorted((e["true"], e["predicted"]))) for e in clf["misclassified"])
    return {
        "segmentation": {
            "test_patients": seg["test_cases"],
            "model": "3D U-Net, 5-fold ensemble, ET post-processing",
            "dice": {"mean": ens["mean"], "WT": ens["WT"], "TC": ens["TC"], "ET": ens["ET"]},
        },
        "classification": {
            "test_images": n,
            "model": "EfficientNetV2-S",
            "checkpoint": served,
            "accuracy": accuracy,
            "accuracy_ci95": wilson(accuracy, n),
            "macro_f1": clf["test"]["test/macro_f1"],
            "accuracy_at_128px": stress.get("tiny_square", {}).get("accuracy"),
            "checkpoint_epoch": clf["checkpoint_epoch"],
            "test_errors": len(clf["misclassified"]),
            "confusions": [{"classes": list(pair), "errors": count}
                           for pair, count in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))],
        },
        "sources": ["reports/results/segmentation_test.json", f"reports/results/{clf_file}",
                    "reports/results/format_stress_test.json"],
    }


@router.get("/results/summary", responses={503: {"model": ErrorResponse}},
            summary="Headline held-out test results for both models")
def results_summary(request: Request):
    return summary(request.app.state.settings.classifier_checkpoint)
