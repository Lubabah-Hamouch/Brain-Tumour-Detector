"""Unit tests for the in-process job store (no models needed)."""
import threading
import time

import pytest

from backend.services.errors import ApiError
from backend.services.jobs import JobStore

STAGES = (("prep", "Prepare", 0.2), ("work", "Work", 0.8))


def wait(store, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = store.get(job_id)
        if snap["status"] in ("complete", "failed"):
            return snap
        time.sleep(0.01)
    raise AssertionError("job did not finish")


@pytest.fixture
def store():
    s = JobStore(keep=4)
    yield s
    s.shutdown()


def test_progress_and_result(store):
    reached, gate = threading.Event(), threading.Event()

    def work(progress):
        progress("prep", 0.0, "starting")
        progress("work", 0.5, "half")
        reached.set()
        gate.wait(2)
        return {"answer": 42}

    job_id = store.submit("test", STAGES, work)
    assert reached.wait(2)
    mid = store.get(job_id)
    gate.set()
    final = wait(store, job_id)
    assert mid["status"] == "running" and mid["stage"] == "work" and mid["stage_detail"] == "half"
    assert [s["status"] for s in mid["stages"]] == ["complete", "running"]
    assert mid["progress"] == pytest.approx(0.2 + 0.8 * 0.5)
    assert final["status"] == "complete" and final["result"] == {"answer": 42} and final["progress"] == 1.0
    assert all(s["status"] == "complete" for s in final["stages"])


def test_api_error_marks_job_and_stage_failed(store):
    def work(progress):
        progress("work", 0.3)
        raise ApiError(422, "bad_volume", "volume is empty")

    final = wait(store, store.submit("test", STAGES, work))
    assert final["status"] == "failed" and final["error"] == {"code": "bad_volume", "message": "volume is empty"}
    assert final["result"] is None
    assert [s["status"] for s in final["stages"]] == ["pending", "failed"]


def test_unexpected_exception_is_internal_error(store):
    final = wait(store, store.submit("test", STAGES, lambda progress: 1 / 0))
    assert final["status"] == "failed" and final["error"]["code"] == "internal_error"


def test_jobs_run_one_at_a_time_in_order(store):
    gate = threading.Event()
    order = []
    first = store.submit("test", STAGES, lambda p: (order.append(1), gate.wait(2)) and {})
    second = store.submit("test", STAGES, lambda p: order.append(2) or {})
    time.sleep(0.1)
    assert store.get(first)["status"] == "running"
    snap = store.get(second)
    assert snap["status"] == "queued" and snap["queue_position"] == 1
    gate.set()
    wait(store, second)
    assert order == [1, 2]


def test_oldest_finished_jobs_are_evicted(store):
    ids = []
    for _ in range(6):
        ids.append(store.submit("test", STAGES, lambda p: {}))
        wait(store, ids[-1])
    # keep=4: eviction runs on submit, so the two oldest finished jobs are gone
    assert [store.get(i) is not None for i in ids] == [False, False, True, True, True, True]
    assert store.get("missing") is None
